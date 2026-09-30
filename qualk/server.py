"""The qualk web app: REST + WebSocket over the run manager, plus the static UI.

Security model: the default bind is loopback with no token. Anything else needs `token`, checked
on every /api call and on the WebSocket (Bearer header, HttpOnly cookie, or `?token=` on the
socket). Secret settings are write-only. See `security.py`.
"""

from __future__ import annotations

import asyncio
import os
import time
from contextlib import asynccontextmanager
from collections import defaultdict, deque
from typing import Any, Dict, Optional

from fastapi import Body, FastAPI, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from . import settings as S
from .runner import EventBus, RealBackend, RunManager, SimulatedBackend
from .security import COOKIE, TokenAuth
from .viewer import load_run

STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
PUBLIC_API = ('/api/auth', '/api/login', '/api/logout')
CSP = ("default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
       "connect-src 'self' ws: wss:; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")


class NoCacheStatic(StaticFiles):
    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        response.headers['Cache-Control'] = 'no-store'
        return response


class LoginLimiter:
    """At most `limit` failed logins per client per minute."""

    def __init__(self, limit: int = 10, window: float = 60.0):
        self.limit, self.window = limit, window
        self.failures: Dict[str, deque] = defaultdict(deque)

    def blocked(self, client: str) -> bool:
        q, now = self.failures[client], time.time()
        while q and now - q[0] > self.window:
            q.popleft()
        return len(q) >= self.limit

    def fail(self, client: str) -> None:
        self.failures[client].append(time.time())


def create_app(runs_root: str = 'runs', token: Optional[str] = None, simulate: bool = False,
               env_path: str = '.env', manager: Optional[RunManager] = None, max_rounds: int = 200,
               sim_delay: float = 0.3) -> FastAPI:
    bus = EventBus()
    backend = SimulatedBackend(delay=sim_delay) if simulate else RealBackend()
    manager = manager or RunManager(runs_root, backend, bus, max_rounds=max_rounds)
    backend, bus = manager.backend, manager.bus
    store = S.SettingsStore(env_path)
    auth = TokenAuth(token)
    limiter = LoginLimiter()
    @asynccontextmanager
    async def lifespan(_app):
        yield
        manager.stop_all(10)

    app = FastAPI(title='qualk', docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    app.state.manager, app.state.bus, app.state.settings, app.state.auth = manager, bus, store, auth

    @app.middleware('http')
    async def guard(request: Request, call_next):
        path = request.url.path
        if path.startswith('/api/') and path not in PUBLIC_API and not auth.from_request(request):
            return JSONResponse({'detail': 'authentication required'}, status_code=401)
        response = await call_next(request)
        response.headers.setdefault('X-Content-Type-Options', 'nosniff')
        response.headers.setdefault('Referrer-Policy', 'no-referrer')
        response.headers.setdefault('X-Frame-Options', 'DENY')
        response.headers.setdefault('Content-Security-Policy', CSP)
        if path.startswith('/api/'):
            response.headers['Cache-Control'] = 'no-store'
        return response

    # --- auth ------------------------------------------------------------------------------

    @app.get('/api/auth')
    def api_auth(request: Request):
        return {'required': auth.required, 'ok': auth.from_request(request)}

    @app.post('/api/login')
    def api_login(request: Request, response: Response, body: Dict[str, Any] = Body(...)):
        client = request.client.host if request.client else '?'
        if limiter.blocked(client):
            raise HTTPException(429, 'too many attempts, wait a minute')
        if not auth.verify(str(body.get('token') or '')):
            limiter.fail(client)
            raise HTTPException(401, 'wrong token')
        if auth.required:
            response.set_cookie(COOKIE, auth.token, httponly=True, samesite='strict',
                                secure=request.url.scheme == 'https', max_age=30 * 24 * 3600, path='/')
        return {'ok': True}

    @app.post('/api/logout')
    def api_logout(response: Response):
        response.delete_cookie(COOKIE, path='/')
        return {'ok': True}

    # --- state and settings ------------------------------------------------------------------

    def state() -> Dict[str, Any]:
        return {'simulate': backend.simulated, 'providers': store.providers(), 'limits': {'max_rounds': max_rounds},
                'status': manager.snapshot()['active'], 'auth_required': auth.required}

    @app.get('/api/state')
    def api_state():
        return state()

    @app.get('/api/settings')
    def api_settings():
        return {'settings': store.view(), 'providers': store.providers(), 'locked': bool(manager.active())}

    @app.put('/api/settings')
    def api_settings_put(body: Dict[str, Any] = Body(...)):
        if manager.active():
            raise HTTPException(409, 'settings cannot change while a run is active')
        try:
            changed = store.update(body.get('set') or {}, body.get('clear') or ())
        except ValueError as e:
            raise HTTPException(400, str(e))
        if 'QUALK_EMBED_MODEL' in changed:
            backend.reset_embedder()
        return {'changed': changed, 'settings': store.view(), 'providers': store.providers()}

    @app.post('/api/settings/test/{what}')
    async def api_settings_test(what: str):
        if backend.simulated:
            return {'ok': True, 'detail': 'simulated sources: nothing to test'}
        if what == 'search':
            return await S.check_search()
        if what == 'llm':
            return await S.check_llm()
        if what == 'atlas':
            return await run_in_threadpool(S.check_atlas)
        if what == 'embed':
            return await run_in_threadpool(S.check_embed, backend.embedder)
        raise HTTPException(404, 'unknown test')

    # --- runs --------------------------------------------------------------------------------

    def run_path(run_id: str) -> str:
        try:
            path = manager.run_dir(run_id)
        except ValueError:
            raise HTTPException(404, 'no such run')
        if not os.path.isdir(path):
            raise HTTPException(404, 'no such run')
        return path

    def guarded(fn, *args):
        try:
            return fn(*args)
        except ValueError as e:
            raise HTTPException(400, S.scrub(e))
        except RuntimeError as e:
            raise HTTPException(409, S.scrub(e))

    @app.get('/api/runs')
    def api_runs():
        return {'runs': manager.list_runs(), 'status': manager.snapshot()['active']}

    @app.get('/api/runs/{run_id}')
    async def api_run(run_id: str):
        path = run_path(run_id)
        data = await run_in_threadpool(load_run, path)
        data['id'] = run_id
        return data

    @app.get('/api/runs/{run_id}/qasm/{round_no}')
    def api_qasm(run_id: str, round_no: int):
        path = os.path.join(run_path(run_id), 'quantum', f'round-{int(round_no):05d}.qasm')
        if not os.path.isfile(path):
            raise HTTPException(404, 'no circuit was saved for that round')
        return FileResponse(path, media_type='text/plain', filename=f'{run_id}-round-{int(round_no):05d}.qasm')

    @app.delete('/api/runs/{run_id}')
    def api_delete(run_id: str):
        run_path(run_id)
        result = guarded(manager.delete, run_id)
        bus.publish({'type': 'deleted', 'runs': result['deleted'], 'leftover': result['leftover']})
        return result

    @app.post('/api/runs')
    def api_create(body: Dict[str, Any] = Body(...)):
        return {'ids': guarded(manager.create, body)}

    @app.post('/api/runs/{run_id}/continue')
    def api_continue(run_id: str, body: Dict[str, Any] = Body(default={})):
        run_path(run_id)
        try:
            rounds = int(body.get('rounds') or 10)
        except (TypeError, ValueError):
            raise HTTPException(400, 'rounds must be a number')
        return {'ids': guarded(manager.resume, run_id, rounds)}

    @app.post('/api/runs/{run_id}/{action}')
    def api_control(run_id: str, action: str, body: Dict[str, Any] = Body(default={})):
        if action not in ('pause', 'resume', 'step', 'stop', 'params', 'pin', 'inject', 'ask', 'focus', 'close_thread'):
            raise HTTPException(404, 'unknown action')
        run_path(run_id)
        return {'ids': guarded(manager.control, run_id, action, body)}

    # --- live events ---------------------------------------------------------------------------

    @app.websocket('/ws')
    async def ws(socket: WebSocket):
        if not auth.from_request(socket):
            await socket.close(code=4401)
            return
        await socket.accept()
        queue = bus.subscribe(asyncio.get_running_loop())
        try:
            await socket.send_json({'type': 'hello', **state(), 'recent': list(bus.recent)})
            while True:
                try:
                    message = await asyncio.wait_for(queue.get(), 20)
                except asyncio.TimeoutError:
                    message = {'type': 'ping'}
                await socket.send_json(message)
        except (WebSocketDisconnect, RuntimeError, asyncio.CancelledError):
            pass
        finally:
            bus.unsubscribe(queue)

    # --- UI ------------------------------------------------------------------------------------

    app.mount('/static', NoCacheStatic(directory=STATIC), name='static')

    @app.get('/')
    def index():
        return FileResponse(os.path.join(STATIC, 'index.html'), headers={'Cache-Control': 'no-store'})

    return app
