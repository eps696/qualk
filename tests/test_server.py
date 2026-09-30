"""The web app: auth, secret handling, SSRF guard, path safety and a full run over REST + WebSocket."""
import asyncio
import json
import os
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import patch

os.environ.pop("MOTH_API_KEY", None)

from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from qualk import security
from qualk.server import create_app

SECRET = 'sk-very-secret-value-zq9x'
BODY = {'seed': 'a seed passage', 'topic': '', 'rounds': 3, 'nodes': 6, 'steps': 3, 'shots': 128,
        'time': 2.0, 'explore': 1.0, 'rng': 3}


def wait_for(cond, timeout=60., what='condition'):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return
        time.sleep(0.03)
    raise AssertionError(f'timed out waiting for {what}')


class AppCase(unittest.TestCase):
    token = None

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix='qualk_server_')
        self.env = os.path.join(self.tmp, 'env')
        self.patch = patch.dict(os.environ, {}, clear=False)
        self.patch.start()
        self.app = create_app(os.path.join(self.tmp, 'runs'), token=self.token, simulate=True,
                              env_path=self.env, max_rounds=20, sim_delay=0.0)
        self.client = TestClient(self.app)

    def tearDown(self):
        self.app.state.manager.stop_all(20)
        self.patch.stop()

    def run_state(self, run_id):
        for s in self.client.get('/api/state').json()['status']:
            if s['run'] == run_id:
                return s['state']


class LifecycleTests(AppCase):
    def test_full_run_over_rest_and_websocket(self):
        with self.client.websocket_connect('/ws') as ws:
            hello = ws.receive_json()
            self.assertEqual(hello['type'], 'hello')
            self.assertTrue(hello['simulate'])
            r = self.client.post('/api/runs', json={**BODY, 'name': 'demo'})
            self.assertEqual(r.status_code, 200, r.text)
            self.assertEqual(r.json()['ids'], ['demo'])
            seen = []
            while True:
                m = ws.receive_json()
                if m['type'] == 'round':
                    seen.append(m['record']['kind'])
                if m['type'] == 'status' and m['state'] == 'done':
                    break
            self.assertEqual(seen, ['seed', 'seed'][:seen.count('seed')] + ['probe'] * 3)
        data = self.client.get('/api/runs/demo').json()
        self.assertEqual(data['id'], 'demo')
        self.assertEqual(sum(1 for r in data['rounds'] if r['kind'] == 'probe'), 3)
        self.assertGreater(len(data['nodes']), 3)
        walked = [r for r in data['rounds'] if r['kind'] == 'probe' and (r['probe']['walk'] or {}).get('qasm_path')]
        self.assertTrue(walked)
        qasm = self.client.get(f"/api/runs/demo/qasm/{walked[0]['round']}")
        self.assertEqual(qasm.status_code, 200)
        self.assertIn('OPENQASM', qasm.text)
        self.assertEqual(self.client.get('/api/runs/demo/qasm/999').status_code, 404)
        wait_for(lambda: not self.app.state.manager.active())
        listing = self.client.get('/api/runs').json()['runs']
        self.assertEqual(listing[0]['id'], 'demo')
        self.assertEqual(listing[0]['round'], 3)

    def test_errors_map_to_http_codes(self):
        self.assertEqual(self.client.post('/api/runs', json={**BODY, 'rounds': 0}).status_code, 400)
        self.assertEqual(self.client.post('/api/runs', json={**BODY, 'rounds': 21}).status_code, 400)
        self.assertEqual(self.client.post('/api/runs/none/pause').status_code, 404)
        self.assertEqual(self.client.get('/api/runs/nope').status_code, 404)
        r = self.client.post('/api/runs', json={**BODY, 'name': 'a', 'start_paused': True, 'rounds': 5})
        self.assertEqual(r.status_code, 200)
        wait_for(lambda: self.run_state('a') == 'paused')
        self.assertEqual(self.client.post('/api/runs', json={**BODY, 'name': 'b'}).status_code, 409)
        self.assertEqual(self.client.post('/api/runs/a/params', json={'steps': 999}).status_code, 400)
        self.assertEqual(self.client.post('/api/runs/a/inject', json={'url': 'http://127.0.0.1/x'}).status_code, 400)
        self.assertEqual(self.client.post('/api/runs/a/explode').status_code, 404)
        self.assertEqual(self.client.post('/api/runs/a/stop').status_code, 200)

    def test_path_traversal_ids_are_refused(self):
        for bad in ('..', '.hidden', 'A B', 'x' * 80):
            self.assertEqual(self.client.get(f'/api/runs/{bad}').status_code, 404, bad)
            self.assertEqual(self.client.get(f'/api/runs/{bad}/qasm/1').status_code, 404, bad)
        self.assertEqual(self.client.get('/api/runs/%2e%2e%2f%2e%2e/qasm/1').status_code, 404)

    def test_ui_files_are_served_with_security_headers(self):
        page = self.client.get('/')
        self.assertEqual(page.status_code, 200)
        self.assertIn('qualk', page.text)
        self.assertIn("script-src 'self'", page.headers['content-security-policy'])
        self.assertEqual(page.headers['x-frame-options'], 'DENY')
        for path in ('/static/app.js', '/static/graph.js', '/static/app.css', '/static/vendor/d3.min.js'):
            self.assertEqual(self.client.get(path).status_code, 200, path)
        self.assertNotIn('<script>', page.text.replace('<script src', ''), 'no inline scripts (CSP)')


class SettingsTests(AppCase):
    def test_secrets_are_write_only(self):
        r = self.client.put('/api/settings', json={'set': {'TAVILY_API_KEY': SECRET, 'QUALK_LLM_MODEL': 'my-model'}})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(sorted(r.json()['changed']), ['QUALK_LLM_MODEL', 'TAVILY_API_KEY'])
        for path in ('/api/settings', '/api/state', '/api/runs'):
            self.assertNotIn(SECRET, self.client.get(path).text, path)
            self.assertNotIn(SECRET[-4:], self.client.get(path).text, path)
        shown = self.client.get('/api/settings').json()['settings']
        self.assertTrue(shown['TAVILY_API_KEY']['set'])
        self.assertNotIn('value', shown['TAVILY_API_KEY'])
        self.assertEqual(shown['QUALK_LLM_MODEL']['value'], 'my-model')
        with open(self.env, encoding='utf-8') as f:
            self.assertIn(f'TAVILY_API_KEY={SECRET}', f.read())
        self.assertEqual(os.environ['TAVILY_API_KEY'], SECRET)

    def test_blank_secret_keeps_value_and_clear_removes_it(self):
        self.client.put('/api/settings', json={'set': {'TAVILY_API_KEY': SECRET}})
        self.client.put('/api/settings', json={'set': {'TAVILY_API_KEY': ''}})
        self.assertEqual(os.environ['TAVILY_API_KEY'], SECRET)
        self.client.put('/api/settings', json={'clear': ['TAVILY_API_KEY']})
        self.assertNotIn('TAVILY_API_KEY', os.environ)
        with open(self.env, encoding='utf-8') as f:
            self.assertNotIn('TAVILY_API_KEY', f.read())

    def test_invalid_settings_are_rejected(self):
        for bad in ({'NOT_A_KEY': 'x'}, {'QUALK_LLM_URL': 'a\nb'}, {'QUALK_LLM_URL': 'x' * 600}, {'QUALK_LLM_URL': 5}):
            self.assertEqual(self.client.put('/api/settings', json={'set': bad}).status_code, 400, bad)

    def test_settings_locked_while_a_run_is_active(self):
        self.client.post('/api/runs', json={**BODY, 'name': 'a', 'start_paused': True, 'rounds': 5})
        wait_for(lambda: self.run_state('a') == 'paused')
        self.assertEqual(self.client.put('/api/settings', json={'set': {'QUALK_LLM_MODEL': 'x'}}).status_code, 409)
        self.assertTrue(self.client.get('/api/settings').json()['locked'])
        self.client.post('/api/runs/a/stop')

    def test_scrub_removes_secrets_from_messages(self):
        from qualk.settings import scrub
        os.environ['MOTH_API_KEY'] = SECRET
        self.assertNotIn(SECRET, scrub(f'HTTPError for key {SECRET} at host'))


class AuthTests(AppCase):
    token = 'correct-horse-battery'

    def test_api_requires_the_token(self):
        self.assertEqual(self.client.get('/api/state').status_code, 401)
        self.assertEqual(self.client.get('/api/runs').status_code, 401)
        self.assertEqual(self.client.post('/api/runs', json=BODY).status_code, 401)
        self.assertEqual(self.client.get('/api/settings').status_code, 401)
        self.assertEqual(self.client.get('/api/auth').json(), {'required': True, 'ok': False})
        self.assertEqual(self.client.get('/').status_code, 200, 'the login page itself is public')

    def test_bearer_and_cookie_login(self):
        self.assertEqual(self.client.get('/api/state', headers={'Authorization': 'Bearer ' + self.token}).status_code, 200)
        self.assertEqual(self.client.get('/api/state', headers={'Authorization': 'Bearer wrong'}).status_code, 401)
        bad = self.client.post('/api/login', json={'token': 'wrong'})
        self.assertEqual(bad.status_code, 401)
        good = self.client.post('/api/login', json={'token': self.token})
        self.assertEqual(good.status_code, 200)
        self.assertIn('httponly', good.headers['set-cookie'].lower())
        self.assertIn('samesite=strict', good.headers['set-cookie'].lower())
        self.assertEqual(self.client.get('/api/state').status_code, 200)
        self.assertEqual(self.client.get('/api/auth').json()['ok'], True)
        self.client.post('/api/logout')
        self.assertEqual(self.client.get('/api/state').status_code, 401)

    def test_login_is_rate_limited(self):
        codes = [self.client.post('/api/login', json={'token': f'x{i}'}).status_code for i in range(12)]
        self.assertEqual(codes[:10], [401] * 10)
        self.assertEqual(codes[10], 429)
        self.assertEqual(self.client.post('/api/login', json={'token': self.token}).status_code, 429)

    def test_websocket_needs_the_token(self):
        with self.assertRaises(WebSocketDisconnect):
            with self.client.websocket_connect('/ws') as ws:
                ws.receive_json()
        with self.client.websocket_connect('/ws?token=' + self.token) as ws:
            self.assertEqual(ws.receive_json()['type'], 'hello')
        with self.assertRaises(WebSocketDisconnect):
            with self.client.websocket_connect('/ws?token=nope') as ws:
                ws.receive_json()

    def test_token_query_is_not_accepted_for_http(self):
        self.assertEqual(self.client.get('/api/state?token=' + self.token).status_code, 401)

    def test_loopback_detection(self):
        for host in ('127.0.0.1', 'localhost', '::1'):
            self.assertTrue(security.is_loopback_host(host), host)
        for host in ('0.0.0.0', '192.168.1.5', 'example.org', '::'):
            self.assertFalse(security.is_loopback_host(host), host)


class SsrfTests(unittest.TestCase):
    def test_check_url(self):
        for bad in ('file:///etc/passwd', 'ftp://example.org', 'http://127.0.0.1/', 'http://10.0.0.5/',
                    'http://169.254.169.254/latest/meta-data', 'http://[::1]/', 'http://user:pw@example.org/',
                    'http://0.0.0.0/', 'javascript:alert(1)', ''):
            with self.assertRaises(ValueError, msg=bad):
                security.check_url(bad)
        self.assertEqual(security.check_url('https://example.org/page'), 'https://example.org/page')

    def test_public_ip(self):
        for ip in ('127.0.0.1', '10.1.2.3', '192.168.0.1', '172.16.0.1', '169.254.169.254', '::1', 'fe80::1',
                   '::ffff:127.0.0.1', '0.0.0.0', 'fc00::1'):
            self.assertFalse(security.public_ip(ip), ip)
        for ip in ('93.184.216.34', '8.8.8.8', '2606:4700:4700::1111'):
            self.assertTrue(security.public_ip(ip), ip)

    def test_resolver_drops_private_answers(self):
        class Inner:
            def __init__(self, hosts):
                self.hosts = hosts

            async def resolve(self, host, port=0, family=0):
                return [{'hostname': host, 'host': h, 'port': port, 'family': 2, 'proto': 0, 'flags': 0} for h in self.hosts]

            async def close(self):
                pass

        async def go():
            mixed = security.PublicOnlyResolver(Inner(['127.0.0.1', '93.184.216.34']))
            self.assertEqual([i['host'] for i in await mixed.resolve('x.test')], ['93.184.216.34'])
            with self.assertRaises(OSError):
                await security.PublicOnlyResolver(Inner(['10.0.0.1', '127.0.0.1'])).resolve('rebind.test')
        asyncio.run(go())

    def test_a_local_server_is_never_contacted(self):
        hits = []

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                hits.append(self.path)
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'internal secret')

            def log_message(self, *a):
                pass

        server = HTTPServer(('127.0.0.1', 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        port = server.server_address[1]
        try:
            for url in (f'http://127.0.0.1:{port}/', f'http://localhost:{port}/'):
                with self.assertRaises(Exception):
                    asyncio.run(security.fetch_public_text(url))
        finally:
            server.shutdown()
            server.server_close()
        self.assertEqual(hits, [], 'the guard must refuse before any connection reaches a local service')


class RealBackendWithoutKeysTests(unittest.TestCase):
    def test_missing_search_key_is_a_clear_run_error(self):
        keys = ('TAVILY_API_KEY', 'SERPER_API_KEY', 'BRAVE_API_KEY')
        with patch.dict(os.environ, {}, clear=False):
            for k in keys:
                os.environ.pop(k, None)
            tmp = tempfile.mkdtemp(prefix='qualk_real_')
            app = create_app(os.path.join(tmp, 'runs'), simulate=False, env_path=os.path.join(tmp, 'env'))
            client = TestClient(app)
            try:
                self.assertFalse(client.get('/api/state').json()['simulate'])
                self.assertEqual(client.get('/api/state').json()['providers']['search'], False)
                self.assertEqual(client.post('/api/runs', json={**BODY, 'name': 'nokeys'}).status_code, 200)
                wait_for(lambda: any(s['state'] == 'error' for s in client.get('/api/state').json()['status']))
                status = client.get('/api/state').json()['status'][0]
                self.assertIn('No search key configured', status['error'])
                self.assertEqual(client.post('/api/settings/test/search').json()['ok'], False)
            finally:
                app.state.manager.stop_all(10)


class LauncherTests(unittest.TestCase):
    def launch(self, argv, env=None):
        import importlib.util
        import io
        from contextlib import redirect_stderr, redirect_stdout
        spec = importlib.util.spec_from_file_location('qualk_app_launcher', os.path.join(os.path.dirname(__file__), '..', 'app.py'))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        captured = {}
        out, err = io.StringIO(), io.StringIO()
        with patch('uvicorn.run', side_effect=lambda app, **kw: captured.update(kw)),                 patch('qualk.server.create_app', side_effect=lambda *a, **kw: captured.update(app_kwargs=kw)),                 patch.dict(os.environ, env or {}), redirect_stdout(out), redirect_stderr(err):
            if not env:
                os.environ.pop('QUALK_TOKEN', None)
            module.main(argv)
        return captured, out.getvalue(), err.getvalue()

    def test_no_login_by_default_on_loopback(self):
        captured, out, err = self.launch(['--host', '127.0.0.1', '--simulate'])
        self.assertIsNone(captured['app_kwargs']['token'])
        self.assertEqual(captured['host'], '127.0.0.1')
        self.assertNotIn('no access token', err)

    def test_no_token_is_generated_even_when_reachable_from_other_machines(self):
        captured, out, err = self.launch(['--simulate'])
        self.assertEqual(captured['host'], '0.0.0.0')
        self.assertIsNone(captured['app_kwargs']['token'])
        self.assertIn('no access token', err)              # said plainly, not enforced
        self.assertNotIn('token:', out)

    def test_a_token_is_still_available_as_an_option(self):
        captured, out, err = self.launch(['--token', 'mine', '--ssl-certfile', 'c.pem', '--ssl-keyfile', 'k.pem'])
        self.assertEqual(captured['app_kwargs']['token'], 'mine')
        self.assertNotIn('mine', out)
        self.assertNotIn('no access token', err)
        self.assertNotIn('no TLS', err)
        self.assertEqual(captured['ssl_certfile'], 'c.pem')

    def test_token_from_the_environment(self):
        captured, out, err = self.launch(['--simulate'], env={'QUALK_TOKEN': 'from-env'})
        self.assertEqual(captured['app_kwargs']['token'], 'from-env')


if __name__ == '__main__':
    unittest.main()
