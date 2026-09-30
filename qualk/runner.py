"""Run orchestration for the web app: one worker thread per run, steered between rounds.

Why a thread per run: the walk blocks (local Qiskit; Atlas polling up to 90 s), the embedder is
CPU-bound and sqlite connections belong to the thread that opened them. Each `RunWorker` therefore
owns its event loop, engine and index; the web server only talks to it through the thread-safe
methods below and receives events through the `EventBus`.

A steering request (pause, parameters, a pinned seed, an injected page) is queued and takes effect
at the next round boundary. Nothing interrupts a round already in flight.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import threading
import time
import traceback
from collections import deque
from dataclasses import asdict, dataclass, fields
from typing import Any, Deque, Dict, List, Optional, Tuple

from .settings import scrub

WALKS = ('quantum', 'diffusion', 'classical')
BACKENDS = ('qiskit', 'atlas')
LIVE_PARAMS = ('walk', 'nodes', 'steps', 'time', 'shots', 'backend', 'explore', 'rounds', 'thread_aim', 'dedupe')
WALK_PARAMS = ('walk', 'nodes', 'steps', 'time', 'shots', 'backend')
MAX_TEXT = 20000
SEED_CHARS = 1600
RUN_ID = re.compile(r'^[a-z0-9][a-z0-9-]{0,63}$')
CTL_SUFFIX = '-ctl'
COPY_FILES = ('world.json', 'world.jsonl', 'semantic.sqlite', 'rounds.jsonl')
# Everything a run writes, and nothing else. Deleting a run removes exactly these; anything else found
# in the folder is left alone and reported.
RUN_FILES = ('config.json', 'world.json', 'world.json.tmp', 'world.jsonl', 'semantic.sqlite', 'semantic.sqlite-journal',
             'rounds.jsonl', 'steer.jsonl', 'index.html')
QASM_FILE = re.compile(r'^round-\d{5}\.qasm$')


def slug(text: str, limit: int = 32) -> str:
    return re.sub(r'[^a-z0-9]+', '-', str(text or '').lower()).strip('-')[:limit].strip('-')


# --- configuration ----------------------------------------------------------------------------

@dataclass
class RunConfig:
    name: str = ''
    topic: str = ''
    seed: str = ''
    rounds: int = 10
    walk: str = 'quantum'
    nodes: int = 12
    steps: int = 8
    time: float = 3.0
    shots: int = 1024
    backend: str = 'qiskit'
    explore: float = 0.4
    rng: int = 0
    paired: bool = False
    start_paused: bool = False
    threads: bool = True
    thread_aim: float = 0.3
    thread_cap: int = 24
    thread_decay: float = 0.95
    dedupe: float = 0.10

    @classmethod
    def from_dict(cls, data: Dict[str, Any], max_rounds: int = 200) -> 'RunConfig':
        known = {f.name: f for f in fields(cls)}
        unknown = set(data or {}) - set(known)
        if unknown:
            raise ValueError('unknown field(s): ' + ', '.join(sorted(unknown)))
        cfg = cls()
        for key, value in (data or {}).items():
            kind = type(getattr(cfg, key))
            try:
                if kind is bool:
                    if not isinstance(value, bool):
                        raise ValueError
                    setattr(cfg, key, value)
                elif kind is int:
                    if isinstance(value, bool):
                        raise ValueError
                    setattr(cfg, key, int(value))
                elif kind is float:
                    setattr(cfg, key, float(value))
                else:
                    setattr(cfg, key, str(value))
            except (TypeError, ValueError):
                raise ValueError(f'{key}: expected {kind.__name__}') from None
        cfg.validate(max_rounds)
        return cfg

    def validate(self, max_rounds: int = 200) -> None:
        if self.walk not in WALKS:
            raise ValueError(f'walk must be one of {", ".join(WALKS)}')
        if self.backend not in BACKENDS:
            raise ValueError(f'backend must be one of {", ".join(BACKENDS)}')
        if not 1 <= self.rounds <= max_rounds:
            raise ValueError(f'rounds must be between 1 and {max_rounds}')
        if not 2 <= self.nodes <= 24:
            raise ValueError('nodes must be between 2 and 24')
        if not 1 <= self.steps <= 20:
            raise ValueError('steps must be between 1 and 20')
        if not 1 <= self.shots <= 65536:
            raise ValueError('shots must be between 1 and 65536')
        if not (self.time > 0 and self.time < 1e6):
            raise ValueError('time must be positive and finite')
        if not 0 <= self.explore <= 1:
            raise ValueError('explore must be between 0 and 1')
        if not 0 <= self.dedupe <= 0.5:
            raise ValueError('dedupe must be between 0 (off) and 0.5')
        if not 0 <= self.thread_aim <= 1:
            raise ValueError('thread_aim must be between 0 and 1')
        if not 1 <= self.thread_cap <= 200:
            raise ValueError('thread_cap must be between 1 and 200')
        if not 0 < self.thread_decay <= 1:
            raise ValueError('thread_decay must be in (0, 1]')
        if len(self.topic) > 500 or len(self.seed) > MAX_TEXT or len(self.name) > 64:
            raise ValueError('topic, seed or name too long')
        if self.paired and self.walk not in ('quantum', 'diffusion'):
            raise ValueError('a paired run compares the quantum walk with its diffusion control')

    def merged(self, changes: Dict[str, Any], max_rounds: int = 200) -> 'RunConfig':
        bad = set(changes) - set(LIVE_PARAMS)
        if bad:
            raise ValueError('cannot change while running: ' + ', '.join(sorted(bad)))
        return RunConfig.from_dict({**asdict(self), **changes}, max_rounds)


def make_walk(cfg: RunConfig, trace_dir: str):
    from .quantum_walk import DiffusionProbeWalk, QuantumProbeWalk
    if cfg.walk == 'quantum':
        return QuantumProbeWalk(max_nodes=cfg.nodes, steps=cfg.steps, time=cfg.time, shots=cfg.shots,
                                backend=cfg.backend, trace_dir=trace_dir)
    if cfg.walk == 'diffusion':
        return DiffusionProbeWalk(max_nodes=cfg.nodes, time=cfg.time)
    return None


# --- events -----------------------------------------------------------------------------------

class EventBus:
    """Thread-safe fan-out from worker threads to asyncio queues (one per WebSocket)."""

    def __init__(self, keep: int = 200):
        self._subs: List[Tuple[asyncio.AbstractEventLoop, asyncio.Queue]] = []
        self._lock = threading.Lock()
        self.recent: Deque[Dict[str, Any]] = deque(maxlen=keep)     # log/error/steer lines for late joiners

    def subscribe(self, loop: asyncio.AbstractEventLoop, maxsize: int = 500) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize)
        with self._lock:
            self._subs.append((loop, queue))
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        with self._lock:
            self._subs = [(l, q) for l, q in self._subs if q is not queue]

    @staticmethod
    def _put(queue: asyncio.Queue, message: Dict[str, Any]) -> None:
        if queue.full():
            try:
                queue.get_nowait()          # a slow client loses its oldest event, never blocks a run
            except asyncio.QueueEmpty:
                pass
        queue.put_nowait(message)

    def publish(self, message: Dict[str, Any]) -> None:
        if message.get('type') in ('log', 'error', 'steer'):
            self.recent.append(message)
        with self._lock:
            subs = list(self._subs)
        dead = []
        for loop, queue in subs:
            try:
                loop.call_soon_threadsafe(self._put, queue, message)
            except RuntimeError:
                dead.append(queue)
        for queue in dead:
            self.unsubscribe(queue)


# --- backends: where the embedder, web and LLM come from ---------------------------------------

class _LockedEmbedder:
    """One encode at a time: two run threads share the model."""

    def __init__(self, inner):
        self._inner, self._lock = inner, threading.Lock()
        self.fingerprint = inner.fingerprint
        self.backend = getattr(inner, 'backend', '')
        self.model_name = getattr(inner, 'model_name', '')

    def encode_text(self, texts):
        with self._lock:
            return self._inner.encode_text(texts)


class RealBackend:
    simulated = False

    def __init__(self):
        self._embedder = None
        self._lock = threading.Lock()

    def embedder(self):
        with self._lock:
            if self._embedder is None:
                from .embed import STEmbedder, configure
                inner = STEmbedder()
                configure(inner)
                self._embedder = _LockedEmbedder(inner)
            return self._embedder

    def reset_embedder(self) -> None:
        with self._lock:
            self._embedder = None

    def parts(self):
        """(embedder, web source, extractor, thread extractor) for one run."""
        from . import web
        from .llm import llm_extractor, thread_extractor
        if not web.search_available():
            raise RuntimeError('No search key configured: open Settings and add one')
        return self.embedder(), web.WebSource(), llm_extractor(), thread_extractor()


class SimulatedBackend:
    """Synthetic embedder, pages and extraction: a keyless demo and the UI test bed."""
    simulated = True
    dedupe_override = 0.005          # the toy embedder's scale: only exact copies are duplicates

    def __init__(self, delay: float = 0.3, mirror_every: int = 5):
        self.delay = delay
        self.mirror_every = mirror_every

    def embedder(self):
        from .fakes import HashEmbedder
        return HashEmbedder()

    def reset_embedder(self) -> None:
        pass

    def parts(self):
        from .fakes import FakeWeb, HashEmbedder, fake_thread_extractor, question_extractor
        return (HashEmbedder(), FakeWeb(delay=self.delay, mirror_every=self.mirror_every), question_extractor(delay=self.delay),
                fake_thread_extractor(delay=self.delay))


# --- one run ----------------------------------------------------------------------------------

class Group:
    """The two runs of a paired experiment: a shared seed, then rounds in lockstep."""

    def __init__(self, fresh: bool):
        self.fresh = fresh
        self.workers: List['RunWorker'] = []
        self.seed_done = threading.Event()
        self.ctl_ready = threading.Event()

    def partner_behind(self, me: 'RunWorker') -> bool:
        return any(w is not me and w.is_alive() and w.state not in ('done', 'stopped', 'error')
                   and w.done_rounds < me.done_rounds for w in self.workers)


class RunWorker(threading.Thread):
    def __init__(self, run_id: str, out_dir: str, cfg: RunConfig, backend, bus: EventBus,
                 role: str = 'main', group: Optional[Group] = None, max_rounds: int = 200):
        super().__init__(daemon=True, name=f'qualk-{run_id}')
        self.run_id, self.out_dir, self.cfg, self.backend, self.bus = run_id, out_dir, cfg, backend, bus
        self.role, self.group, self.max_rounds = role, group, max_rounds
        self.state = 'starting'
        self.round = 0
        self.done_rounds = 0
        self.total = cfg.rounds
        self.error = ''
        self._cv = threading.Condition()
        self._paused = cfg.start_paused
        self._steps = 0
        self._halt = False
        self._actions: Deque[Tuple[str, Any]] = deque()
        self._pinned: Optional[str] = None
        self.engine = None

    # -- thread-safe control (called from the web server) --------------------------------

    def pause(self) -> None:
        with self._cv:
            self._paused = True
            self._cv.notify_all()

    def resume(self) -> None:
        with self._cv:
            self._paused = False
            self._cv.notify_all()

    def step(self) -> None:
        with self._cv:
            if self._paused:
                self._steps += 1
            self._cv.notify_all()

    def stop(self) -> None:
        with self._cv:
            self._halt = True
            self._cv.notify_all()
        if self.state not in ('done', 'stopped', 'error'):
            self._set_state('stopping')

    def submit(self, kind: str, payload: Any) -> None:
        with self._cv:
            self._actions.append((kind, payload))
            self._cv.notify_all()

    def status(self) -> Dict[str, Any]:
        return {'type': 'status', 'run': self.run_id, 'role': self.role, 'state': self.state,
                'round': self.round, 'done': self.done_rounds, 'total': self.total,
                'paired_with': [w.run_id for w in self.group.workers if w is not self] if self.group else [],
                'cfg': asdict(self.cfg), 'error': self.error, 'pending': len(self._actions),
                'pinned': self._pinned, 'focus': self.engine.focus_thread if self.engine else None}

    # -- internals -----------------------------------------------------------------------

    def _emit_status(self) -> None:
        self.bus.publish(self.status())

    def _set_state(self, state: str) -> None:
        if self.state != state:
            self.state = state
            self._emit_status()

    def _log(self, text: str, kind: str = 'log') -> None:
        self.bus.publish({'type': kind, 'run': self.run_id, 't': time.time(), 'text': scrub(text)})

    def _steer(self, kind: str, **payload) -> None:
        entry = {'t': round(time.time(), 2), 'round': self.round, 'kind': kind, **payload}
        with open(os.path.join(self.out_dir, 'steer.jsonl'), 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        self.bus.publish({'type': 'steer', 'run': self.run_id, **entry})

    def _write_config(self) -> None:
        path = os.path.join(self.out_dir, 'config.json')
        data = {'cfg': asdict(self.cfg), 'role': self.role,
                'paired_with': [w.run_id for w in self.group.workers if w is not self] if self.group else []}
        with open(path + '.tmp', 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        os.replace(path + '.tmp', path)

    def _gate(self) -> bool:
        """Block while paused (a queued step lets exactly one round through). False: stop."""
        with self._cv:
            while True:
                if self._halt:
                    return False
                if not self._paused:
                    break
                if self._steps > 0:
                    self._steps -= 1
                    break
                self._set_state('paused')
                self._cv.wait(0.25)
        self._set_state('running')
        return True

    def _build(self):
        from .engine import Engine
        from .semantic import SemanticIndex
        embedder, web, extractor, thread_extractor = self.backend.parts()
        index = SemanticIndex(os.path.join(self.out_dir, 'semantic.sqlite'), embedder)
        walk = make_walk(self.cfg, os.path.join(self.out_dir, 'quantum'))
        self.engine = Engine(self.out_dir, extractor, index, web, walk=walk,
                             explore=self.cfg.explore, seed=self.cfg.rng,
                             thread_extractor=thread_extractor, threads=self.cfg.threads,
                             thread_aim=self.cfg.thread_aim, thread_cap=self.cfg.thread_cap,
                             thread_decay=self.cfg.thread_decay,
                             dedupe=getattr(self.backend, 'dedupe_override', None) or self.cfg.dedupe)
        self.round = self.engine.round
        return index

    async def _seed(self) -> None:
        from .web import Parcel
        engine = self.engine
        parcels = []
        if self.cfg.seed.strip():
            parcels.append(Parcel(text=self.cfg.seed.strip()[:SEED_CHARS], origin='seed:text', source='seed'))
        if self.cfg.topic.strip():
            for i in range(2):
                got = await engine.web.harvest(self.cfg.topic.strip(), accept=engine._duplicate_check)
                for page in got:
                    engine.remember_page(page, f'seedweb{i}')
                parcels += got
        if not parcels:
            raise RuntimeError('nothing to seed the graph from: the topic search returned no page and no seed text was given')
        self._log(f'seeding the graph from {len(parcels)} passage(s)')
        for record in await engine.seed(parcels):
            self.bus.publish({'type': 'round', 'run': self.run_id, 'record': record})
        self.round = engine.round

    async def _apply_actions(self) -> None:
        from .web import Parcel
        while True:
            with self._cv:
                if not self._actions:
                    return
                kind, payload = self._actions.popleft()
            try:
                if kind == 'params':
                    changes = dict(payload)
                    if self.role == 'ctl':
                        changes.pop('walk', None)          # the control stays a diffusion walk
                    new = self.cfg.merged(changes, self.max_rounds)
                    if new.rounds < self.done_rounds:
                        raise ValueError(f'rounds cannot be below the {self.done_rounds} already run')
                    if any(getattr(new, k) != getattr(self.cfg, k) for k in WALK_PARAMS):
                        self.engine.set_walk(make_walk(new, os.path.join(self.out_dir, 'quantum')))
                    self.engine.set_explore(new.explore)
                    self.engine.thread_aim = new.thread_aim
                    if not getattr(self.backend, 'dedupe_override', None):
                        self.engine.set_dedupe(new.dedupe)
                    self.cfg, self.total = new, new.rounds
                    self._write_config()
                    self._steer('params', changes=changes)
                elif kind == 'pin':
                    node = self.engine.graph.nodes.get(payload)
                    if node is None or node.kind == 'thread':
                        raise ValueError(f'no such concept: {payload}')
                    self._pinned = payload
                    self._steer('pin', node=payload, name=node.name)
                elif kind in ('ask', 'focus', 'close_thread'):
                    self._thread_action(kind, payload)
                elif kind == 'inject':
                    text = payload['text']
                    origin = 'inject:text'
                    if payload.get('url'):
                        from .security import fetch_public_text
                        text = await fetch_public_text(payload['url'])
                        origin = 'inject:web:' + payload['url']
                    self._steer('inject', source=origin, chars=len(text))
                    record = await self.engine.inject(Parcel(text=text, origin=origin, source='inject'))
                    self.round = self.engine.round
                    self.bus.publish({'type': 'round', 'run': self.run_id, 'record': record})
                self._emit_status()
            except Exception as e:
                self._log(f'{kind} rejected: {e}', 'error')

    def _publish_threads(self) -> None:
        """Right after a user action on threads, before any round record carries the new table."""
        self.bus.publish({'type': 'threads', 'run': self.run_id, 'round': self.engine.round,
                          'table': self.engine.keeper.table(self.engine.graph, self.engine.round),
                          'focus': self.engine.focus_thread})

    def _thread_action(self, kind: str, payload: Dict[str, Any]) -> None:
        engine = self.engine
        if not engine.threads_on:
            raise ValueError('threads are switched off for this run')
        if kind == 'ask':
            tid = engine.ask(payload['question'], payload.get('involves'))
            self._steer('ask', thread=tid, question=payload['question'][:200])
            if payload.get('focus'):
                engine.set_focus(tid)
                self._steer('focus', thread=tid)
        elif kind == 'focus':
            tid = payload.get('thread_id') or None
            engine.set_focus(tid)
            self._steer('focus', thread=tid)
        else:
            tid = payload['thread_id']
            engine.close_thread(tid)
            if engine.focus_thread == tid:
                engine.focus_thread = None
            self._steer('close_thread', thread=tid)
        self._publish_threads()

    async def _rounds(self) -> None:
        engine = self.engine
        while self.done_rounds < self.total:
            if not self._gate():
                return
            await self._apply_actions()
            if self.done_rounds >= self.total:
                return
            pinned, self._pinned = self._pinned, None
            record = await engine.step(pinned_seed=pinned)
            self.round, self.done_rounds = engine.round, self.done_rounds + 1
            self.bus.publish({'type': 'round', 'run': self.run_id, 'record': record})
            self._emit_status()
            while (self.group and self.group.partner_behind(self)) and not self._halt:
                time.sleep(0.05)              # lockstep with the paired run

    async def _main(self) -> None:
        index = None
        try:
            group = self.group
            if group and self.role == 'ctl' and group.fresh:
                if not group.seed_done.wait(timeout=1800):
                    raise RuntimeError('the main run never finished seeding')
                main_dir = next(w.out_dir for w in group.workers if w.role == 'main')
                if self._halt:
                    return
                for name in COPY_FILES:
                    if os.path.isfile(os.path.join(main_dir, name)):
                        shutil.copy2(os.path.join(main_dir, name), os.path.join(self.out_dir, name))
            index = self._build()
            if self.engine.graph.nodes:
                self.total = self.done_rounds + self.cfg.rounds
            elif self.role == 'ctl':
                raise RuntimeError('the control has no seed graph to start from')
            else:
                self._set_state('seeding')
                await self._seed()
            if group:
                if self.role == 'main':
                    group.seed_done.set()
                    if group.fresh and not group.ctl_ready.wait(timeout=300):
                        raise RuntimeError('the control run did not start')
                else:
                    group.ctl_ready.set()
            self._write_config()
            self._log(f'{self.run_id}: starting {self.cfg.rounds} round(s), {self.cfg.walk} walk')
            await self._rounds()
            self._set_state('stopped' if self._halt else 'done')
        except Exception as e:
            self.error = scrub(f'{type(e).__name__}: {e}')
            self._log(self.error + '\n' + scrub(''.join(traceback.format_exc().splitlines(True)[-6:])), 'error')
            self._set_state('error')
        finally:
            if self.group:
                self.group.seed_done.set()
                self.group.ctl_ready.set()
            if index is not None:
                index.close()
            self._emit_status()

    def run(self) -> None:
        asyncio.run(self._main())


# --- the set of runs --------------------------------------------------------------------------

class RunManager:
    def __init__(self, runs_root: str, backend, bus: EventBus, max_rounds: int = 200):
        self.runs_root = os.path.abspath(runs_root)
        os.makedirs(self.runs_root, exist_ok=True)
        self.backend, self.bus, self.max_rounds = backend, bus, max_rounds
        self.workers: Dict[str, RunWorker] = {}
        self._lock = threading.Lock()

    # -- lookup ---------------------------------------------------------------------------

    def run_dir(self, run_id: str) -> str:
        if not RUN_ID.match(str(run_id or '')):
            raise ValueError('invalid run id')
        path = os.path.abspath(os.path.join(self.runs_root, run_id))
        if os.path.dirname(path) != self.runs_root:
            raise ValueError('invalid run id')
        return path

    def active(self) -> List[RunWorker]:
        return [w for w in self.workers.values() if w.is_alive()]

    def snapshot(self) -> Dict[str, Any]:
        return {'active': [w.status() for w in self.workers.values()]}

    def _group_of(self, run_id: str) -> List[RunWorker]:
        worker = self.workers.get(run_id)
        if worker is None or not worker.is_alive():
            raise RuntimeError('that run is not active')
        return worker.group.workers if worker.group else [worker]

    def list_runs(self) -> List[Dict[str, Any]]:
        out = []
        for name in sorted(os.listdir(self.runs_root)):
            path = os.path.join(self.runs_root, name)
            if not RUN_ID.match(name) or not os.path.isdir(path):
                continue
            entry: Dict[str, Any] = {'id': name, 'state': 'idle', 'round': 0, 'nodes': 0, 'relations': 0,
                                     'role': 'main', 'paired_with': [], 'cfg': {},
                                     'mtime': os.path.getmtime(path)}
            cfg_path = os.path.join(path, 'config.json')
            if os.path.isfile(cfg_path):
                try:
                    with open(cfg_path, encoding='utf-8') as f:
                        meta = json.load(f)
                    entry.update(role=meta.get('role', 'main'), paired_with=meta.get('paired_with', []),
                                 cfg=meta.get('cfg', {}))
                except (OSError, json.JSONDecodeError):
                    pass
            last = _last_record(os.path.join(path, 'rounds.jsonl'))
            if last:
                entry.update(round=last.get('round', 0), nodes=(last.get('graph') or {}).get('nodes', 0),
                             relations=(last.get('graph') or {}).get('relations', 0))
            worker = self.workers.get(name)
            if worker is not None and worker.is_alive():
                entry['state'] = worker.state
            out.append(entry)
        out.sort(key=lambda e: -e['mtime'])
        return out

    # -- lifecycle ------------------------------------------------------------------------

    def create(self, data: Dict[str, Any]) -> List[str]:
        cfg = RunConfig.from_dict(data, self.max_rounds)
        if cfg.paired and cfg.walk != 'quantum':
            raise ValueError('a paired run compares the quantum walk with its diffusion control: walk must be quantum')
        with self._lock:
            if self.active():
                raise RuntimeError('a run is already active: stop it first')
            base = slug(cfg.name) or slug(cfg.topic, 24) or 'run'
            run_id = base if cfg.name.strip() else f'{base}-{time.strftime("%m%d-%H%M%S")}'
            if not RUN_ID.match(run_id):
                raise ValueError('invalid run name')
            ids = [run_id] + ([run_id + CTL_SUFFIX] if cfg.paired else [])
            for rid in ids:
                if os.path.exists(self.run_dir(rid)):
                    raise ValueError(f'a run named {rid!r} already exists')
            group = Group(fresh=True) if cfg.paired else None
            workers = []
            for rid in ids:
                ctl = rid.endswith(CTL_SUFFIX)
                wcfg = RunConfig.from_dict({**asdict(cfg), 'walk': 'diffusion' if ctl else cfg.walk,
                                            'paired': cfg.paired}, self.max_rounds) if ctl else cfg
                os.makedirs(self.run_dir(rid))
                workers.append(RunWorker(rid, self.run_dir(rid), wcfg, self.backend, self.bus,
                                         role='ctl' if ctl else 'main', group=group, max_rounds=self.max_rounds))
            if group:
                group.workers = workers
            self._start(workers)
            return ids

    def resume(self, run_id: str, rounds: int = 10) -> List[str]:
        with self._lock:
            if self.active():
                raise RuntimeError('a run is already active: stop it first')
            path = self.run_dir(run_id)
            cfg_path = os.path.join(path, 'config.json')
            if not os.path.isfile(cfg_path):
                raise ValueError('that run has no saved configuration')
            with open(cfg_path, encoding='utf-8') as f:
                meta = json.load(f)
            ids = [run_id] if not meta.get('paired_with') else sorted({run_id, *meta['paired_with']})
            group = Group(fresh=False) if len(ids) > 1 else None
            workers = []
            for rid in ids:
                with open(os.path.join(self.run_dir(rid), 'config.json'), encoding='utf-8') as f:
                    m = json.load(f)
                cfg = RunConfig.from_dict({**m['cfg'], 'rounds': rounds, 'start_paused': False}, self.max_rounds)
                workers.append(RunWorker(rid, self.run_dir(rid), cfg, self.backend, self.bus,
                                         role=m.get('role', 'main'), group=group, max_rounds=self.max_rounds))
            if group:
                group.workers = workers
            self._start(workers)
            return ids

    def _start(self, workers: List[RunWorker]) -> None:
        for w in workers:
            self.workers[w.run_id] = w
        for w in workers:
            w.start()
        for w in workers:
            w._emit_status()

    # -- steering -------------------------------------------------------------------------

    def control(self, run_id: str, action: str, payload: Optional[Dict[str, Any]] = None) -> List[str]:
        group = self._group_of(run_id)
        payload = payload or {}
        if action in ('pause', 'resume', 'step', 'stop'):
            for w in group:
                getattr(w, action)()
        elif action == 'params':
            if not isinstance(payload, dict) or not payload:
                raise ValueError('no parameters given')
            for w in group:          # validate against the first worker's config before queueing on any
                w.cfg.merged({k: v for k, v in payload.items() if not (w.role == 'ctl' and k == 'walk')},
                             self.max_rounds)
            for w in group:
                w.submit('params', payload)
        elif action == 'pin':
            node = str(payload.get('node_id') or '')
            if not node:
                raise ValueError('node_id is required')
            for w in group:
                w.submit('pin', node)
        elif action == 'ask':
            question = str(payload.get('question') or '').strip()
            involves = payload.get('involves') or []
            if not 4 <= len(question) <= 300:
                raise ValueError('the question must be between 4 and 300 characters')
            if not isinstance(involves, list) or len(involves) > 8 or not all(isinstance(x, str) for x in involves):
                raise ValueError('involves must be a list of up to 8 concept names')
            for w in group:
                w.submit('ask', {'question': question, 'involves': involves, 'focus': bool(payload.get('focus'))})
        elif action == 'focus':
            tid = payload.get('thread_id')
            if tid is not None and not isinstance(tid, str):
                raise ValueError('thread_id must be a string or null')
            for w in group:
                w.submit('focus', {'thread_id': tid})
        elif action == 'close_thread':
            tid = str(payload.get('thread_id') or '')
            if not tid:
                raise ValueError('thread_id is required')
            for w in group:
                w.submit('close_thread', {'thread_id': tid})
        elif action == 'inject':
            text, url = str(payload.get('text') or '').strip(), str(payload.get('url') or '').strip()
            if bool(text) == bool(url):
                raise ValueError('give either text or a url')
            if len(text) > MAX_TEXT:
                raise ValueError('text too long')
            if url:
                from .security import check_url
                check_url(url)
            for w in group:
                w.submit('inject', {'text': text[:MAX_TEXT], 'url': url})
        else:
            raise ValueError(f'unknown action {action!r}')
        return [w.run_id for w in group]

    def delete(self, run_id: str) -> Dict[str, Any]:
        """Delete a finished run, and its paired control if it has one. Only the files a run writes are
        removed (never a recursive delete); a run that is active, a symlink, or outside the runs folder
        is refused. Returns {'deleted': [ids], 'leftover': {id: [names not recognised, so kept]}}."""
        with self._lock:
            path = self.run_dir(run_id)
            if not os.path.isdir(path):
                raise ValueError('no such run')
            members = [run_id]
            cfg_path = os.path.join(path, 'config.json')
            if os.path.isfile(cfg_path):
                try:
                    with open(cfg_path, encoding='utf-8') as f:
                        members += [m for m in json.load(f).get('paired_with', []) if RUN_ID.match(str(m))]
                except (OSError, json.JSONDecodeError):
                    pass
            members = [m for m in dict.fromkeys(members) if os.path.isdir(self.run_dir(m))]
            busy = [m for m in members if m in self.workers and self.workers[m].is_alive()]
            if busy:
                raise RuntimeError('stop the run before deleting it: ' + ', '.join(busy))
            result: Dict[str, Any] = {'deleted': [], 'leftover': {}}
            for member in members:
                left = self._delete_one(self.run_dir(member))
                if left:
                    result['leftover'][member] = left
                else:
                    result['deleted'].append(member)
                self.workers.pop(member, None)
            return result

    def _delete_one(self, path: str) -> List[str]:
        expected = os.path.join(os.path.realpath(self.runs_root), os.path.basename(path))
        if os.path.islink(path) or os.path.normcase(os.path.realpath(path)) != os.path.normcase(expected):
            raise ValueError('refusing to delete a folder that is not a plain run folder inside the runs folder')
        names = set(os.listdir(path))
        if not names & {'config.json', 'rounds.jsonl', 'world.json'}:
            raise ValueError('that folder does not look like a run')
        for name in RUN_FILES:
            target = os.path.join(path, name)
            if os.path.isfile(target) and not os.path.islink(target):
                os.remove(target)
        circuits = os.path.join(path, 'quantum')
        if os.path.isdir(circuits) and not os.path.islink(circuits):
            for name in os.listdir(circuits):
                target = os.path.join(circuits, name)
                if QASM_FILE.match(name) and os.path.isfile(target) and not os.path.islink(target):
                    os.remove(target)
            try:
                os.rmdir(circuits)
            except OSError:
                pass
        try:
            os.rmdir(path)                        # succeeds only if the folder is now empty
            return []
        except OSError:
            return sorted(os.listdir(path))

    def stop_all(self, wait: float = 5.0) -> None:
        for w in self.active():
            w.stop()
        deadline = time.time() + wait
        for w in self.active():
            w.join(max(0., deadline - time.time()))


def _last_record(path: str) -> Optional[Dict[str, Any]]:
    """The last complete JSON line of a jsonl file, without reading all of it."""
    if not os.path.isfile(path):
        return None
    try:
        with open(path, 'rb') as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 65536))
            tail = f.read().decode('utf-8', errors='replace').splitlines()
    except OSError:
        return None
    for line in reversed(tail):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return None
