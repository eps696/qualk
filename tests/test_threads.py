"""Threads: the keeper's hygiene, the lifecycle in the engine, thread-aimed probes and steering."""
import asyncio
import json
import os
import random
import tempfile
import time
import unittest
from types import SimpleNamespace

os.environ.pop("MOTH_API_KEY", None)

from qualk.fakes import FakeWeb, HashEmbedder, fake_thread_extractor, question_extractor, ring_extractor
from qualk.threads import THREAD_FLOOR, ThreadKeeper
from qualk.world.store import WorldGraph


def graph_with(root, ops, at=1):
    g = WorldGraph(root, fuzzy_names=False)
    g.apply(ops, at=at, by='test')
    return g


def node(name, gist=''):
    return {'op': 'node', 'kind': 'motif', 'name': name, 'gist': gist or name}


def thread(name, involves, state='open', pressure=0.6):
    return {'op': 'thread', 'name': name, 'state': state, 'pressure': pressure, 'involves': involves, 'gist': name}


class KeeperTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix='qualk_threads_')
        self.keeper = ThreadKeeper()

    def test_open_thread_needs_a_grounded_element(self):
        g = graph_with(self.root, [node('alpha')])
        kept = self.keeper.filter_ops([thread('Q1', ['alpha']), thread('Q2', ['nothing known'])], g)
        self.assertEqual([op['name'] for op in kept], ['Q1'])

    def test_batch_elements_ground_a_thread(self):
        g = graph_with(self.root, [])
        kept = self.keeper.filter_ops([node('fresh'), thread('Q', ['fresh'])], g)
        self.assertEqual(len(kept), 2)

    def test_pressure_is_mechanical_not_model_estimated(self):
        g = graph_with(self.root, [node('a'), thread('Q', ['a'], pressure=0.6)])
        opened = self.keeper.filter_ops([thread('New', ['a'], pressure=0.99)], g)[0]
        self.assertEqual(opened['pressure'], 0.6)
        advanced = self.keeper.filter_ops([thread('Q', ['a'], state='complicated', pressure=0.01)], g)[0]
        self.assertAlmostEqual(advanced['pressure'], 0.75)

    def test_pressure_decays_with_age_and_sweep_closes_dormant_threads(self):
        g = graph_with(self.root, [node('a'), thread('Old', ['a'], pressure=0.6)], at=1)
        old = g.nodes[g.resolve('Old', 'thread')]
        self.assertLess(self.keeper.pressure_eff(old, 60), THREAD_FLOOR)
        closed = self.keeper.sweep(g, 60)
        self.assertEqual(closed['dormant'], 1)
        self.assertEqual(closed['detail'], [(old.id, 'dormant')])
        self.assertEqual(old.traits['state'], 'resolved')

    def test_sweep_caps_the_pool_by_dropping_the_weakest(self):
        keeper = ThreadKeeper(cap=2)
        ops = [node('a')] + [thread(f'Q{i}', ['a'], pressure=0.3 + 0.1 * i) for i in range(4)]
        g = graph_with(self.root, ops)
        closed = keeper.sweep(g, 1)
        self.assertEqual(closed['cap'], 2)
        self.assertEqual(sorted(n.name for n in keeper.open_threads(g)), ['Q2', 'Q3'])

    def test_involved_ids_and_table(self):
        g = graph_with(self.root, [node('a'), node('b'), thread('Q', ['a', 'b'])])
        tid = g.resolve('Q', 'thread')
        self.assertEqual(sorted(self.keeper.involved_ids(g, tid)), ['motif:a', 'motif:b'])
        row = self.keeper.table(g, 2)[0]
        self.assertEqual((row['id'], row['state'], row['name']), (tid, 'open', 'Q'))
        self.assertEqual(sorted(row['involves']), ['motif:a', 'motif:b'])

    def test_pick_prefers_pressure(self):
        g = graph_with(self.root, [node('a'), thread('Low', ['a'], pressure=0.05), thread('High', ['a'], pressure=1.0)])
        rng = random.Random(0)
        picks = [self.keeper.pick(g, rng, 1) for _ in range(200)]
        self.assertGreater(picks.count(g.resolve('High', 'thread')), 150)


def make_engine(root, threads=True, aim=0.0, walk=None, extractor=None):
    from qualk.engine import Engine
    from qualk.semantic import SemanticIndex
    index = SemanticIndex(os.path.join(root, 'semantic.sqlite'), HashEmbedder())
    engine = Engine(root, extractor or question_extractor(), index, FakeWeb(), walk=walk, explore=1.0, seed=3,
                    thread_extractor=fake_thread_extractor(), threads=threads, thread_aim=aim)
    return engine, index


def seed_and_run(engine, rounds):
    from qualk.web import Parcel
    records = asyncio.run(engine.seed([Parcel('seed text', 'seed:test', 'seed')]))
    return records + [asyncio.run(engine.step()) for _ in range(rounds)]


class EngineThreadTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix='qualk_engine_threads_')

    def test_questions_open_advance_and_resolve_and_are_logged(self):
        engine, index = make_engine(self.root)
        records = seed_and_run(engine, 8)
        index.close()
        events = [e for r in records for e in (r['threads'] or {}).get('events', [])]
        kinds = {e['event'] for e in events}
        self.assertLessEqual({'opened', 'advanced', 'resolved'}, kinds)
        first = next(e for e in events if e['event'] == 'opened')
        later = [e['event'] for e in events if e['id'] == first['id']]
        self.assertEqual(later[0], 'opened')
        self.assertLess(later.index('advanced') if 'advanced' in later else 99,
                        later.index('resolved') if 'resolved' in later else 100, 'advanced before resolved')
        last = records[-1]['threads']
        self.assertTrue(last['table'])
        self.assertEqual(sum(1 for t in last['table'] if t['state'] != 'resolved'), last['open'])
        self.assertTrue(all(t['involves'] for t in last['table']))

    def test_threads_off_leaves_records_and_graph_free_of_threads(self):
        engine, index = make_engine(self.root, threads=False, extractor=ring_extractor())
        records = seed_and_run(engine, 3)
        index.close()
        self.assertTrue(all(r.get('threads') is None for r in records))
        self.assertFalse(any(n.kind == 'thread' for n in engine.graph.nodes.values()))

    def test_thread_aimed_probe_starts_from_the_threads_own_concepts(self):
        from qualk.quantum_walk import DiffusionProbeWalk
        engine, index = make_engine(self.root, aim=1.0, walk=DiffusionProbeWalk(max_nodes=6, time=2.))
        records = seed_and_run(engine, 6)
        index.close()
        aimed = [r for r in records if r['kind'] == 'probe' and r['probe']['thread']]
        self.assertTrue(aimed, 'with thread_aim=1 probes must be aimed at threads once some are open')
        table = {t['id']: t for r in records for t in (r['threads'] or {}).get('table', [])}
        for r in aimed:
            thread_row = table[r['probe']['thread']]
            self.assertIn(r['probe']['components'][0]['id'], thread_row['involves'])
            self.assertTrue(r['probe']['selection'].startswith('thread'))
            self.assertEqual(r['threads']['aimed'], r['probe']['thread'])

    def test_a_threads_own_text_never_becomes_the_query(self):
        from qualk.quantum_walk import DiffusionProbeWalk
        engine, index = make_engine(self.root, aim=1.0, walk=DiffusionProbeWalk(max_nodes=6, time=2.))
        records = seed_and_run(engine, 5)
        index.close()
        for r in records:
            if r['kind'] == 'probe' and r['probe']['thread']:
                self.assertNotIn('Why does', r['query'])

    def test_ask_grounds_a_user_question_and_focus_steers_probes(self):
        from qualk.quantum_walk import DiffusionProbeWalk
        engine, index = make_engine(self.root, walk=DiffusionProbeWalk(max_nodes=6, time=2.))
        seed_and_run(engine, 2)
        tid = engine.ask('What links idea 1 to idea 2?', involves=['idea 1', 'idea 2'])
        node = engine.graph.nodes[tid]
        self.assertEqual(node.traits['state'], 'open')
        self.assertEqual(sorted(engine.keeper.involved_ids(engine.graph, tid)), ['motif:idea-1', 'motif:idea-2'])
        engine.set_focus(tid)
        records = [asyncio.run(engine.step()) for _ in range(2)]
        index.close()
        for r in records:
            if r['kind'] == 'probe':
                self.assertEqual(r['probe']['thread'], tid)
                self.assertEqual(r['probe']['selection'], 'thread: steered focus')
        self.assertTrue(any(e['event'] == 'opened' and e['id'] == tid for r in records for e in r['threads']['events']))

    def test_ask_falls_back_to_the_nearest_concepts_and_validates(self):
        engine, index = make_engine(self.root)
        seed_and_run(engine, 1)
        tid = engine.ask('Something about idea 2 and idea 3 maybe?')
        self.assertTrue(engine.keeper.involved_ids(engine.graph, tid))
        with self.assertRaises(ValueError):
            engine.ask('no')
        with self.assertRaises(ValueError):
            engine.set_focus('thread:does-not-exist')
        index.close()

    def test_user_close_is_reported_as_closed_and_ends_focus(self):
        engine, index = make_engine(self.root)
        seed_and_run(engine, 1)
        tid = engine.ask('Will this be closed by the user?', involves=['idea 1'])
        engine.set_focus(tid)
        engine.close_thread(tid)
        record = asyncio.run(engine.step())
        index.close()
        self.assertIn(('closed', tid), [(e['event'], e['id']) for e in record['threads']['events']])
        self.assertEqual(record['threads']['focus_ended'], tid)
        self.assertIsNone(engine.focus_thread)

    def test_resume_does_not_replay_old_events(self):
        engine, index = make_engine(self.root)
        seed_and_run(engine, 4)
        index.close()
        again, index2 = make_engine(self.root)
        record = asyncio.run(again.step())
        index2.close()
        opened_before = {e['id'] for r in [json.loads(l) for l in open(os.path.join(self.root, 'rounds.jsonl'), encoding='utf-8')]
                         if r.get('threads') and r['round'] < record['round'] for e in r['threads']['events'] if e['event'] == 'opened'}
        reopened = [e for e in record['threads']['events'] if e['event'] == 'opened' and e['id'] in opened_before]
        self.assertEqual(reopened, [])

    def test_probe_archive_snapshot_restores_thread_field(self):
        from qualk.exploration import ProbeArchive
        from qualk.quantum_walk import DiffusionProbeWalk
        engine, index = make_engine(self.root, aim=1.0, walk=DiffusionProbeWalk(max_nodes=6, time=2.))
        seed_and_run(engine, 4)
        snap = json.loads(json.dumps(engine.archive.snapshot()))
        fresh = ProbeArchive(random.Random(0))
        fresh.restore(snap)
        index.close()
        self.assertEqual([p.thread for p in fresh.probes], [p.thread for p in engine.archive.probes])


class RunnerThreadTests(unittest.TestCase):
    def setUp(self):
        from qualk.runner import EventBus, RunManager, SimulatedBackend
        self.root = tempfile.mkdtemp(prefix='qualk_runner_threads_')
        self.bus = EventBus()
        self.events = []
        publish = self.bus.publish
        self.bus.publish = lambda m: (self.events.append(m), publish(m))
        self.manager = RunManager(self.root, SimulatedBackend(delay=0.0), self.bus, max_rounds=30)

    def tearDown(self):
        self.manager.stop_all(20)

    def wait(self, cond, timeout=60.):
        end = time.time() + timeout
        while time.time() < end:
            if cond():
                return
            time.sleep(0.02)
        raise AssertionError('timed out')

    def start(self, **kw):
        ids = self.manager.create({'name': 't1', 'seed': 'a seed', 'nodes': 6, 'steps': 3, 'shots': 128,
                                   'explore': 1.0, 'rng': 3, 'rounds': 6, 'start_paused': True, **kw})
        return self.manager.workers[ids[0]]

    def test_ask_focus_and_close_through_the_manager(self):
        w = self.start()
        self.wait(lambda: w.state == 'paused')
        self.manager.control('t1', 'ask', {'question': 'Does steering work end to end?', 'involves': ['idea 1'], 'focus': True})
        self.manager.control('t1', 'step')
        self.wait(lambda: w.done_rounds == 1)
        tid = w.engine.graph.resolve('Does steering work end to end?', 'thread')
        self.assertEqual(w.engine.focus_thread, tid)
        self.assertTrue(any(e['type'] == 'threads' and e['focus'] == tid for e in self.events))
        self.assertEqual(w.status()['focus'], tid)
        self.manager.control('t1', 'focus', {'thread_id': None})
        self.manager.control('t1', 'close_thread', {'thread_id': tid})
        self.manager.control('t1', 'step')
        self.wait(lambda: w.done_rounds == 2)
        self.assertEqual(w.engine.graph.nodes[tid].traits['state'], 'resolved')
        kinds = [json.loads(l)['kind'] for l in open(os.path.join(w.out_dir, 'steer.jsonl'), encoding='utf-8')]
        self.assertEqual(kinds, ['ask', 'focus', 'focus', 'close_thread'])

    def test_thread_actions_are_validated_and_can_be_rejected_at_the_boundary(self):
        w = self.start()
        self.wait(lambda: w.state == 'paused')
        for action, body in (('ask', {'question': 'x'}), ('ask', {'question': 'long enough?', 'involves': 'idea 1'}),
                             ('focus', {'thread_id': 5}), ('close_thread', {})):
            with self.assertRaises(ValueError, msg=action):
                self.manager.control('t1', action, body)
        self.manager.control('t1', 'focus', {'thread_id': 'thread:nope'})
        self.manager.control('t1', 'step')
        self.wait(lambda: w.done_rounds == 1)
        self.assertTrue(any(e['type'] == 'error' and 'not open' in e['text'] for e in self.events))

    def test_thread_aim_is_a_live_parameter_and_threads_can_be_switched_off(self):
        w = self.start()
        self.wait(lambda: w.state == 'paused')
        self.manager.control('t1', 'params', {'thread_aim': 0.9})
        self.manager.control('t1', 'step')
        self.wait(lambda: w.done_rounds == 1)
        self.assertEqual(w.engine.thread_aim, 0.9)
        self.manager.control('t1', 'stop')
        self.wait(lambda: not w.is_alive())
        off = self.manager.create({'name': 't2', 'seed': 'a seed', 'nodes': 6, 'steps': 3, 'shots': 128, 'rounds': 2,
                                   'threads': False})
        w2 = self.manager.workers[off[0]]
        self.wait(lambda: w2.state == 'done')
        self.manager.control if False else None
        records = [json.loads(l) for l in open(os.path.join(w2.out_dir, 'rounds.jsonl'), encoding='utf-8')]
        self.assertTrue(all(r.get('threads') is None for r in records))


if __name__ == '__main__':
    unittest.main()
