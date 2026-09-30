"""Run orchestration offline: simulated backend, real graph, walk and circuit."""
import json
import os
import tempfile
import time
import unittest

os.environ.pop("MOTH_API_KEY", None)

from qualk.runner import EventBus, RunConfig, RunManager, SimulatedBackend

FAST = dict(nodes=6, steps=3, shots=128, time=2.0, explore=1.0, rng=3)


def wait_for(cond, timeout=60., what='condition'):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return
        time.sleep(0.02)
    raise AssertionError(f'timed out waiting for {what}')


def read_jsonl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(l) for l in f if l.strip()]


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix='qualk_runner_')
        self.bus = EventBus()
        self.events = []
        self.bus.publish = self._capture(self.bus.publish)
        self.manager = RunManager(self.root, SimulatedBackend(delay=0.0), self.bus, max_rounds=50)

    def _capture(self, publish):
        def wrapper(message):
            self.events.append(message)
            publish(message)
        return wrapper

    def tearDown(self):
        self.manager.stop_all(20)

    def start(self, **kw):
        ids = self.manager.create({'name': kw.pop('name', 'r1'), 'seed': 'a seed passage', **FAST, **kw})
        return ids, [self.manager.workers[i] for i in ids]

    def test_run_completes_and_logs_walks(self):
        _ids, (w,) = self.start(rounds=4)
        wait_for(lambda: w.state == 'done', what='done')
        records = read_jsonl(os.path.join(w.out_dir, 'rounds.jsonl'))
        self.assertEqual([r['kind'] for r in records][:1], ['seed'])
        probes = [r for r in records if r['kind'] == 'probe']
        self.assertEqual(len(probes), 4)
        self.assertTrue(any((r['probe']['walk'] or {}).get('method') == 'qiskit-statevector' for r in probes))
        self.assertEqual(sum(1 for e in self.events if e['type'] == 'round' and e['run'] == 'r1'), 5)
        self.assertTrue(os.path.isfile(os.path.join(w.out_dir, 'config.json')))

    def test_pause_step_resume(self):
        _ids, (w,) = self.start(rounds=3, start_paused=True)
        wait_for(lambda: w.state == 'paused', what='paused')
        time.sleep(0.3)
        self.assertEqual(w.done_rounds, 0)
        self.manager.control('r1', 'step')
        wait_for(lambda: w.done_rounds == 1 and w.state == 'paused', what='one step')
        time.sleep(0.3)
        self.assertEqual(w.done_rounds, 1, 'a step runs exactly one round')
        self.manager.control('r1', 'resume')
        wait_for(lambda: w.state == 'done', what='done')
        self.assertEqual(w.done_rounds, 3)

    def test_params_apply_at_the_next_round_only(self):
        _ids, (w,) = self.start(rounds=3, start_paused=True)
        wait_for(lambda: w.state == 'paused')
        before = w.engine.walk
        self.manager.control('r1', 'params', {'steps': 5, 'explore': 0.9, 'nodes': 4})
        time.sleep(0.3)
        self.assertIs(w.engine.walk, before, 'nothing changes while paused between rounds')
        self.manager.control('r1', 'step')
        wait_for(lambda: w.done_rounds == 1)
        self.assertEqual((w.engine.walk.steps, w.engine.walk.max_nodes, w.cfg.explore), (5, 4, 0.9))
        self.assertEqual(w.engine.archive.explore, 0.9)
        steer = read_jsonl(os.path.join(w.out_dir, 'steer.jsonl'))
        self.assertEqual(steer[0]['kind'], 'params')
        with open(os.path.join(w.out_dir, 'config.json'), encoding='utf-8') as f:
            self.assertEqual(json.load(f)['cfg']['steps'], 5)

    def test_invalid_params_are_rejected_immediately(self):
        _ids, (w,) = self.start(rounds=2, start_paused=True)
        wait_for(lambda: w.state == 'paused')
        with self.assertRaises(ValueError):
            self.manager.control('r1', 'params', {'steps': 999})
        with self.assertRaises(ValueError):
            self.manager.control('r1', 'params', {'topic': 'not live'})

    def test_pin_sets_the_next_probe_seed(self):
        _ids, (w,) = self.start(rounds=2, start_paused=True)
        wait_for(lambda: w.state == 'paused')
        pinned = sorted(i for i in w.engine.graph.nodes if not i.startswith('thread:'))[-1]
        self.manager.control('r1', 'pin', {'node_id': pinned})
        self.manager.control('r1', 'step')
        wait_for(lambda: w.done_rounds == 1)
        records = read_jsonl(os.path.join(w.out_dir, 'rounds.jsonl'))
        probe = [r for r in records if r['kind'] == 'probe'][0]['probe']
        self.assertEqual(probe['components'][0]['id'], pinned)
        self.assertIn('pinned', probe['selection'])

    def test_pin_of_unknown_concept_is_logged_not_fatal(self):
        _ids, (w,) = self.start(rounds=2, start_paused=True)
        wait_for(lambda: w.state == 'paused')
        self.manager.control('r1', 'pin', {'node_id': 'motif:nope'})
        self.manager.control('r1', 'resume')
        wait_for(lambda: w.state == 'done')
        self.assertTrue(any(e['type'] == 'error' and 'no such concept' in e['text'] for e in self.events))

    def test_inject_text_becomes_a_round_and_grows_the_graph(self):
        _ids, (w,) = self.start(rounds=1, start_paused=True)
        wait_for(lambda: w.state == 'paused')
        n0 = len(w.engine.graph.nodes)
        self.manager.control('r1', 'inject', {'text': 'A manually supplied passage about pulsars.'})
        self.manager.control('r1', 'step')
        wait_for(lambda: w.state == 'done')
        records = read_jsonl(os.path.join(w.out_dir, 'rounds.jsonl'))
        kinds = [r['kind'] for r in records]
        self.assertEqual(kinds, ['seed', 'inject', 'probe'])
        self.assertGreater(len(w.engine.graph.nodes), n0)

    def test_inject_requires_exactly_one_of_text_or_url(self):
        _ids, (w,) = self.start(rounds=1, start_paused=True)
        wait_for(lambda: w.state == 'paused')
        for bad in ({}, {'text': 'x', 'url': 'http://example.org'}, {'url': 'ftp://example.org'},
                    {'url': 'http://127.0.0.1/admin'}):
            with self.assertRaises(ValueError, msg=str(bad)):
                self.manager.control('r1', 'inject', bad)

    def test_only_one_active_run_at_a_time(self):
        _ids, (w,) = self.start(rounds=2, start_paused=True)
        wait_for(lambda: w.state == 'paused')
        with self.assertRaises(RuntimeError):
            self.manager.create({'name': 'r2', 'seed': 'x', **FAST})
        self.manager.control('r1', 'stop')
        wait_for(lambda: w.state == 'stopped')
        wait_for(lambda: not w.is_alive())
        self.manager.create({'name': 'r2', 'seed': 'x', **FAST, 'rounds': 1})

    def test_stop_ends_early_and_resume_continues_numbering(self):
        _ids, (w,) = self.start(rounds=40, start_paused=True)
        wait_for(lambda: w.state == 'paused')
        self.manager.control('r1', 'step')
        wait_for(lambda: w.done_rounds == 1)
        self.manager.control('r1', 'stop')
        wait_for(lambda: w.state == 'stopped')
        wait_for(lambda: not w.is_alive())
        self.assertEqual(w.done_rounds, 1)
        self.manager.resume('r1', rounds=2)
        w2 = self.manager.workers['r1']
        wait_for(lambda: w2.state == 'done')
        rounds = [r['round'] for r in read_jsonl(os.path.join(w.out_dir, 'rounds.jsonl')) if r['kind'] == 'probe']
        self.assertEqual(rounds, [1, 2, 3])

    def test_paired_run_shares_the_seed_and_runs_in_lockstep(self):
        ids, (main, ctl) = self.start(rounds=4, paired=True)
        self.assertEqual(ids, ['r1', 'r1-ctl'])
        wait_for(lambda: main.state == 'done' and ctl.state == 'done', what='both done')
        a = read_jsonl(os.path.join(main.out_dir, 'rounds.jsonl'))
        b = read_jsonl(os.path.join(ctl.out_dir, 'rounds.jsonl'))
        self.assertEqual(a[0]['kind'], 'seed')
        self.assertEqual(a[0], b[0], 'the control starts from the identical seed record')
        methods_a = {(r['probe']['walk'] or {}).get('method') for r in a if r['kind'] == 'probe'}
        methods_b = {(r['probe']['walk'] or {}).get('method') for r in b if r['kind'] == 'probe'}
        self.assertIn('qiskit-statevector', methods_a)
        self.assertEqual(methods_b - {None}, {'diffusion'})
        self.assertEqual(main.done_rounds, ctl.done_rounds)

    def test_paired_pause_applies_to_both(self):
        _ids, (main, ctl) = self.start(rounds=6, paired=True, start_paused=True)
        wait_for(lambda: main.state == 'paused' and ctl.state == 'paused')
        self.manager.control('r1', 'step')
        wait_for(lambda: main.done_rounds == 1 and ctl.done_rounds == 1)
        self.manager.control('r1-ctl', 'params', {'walk': 'quantum', 'steps': 4})
        self.manager.control('r1', 'step')
        wait_for(lambda: main.done_rounds == 2 and ctl.done_rounds == 2)
        self.assertEqual(ctl.cfg.walk, 'diffusion', 'the control never becomes a quantum walk')
        self.assertEqual(main.cfg.steps, 4)
        self.manager.control('r1', 'stop')

    def test_invalid_config_and_run_ids(self):
        for bad in ({'rounds': 0}, {'rounds': 999}, {'walk': 'grover'}, {'nodes': 1}, {'time': -1},
                    {'explore': 2}, {'paired': True, 'walk': 'diffusion'}, {'bogus': 1}, {'rounds': 'x'}):
            with self.assertRaises(ValueError, msg=str(bad)):
                self.manager.create({'seed': 'x', **bad})
        for bad in ('../x', 'a/b', '', 'UPPER', '.hidden'):
            with self.assertRaises(ValueError, msg=bad):
                self.manager.run_dir(bad)

    def test_list_runs_reports_state_and_counts(self):
        _ids, (w,) = self.start(rounds=2)
        wait_for(lambda: w.state == 'done')
        wait_for(lambda: not w.is_alive())
        runs = self.manager.list_runs()
        self.assertEqual([r['id'] for r in runs], ['r1'])
        self.assertEqual(runs[0]['round'], 2)
        self.assertGreater(runs[0]['nodes'], 0)


class ConfigTests(unittest.TestCase):
    def test_defaults_validate(self):
        RunConfig().validate()

    def test_merged_only_allows_live_params(self):
        cfg = RunConfig()
        self.assertEqual(cfg.merged({'steps': 4}).steps, 4)
        with self.assertRaises(ValueError):
            cfg.merged({'topic': 'x'})


if __name__ == '__main__':
    unittest.main()
