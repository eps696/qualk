"""Deleting runs: only what a run wrote, never anything else, never while active."""
import json
import os
import tempfile
import time
import unittest

os.environ.pop("MOTH_API_KEY", None)

from fastapi.testclient import TestClient

from qualk.runner import EventBus, RunManager, SimulatedBackend
from qualk.server import create_app

BODY = {'seed': 'a seed passage', 'rounds': 2, 'nodes': 6, 'steps': 3, 'shots': 128, 'time': 2.0, 'explore': 1.0, 'rng': 3}


def wait_for(cond, timeout=60.):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return
        time.sleep(0.03)
    raise AssertionError('timed out')


class DeleteTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix='qualk_delete_')
        self.manager = RunManager(os.path.join(self.root, 'runs'), SimulatedBackend(delay=0.0), EventBus(), max_rounds=20)

    def tearDown(self):
        self.manager.stop_all(20)

    def finish(self, **kw):
        ids = self.manager.create({**BODY, **kw})
        wait_for(lambda: all(not self.manager.workers[i].is_alive() for i in ids))
        return ids

    def path(self, run_id):
        return os.path.join(self.manager.runs_root, run_id)

    def test_deletes_a_finished_run_completely(self):
        (rid,) = self.finish(name='gone')
        self.assertTrue(os.path.isdir(self.path(rid)))
        self.assertTrue(os.listdir(os.path.join(self.path(rid), 'quantum')))
        result = self.manager.delete(rid)
        self.assertEqual(result, {'deleted': ['gone'], 'leftover': {}})
        self.assertFalse(os.path.exists(self.path(rid)))
        self.assertEqual(self.manager.list_runs(), [])

    def test_a_paired_run_is_deleted_together_and_siblings_are_untouched(self):
        keep = self.finish(name='keep')[0]
        ids = self.finish(name='pair', paired=True)
        self.assertEqual(ids, ['pair', 'pair-ctl'])
        result = self.manager.delete('pair-ctl')                 # either member deletes both
        self.assertEqual(sorted(result['deleted']), ['pair', 'pair-ctl'])
        self.assertFalse(os.path.exists(self.path('pair')) or os.path.exists(self.path('pair-ctl')))
        self.assertTrue(os.path.isfile(os.path.join(self.path(keep), 'rounds.jsonl')), 'other runs are not touched')

    def test_unrecognised_files_are_kept_and_reported(self):
        (rid,) = self.finish(name='mixed')
        stray = os.path.join(self.path(rid), 'my-notes.txt')
        with open(stray, 'w') as f:
            f.write('do not delete me')
        extra = os.path.join(self.path(rid), 'quantum', 'notes.txt')
        with open(extra, 'w') as f:
            f.write('nor me')
        result = self.manager.delete(rid)
        self.assertEqual(result['deleted'], [])
        self.assertEqual(result['leftover'], {'mixed': ['my-notes.txt', 'quantum']})
        self.assertTrue(os.path.isfile(stray) and os.path.isfile(extra))
        self.assertFalse(os.path.exists(os.path.join(self.path(rid), 'world.json')), 'the run\'s own files are gone')
        self.assertEqual(self.manager.list_runs(), [], 'a folder with nothing of a run left is not listed as a run')

    def test_an_active_run_cannot_be_deleted(self):
        ids = self.manager.create({**BODY, 'name': 'busy', 'rounds': 5, 'start_paused': True})
        worker = self.manager.workers['busy']
        wait_for(lambda: worker.state == 'paused')
        with self.assertRaises(RuntimeError):
            self.manager.delete('busy')
        self.assertTrue(os.path.isdir(self.path('busy')))
        self.manager.control('busy', 'stop')
        wait_for(lambda: not worker.is_alive())
        self.manager.delete('busy')
        self.assertFalse(os.path.exists(self.path('busy')))

    def test_invalid_and_foreign_targets_are_refused(self):
        for bad in ('..', '../x', 'a/b', '', 'UPPER', '.hidden'):
            with self.assertRaises(ValueError, msg=bad):
                self.manager.delete(bad)
        with self.assertRaises(ValueError):
            self.manager.delete('never-existed')
        foreign = self.path('not-a-run')                        # a folder that is not a run at all
        os.makedirs(foreign)
        with open(os.path.join(foreign, 'precious.txt'), 'w') as f:
            f.write('x')
        with self.assertRaises(ValueError):
            self.manager.delete('not-a-run')
        self.assertTrue(os.path.isfile(os.path.join(foreign, 'precious.txt')))

    def test_a_symlinked_run_folder_is_refused(self):
        (rid,) = self.finish(name='real')
        link = self.path('linked')
        try:
            os.symlink(self.path(rid), link, target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest('symlinks are not available here')
        with self.assertRaises(ValueError):
            self.manager.delete('linked')
        self.assertTrue(os.path.isfile(os.path.join(self.path(rid), 'rounds.jsonl')))


class ServerDeleteTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix='qualk_delete_srv_')
        self.app = create_app(os.path.join(self.tmp, 'runs'), token='secret-token', simulate=True,
                              env_path=os.path.join(self.tmp, 'env'), max_rounds=20, sim_delay=0.0)
        self.client = TestClient(self.app)
        self.auth = {'Authorization': 'Bearer secret-token'}

    def tearDown(self):
        self.app.state.manager.stop_all(20)

    def test_delete_requires_the_token_and_reports_and_broadcasts(self):
        self.assertEqual(self.client.post('/api/runs', json={**BODY, 'name': 'x1'}, headers=self.auth).status_code, 200)
        manager = self.app.state.manager
        wait_for(lambda: not manager.workers['x1'].is_alive())
        self.assertEqual(self.client.delete('/api/runs/x1').status_code, 401)
        self.assertTrue(os.path.isdir(os.path.join(manager.runs_root, 'x1')))
        with self.client.websocket_connect('/ws?token=secret-token') as ws:
            ws.receive_json()
            r = self.client.delete('/api/runs/x1', headers=self.auth)
            self.assertEqual(r.status_code, 200)
            self.assertEqual(r.json(), {'deleted': ['x1'], 'leftover': {}})
            seen = [ws.receive_json() for _ in range(3)]
            self.assertIn({'type': 'deleted', 'runs': ['x1'], 'leftover': {}}, seen)
        self.assertFalse(os.path.exists(os.path.join(manager.runs_root, 'x1')))
        self.assertEqual(self.client.delete('/api/runs/x1', headers=self.auth).status_code, 404)

    def test_active_runs_and_bad_ids_map_to_http_errors(self):
        self.client.post('/api/runs', json={**BODY, 'name': 'busy', 'rounds': 5, 'start_paused': True}, headers=self.auth)
        manager = self.app.state.manager
        wait_for(lambda: manager.workers['busy'].state == 'paused')
        self.assertEqual(self.client.delete('/api/runs/busy', headers=self.auth).status_code, 409)
        self.assertEqual(self.client.delete('/api/runs/..', headers=self.auth).status_code, 404)
        self.client.post('/api/runs/busy/stop', headers=self.auth)


if __name__ == '__main__':
    unittest.main()
