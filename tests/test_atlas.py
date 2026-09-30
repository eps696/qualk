"""Contracts for running dog's quantum walk on Moth's Atlas platform, with local Qiskit as fallback."""
import io
import json
import os
import random
import re
import tempfile
import threading
import unittest
from contextlib import redirect_stdout
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

KEY = 'moth_secret_test_key'


class FakeAtlas:
    """The slice of the Atlas API the client uses: submit -> status -> result, Bearer auth."""

    def __init__(self, counts=None, fail_status=None, job_error=None, polls_before_done=1, key=KEY):
        outer = self
        self.counts = counts if counts is not None else {'0100': 700, '0010': 200, '0001': 100}
        self.fail_status, self.job_error, self.polls_before_done, self.key = fail_status, job_error, polls_before_done, key
        self.submits, self.bodies, self.polls, self.auth_seen = 0, [], 0, []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, code, payload):
                body = json.dumps(payload).encode()
                self.send_response(code)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _authorised(self):
                outer.auth_seen.append(self.headers.get('Authorization'))
                if self.headers.get('Authorization') != 'Bearer ' + outer.key:
                    self._send(401, {'title': 'Unauthorized', 'detail': 'bad key'})
                    return False
                return True

            def do_POST(self):
                if not self._authorised():
                    return
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                outer.submits += 1
                outer.bodies.append((self.path, body))
                if outer.fail_status:
                    return self._send(outer.fail_status, {'title': 'boom', 'detail': 'engine unavailable'})
                self._send(202, {'job_id': 'job-1', 'status': 'queued', 'submitted_at': 'now'})

            def do_GET(self):
                if not self._authorised():
                    return
                if self.path.endswith('/status'):
                    outer.polls += 1
                    if outer.job_error:
                        return self._send(200, {'job_id': 'job-1', 'status': 'failed',
                                                'error': {'type': 'X', 'message': outer.job_error}})
                    done = outer.polls > outer.polls_before_done
                    return self._send(200, {'job_id': 'job-1', 'status': 'completed' if done else 'processing'})
                n = len(next(iter(outer.counts)))
                self._send(200, {'result': {'measurements': {
                    'X' * n: {'counts': {'0' * n: 4096}, 'shots': 4096},
                    'Z' * n: {'counts': outer.counts, 'shots': sum(outer.counts.values())}}}})
        self.server = HTTPServer(('127.0.0.1', 0), Handler)
        self.url = 'http://127.0.0.1:%d' % self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


QASM = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[4];\nx q[0];\n'


class ClientTests(unittest.TestCase):
    def client(self, server, **kw):
        from qualk.atlas import AtlasClient
        return AtlasClient(KEY, base_url=server.url, poll_seconds=.01, timeout_seconds=kw.pop('timeout_seconds', 5), **kw)

    def test_z_counts_submits_the_circuit_and_returns_the_z_basis_counts(self):
        server = FakeAtlas()
        try:
            counts, meta = self.client(server).z_counts(QASM, 4, 2048)
        finally:
            server.close()
        self.assertEqual(counts, {'0100': 700, '0010': 200, '0001': 100})
        self.assertEqual(meta['job_id'], 'job-1')
        path, body = server.bodies[0]
        self.assertEqual(path, '/api/v1/engines/tomography-api-v2/process')
        params = body['params']
        self.assertEqual((params['circuit_qasm'], params['shots'], params['provider_name']), (QASM, 2048, 'aer'))
        self.assertEqual(params['qubit_list'], [0, 1, 2, 3])
        self.assertFalse(params['double_tomography'] or params['mutual_information'])
        self.assertTrue(all(a == 'Bearer ' + KEY for a in server.auth_seen))

    def test_the_simulation_method_is_configurable(self):
        server = FakeAtlas()
        try:
            self.client(server).z_counts(QASM, 4, 100)
            self.client(server, backend_name='matrix_product_state').z_counts(QASM, 4, 100)
        finally:
            server.close()
        self.assertEqual([b['params']['backend_name'] for _p, b in server.bodies],
                         ['automatic', 'matrix_product_state'])

    def test_from_env_reads_the_backend_override(self):
        from qualk.atlas import AtlasClient
        old = {k: os.environ.get(k) for k in ('MOTH_API_KEY', 'MOTH_ATLAS_BACKEND')}
        try:
            os.environ['MOTH_API_KEY'] = KEY
            os.environ['MOTH_ATLAS_BACKEND'] = 'matrix_product_state'
            self.assertEqual(AtlasClient.from_env().backend_name, 'matrix_product_state')
        finally:
            for k, v in old.items():
                os.environ.pop(k, None)
                if v is not None:
                    os.environ[k] = v

    def test_failures_raise_atlas_error_without_leaking_the_key(self):
        from qualk.atlas import AtlasError
        cases = [FakeAtlas(fail_status=500), FakeAtlas(key='another'), FakeAtlas(job_error='circuit rejected')]
        for server in cases:
            try:
                with self.assertRaises(AtlasError) as raised:
                    self.client(server).z_counts(QASM, 4, 1024)
            finally:
                server.close()
            self.assertNotIn(KEY, str(raised.exception))
        self.assertIn('circuit rejected', str(AtlasErrorFor(cases[2])))

    def test_a_job_that_never_completes_times_out(self):
        from qualk.atlas import AtlasError
        server = FakeAtlas(polls_before_done=10 ** 6)
        try:
            with self.assertRaisesRegex(AtlasError, 'timed out'):
                self.client(server, timeout_seconds=.3).z_counts(QASM, 4, 1024)
        finally:
            server.close()

    def test_unreachable_server_raises_atlas_error(self):
        from qualk.atlas import AtlasClient, AtlasError
        with self.assertRaises(AtlasError):
            AtlasClient(KEY, base_url='http://127.0.0.1:9', poll_seconds=.01, timeout_seconds=1).z_counts(QASM, 4, 10)

    def test_from_env_needs_the_key_and_honours_the_url_override(self):
        from qualk.atlas import AtlasClient
        old = {k: os.environ.pop(k, None) for k in ('MOTH_API_KEY', 'MOTH_API_URL')}
        try:
            self.assertIsNone(AtlasClient.from_env())
            os.environ['MOTH_API_KEY'] = KEY
            os.environ['MOTH_API_URL'] = 'http://example.invalid'
            client = AtlasClient.from_env()
            self.assertEqual(client.base_url, 'http://example.invalid')
            self.assertNotIn(KEY, repr(client))
        finally:
            for k, v in old.items():
                os.environ.pop(k, None)
                if v is not None:
                    os.environ[k] = v


def AtlasErrorFor(server):
    from qualk.atlas import AtlasClient, AtlasError
    server2 = FakeAtlas(job_error='circuit rejected')
    try:
        AtlasClient(KEY, base_url=server2.url, poll_seconds=.01, timeout_seconds=3).z_counts(QASM, 4, 10)
    except AtlasError as exc:
        return exc
    finally:
        server2.close()


def ring_graph():
    from tests.test_quantum_walk import walk_graph
    return walk_graph([('a', 'b', 'caused', .9, 0., .9), ('b', 'c', 'caused', .8, 0., .9),
                       ('c', 'd', 'caused', .7, 0., .9), ('a', 'd', 'enables', .6, 0., .9)])


class WalkOnAtlasTests(unittest.TestCase):
    def walk(self, server, **kw):
        from qualk.atlas import AtlasClient
        from qualk.quantum_walk import QuantumProbeWalk
        client = AtlasClient(KEY, base_url=server.url, poll_seconds=.01, timeout_seconds=5)
        return QuantumProbeWalk(max_nodes=4, steps=5, time=2.0, shots=1024, atlas=client, **kw)

    def test_the_pick_comes_from_the_atlas_counts_and_the_trace_says_so(self):
        graph = ring_graph()
        with tempfile.TemporaryDirectory() as root:
            server = FakeAtlas(counts={'0100': 900, '0010': 100})       # node index 2 dominates
            try:
                walk = self.walk(server, trace_dir=Path(root))
                target, trace = walk.select(graph, 'a', graph.nodes, random.Random(1), exclude=('a',), round_idx=3)
            finally:
                server.close()
            self.assertEqual(trace['method'], 'atlas-aer')
            self.assertEqual(trace['atlas_job_id'], 'job-1')
            self.assertEqual(trace['sampled_counts'][2], 900)
            self.assertEqual(target, trace['nodes'][2] if trace['nodes'][2] != 'a' else target)
            self.assertAlmostEqual(sum(trace['quantum_probabilities']), 1., places=9)
            self.assertLess(trace['trotter_error'], .1)
            self.assertIn('sampling_error', trace)
            qasm = Path(root, 'round-00003.qasm').read_text(encoding='utf-8')
            self.assertTrue(qasm.startswith('OPENQASM 2.0'))
            self.assertNotIn('measure', qasm)
            self.assertEqual(server.submits, 1)
            self.assertNotIn(KEY, json.dumps(trace))
            self.assertNotIn(KEY, qasm)

    def test_atlas_failure_falls_back_to_local_qiskit_for_that_pick(self):
        graph = ring_graph()
        server = FakeAtlas(fail_status=500)
        out = io.StringIO()
        try:
            walk = self.walk(server)
            with redirect_stdout(out):
                target, trace = walk.select(graph, 'a', graph.nodes, random.Random(1), exclude=('a',))
        finally:
            server.close()
        self.assertEqual(trace['method'], 'qiskit-statevector')
        self.assertIn('500', trace['atlas_error'])
        self.assertIn(target, ('b', 'c', 'd'))
        self.assertIn('falling back to local Qiskit', out.getvalue())

    def test_repeated_failures_stop_calling_atlas_for_the_rest_of_the_run(self):
        graph = ring_graph()
        server = FakeAtlas(fail_status=500)
        try:
            walk = self.walk(server)
            traces = []
            with redirect_stdout(io.StringIO()):
                for at in range(6):
                    traces.append(walk.select(graph, 'a', graph.nodes, random.Random(at), exclude=('a',))[1])
        finally:
            server.close()
        self.assertEqual(server.submits, 3)
        self.assertTrue(traces[-1]['atlas_disabled'])

    def test_a_success_resets_the_failure_count(self):
        graph = ring_graph()
        server = FakeAtlas(fail_status=500)
        try:
            walk = self.walk(server)
            with redirect_stdout(io.StringIO()):
                for at in range(2):
                    walk.select(graph, 'a', graph.nodes, random.Random(at), exclude=('a',))
                server.fail_status = None
                trace = walk.select(graph, 'a', graph.nodes, random.Random(9), exclude=('a',))[1]
                self.assertEqual(trace['method'], 'atlas-aer')
                server.fail_status = 500
                for at in range(2):
                    walk.select(graph, 'a', graph.nodes, random.Random(at), exclude=('a',))
                self.assertFalse(walk.atlas_disabled)
        finally:
            server.close()

    def wide_graph(self, n=20):
        from tests.test_quantum_walk import walk_graph
        return walk_graph([(f'n{i:02d}', f'n{i + 1:02d}', 'caused', .9 - .01 * i, 0., .9) for i in range(n - 1)])

    def test_a_window_wider_than_local_qiskit_asks_atlas_for_the_mps_simulator(self):
        graph = self.wide_graph(20)
        n = 20
        counts = {'0' * (n - 1) + '1': 500, '0' * (n - 2) + '10': 500}
        server = FakeAtlas(counts=counts)
        try:
            from qualk.atlas import AtlasClient
            from qualk.quantum_walk import QuantumProbeWalk
            walk = QuantumProbeWalk(max_nodes=n, steps=3, atlas=AtlasClient(
                KEY, base_url=server.url, poll_seconds=.01, timeout_seconds=5))
            target, trace = walk.select(graph, 'n00', graph.nodes, random.Random(1), exclude=('n00',))
        finally:
            server.close()
        self.assertEqual(len(trace['nodes']), 20)
        self.assertEqual(server.bodies[0][1]['params']['backend_name'], 'matrix_product_state')
        self.assertEqual(trace['method'], 'atlas-aer')
        small = FakeAtlas()
        try:
            ring = ring_graph()
            walk = self.walk(small)
            walk.select(ring, 'a', ring.nodes, random.Random(1), exclude=('a',))
        finally:
            small.close()
        self.assertEqual(small.bodies[0][1]['params']['backend_name'], 'automatic')

    def test_a_wide_window_without_atlas_is_simulated_exactly_with_numpy(self):
        from unittest.mock import patch
        from qualk.quantum_walk import QuantumProbeWalk
        graph = self.wide_graph(22)
        walk = QuantumProbeWalk(max_nodes=22, steps=3, backend='qiskit')
        with patch.object(walk.Statevector, 'from_instruction', side_effect=AssertionError('statevector used')):
            target, trace = walk.select(graph, 'n00', graph.nodes, random.Random(2), exclude=('n00',))
        self.assertEqual(trace['method'], 'numpy-trotter')
        self.assertEqual(len(trace['nodes']), 22)
        self.assertIsNotNone(target)
        self.assertAlmostEqual(sum(trace['quantum_probabilities']), 1., places=9)
        self.assertEqual(sum(trace['sampled_counts']), walk.shots)
        self.assertLess(trace['trotter_error'], .2)

    def test_the_qiskit_backend_never_contacts_atlas(self):
        graph = ring_graph()
        server = FakeAtlas()
        try:
            from qualk.atlas import AtlasClient
            from qualk.quantum_walk import QuantumProbeWalk
            walk = QuantumProbeWalk(max_nodes=4, backend='qiskit',
                                    atlas=AtlasClient(KEY, base_url=server.url))
            trace = walk.select(graph, 'a', graph.nodes, random.Random(1), exclude=('a',))[1]
        finally:
            server.close()
        self.assertEqual((trace['method'], server.submits), ('qiskit-statevector', 0))

    def test_without_a_key_the_default_backend_uses_qiskit_and_says_so_once(self):
        from qualk.quantum_walk import QuantumProbeWalk
        old = os.environ.pop('MOTH_API_KEY', None)
        out = io.StringIO()
        try:
            with redirect_stdout(out):
                walk = QuantumProbeWalk(max_nodes=4)
                trace = walk.select(ring_graph(), 'a', ring_graph().nodes, random.Random(1), exclude=('a',))[1]
        finally:
            if old is not None:
                os.environ['MOTH_API_KEY'] = old
        self.assertEqual(trace['method'], 'qiskit-statevector')
        self.assertEqual(out.getvalue().count('MOTH_API_KEY'), 1, out.getvalue())

    def test_backend_must_be_atlas_or_qiskit(self):
        from qualk.quantum_walk import QuantumProbeWalk
        with self.assertRaises(ValueError):
            QuantumProbeWalk(backend='ibm')


if __name__ == '__main__':
    unittest.main()
