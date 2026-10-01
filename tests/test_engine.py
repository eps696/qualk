"""Full rounds offline: fake embedder, fake web, fake LLM - real graph, walk and circuit."""
import asyncio
import json
import os
import random
import tempfile
import unittest

from qualk.fakes import FakeWeb, HashEmbedder, chain_extractor

os.environ.pop("MOTH_API_KEY", None)


class EngineTests(unittest.TestCase):
    def run_engine(self, walk_kind, rounds=6):
        from qualk.engine import Engine
        from qualk.quantum_walk import DiffusionProbeWalk, QuantumProbeWalk
        from qualk.semantic import SemanticIndex
        from qualk.web import Parcel
        root = tempfile.mkdtemp(prefix='qualk_test_')
        walk = {'quantum': lambda: QuantumProbeWalk(max_nodes=6, steps=3, time=2., shots=256,
                                                    backend='qiskit', trace_dir=os.path.join(root, 'quantum')),
                'diffusion': lambda: DiffusionProbeWalk(max_nodes=6, time=2.)}[walk_kind]()
        index = SemanticIndex(os.path.join(root, 'semantic.sqlite'), HashEmbedder())
        web = FakeWeb()
        engine = Engine(root, chain_extractor(), index, web, walk=walk, explore=1.0, seed=3)
        asyncio.run(engine.seed([Parcel('seed text', 'seed:test', 'seed')]))
        asyncio.run(engine.run(rounds))
        index.close()
        with open(os.path.join(root, 'rounds.jsonl'), encoding='utf-8') as f:
            records = [json.loads(l) for l in f]
        return root, engine, web, records

    def test_quantum_rounds_grow_the_graph_and_log_the_circuit(self):
        root, engine, web, records = self.run_engine('quantum')
        probes = [r for r in records if r['kind'] == 'probe']
        self.assertEqual(len(probes), 6)
        sizes = [r['graph']['nodes'] for r in records]
        self.assertEqual(sizes, sorted(sizes))
        self.assertGreater(sizes[-1], sizes[0])
        walks = [r['probe']['walk'] for r in probes if r['probe']['walk'].get('target')]
        self.assertTrue(walks, 'once a relation cluster exists the walk must run')
        w = walks[-1]
        self.assertEqual(w['method'], 'qiskit-statevector')
        self.assertAlmostEqual(sum(w['quantum_probabilities']), 1., places=6)
        self.assertEqual(sum(w['sampled_counts']) + w['rejected_shots'], 256)
        self.assertTrue(os.path.isfile(os.path.join(root, 'quantum', os.path.basename(w['qasm_path']))))
        # the walk's target became a query term
        target = engine.graph.nodes[w['target']].name
        self.assertTrue(any(target in q for q in web.queries))

    def test_diffusion_control_runs_without_a_circuit(self):
        _root, _engine, _web, records = self.run_engine('diffusion')
        walks = [r['probe']['walk'] for r in records if r['kind'] == 'probe' and r['probe']['walk'].get('method')]
        self.assertTrue(walks)
        self.assertEqual(walks[-1]['method'], 'diffusion')

    def test_resume_continues_numbering_and_archive(self):
        from qualk.engine import Engine
        from qualk.semantic import SemanticIndex
        root, engine, web, records = self.run_engine('diffusion', rounds=3)
        index = SemanticIndex(os.path.join(root, 'semantic.sqlite'), HashEmbedder())
        again = Engine(root, chain_extractor(), index, FakeWeb(), walk=None, explore=1.0, seed=3)
        self.assertEqual(again.round, records[-1]['round'])
        self.assertEqual(len(again.archive.probes), len(engine.archive.probes))
        self.assertEqual(set(again.graph.nodes), set(engine.graph.nodes))
        index.close()


if __name__ == '__main__':
    unittest.main()
