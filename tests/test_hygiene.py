"""Graph hygiene: the node gate, orphan focus and probe spread (ported from assembly's
stim_nodegate_test.py and extended for this repo's engine, records and configuration)."""
import asyncio
import json
import os
import random
import tempfile
import unittest
from collections import Counter

os.environ.pop("MOTH_API_KEY", None)

from qualk.digest import Digester
from qualk.nodegate import NodeGate
from qualk.web import Parcel
from qualk.world.store import WorldGraph


def node(name, gist='', kind='motif'):
    return {'op': 'node', 'kind': kind, 'name': name, 'gist': gist}


def claim(s, p, o):
    return {'op': 'assert', 'subject': s, 'pred': p, 'object': o}


class NodeGateTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.g = WorldGraph(self._tmp.name, fuzzy_names=False)
        self.g.apply([node('the lamp', 'a lamp that never goes out')], at=1, by='fixture')
        self.gate = NodeGate()

    def tearDown(self):
        self._tmp.cleanup()

    def test_node_without_gist_is_dropped(self):
        out = self.gate.filter_ops([node('five lux'), node('the door', 'a door')], self.g)
        self.assertEqual([o['name'] for o in out], ['the door'])
        self.assertEqual(self.gate.last['nodes'], 1)

    def test_gistless_restatement_of_a_gisted_node_passes(self):
        self.assertEqual(len(self.gate.filter_ops([node('the lamp')], self.g)), 1)

    def test_claim_needs_both_ends_gisted(self):
        ops = [node('the door', 'a door'),
               claim('the door', 'part_of', 'the lamp'),       # declared + existing
               claim('the door', 'part_of', 'some figure'),    # unknown end
               claim('ghost', 'caused', 'the lamp')]           # unknown subject
        out = self.gate.filter_ops(ops, self.g)
        self.assertEqual(len([o for o in out if o['op'] == 'assert']), 1)
        self.assertEqual(self.gate.last['claims'], 2)
        self.assertEqual(self.gate.last['ops'], 4)

    def test_literal_object_is_not_an_end(self):
        out = self.gate.filter_ops([claim('the lamp', 'attr', 'five lux')], self.g)
        self.assertEqual(len(out), 1)

    def test_threads_and_other_ops_pass(self):
        t = {'op': 'thread', 'name': 'what keeps it lit', 'state': 'open', 'involves': ['the lamp']}
        self.assertEqual(self.gate.filter_ops([t], self.g), [t])

    def test_no_bare_node_can_reach_the_graph(self):
        ops = self.gate.filter_ops([node('five lux'), claim('the lamp', 'has', 'five lux'),
                                    claim('the lamp', 'attr', 'warm')], self.g)
        self.g.apply(ops, at=2, by='fixture')
        self.assertTrue(all(n.gist for n in self.g.nodes.values() if n.kind != 'thread'))

    def test_counts_describe_only_the_latest_batch(self):
        self.gate.filter_ops([node('five lux')], self.g)
        self.assertEqual(self.gate.last['nodes'], 1)
        self.gate.filter_ops([node('the door', 'a door')], self.g)
        self.assertEqual(self.gate.last, {'ops': 1, 'nodes': 0, 'claims': 0})


class FixtureIndex:
    def encode(self, text):
        return [1.0, 0.0]

    def sync_graph(self, graph):
        pass

    def retrieve_nodes(self, vector, graph, k=6):
        return []

    def search(self, vector, prefix='', k=8, **kw):
        return [{'metadata': {'node_id': n.id}, 'similarity': 1.0 - i / 100, 'key': n.id}
                for i, n in enumerate(self.g.nodes.values())]


class OrphanFocusTests(unittest.TestCase):
    def test_nearest_unlinked_gisted_nodes_join_the_focus(self):
        with tempfile.TemporaryDirectory() as d:
            g = WorldGraph(d, fuzzy_names=False)
            g.apply([node('linked a', 'a'), node('linked b', 'b'), node('orphan one', 'o1'),
                     node('orphan two', 'o2'), node('bare')], at=1, by='fixture')
            g.apply([claim('linked a', 'caused', 'linked b'), claim('orphan one', 'part_of', 'linked a')],
                    at=1, by='fixture')
            idx = FixtureIndex()
            idx.g = g
            seen = {}

            async def extract(text, known):
                seen['present'] = [p['name'] for p in known['present']]
                return []

            dig = Digester(g, extract, semantic_index=idx, orphan_focus=2)
            asyncio.run(dig.feed(Parcel('x', 'fixture', 'folder'), at=2))
            self.assertIn('orphan one', seen['present'])      # scaffold-only (part_of) counts as an orphan
            self.assertIn('orphan two', seen['present'])
            self.assertNotIn('bare', seen['present'])         # no gist: never offered
            off = Digester(g, extract, semantic_index=idx)    # off by default
            off._nearest_orphans = lambda *a, **k: self.fail('orphan lookup ran while off')
            asyncio.run(off.feed(Parcel('x', 'fixture', 'folder'), at=3))

    def test_a_non_finite_embedding_costs_only_the_retrieval(self):
        with tempfile.TemporaryDirectory() as d:
            g = WorldGraph(d, fuzzy_names=False)
            g.apply([node('a', 'a')], at=1, by='fixture')

            class BrokenIndex(FixtureIndex):
                def retrieve_nodes(self, vector, graph, k=6):
                    raise ValueError('semantic vector must be nonempty and finite')

            idx = BrokenIndex()
            idx.g = g
            calls = []

            async def extract(text, known):
                calls.append(known)
                return []
            asyncio.run(Digester(g, extract, semantic_index=idx).feed(Parcel('x', 'fixture', 'folder'), at=2))
            self.assertEqual(len(calls), 1, 'extraction still ran')


class ProbeSpreadTests(unittest.TestCase):
    def archive_and_graph(self, names_and_gists, visits=None):
        from types import SimpleNamespace
        from qualk.exploration import ProbeArchive
        from qualk.world.model import Node
        nodes = {n: Node.make('motif', n, id=n, gist=g) for n, g in names_and_gists.items()}
        graph = SimpleNamespace(nodes=nodes, assertions={})
        archive = ProbeArchive(random.Random(1), explore=1.0)
        archive.visits = dict(visits or {})
        return archive, graph

    def test_constants_match_assembly(self):
        from qualk import exploration
        self.assertEqual((exploration.TARGET_COOLDOWN, exploration.VISIT_POWER), (24, 2))

    def test_only_described_concepts_are_probe_candidates(self):
        archive, graph = self.archive_and_graph({'a': 'about a', 'b': 'about b', 'c': 'about c', 'bare': ''})
        used = set()
        for i in range(30):
            probe = archive.next(graph, object(), i + 1)
            used.update(probe.components)
            archive.probes.clear()
        self.assertNotIn('bare', used)

    def test_when_too_few_are_described_nothing_is_filtered(self):
        archive, graph = self.archive_and_graph({'a': 'about a', 'bare1': '', 'bare2': ''})
        used = set()
        for i in range(30):
            used.update(archive.next(graph, object(), i + 1).components)
            archive.probes.clear()
        self.assertTrue({'bare1', 'bare2'} & used)

    def test_visit_pressure_is_squared(self):
        archive, graph = self.archive_and_graph({'fresh': 'f', 'seen': 's'}, visits={'seen': 3})
        counts = Counter()
        for i in range(4000):
            probe = archive.next(graph, object(), i + 1)
            counts[probe.components[0]] += 1
            archive.probes.clear()
            archive.visits = {'seen': 3}                      # keep the pressure fixed while sampling
        # weights 1 vs 1/16: fresh should win about 16:1, far beyond the 4:1 of the old linear rule
        self.assertGreater(counts['fresh'] / max(1, counts['seen']), 8)


class EngineGateTests(unittest.TestCase):
    def run_round(self, node_gate):
        from qualk.engine import Engine
        from qualk.fakes import FakeWeb, HashEmbedder
        from qualk.semantic import SemanticIndex

        async def sloppy(text, known):
            return {'ops': [node('real concept', 'a concept with a description'), node('figure 7'),
                            claim('real concept', 'precedes', 'the year 1998'),
                            claim('real concept', 'enables', 'real concept two'),
                            node('real concept two', 'another described concept')]}
        root = tempfile.mkdtemp(prefix='qualk_gate_')
        index = SemanticIndex(os.path.join(root, 'semantic.sqlite'), HashEmbedder())
        engine = Engine(root, sloppy, index, FakeWeb(), walk=None, explore=1.0, seed=3, node_gate=node_gate)
        asyncio.run(engine.seed([Parcel('seed text', 'seed:test', 'seed')]))
        record = asyncio.run(engine.step())
        index.close()
        return engine, record

    def test_the_gate_keeps_bare_names_out_and_reports_what_it_dropped(self):
        engine, record = self.run_round(True)
        self.assertTrue(all(n.gist for n in engine.graph.nodes.values() if n.kind != 'thread'))
        self.assertEqual(record['gated']['nodes'], 1)
        self.assertEqual(record['gated']['claims'], 1)
        self.assertNotIn('year', ' '.join(engine.graph.nodes))

    def test_without_the_gate_the_bare_nodes_get_in(self):
        engine, record = self.run_round(False)
        self.assertIsNone(record['gated'])
        self.assertTrue(any(not n.gist for n in engine.graph.nodes.values()))

    def test_a_clean_round_reports_nothing(self):
        from qualk.engine import Engine
        from qualk.fakes import FakeWeb, HashEmbedder, chain_extractor
        from qualk.semantic import SemanticIndex
        root = tempfile.mkdtemp(prefix='qualk_gate_')
        index = SemanticIndex(os.path.join(root, 'semantic.sqlite'), HashEmbedder())
        engine = Engine(root, chain_extractor(), index, FakeWeb(), walk=None, explore=1.0, seed=3, node_gate=True)
        asyncio.run(engine.seed([Parcel('seed text', 'seed:test', 'seed')]))
        record = asyncio.run(engine.step())
        index.close()
        self.assertIsNone(record['gated'])

    def test_the_gate_runs_before_the_thread_filter(self):
        """A thread may only be grounded on concepts the gate has already accepted."""
        from qualk.engine import Engine
        from qualk.fakes import FakeWeb, HashEmbedder
        from qualk.semantic import SemanticIndex

        async def extract(text, known):
            return {'ops': [node('bare one'), {'op': 'thread', 'name': 'What is the bare one?', 'state': 'open',
                                               'involves': ['bare one']}]}
        root = tempfile.mkdtemp(prefix='qualk_gate_')
        index = SemanticIndex(os.path.join(root, 'semantic.sqlite'), HashEmbedder())
        engine = Engine(root, extract, index, FakeWeb(), walk=None, explore=1.0, seed=3, node_gate=True, threads=True)
        asyncio.run(engine.seed([Parcel('seed text', 'seed:test', 'seed')]))
        index.close()
        self.assertFalse(any(n.kind == 'thread' for n in engine.graph.nodes.values()),
                         'a thread resting on a dropped concept must not open')


class ConfigTests(unittest.TestCase):
    def test_defaults_follow_assembly(self):
        from qualk.runner import RunConfig
        cfg = RunConfig()
        self.assertEqual((cfg.explore, cfg.thread_decay, cfg.orphan_focus, cfg.node_gate), (0.7, 0.93, 3, True))

    def test_orphan_focus_is_validated_and_live(self):
        from qualk.runner import RunConfig
        for bad in (-1, 11):
            with self.assertRaises(ValueError):
                RunConfig.from_dict({'orphan_focus': bad})
        self.assertEqual(RunConfig().merged({'orphan_focus': 0}).orphan_focus, 0)
        with self.assertRaises(ValueError):
            RunConfig().merged({'node_gate': False})          # not a live parameter


class SimulationShowsTheGate(unittest.TestCase):
    def test_simulated_runs_exercise_the_gate(self):
        from qualk.runner import EventBus, RunManager, SimulatedBackend
        import time
        root = tempfile.mkdtemp(prefix='qualk_gate_sim_')
        manager = RunManager(os.path.join(root, 'runs'), SimulatedBackend(delay=0.0), EventBus(), max_rounds=20)
        try:
            (rid,) = manager.create({'name': 'g1', 'seed': 'a seed', 'rounds': 6, 'nodes': 6, 'steps': 3, 'shots': 128, 'rng': 3})
            worker = manager.workers[rid]
            end = time.time() + 60
            while worker.is_alive() and time.time() < end:
                time.sleep(0.05)
            with open(os.path.join(worker.out_dir, 'rounds.jsonl'), encoding='utf-8') as f:
                records = [json.loads(l) for l in f]
            self.assertTrue(any(r.get('gated') for r in records))
            self.assertTrue(all(n.gist for n in worker.engine.graph.nodes.values() if n.kind != 'thread'))
        finally:
            manager.stop_all(10)


if __name__ == '__main__':
    unittest.main()
