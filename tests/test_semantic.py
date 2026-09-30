"""Edge affinity: the semantic index gives every relation edge the cosine of its endpoints."""
import os
import unittest

os.environ.pop("MOTH_API_KEY", None)   # tests must never reach the real Atlas platform


class AffinityFieldTests(unittest.TestCase):
    def test_affinity_is_omitted_when_unset_so_other_modes_serialise_unchanged(self):
        from qualk.world.model import Assertion
        a = Assertion('a1', 'x', 'caused', 'y')
        self.assertNotIn('affinity', a.to_dict())
        a.affinity = 0.42
        self.assertEqual(a.to_dict()['affinity'], 0.42)
        self.assertEqual(Assertion.from_dict(a.to_dict()).affinity, 0.42)
        self.assertIsNone(Assertion.from_dict({'id': 'a2', 'subject': 'x', 'pred': 'p',
                                               'object': 'y'}).affinity)

    def test_affinity_is_not_part_of_the_claim_identity(self):
        from qualk.world.model import Assertion
        a = Assertion('a1', 'x', 'caused', 'y')
        b = Assertion('a2', 'x', 'caused', 'y', affinity=0.9)
        self.assertTrue(a.same_values(b))
        self.assertEqual(a.key(), b.key())


class CountingEncoder:
    """Vectors chosen by node name so cosines are known; counts encode calls."""
    fingerprint = 'fixture:affinity'
    backend = 'fixture'
    VECTORS = {'alpha': [1., 0.], 'beta': [1., 1.], 'gamma': [0., 1.], 'moved': [-1., 1.]}

    def __init__(self):
        self.calls = 0

    def encode_text(self, texts):
        self.calls += len(texts)
        out = []
        for text in texts:
            key = 'moved' if 'moved' in text else next(
                (k for k in ('alpha', 'beta', 'gamma') if text.startswith(k)), 'gamma')
            out.append(list(self.VECTORS[key]))
        return out


def affinity_graph(root):
    from qualk.world.store import WorldGraph
    graph = WorldGraph(root, fuzzy_names=False)
    graph.apply([{'op': 'node', 'name': n, 'kind': 'motif', 'gist': f'{n} gist'}
                 for n in ('alpha', 'beta', 'gamma')] +
                [{'op': 'assert', 'subject': 'alpha', 'pred': 'caused', 'object': 'beta'},
                 {'op': 'assert', 'subject': 'beta', 'pred': 'triggers', 'object': 'gamma'},
                 {'op': 'assert', 'subject': 'alpha', 'pred': 'enables', 'object': 'gamma'},
                 {'op': 'assert', 'subject': 'alpha', 'pred': 'attr', 'object': 'brittle'}],
                at=0, by='web:fixture')
    return graph


def edge(graph, subject, pred, obj):
    ids = {n.name: n.id for n in graph.nodes.values()}
    return next(a for a in graph.assertions.values()
                if a.subject == ids[subject] and a.pred == pred and a.object == ids.get(obj, obj))


class AffinityIndexTests(unittest.TestCase):
    def sync(self, root, encoder):
        from qualk.semantic import SemanticIndex
        import os
        return SemanticIndex(os.path.join(root, 'semantic.sqlite'), encoder)

    def test_sync_fills_affinity_once_from_the_endpoint_vectors(self):
        import tempfile
        with tempfile.TemporaryDirectory() as root:
            graph, encoder = affinity_graph(root), CountingEncoder()
            with self.sync(root, encoder) as index:
                self.assertEqual(index.sync_graph(graph), 3)
                self.assertAlmostEqual(edge(graph, 'alpha', 'caused', 'beta').affinity, .7071, places=3)
                self.assertAlmostEqual(edge(graph, 'beta', 'triggers', 'gamma').affinity, .7071, places=3)
                self.assertAlmostEqual(edge(graph, 'alpha', 'enables', 'gamma').affinity, 0., places=3)
                self.assertIsNone(edge(graph, 'alpha', 'attr', 'brittle').affinity)
                calls = encoder.calls
                self.assertEqual(index.sync_graph(graph), 0)
                self.assertEqual(encoder.calls, calls)

    def test_revising_a_node_refreshes_only_edges_touching_it(self):
        import tempfile
        with tempfile.TemporaryDirectory() as root:
            graph, encoder = affinity_graph(root), CountingEncoder()
            with self.sync(root, encoder) as index:
                index.sync_graph(graph)
                beta = next(n for n in graph.nodes.values() if n.name == 'beta')
                beta.gist = 'moved on'
                self.assertEqual(index.sync_graph(graph), 2)
                self.assertAlmostEqual(edge(graph, 'alpha', 'caused', 'beta').affinity, -.7071, places=3)
                self.assertAlmostEqual(edge(graph, 'alpha', 'enables', 'gamma').affinity, 0., places=3)

    def test_closed_edges_and_edges_without_vectors_are_left_alone(self):
        import tempfile
        with tempfile.TemporaryDirectory() as root:
            graph, encoder = affinity_graph(root), CountingEncoder()
            edge(graph, 'alpha', 'enables', 'gamma').until = 1
            with self.sync(root, encoder) as index:
                self.assertEqual(index.sync_graph(graph), 2)
                self.assertIsNone(edge(graph, 'alpha', 'enables', 'gamma').affinity)

    def test_affinity_survives_graph_save_and_load(self):
        import tempfile
        from qualk.world.store import WorldGraph
        with tempfile.TemporaryDirectory() as root:
            graph, encoder = affinity_graph(root), CountingEncoder()
            with self.sync(root, encoder) as index:
                index.sync_graph(graph)
            graph.save()
            again = WorldGraph(root, fuzzy_names=False).load()
            self.assertAlmostEqual(edge(again, 'alpha', 'caused', 'beta').affinity, .7071, places=3)

