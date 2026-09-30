"""Offline quantum-vs-diffusion analysis."""
import os
import unittest

os.environ.pop("MOTH_API_KEY", None)   # tests must never reach the real Atlas platform


from tests.test_quantum_walk import cluster_graph, walk_graph


class ReportTests(unittest.TestCase):
    def test_fill_affinity_computes_missing_values_from_vectors_without_touching_set_ones(self):
        from qualk.report import fill_affinity
        graph = walk_graph([('a', 'b', 'caused', .9, 0., None), ('b', 'c', 'caused', .9, 0., .33)])
        filled = fill_affinity(graph, {'a': [1., 0.], 'b': [1., 1.], 'c': [0., 1.]})
        self.assertEqual(filled, 1)
        self.assertAlmostEqual(graph.assertions['0'].affinity, .7071, places=3)
        self.assertEqual(graph.assertions['1'].affinity, .33)

    def test_analyse_summarises_windows_and_quantum_vs_diffusion(self):
        import random
        from qualk.report import analyse
        graph = cluster_graph()
        result = analyse(graph, times=(1., 2.), max_nodes=4, seeds=10, rng=random.Random(1))
        self.assertEqual(result['graph']['seedable'], 4)
        self.assertEqual(result['graph']['relation_edges'], 4)
        self.assertEqual(result['windows'], 4)                  # one per seedable node
        self.assertAlmostEqual(result['loop_share'], 1.)       # the 4-chord cluster has a loop
        for row in result['by_time']:
            self.assertTrue(0. <= row['tvd'] <= 1.)
            self.assertTrue(0. <= row['far_quantum'] <= 1.)
            self.assertTrue(0. <= row['suppressed_share'] <= 1.)
        self.assertEqual([row['time'] for row in result['by_time']], [1., 2.])

