"""The wave view's data: how a walk spreads over its window before it is measured."""
import asyncio
import math
import os
import tempfile
import unittest

import numpy as np

os.environ.pop("MOTH_API_KEY", None)

from qualk.evolution import evolution_series
from qualk.quantum_walk import coupling_matrix, diffusion_distribution, quantum_distribution


def ring(n, weight, flipped=()):
    """A window shaped like a ring of n concepts (node 0 is the seed)."""
    edges = []
    for i in range(n):
        j = (i + 1) % n
        w = -weight if (i, j) in flipped else weight
        edges.append({'u': min(i, j), 'v': max(i, j), 'weight': w})
    return {'nodes': [f'n{i}' for i in range(n)], 'edges': edges}


class EvolutionTests(unittest.TestCase):
    def test_shape_and_start(self):
        window = ring(4, 0.8)
        series = evolution_series(window, 2.0, points=9)
        self.assertEqual(len(series['times']), 9)
        self.assertEqual((series['times'][0], series['times'][-1]), (0.0, 2.0))
        for key in ('quantum', 'classical'):
            self.assertEqual(len(series[key]), 9)
            self.assertTrue(all(len(row) == 4 for row in series[key]))
            self.assertEqual(series[key][0], [1.0, 0.0, 0.0, 0.0], 'the walker starts on the seed')

    def test_probabilities_are_normalised_and_end_at_the_recorded_distributions(self):
        window = ring(6, 0.7, flipped={(1, 2)})
        T = 3.0
        series = evolution_series(window, T)
        A = coupling_matrix(window)
        for row in series['quantum']:
            self.assertAlmostEqual(sum(row), 1.0, places=2)
        for row in series['classical']:
            self.assertTrue(all(p >= 0 for p in row))
            self.assertAlmostEqual(sum(row), 1.0, places=2)
        np.testing.assert_allclose(series['quantum'][-1], quantum_distribution(A, T), atol=1e-4)
        np.testing.assert_allclose(series['classical'][-1], np.maximum(0, diffusion_distribution(A, T)), atol=1e-4)

    def test_ring_transfer_quantum_versus_diffusion(self):
        w = 0.8
        T = math.pi / (2 * w)                      # perfect state transfer on a 4-ring
        series = evolution_series(ring(4, w), T, points=12)
        self.assertGreater(series['quantum'][-1][2], 0.99)
        self.assertLess(series['classical'][-1][2], 0.3)
        # diffusion only ever approaches its limit, the quantum wave swings: it is not monotone
        opposite = [row[2] for row in series['quantum']]
        self.assertGreater(max(opposite[:-1]), 0.0)
        self.assertEqual(opposite[-1], max(opposite))

    def test_a_hostile_edge_changes_the_wave_but_not_diffusion(self):
        base = evolution_series(ring(4, 0.8), 2.0)
        flipped = evolution_series(ring(4, 0.8, flipped={(1, 2)}), 2.0)
        self.assertNotEqual(base['quantum'][-1], flipped['quantum'][-1])
        self.assertEqual(base['classical'], flipped['classical'])

    def test_validation(self):
        with self.assertRaises(ValueError):
            evolution_series(ring(4, 0.8), 2.0, points=1)


class EngineRecordsTheWave(unittest.TestCase):
    def run_rounds(self, rounds=4):
        from qualk.engine import Engine
        from qualk.fakes import FakeWeb, HashEmbedder, chain_extractor
        from qualk.quantum_walk import DiffusionProbeWalk
        from qualk.semantic import SemanticIndex
        from qualk.web import Parcel
        root = tempfile.mkdtemp(prefix='qualk_evo_')
        index = SemanticIndex(os.path.join(root, 'semantic.sqlite'), HashEmbedder())
        engine = Engine(root, chain_extractor(), index, FakeWeb(), walk=DiffusionProbeWalk(max_nodes=6, time=2.0),
                        explore=1.0, seed=3)
        asyncio.run(engine.seed([Parcel('seed text', 'seed:test', 'seed')]))
        records = [asyncio.run(engine.step()) for _ in range(rounds)]
        index.close()
        return records

    def test_a_walked_probe_carries_its_evolution(self):
        records = self.run_rounds(5)
        walked = [r['probe']['walk'] for r in records if r['probe']['walk'].get('target')]
        self.assertTrue(walked)
        for w in walked:
            evo = w['evolution']
            self.assertEqual(len(evo['quantum'][0]), len(w['nodes']))
            self.assertEqual(evo['times'][-1], w['time'])
            np.testing.assert_allclose(evo['classical'][-1], w['classical_probabilities'], atol=1e-3)

    def test_probes_without_a_walk_carry_none(self):
        records = self.run_rounds(5)
        for r in records:
            w = r['probe']['walk']
            if not w.get('quantum_probabilities'):
                self.assertNotIn('evolution', w)


if __name__ == '__main__':
    unittest.main()
