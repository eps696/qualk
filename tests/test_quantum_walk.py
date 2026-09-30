"""Contracts for the graph walk: edge roles, window, quantum vs diffusion, archive integration."""
import os
import unittest

os.environ.pop("MOTH_API_KEY", None)   # tests must never reach the real Atlas platform


class EdgeRoleTests(unittest.TestCase):
    def test_core_families_split_into_scaffold_and_relation(self):
        from qualk.world.ops import edge_role
        for pred in ('is_a', 'in', 'part_of', 'has', 'at', 'attr'):
            self.assertEqual(edge_role(pred), 'scaffold', pred)
        for pred in ('caused', 'enables', 'precedes', 'fears', 'opposes', 'echoes',
                     'conceals', 'knows', 'contradicts'):
            self.assertEqual(edge_role(pred), 'relation', pred)

    def test_invented_containment_and_property_predicates_are_scaffold(self):
        from qualk.world.ops import edge_role
        for pred in ('has_internal_id', 'includes', 'contains', 'has_attr', 'on', 'covers',
                     'wears', 'date', 'above', 'in_front_of', 'has_mood', 'has_shape',
                     'is_lined_with', 'is_type_of', 'contains_icon', 'located_near'):
            self.assertEqual(edge_role(pred), 'scaffold', pred)

    def test_invented_acting_predicates_are_relation(self):
        from qualk.world.ops import edge_role
        for pred in ('triggers', 'reduces', 'provides', 'supports', 'explains', 'suppresses',
                     'lets_in', 'inhibits', 'evokes', 'suitable_for', 'uses', 'partners_with',
                     'commissioned_work_by', 'argues_against'):
            self.assertEqual(edge_role(pred), 'relation', pred)


def walk_graph(edges, threads=()):
    """edges: (u, v, pred, conf, valence, affinity[, frame[, until]]) tuples."""
    from types import SimpleNamespace
    from qualk.world.model import Assertion, Node
    names = sorted({name for edge in edges for name in edge[:2]} | set(threads))
    nodes = {n: Node.make('thread' if n in threads else 'motif', n, id=n, by='web:fixture')
             for n in names}
    assertions = {}
    for i, (u, v, pred, conf, valence, affinity, *rest) in enumerate(edges):
        assertions[str(i)] = Assertion(str(i), u, pred, v, conf=conf, valence=valence,
                                       affinity=affinity,
                                       frame=rest[0] if rest else 'world',
                                       until=rest[1] if len(rest) > 1 else None)
    return SimpleNamespace(nodes=nodes, assertions=assertions)


class RelationEdgeTests(unittest.TestCase):
    def test_only_open_world_relation_edges_between_nonthreads_enter(self):
        from qualk.quantum_walk import relation_edges
        graph = walk_graph([
            ('a', 'b', 'caused', .8, 0., .8),
            ('a', 'c', 'part_of', .9, 0., .9),            # scaffold
            ('a', 't', 'caused', .9, 0., .9),             # thread endpoint
            ('b', 'c', 'caused', .9, 0., .9, 'persona'),  # another standpoint
            ('b', 'd', 'caused', .9, 0., .9, 'world', 3), # closed
            ('c', 'c', 'caused', .9, 0., .9),             # self loop
        ], threads=('t',))
        edges = relation_edges(graph, graph.nodes)
        self.assertEqual(list(edges), [('a', 'b')])

    def test_weight_is_confidence_scaled_by_affinity_and_signed_by_valence(self):
        from qualk.quantum_walk import relation_edges
        graph = walk_graph([
            ('a', 'b', 'caused', .8, 0., 1.0),
            ('a', 'c', 'caused', .8, 0., .5),      # below the floor: weakest but present
            ('a', 'd', 'caused', .8, 0., None),    # unknown affinity is neutral
            ('a', 'e', 'opposes', .8, -.5, 1.0),   # hostile: negative coupling
        ])
        edges = relation_edges(graph, graph.nodes)
        self.assertAlmostEqual(edges[('a', 'b')]['weight'], .8)
        self.assertAlmostEqual(edges[('a', 'c')]['weight'], .32)
        self.assertAlmostEqual(edges[('a', 'd')]['weight'], .56)
        self.assertAlmostEqual(edges[('a', 'e')]['weight'], -.8)
        self.assertEqual(edges[('a', 'b')]['affinity'], 1.0)
        self.assertIsNone(edges[('a', 'd')]['affinity'])

    def test_strongest_of_parallel_edges_wins(self):
        from qualk.quantum_walk import relation_edges
        graph = walk_graph([('a', 'b', 'caused', .5, 0., 1.), ('b', 'a', 'enables', .9, 0., 1.)])
        edge = relation_edges(graph, graph.nodes)[('a', 'b')]
        self.assertAlmostEqual(edge['weight'], .9)
        self.assertEqual(edge['predicates'], ['enables'])


class SeedableTests(unittest.TestCase):
    def test_seedable_nodes_have_a_relation_cluster_of_at_least_three(self):
        from qualk.quantum_walk import seedable_nodes
        graph = walk_graph([
            ('a', 'b', 'caused', .8, 0., .8), ('b', 'c', 'caused', .8, 0., .8),   # 3-cluster
            ('x', 'y', 'caused', .8, 0., .8),                                     # pair
            ('p', 'q', 'part_of', .9, 0., .9), ('q', 'r', 'part_of', .9, 0., .9),  # scaffold only
        ])
        self.assertEqual(seedable_nodes(graph, graph.nodes), {'a', 'b', 'c'})


class WindowTests(unittest.TestCase):
    def test_window_grows_by_strongest_edge_and_is_capped_and_deterministic(self):
        from qualk.quantum_walk import project_window
        graph = walk_graph([('a', 'b', 'caused', .9, 0., 1.), ('a', 'c', 'caused', .7, 0., 1.),
                            ('b', 'd', 'caused', .8, 0., 1.), ('c', 'e', 'caused', .6, 0., 1.)])
        first = project_window(graph, 'a', graph.nodes, max_nodes=3)
        self.assertEqual(first['nodes'], ['a', 'b', 'd'])
        self.assertEqual(first, project_window(graph, 'a', graph.nodes, max_nodes=3))
        self.assertEqual({(e['a'], e['b']) for e in first['edges']}, {('a', 'b'), ('b', 'd')})

    def test_window_of_an_ineligible_or_isolated_seed_has_no_edges(self):
        from qualk.quantum_walk import project_window
        graph = walk_graph([('a', 'b', 'part_of', .9, 0., 1.)])
        self.assertEqual(project_window(graph, 'zzz', graph.nodes, 4), {'nodes': [], 'edges': []})
        self.assertEqual(project_window(graph, 'a', graph.nodes, 4)['edges'], [])


def ring(n, weight, flipped=()):
    import numpy as np
    A = np.zeros((n, n))
    for i in range(n):
        j = (i + 1) % n
        A[i, j] = A[j, i] = -weight if i in flipped else weight
    return A


class DistributionTests(unittest.TestCase):
    def test_ring_focuses_quantumly_but_diffuses_classically(self):
        import math
        from qualk.quantum_walk import diffusion_distribution, quantum_distribution
        A = ring(4, .8)
        t = math.pi / (2 * .8)
        quantum, classical = quantum_distribution(A, t), diffusion_distribution(A, t)
        self.assertAlmostEqual(quantum[2], 1., places=6)
        self.assertAlmostEqual(sum(classical), 1., places=9)
        self.assertLess(classical[2], .3)

    def test_a_frustrated_ring_changes_quantum_but_not_diffusion(self):
        from qualk.quantum_walk import diffusion_distribution, quantum_distribution
        plain, frustrated = ring(4, 1.), ring(4, 1., flipped=(2,))
        q1, q2 = quantum_distribution(plain, 1.3), quantum_distribution(frustrated, 1.3)
        self.assertGreater(abs(q1[2] - q2[2]), .5)
        d1, d2 = diffusion_distribution(plain, 1.3), diffusion_distribution(frustrated, 1.3)
        self.assertTrue(all(abs(a - b) < 1e-9 for a, b in zip(d1, d2)))

    def test_flipping_a_triangle_changes_nothing(self):
        import numpy as np
        from qualk.quantum_walk import quantum_distribution
        tri = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], float)
        flipped = np.array([[0, 1, 1], [1, 0, -1], [1, -1, 0]], float)
        for a, b in zip(quantum_distribution(tri, 1.2), quantum_distribution(flipped, 1.2)):
            self.assertAlmostEqual(a, b, places=9)


class TrotterNumpyTests(unittest.TestCase):
    def test_numpy_trotter_equals_the_qiskit_circuit_output(self):
        from qualk.quantum_walk import QuantumProbeWalk, trotter_distribution
        window = {'nodes': list('abcd'), 'edges': [
            {'u': 0, 'v': 1, 'weight': .8}, {'u': 1, 'v': 2, 'weight': -.6},
            {'u': 2, 'v': 3, 'weight': .7}, {'u': 0, 'v': 3, 'weight': .5}, {'u': 1, 'v': 3, 'weight': .4}]}
        walk = QuantumProbeWalk(max_nodes=4, steps=4, time=1.7)
        circuit = walk.circuit_probabilities(window)
        numpy_p = trotter_distribution(window, 1.7, 4)
        self.assertAlmostEqual(sum(numpy_p), 1., places=9)
        for a, b in zip(circuit, numpy_p):
            self.assertAlmostEqual(a, b, places=9)


def cluster_graph():
    """a - b - c - d chain plus a-c chord: one 4-concept relation cluster, one isolated node."""
    graph = walk_graph([('a', 'b', 'caused', .9, 0., .9), ('b', 'c', 'caused', .8, 0., .9),
                        ('c', 'd', 'caused', .7, 0., .9), ('a', 'c', 'enables', .6, 0., .9),
                        ('z', 'a', 'part_of', .9, 0., .9)])
    return graph


class QuantumWalkTests(unittest.TestCase):
    def test_parse_shot_accepts_exactly_one_excitation(self):
        from qualk.quantum_walk import parse_shot
        self.assertEqual(parse_shot('0001', 4), 0)
        self.assertEqual(parse_shot('1000', 4), 3)
        for bad in ('0000', '0110', '1111'):
            self.assertIsNone(parse_shot(bad, 4))

    def test_circuit_matches_the_exact_walk_and_stays_in_one_excitation_space(self):
        import math
        from qualk.quantum_walk import QuantumProbeWalk, quantum_distribution
        window = {'nodes': list('abcd'), 'edges': [
            {'u': i, 'v': (i + 1) % 4, 'weight': .8} for i in range(4)]}
        A = ring(4, .8)
        walk = QuantumProbeWalk(max_nodes=4, steps=5, time=2.0)
        circuit_p = walk.circuit_probabilities(window)
        self.assertAlmostEqual(sum(circuit_p), 1., places=9)
        exact = quantum_distribution(A, 2.0)
        self.assertLess(0.5 * sum(abs(a - b) for a, b in zip(circuit_p, exact)), .03)

    def test_select_measures_an_eligible_concept_and_logs_the_evidence(self):
        import random
        import tempfile
        from pathlib import Path
        from qualk.quantum_walk import QuantumProbeWalk
        graph = cluster_graph()
        with tempfile.TemporaryDirectory() as root:
            walk = QuantumProbeWalk(max_nodes=4, steps=5, time=2.0, shots=2048,
                                    trace_dir=Path(root))
            target, trace = walk.select(graph, 'a', graph.nodes, random.Random(3),
                                        exclude=('a',), round_idx=7)
            self.assertIn(target, ('b', 'c', 'd'))
            self.assertEqual(trace['method'], 'qiskit-statevector')
            self.assertEqual(trace['nodes'][0], 'a')
            self.assertNotIn('z', trace['nodes'])                # scaffold neighbour excluded
            self.assertAlmostEqual(sum(trace['quantum_probabilities']), 1., places=9)
            self.assertAlmostEqual(sum(trace['classical_probabilities']), 1., places=9)
            self.assertGreater(trace['total_variation_distance'], 0.)
            self.assertLess(trace['trotter_error'], .1)
            self.assertFalse(trace['trotter_warning'])
            self.assertEqual(sum(trace['sampled_counts']) + trace['rejected_shots'], 2048)
            self.assertEqual(trace['rejected_shots'], 0)
            self.assertFalse(trace['forced'])
            edge = trace['edges'][0]
            self.assertTrue({'conf', 'affinity', 'weight', 'predicates'} <= set(edge))
            self.assertIn('OPENQASM 3', Path(root, 'round-00007.qasm').read_text(encoding='utf-8'))
            again, repeated = walk.select(graph, 'a', graph.nodes, random.Random(3),
                                          exclude=('a',), round_idx=7)
            self.assertEqual((again, repeated['sampled_counts']), (target, trace['sampled_counts']))

    def test_single_option_is_flagged_forced(self):
        import random
        from qualk.quantum_walk import QuantumProbeWalk
        graph = walk_graph([('a', 'b', 'caused', .9, 0., .9), ('b', 'c', 'caused', .9, 0., .9)])
        target, trace = QuantumProbeWalk(max_nodes=3).select(
            graph, 'a', graph.nodes, random.Random(1), exclude=('a', 'b'))
        self.assertEqual(target, 'c')
        self.assertTrue(trace['forced'])

    def test_fallback_reasons_are_explicit(self):
        import random
        from qualk.quantum_walk import QuantumProbeWalk
        graph = cluster_graph()
        walk = QuantumProbeWalk(max_nodes=4)
        self.assertEqual(walk.select(graph, 'nope', graph.nodes, random.Random(1))[1]['fallback'],
                         'ineligible_seed')
        self.assertEqual(walk.select(graph, 'z', graph.nodes, random.Random(1))[1]['fallback'],
                         'isolated_seed')
        self.assertEqual(walk.select(graph, 'a', graph.nodes, random.Random(1),
                                     exclude=('a', 'b', 'c', 'd'))[1]['fallback'],
                         'no_eligible_target')

    def test_parameters_are_validated(self):
        from qualk.quantum_walk import QuantumProbeWalk
        for options in ({'max_nodes': 1}, {'max_nodes': 25}, {'steps': 0}, {'steps': 21},
                        {'shots': 0}, {'time': float('nan')}, {'time': 0}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                QuantumProbeWalk(**options)
        QuantumProbeWalk(max_nodes=24)

    def test_seed_candidates_are_the_seedable_nodes(self):
        from qualk.quantum_walk import QuantumProbeWalk
        graph = cluster_graph()
        self.assertEqual(QuantumProbeWalk().seed_candidates(graph, graph.nodes), {'a', 'b', 'c', 'd'})


class DiffusionWalkTests(unittest.TestCase):
    def test_diffusion_control_needs_no_qiskit_and_uses_the_same_window(self):
        import builtins
        import random
        from unittest.mock import patch
        from qualk.quantum_walk import DiffusionProbeWalk, project_window
        graph = cluster_graph()
        real_import = builtins.__import__

        def without_qiskit(name, *args, **kwargs):
            if name == 'qiskit' or name.startswith('qiskit.'):
                raise ImportError('fixture: qiskit unavailable')
            return real_import(name, *args, **kwargs)
        with patch('builtins.__import__', side_effect=without_qiskit):
            walk = DiffusionProbeWalk(max_nodes=4, time=2.0)
            target, trace = walk.select(graph, 'a', graph.nodes, random.Random(5), exclude=('a',))
        self.assertIn(target, ('b', 'c', 'd'))
        self.assertEqual(trace['method'], 'diffusion')
        window = project_window(graph, 'a', graph.nodes, 4)
        self.assertEqual(trace['nodes'], window['nodes'])
        self.assertEqual(len(trace['edges']), len(window['edges']))
        self.assertAlmostEqual(sum(trace['classical_probabilities']), 1., places=9)
        self.assertAlmostEqual(sum(trace['quantum_probabilities']), 1., places=9)
        self.assertNotIn('trotter_error', trace)

    def test_diffusion_shares_fallbacks_and_seed_rule_with_the_quantum_walk(self):
        import random
        from qualk.quantum_walk import DiffusionProbeWalk
        graph = cluster_graph()
        walk = DiffusionProbeWalk(max_nodes=4)
        self.assertEqual(walk.select(graph, 'z', graph.nodes, random.Random(1))[1]['fallback'],
                         'isolated_seed')
        self.assertEqual(walk.seed_candidates(graph, graph.nodes), {'a', 'b', 'c', 'd'})

    def test_diffusion_selection_replays_with_the_same_rng(self):
        import random
        from qualk.quantum_walk import DiffusionProbeWalk
        graph = cluster_graph()
        walk = DiffusionProbeWalk(max_nodes=4)
        picks = {walk.select(graph, 'a', graph.nodes, random.Random(9), exclude=('a',))[0]
                 for _ in range(3)}
        self.assertEqual(len(picks), 1)


class SeedRuleTests(unittest.TestCase):
    def graph(self):
        from qualk.world.model import Node
        graph = cluster_graph()
        for name in ('lone1', 'lone2', 'lone3'):        # nodes no relation reaches
            graph.nodes[name] = Node.make('motif', name, id=name, by='web:fixture')
        return graph

    def test_fresh_walk_probe_is_seeded_only_where_a_relation_cluster_exists(self):
        import random
        from qualk.exploration import ProbeArchive
        from qualk.quantum_walk import DiffusionProbeWalk
        graph = self.graph()
        walk = DiffusionProbeWalk(max_nodes=4)
        for seed in range(25):
            probe = ProbeArchive(random.Random(seed), explore=1.).next(
                graph, object(), 1, quantum_walk=walk)
            self.assertIn(probe.components[0], {'a', 'b', 'c', 'd'})
            self.assertEqual(probe.walk['method'], 'diffusion')
            self.assertEqual(len(probe.components), 2)

    def test_without_a_relation_cluster_the_classical_draw_runs_and_says_why(self):
        import random
        from qualk.exploration import ProbeArchive
        from qualk.quantum_walk import DiffusionProbeWalk
        graph = walk_graph([('x', 'y', 'caused', .9, 0., .9)])       # a pair: no arena
        probe = ProbeArchive(random.Random(1), explore=1.).next(
            graph, object(), 1, quantum_walk=DiffusionProbeWalk(max_nodes=4))
        self.assertEqual(probe.walk, {'fallback': 'no_seedable_cluster'})
        self.assertTrue(probe.components)

    def test_selection_reason_names_the_walk_not_a_quantum_claim(self):
        import random
        from qualk.exploration import ProbeArchive
        from qualk.quantum_walk import DiffusionProbeWalk
        probe = ProbeArchive(random.Random(2), explore=1.).next(
            self.graph(), object(), 1, quantum_walk=DiffusionProbeWalk(max_nodes=4))
        self.assertIn('graph walk', probe.selection)
        self.assertNotIn('coherent', probe.selection)


class TargetCooldownTests(unittest.TestCase):
    def star(self):
        return walk_graph([('hub', f'leaf{i}', 'caused', .9, 0., .9) for i in range(6)])

    def test_a_hub_is_not_picked_as_walk_target_again_within_the_cooldown(self):
        import random
        from qualk.exploration import ProbeArchive, TARGET_COOLDOWN
        from qualk.quantum_walk import DiffusionProbeWalk
        graph = self.star()
        archive = ProbeArchive(random.Random(3), explore=1.)
        walk = DiffusionProbeWalk(max_nodes=7)
        targets = []
        for at in range(1, TARGET_COOLDOWN + 1):
            probe = archive.next(graph, object(), at, quantum_walk=walk)
            targets.append(probe.walk.get('target'))
        self.assertLessEqual(targets.count('hub'), 1, targets)

    def test_recent_probe_components_are_excluded_as_targets(self):
        import random
        from qualk.exploration import Probe, ProbeArchive
        graph = self.star()
        seen = {}

        class Walk:
            def seed_candidates(self, graph, eligible_ids):
                return set(eligible_ids)

            def select(self, graph, seed_id, eligible_ids, rng, exclude=(), round_idx=None):
                seen['seed'], seen['exclude'] = seed_id, set(exclude)
                return None, {'fallback': 'fixture'}
        archive = ProbeArchive(random.Random(1), explore=1.)
        archive.probes = [Probe('p1', 0, ['leaf1'], 'x'), Probe('p2', 0, ['leaf2', 'leaf3'], 'y')]
        archive.sequence = 2
        archive.next(graph, object(), 3, quantum_walk=Walk())
        self.assertTrue({'leaf1', 'leaf2', 'leaf3', seen['seed']} <= seen['exclude'])

    def test_cooldown_does_not_apply_without_a_walk(self):
        import random
        from qualk.exploration import ProbeArchive
        archive = ProbeArchive(random.Random(2), explore=1.)
        graph = self.star()
        for at in range(1, 6):
            self.assertIsNotNone(archive.next(graph, object(), at))


class ArchiveSnapshotTests(unittest.TestCase):
    def test_snapshot_omits_walk_traces_but_restore_still_works(self):
        import json
        import random
        from qualk.exploration import Probe, ProbeArchive
        archive = ProbeArchive(random.Random(1))
        archive.probes = [Probe('p00001', 0, ['a'], 'a', walk={'big': list(range(500))})]
        archive.sequence = 1
        snapshot = json.loads(json.dumps(archive.snapshot()))
        self.assertNotIn('walk', snapshot['probes'][0])
        restored = ProbeArchive(random.Random(2))
        restored.restore(snapshot)
        self.assertEqual(restored.probes[0].walk, {})

