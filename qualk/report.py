"""
Offline evidence for the graph walk - no LLM, no web, no Qiskit, read-only on the run.

For a finished run's `world.json` (+ `semantic.sqlite` for edges that predate the stored
`affinity` parameter) it builds the walk windows the live loop would build, and for several
evolution times compares the exact quantum walk with its classical twin (diffusion on the same
couplings). Output is the results table for the write-up:

    python -m qualk.report -i runs/demo
    python -m qualk.report -i runs/demo --times 1,2,3 --nodes 12 --seeds 300

Columns (per evolution time t, averaged over the sampled seeds):
  TVD        total-variation distance between the two distributions over the non-seed concepts
             (0 = identical, 1 = disjoint). The quantum walk is only distinguishable where this
             is clearly above 0.
  far q/d    probability mass on concepts two or more relation hops from the seed, quantum vs
             diffusion. Quantum walks spread further than diffusion.
  suppressed / amplified  share of windows where some concept (with at least 5% diffusion mass)
             is made less than half / more than twice as likely by the quantum walk - the
             signature of destructive / constructive interference.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sqlite3
import sys
from contextlib import closing

import numpy as np


from .quantum_walk import (coupling_matrix, diffusion_distribution, project_window,
                               quantum_distribution, relation_edges, seedable_nodes)
from .semantic import cosine
from .world.ops import LITERAL_PREDS, edge_role


def load_vectors(run_dir):
    """{node_id: vector} from the run's semantic index, opened read-only."""
    path = os.path.join(run_dir, 'semantic.sqlite')
    if not os.path.isfile(path):
        return {}
    with closing(sqlite3.connect(path)) as db:
        db.execute('PRAGMA query_only=ON')
        return {key[len('node:'):]: json.loads(vector) for key, vector in
                db.execute("SELECT key, vector FROM vectors WHERE key LIKE 'node:%'")}


def fill_affinity(graph, vectors):
    """Give open edges without a stored affinity one from the node vectors, in memory only
    (the report never writes the run). Returns how many were filled."""
    filled = 0
    for a in graph.assertions.values():
        if a.until is not None or a.affinity is not None or a.pred in LITERAL_PREDS:
            continue
        if a.subject in vectors and a.object in vectors:
            a.affinity = round(cosine(vectors[a.subject], vectors[a.object]), 4)
            filled += 1
    return filled


def _hops(window):
    """Relation-hop distance of every window node from the seed (node 0)."""
    n = len(window['nodes'])
    distance = [None] * n
    distance[0] = 0
    frontier = [0]
    while frontier:
        nxt = []
        for u in frontier:
            for e in window['edges']:
                for a, b in ((e['u'], e['v']), (e['v'], e['u'])):
                    if a == u and distance[b] is None:
                        distance[b] = distance[u] + 1
                        nxt.append(b)
        frontier = nxt
    return distance


def analyse(graph, times=(1., 2., 3.), max_nodes=12, seeds=300, rng=None):
    rng = rng or random.Random(0)
    eligible = [i for i, n in graph.nodes.items() if n.kind != 'thread']
    pairs = relation_edges(graph, eligible)
    seedable = sorted(seedable_nodes(graph, eligible))
    sample = seedable if len(seedable) <= seeds else sorted(rng.sample(seedable, seeds))
    scaffold = sum(1 for a in graph.assertions.values()
                   if a.until is None and a.frame == 'world' and a.pred not in LITERAL_PREDS
                   and a.subject in graph.nodes and a.object in graph.nodes
                   and edge_role(a.pred) == 'scaffold')
    windows = []
    for seed in sample:
        window = project_window(graph, seed, eligible, max_nodes)
        if len(window['nodes']) >= 2 and window['edges']:
            windows.append((window, coupling_matrix(window), _hops(window)))
    result = {
        'graph': {'nodes': len(eligible), 'seedable': len(seedable),
                  'relation_edges': len(pairs), 'scaffold_edges': scaffold,
                  'affinity_known': sum(1 for e in pairs.values() if e['affinity'] is not None)},
        'windows': len(windows), 'by_time': [],
        'mean_window': float(np.mean([len(w['nodes']) for w, _, _ in windows])) if windows else 0.,
        'loop_share': float(np.mean([len(w['edges']) >= len(w['nodes']) for w, _, _ in windows])) if windows else 0.,
        'hostile_share': float(np.mean([any(e['weight'] < 0 for e in w['edges']) for w, _, _ in windows])) if windows else 0.,
    }
    for t in times:
        tvd, far_q, far_c, suppressed, amplified = [], [], [], [], []
        for window, A, hops in windows:
            q, c = quantum_distribution(A, t), diffusion_distribution(A, t)
            q_rest, c_rest = np.array(q[1:]), np.array(c[1:])
            if q_rest.sum() <= 0 or c_rest.sum() <= 0:
                continue
            qn, cn = q_rest / q_rest.sum(), c_rest / c_rest.sum()
            tvd.append(0.5 * float(np.abs(qn - cn).sum()))
            far = np.array([h is None or h >= 2 for h in hops[1:]])
            far_q.append(float(qn[far].sum()))
            far_c.append(float(cn[far].sum()))
            ratio = qn / np.maximum(cn, 1e-9)
            considered = cn >= 0.05
            suppressed.append(bool(np.any(ratio[considered] < .5)))
            amplified.append(bool(np.any(ratio[considered] > 2.)))
        mean = lambda values: float(np.mean(values)) if values else 0.
        result['by_time'].append({'time': float(t), 'tvd': mean(tvd), 'far_quantum': mean(far_q),
                                  'far_diffusion': mean(far_c),
                                  'suppressed_share': mean(suppressed),
                                  'amplified_share': mean(amplified)})
    return result


def format_report(result, run_dir='', max_nodes=12):
    g = result['graph']
    lines = [f'graph walk report {run_dir}'.rstrip(),
             f"  non-thread nodes {g['nodes']}; relation edges {g['relation_edges']} "
             f"(scaffold edges, never walked: {g['scaffold_edges']}); "
             f"relation edges with affinity {g['affinity_known']}",
             f"  seedable nodes (relation cluster of >= 3): {g['seedable']} "
             f"({100 * g['seedable'] / max(1, g['nodes']):.0f}% of nodes)",
             f"  windows sampled {result['windows']} (cap {max_nodes} concepts): mean size "
             f"{result['mean_window']:.1f}, with a loop {100 * result['loop_share']:.0f}%, "
             f"with a hostile relation {100 * result['hostile_share']:.0f}%",
             '',
             '   t    TVD   far q / d     suppressed  amplified']
    for row in result['by_time']:
        lines.append(f"  {row['time']:>4g}  {row['tvd']:.2f}   {row['far_quantum']:.2f} / "
                     f"{row['far_diffusion']:.2f}    {100 * row['suppressed_share']:>6.0f}%   "
                     f"{100 * row['amplified_share']:>6.0f}%")
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('-i', '--run', required=True, help='dog run directory (world.json, semantic.sqlite)')
    parser.add_argument('--times', default='1,2,3', help='comma-separated evolution times')
    parser.add_argument('--nodes', type=int, default=12, help='window cap = qubits (2-16)')
    parser.add_argument('--seeds', type=int, default=300, help='seeds to sample')
    parser.add_argument('--seed', type=int, default=0, help='sampling seed')
    a = parser.parse_args(argv)
    from .world.store import WorldGraph
    graph = WorldGraph(a.run, fuzzy_names=False).load()
    if not graph.nodes:
        parser.error(f'no world graph in {a.run}')
    fill_affinity(graph, load_vectors(a.run))
    times = [float(t) for t in a.times.split(',') if t.strip()]
    print(format_report(analyse(graph, times, a.nodes, a.seeds, random.Random(a.seed)), a.run, a.nodes))


if __name__ == '__main__':
    main()
