"""qualk (command line; the web app is `python app.py`): a quantum walk that chooses where an explorer looks next, populating a world graph.

    python run.py run  -o runs/demo --topic "Voynich manuscript" -n 8 --probe_walk quantum
    python run.py view -o runs/demo            # builds runs/demo/index.html
    python -m qualk.report -i runs/demo        # offline quantum-vs-diffusion analysis
"""

import argparse
import asyncio
import os
import random
import sys


def get_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('cmd', choices=['run', 'view'])
    p.add_argument('-o', '--out_dir', default='runs/demo')
    p.add_argument('-n', '--rounds', type=int, default=10)
    p.add_argument('--seed', default='', help='seed text, a .txt/.md file or a folder of them')
    p.add_argument('--topic', default='', help='seed the graph from a first web search on this topic')
    p.add_argument('--rng', type=int, default=0)
    p.add_argument('--explore', type=float, default=0.7)
    p.add_argument('--probe_walk', choices=['quantum', 'diffusion', 'classical'], default='quantum')
    p.add_argument('--quantum_nodes', type=int, default=12)
    p.add_argument('--quantum_steps', type=int, default=8)
    p.add_argument('--quantum_time', type=float, default=3.0)
    p.add_argument('--quantum_shots', type=int, default=1024)
    p.add_argument('--quantum_backend', choices=['atlas', 'qiskit'], default='atlas')
    p.add_argument('--search_order', default='')
    p.add_argument('--threads', action=argparse.BooleanOptionalAction, default=True,
                   help='track open questions: raise, advance, answer them, and aim probes at them')
    p.add_argument('--thread_aim', type=float, default=0.3, help='chance a fresh probe is aimed at an open question')
    p.add_argument('--thread_cap', type=int, default=24)
    p.add_argument('--node_gate', action=argparse.BooleanOptionalAction, default=True,
                   help='drop concepts without a description and claims naming unestablished concepts')
    p.add_argument('--orphan_focus', type=int, default=3,
                   help='show the extractor the nearest N concepts that have no relation yet (0 = off)')
    p.add_argument('--dedupe', type=float, default=0.10,
                   help='skip a fetched page closer than this (1 - cosine) to a page already read; 0 = off')
    return p.parse_args(argv)


def build_walk(a):
    from qualk.quantum_walk import DiffusionProbeWalk, QuantumProbeWalk
    if a.probe_walk == 'quantum':
        return QuantumProbeWalk(max_nodes=a.quantum_nodes, steps=a.quantum_steps, time=a.quantum_time,
                                shots=a.quantum_shots, backend=a.quantum_backend,
                                trace_dir=os.path.join(a.out_dir, 'quantum'))
    if a.probe_walk == 'diffusion':
        return DiffusionProbeWalk(max_nodes=a.quantum_nodes, time=a.quantum_time)
    return None


async def run(a):
    from qualk import web as W
    from qualk.embed import STEmbedder, configure
    from qualk.engine import Engine, seed_texts
    from qualk.llm import llm_extractor, thread_extractor
    from qualk.semantic import SemanticIndex
    if not W.search_available():
        sys.exit('No search key: set TAVILY_API_KEY, SERPER_API_KEY or BRAVE_API_KEY (see .env.example)')
    if a.search_order:
        W.set_provider_order(a.search_order)
    embedder = STEmbedder()
    print('affinity floor / duplicate distance:', configure(embedder))
    os.makedirs(a.out_dir, exist_ok=True)
    index = SemanticIndex(os.path.join(a.out_dir, 'semantic.sqlite'), embedder)
    engine = Engine(a.out_dir, llm_extractor(), index, W.WebSource(), walk=build_walk(a),
                    explore=a.explore, seed=a.rng, thread_extractor=thread_extractor(), threads=a.threads,
                    thread_aim=a.thread_aim, thread_cap=a.thread_cap, dedupe=a.dedupe,
                    node_gate=a.node_gate, orphan_focus=a.orphan_focus)
    if not engine.graph.nodes:
        parcels = seed_texts(a.seed, random.Random(a.rng))
        if a.topic:
            for _ in range(2):
                parcels += await engine.web.harvest(a.topic)
                engine.web.seen_urls.update(p.origin[4:] for p in parcels if p.origin.startswith('web:http'))
        if not parcels:
            sys.exit('Empty graph: give --seed or --topic')
        await engine.seed(parcels)
    def show(r):
        g = r.get('graph', {})
        walk = (r.get('probe') or {}).get('walk') or {}
        th = r.get('threads') or {}
        print(f"r{r['round']:03d} {r.get('kind')} nodes={g.get('nodes')} rel={g.get('relations')} "
              f"walk={walk.get('method') or walk.get('fallback', '-')} tvd={walk.get('total_variation_distance', '-')} "
              f"questions open={th.get('open', '-')} "
              f"q={r.get('query', '')[:60]!r}")
    await engine.run(a.rounds, on_round=show)


if __name__ == '__main__':
    from qualk.settings import load_env
    load_env()
    args = get_args()
    if args.cmd == 'run':
        asyncio.run(run(args))
    else:
        from qualk.viewer import build
        print('wrote', build(args.out_dir))
