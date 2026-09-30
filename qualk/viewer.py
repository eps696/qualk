"""Build `index.html` for a finished (or running) run: a replay of the graph populating itself,
with the quantum walk's choice at every round. Self-contained except for d3 (cdnjs)."""

import json
import os
from pathlib import Path

from .world.ops import edge_role
from .world.store import WorldGraph

TEMPLATE = Path(__file__).with_name('viewer_template.html')


def load_run(out_dir):
    graph = WorldGraph(out_dir, fuzzy_names=False).load()
    nodes = [{'id': n.id, 'kind': n.kind, 'name': n.name, 'gist': n.gist, 'first': n.first_seen,
              'by': n.by} for n in graph.nodes.values()]
    edges = [{'id': a.id, 's': a.subject, 'p': a.pred, 'o': a.object, 'conf': a.conf,
              'val': a.valence, 'aff': a.affinity, 'since': a.since, 'until': a.until,
              'role': edge_role(a.pred)} for a in graph.assertions.values() if a.frame == 'world']
    rounds = []
    path = os.path.join(out_dir, 'rounds.jsonl')
    if os.path.isfile(path):
        with open(path, encoding='utf-8') as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rec.pop('fitness', None)                  # archive snapshots are large and not shown
                probe = rec.get('probe')
                if probe:
                    probe.pop('parents', None)
                rounds.append(rec)
    steer = []
    path = os.path.join(out_dir, 'steer.jsonl')
    if os.path.isfile(path):
        with open(path, encoding='utf-8') as f:
            for line in f:
                try:
                    steer.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    config = {}
    path = os.path.join(out_dir, 'config.json')
    if os.path.isfile(path):
        try:
            with open(path, encoding='utf-8') as f:
                config = json.load(f)
        except (OSError, json.JSONDecodeError):
            pass
    return {'nodes': nodes, 'edges': edges, 'rounds': rounds, 'steer': steer, 'config': config,
            'title': os.path.basename(os.path.abspath(out_dir))}


def build(out_dir):
    data = load_run(out_dir)
    graph_js = (Path(__file__).with_name('static') / 'graph.js').read_text(encoding='utf-8')
    page = TEMPLATE.read_text(encoding='utf-8').replace('/*__GRAPH_JS__*/', graph_js.replace('</script', '<\\/script'))
    page = page.replace('__DATA__', json.dumps(data, ensure_ascii=False).replace('</', '<\\/'))
    target = os.path.join(out_dir, 'index.html')
    with open(target, 'w', encoding='utf-8') as f:
        f.write(page)
    return target
