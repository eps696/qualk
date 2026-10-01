"""Minimal dog mode: an explorer that populates its own world graph.

One round:  world graph -> probe (the quantum walk picks a concept) -> web search -> LLM
extraction -> new nodes and relations in the graph -> novelty reward -> next probe.

Modelled on assembly's `kernel/dogloop.py` `DogEngine.step`, without the attention field,
narration, images, threads or UI. Every round is one line of `rounds.jsonl`, with the walk's full
trace (window, couplings, quantum vs diffusion probabilities, counts, circuit errors, backend).
"""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from .digest import Digester
from .evolution import evolution_series
from .nodegate import NodeGate
from .semantic import cosine
from .threads import ThreadKeeper
from .exploration import ProbeArchive, observation_novelty, score_observation
from .web import Parcel, WebSource, compose_query_terms
from .world.ops import edge_role, parse_ops
from .world.store import WorldGraph

NOVELTY_WINDOW = 32         # observations compared against when scoring novelty
DEDUPE_WINDOW = 400         # pages remembered when deciding that a new page is a duplicate
SEED_SPAN_CHARS = 1600


def _slim(node) -> Dict[str, Any]:
    return {'id': node.id, 'kind': node.kind, 'name': node.name, 'gist': node.gist}


def seed_texts(seed: str, rng: random.Random, spans: int = 2) -> List[Parcel]:
    """Seed material: a text file, a folder of .txt/.md files (random spans), or literal text."""
    if not seed:
        return []
    files: List[str] = []
    if os.path.isfile(seed):
        files = [seed]
    elif os.path.isdir(seed):
        for root, _dirs, names in os.walk(seed):
            files += [os.path.join(root, n) for n in names if n.lower().endswith(('.txt', '.md'))]
    if not files:
        return [Parcel(text=seed, origin='seed:text', source='seed')]
    out = []
    for _ in range(spans):
        path = rng.choice(files)
        with open(path, encoding='utf-8', errors='replace') as f:
            lines = [l for l in f.read().splitlines() if l.strip()]
        if not lines:
            continue
        width = min(rng.randint(3, 12), len(lines))
        start = rng.randint(0, len(lines) - width)
        text = '\n'.join(lines[start:start + width]).strip()[:SEED_SPAN_CHARS]
        out.append(Parcel(text=text, origin=f'seed:{os.path.basename(path)}#L{start + 1}-{start + width}',
                          source='seed'))
    return out


class Engine:
    def __init__(self, out_dir: str, extractor, index, web: Optional[WebSource], walk=None,
                 explore: float = 0.4, seed: int = 0, thread_extractor=None, threads: bool = False,
                 thread_aim: float = 0.3, thread_cap: int = 24, thread_decay: float = 0.95,
                 dedupe: float = 0.0, node_gate: bool = False, orphan_focus: int = 0):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.rng = random.Random(seed)
        self.graph = WorldGraph(out_dir, fuzzy_names=True, resolve_fuzz=0.90,
                                allow_gist_revision=True).load()
        self.index = index
        self.web = web
        self.walk = walk
        self.archive = ProbeArchive(self.rng, explore=explore)
        # Threads: open questions with a lifecycle. `keeper` is always there (it ranks the open
        # list the extractor sees); the filter, the thread-upd pass, the sweep and aiming probes
        # at threads only run when `threads` is on.
        self.threads_on = bool(threads)
        # A fetched page closer than this (1 - cosine) to a page already read is skipped, not digested.
        self.dedupe = max(0., float(dedupe))
        self._vectors: Dict[str, Any] = {}       # page text -> embedding, so a page is embedded once
        self.thread_aim = float(thread_aim)
        self.thread_extractor = thread_extractor
        self.keeper = ThreadKeeper(decay=thread_decay, cap=thread_cap)
        self.focus_thread: Optional[str] = None       # steered: aim every probe at this thread
        self._user_closed: set = set()
        # Graph hygiene. The gate drops nodes without a description and claims whose ends are not
        # established; it runs first, so a thread is grounded only on concepts the gate has vetted.
        self.gate = NodeGate() if node_gate else None

        def op_filter(ops, graph, index=None):
            if self.gate is not None:
                ops = self.gate.filter_ops(ops, graph, index)
            return self.keeper.filter_ops(ops, graph, index) if self.threads_on else ops
        self.digester = Digester(self.graph, extractor, semantic_index=index,
                                 op_filter=op_filter if (node_gate or self.threads_on) else None,
                                 thread_ranker=self.keeper, orphan_focus=orphan_focus)
        self.rounds_path = os.path.join(out_dir, 'rounds.jsonl')
        self.round = 0
        self._resume()
        self._tsnap = self._thread_state()

    def _resume(self) -> None:
        """Continue a finished run: last round number, probe archive, digested pages."""
        if not os.path.isfile(self.rounds_path):
            return
        last = None
        with open(self.rounds_path, encoding='utf-8') as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                last = rec
                url = (rec.get('source') or '')
                if url.startswith('web:http') and self.web is not None:
                    self.web.seen_urls.add(url[4:])
        if last:
            self.round = int(last.get('round') or 0)
            if last.get('fitness'):
                self.archive.restore(last['fitness'])

    # --- one round -----------------------------------------------------------------------

    def _log(self, record: Dict[str, Any]) -> None:
        with open(self.rounds_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + '\n')

    def _snapshot_ids(self):
        return set(self.graph.nodes), set(self.graph.assertions)

    async def seed(self, parcels: List[Parcel]) -> List[Dict[str, Any]]:
        """Round 0: whatever the run starts from goes through the same extraction pass."""
        records = []
        for parcel in parcels:
            before = self._snapshot_ids()
            t0 = time.time()
            result = await self.digester.feed(parcel, at=0)
            self.index.put(f'observation:seed{len(records):02d}', vector=self._embed(parcel.text),
                           metadata={'round': 0, 'label': parcel.origin})
            self.index.sync_graph(self.graph)
            threads = self._threads_block(0, parcel.origin)
            self.graph.save()
            record = {'round': 0, 'kind': 'seed', 'source': parcel.origin, 'excerpt': parcel.text[:400],
                      'kept': result.kept, 'summary': result.summary, 'error': result.error,
                      'delta': self._delta(before), 'graph': self._stats(), 'threads': threads,
                      'gated': self._gated(), 'seconds': round(time.time() - t0, 2)}
            self._log(record)
            records.append(record)
        return records

    def _gated(self) -> Optional[Dict[str, int]]:
        """What the node gate dropped from the latest extraction, or None if it dropped nothing."""
        if self.gate is None:
            return None
        last = self.gate.last
        return dict(last) if (last['nodes'] or last['claims']) else None

    def _delta(self, before) -> Dict[str, Any]:
        nodes0, edges0 = before
        new_nodes = [self.graph.nodes[i] for i in self.graph.nodes if i not in nodes0]
        new_edges = [a for i, a in self.graph.assertions.items() if i not in edges0]
        return {'nodes': [_slim(n) for n in new_nodes],
                'assertions': [{'id': a.id, 'subject': a.subject, 'pred': a.pred, 'object': a.object,
                                'conf': a.conf, 'valence': a.valence, 'affinity': a.affinity,
                                'role': edge_role(a.pred), 'frame': a.frame}
                               for a in new_edges]}

    def _stats(self) -> Dict[str, int]:
        open_edges = [a for a in self.graph.assertions.values() if a.open]
        return {'nodes': len(self.graph.nodes), 'assertions': len(open_edges),
                'relations': sum(1 for a in open_edges if edge_role(a.pred) == 'relation')}

    # --- threads ---------------------------------------------------------------------------

    def _thread_state(self) -> Dict[str, tuple]:
        return {n.id: (n.traits.get('state', 'open'), n.gist) for n in self.graph.nodes.values() if n.kind == 'thread'}

    def _threads_block(self, r: int, source: str, aimed: str = '', groundings=(), closed=None) -> Optional[Dict[str, Any]]:
        """What happened to the questions this round, as data: events (opened / advanced / resolved /
        dormant / capped / closed), counts, pool diagnostics and the full thread table."""
        if not self.threads_on:
            return None
        now = self._thread_state()
        why = dict((tid, kind) for tid, kind in (closed or {}).get('detail', []))
        events = []
        for tid, (state, gist) in now.items():
            before = self._tsnap.get(tid)
            node = self.graph.nodes[tid]
            kinds = []
            closing = {'dormant': 'dormant', 'cap': 'capped'}.get(why.get(tid)) or (
                'closed' if tid in self._user_closed else 'resolved')
            if before is None:
                kinds.append('opened')               # a thread born and closed within one interval: both events
                if state == 'resolved':
                    kinds.append(closing)
            elif before[0] != 'resolved' and state == 'resolved':
                kinds.append(closing)
            elif before[0] != 'resolved' and (before[1] != gist or before[0] != state):
                kinds.append('advanced')
            for kind in kinds:
                events.append({'id': tid, 'name': node.name, 'event': kind, 'gist': gist, 'source': source,
                               'involves': [self.graph.name_of(i) for i in self.keeper.involved_ids(self.graph, tid)]})
        self._tsnap = now
        counts = {k: sum(1 for e in events if e['event'] == k)
                  for k in ('opened', 'advanced', 'resolved', 'dormant', 'capped', 'closed')}
        opens = self.keeper.open_threads(self.graph)
        spread = None
        vecs = [v for v in (self.index.get('node:' + n.id) for n in opens) if v is not None]
        if len(vecs) >= 2:
            pairs = [cosine(vecs[i], vecs[j]) for i in range(len(vecs)) for j in range(i + 1, len(vecs))]
            spread = round(sum(pairs) / len(pairs), 3)
        focus_ended = None
        if self.focus_thread and (self.focus_thread not in now or now[self.focus_thread][0] == 'resolved'):
            focus_ended, self.focus_thread = self.focus_thread, None
        return {'open': len(opens), 'events': events, 'counts': counts, 'spread': spread,
                'grounding': round(sum(groundings) / len(groundings), 3) if groundings else None,
                'aimed': aimed, 'focus': self.focus_thread, 'focus_ended': focus_ended,
                'table': self.keeper.table(self.graph, r)}

    async def _update_threads(self, parcel: Parcel, at: int, vector, aimed: str = ''):
        """Ask whether this page advances or resolves already-open threads (never opens one: that is
        the world extractor's job). Candidates: the probe's own aimed thread first, then the open
        threads semantically nearest the page."""
        if not self.threads_on or self.thread_extractor is None:
            return []
        cands: Dict[str, Any] = {}
        aimed_node = self.graph.nodes.get(aimed) if aimed else None
        if aimed_node is not None and aimed_node.traits.get('state') != 'resolved':
            cands[aimed] = aimed_node
        for h in self.index.search(vector, prefix='node:', k=8):
            nid = h['metadata'].get('node_id')
            n = self.graph.nodes.get(nid) if nid else None
            if n is not None and n.kind == 'thread' and n.traits.get('state') != 'resolved':
                cands.setdefault(nid, n)
        for n in self.keeper.open_threads(self.graph):
            if len(cands) >= 5:
                break
            cands.setdefault(n.id, n)
        cand_list = list(cands.values())[:5]
        if not cand_list:
            return []
        payload = [{'name': n.name, 'gist': n.gist,
                    'involves': [self.graph.name_of(i) for i in self.keeper.involved_ids(self.graph, n.id)]}
                   for n in cand_list]
        try:
            raw = await self.thread_extractor(parcel.text, payload)
        except Exception as e:
            print(f'!! thread update failed ({str(e)[:120]}); skipping')
            return []
        valid = {n.name.strip().lower(): n for n in cand_list}
        ops = [op for op in parse_ops(raw) if op.get('op') == 'thread' and op.get('state') in ('complicated', 'resolved')
               and str(op.get('name', '')).strip().lower() in valid]
        ops = self.keeper.filter_ops(ops, self.graph, self.index)
        if not ops:
            return []
        groundings = []
        for op in ops:
            node = valid.get(str(op.get('name', '')).strip().lower())
            nv = self.index.get('node:' + node.id) if node is not None else None
            if nv is not None:
                groundings.append(round(cosine(vector, nv), 3))
        self.graph.apply(ops, at=at, by=parcel.origin)
        return groundings

    async def _thread_pass(self, parcel: Parcel, r: int, vector, aimed: str = '', source: str = '') -> Optional[Dict[str, Any]]:
        if not self.threads_on:
            return None
        groundings = await self._update_threads(parcel, r, vector, aimed)
        closed = self.keeper.sweep(self.graph, r)
        return self._threads_block(r, source or parcel.origin, aimed, groundings, closed)

    def ask(self, question: str, involves=None) -> str:
        """The user's own open question: a thread grounded on the named concepts (or, failing that,
        the concepts nearest the question). Returns the thread id."""
        question = ' '.join(str(question or '').split())[:200]
        if len(question) < 4:
            raise ValueError('the question is too short')
        grounded = []
        for name in involves or []:
            nid = self.graph.resolve(str(name))
            if nid and self.graph.nodes[nid].kind != 'thread' and nid not in grounded:
                grounded.append(nid)
        if not grounded:
            for h in self.index.search(self.index.encode(question), prefix='node:', k=12):
                nid = h['metadata'].get('node_id')
                if nid in self.graph.nodes and self.graph.nodes[nid].kind != 'thread' and nid not in grounded:
                    grounded.append(nid)
                if len(grounded) >= 3:
                    break
        if not grounded:
            raise ValueError('there is no concept in the graph to ground that question on yet')
        self.graph.apply([{'op': 'thread', 'name': question, 'gist': question, 'state': 'open', 'pressure': 0.9,
                           'involves': [self.graph.nodes[i].name for i in grounded]}], at=self.round, by='user')
        self.index.sync_graph(self.graph)
        self.graph.save()
        return self.graph.resolve(question, 'thread')

    def close_thread(self, tid: str) -> None:
        node = self.graph.nodes.get(tid)
        if node is None or node.kind != 'thread':
            raise ValueError('no such thread')
        self._user_closed.add(tid)
        self.graph.apply([{'op': 'thread', 'name': node.name, 'state': 'resolved'}], at=self.round, by='user')
        self.graph.save()

    def set_focus(self, tid: Optional[str]) -> None:
        if tid is not None:
            node = self.graph.nodes.get(tid)
            if node is None or node.kind != 'thread' or node.traits.get('state') == 'resolved':
                raise ValueError('that thread is not open')
        self.focus_thread = tid

    # --- live steering (called between rounds) ------------------------------------------

    def set_walk(self, walk) -> None:
        self.walk = walk

    def _embed(self, text: str):
        key = hash(text)
        if key not in self._vectors:
            if len(self._vectors) > 64:
                self._vectors.clear()
            self._vectors[key] = self.index.encode(text)
        return self._vectors[key]

    def remember_page(self, parcel: Parcel, key: str) -> None:
        """Count a page as read for the duplicate filter (used for the pages that seed a run)."""
        self.index.put(f'observation:{key}', vector=self._embed(parcel.text),
                       metadata={'round': 0, 'label': parcel.origin})

    def _duplicate_check(self, parcel: Parcel) -> Optional[Dict[str, Any]]:
        """None if the page is new enough; otherwise why it is skipped. Compares against every page
        read so far (seeds, searches, injections), not just the recent window used for novelty."""
        if self.dedupe <= 0:
            return None
        hits = self.index.search(self._embed(parcel.text), prefix='observation:', k=1, recent=DEDUPE_WINDOW)
        if not hits:
            return None
        distance = 1.0 - max(0., hits[0]['similarity'])
        if distance < self.dedupe:
            return {'reason': 'near-duplicate', 'distance': round(distance, 4),
                    'of': hits[0]['metadata'].get('label') or hits[0]['key'].replace('observation:', '')}
        return None

    def set_dedupe(self, distance: float) -> None:
        self.dedupe = max(0., float(distance))

    def set_explore(self, explore: float) -> None:
        self.archive.explore = max(0., min(1., float(explore)))

    async def inject(self, parcel: Parcel) -> Dict[str, Any]:
        """A manual observation (pasted text or a fetched URL): digested like any page, logged as
        a round of kind 'inject'. It rewards no probe."""
        self.round += 1
        r, t0 = self.round, time.time()
        before = self._snapshot_ids()
        vector = self._embed(parcel.text)
        result = await self.digester.feed(parcel, at=r, query_vector=vector)
        self.index.put(f'observation:r{r:05d}', vector=vector, metadata={'round': r, 'label': parcel.origin})
        threads = await self._thread_pass(parcel, r, vector)
        self.index.sync_graph(self.graph)
        self.graph.save()
        record = {'round': r, 'kind': 'inject', 'source': parcel.origin, 'excerpt': parcel.text[:400],
                  'kept': result.kept, 'summary': result.summary, 'error': result.error,
                  'delta': self._delta(before), 'graph': self._stats(), 'threads': threads,
                  'gated': self._gated(), 'fitness': self.archive.snapshot(), 'seconds': round(time.time() - t0, 2)}
        self._log(record)
        return record

    async def step(self, pinned_seed: Optional[str] = None) -> Dict[str, Any]:
        self.round += 1
        r, t0 = self.round, time.time()
        self.index.sync_graph(self.graph)
        probe = self.archive.next(self.graph, self.index, r, quantum_walk=self.walk,
                                  seed_override=pinned_seed,
                                  threads=self.keeper if self.threads_on else None,
                                  thread_aim=self.thread_aim, thread_focus=self.focus_thread)
        if probe is None:
            record = {'round': r, 'kind': 'idle', 'note': 'empty graph: nothing to probe',
                      'graph': self._stats()}
            self._log(record)
            return record
        walk_trace = probe.walk
        if walk_trace.get('quantum_probabilities') and walk_trace.get('edges') and walk_trace.get('time'):
            # how the walk spread over its window before it was measured, for the wave view in the UI
            walk_trace['evolution'] = evolution_series(
                {'nodes': walk_trace['nodes'], 'edges': walk_trace['edges']}, walk_trace['time'])
        pairs = [(self.graph.nodes[c].name, self.graph.nodes[c].gist) for c in probe.components]
        query = ' '.join(compose_query_terms(pairs))
        parcels = await self.web.harvest(query, accept=self._duplicate_check) if self.web is not None else []
        attempt = dict(self.web.last_attempt) if self.web is not None else {}
        record: Dict[str, Any] = {
            'round': r, 'kind': 'probe', 'query': query, 'harvest': attempt,
            'probe': {**{k: v for k, v in asdict(probe).items() if k != 'outcomes'},
                      'components': [_slim(self.graph.nodes[c]) for c in probe.components]}}
        threads = None
        if not parcels:
            self.archive.reward(probe.id, novelty=0.0)
            record.update(source='', kept=False, delta={'nodes': [], 'assertions': []})
            if self.threads_on:
                self.keeper.sweep(self.graph, r)
            threads = self._threads_block(r, '', probe.thread)
        else:
            parcel = parcels[0]
            before = self._snapshot_ids()
            vector = self._embed(parcel.text)
            result = await self.digester.feed(parcel, at=r, query_vector=vector)
            prior = self.index.search(vector, prefix='observation:', k=1, recent=NOVELTY_WINDOW)
            distance = 1.0 - max(0., prior[0]['similarity']) if prior else None
            novelty = observation_novelty(distance)
            metrics = score_observation(novelty, 1.0 if result.kept else 0.0)
            self.index.put(f'observation:r{r:05d}', vector=vector, metadata={'round': r, 'label': parcel.origin})
            self.archive.reward(probe.id, novelty=novelty, attention=metrics['score'])
            record.update(source=parcel.origin, excerpt=parcel.text[:400], kept=result.kept,
                          summary=result.summary, error=result.error, delta=self._delta(before),
                          novelty=round(novelty, 4), distance=None if distance is None else round(distance, 4),
                          score=round(metrics['score'], 4), gated=self._gated())
            threads = await self._thread_pass(parcel, r, vector, aimed=probe.thread)
        self.index.sync_graph(self.graph)
        self.graph.save()
        record.update(graph=self._stats(), threads=threads, fitness=self.archive.snapshot(),
                      seconds=round(time.time() - t0, 2))
        self._log(record)
        return record

    async def run(self, rounds: int, on_round=None) -> None:
        for _ in range(rounds):
            record = await self.step()
            if on_round:
                on_round(record)
