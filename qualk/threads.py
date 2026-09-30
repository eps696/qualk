"""
Thread lifecycle (extracted from assembly's `stim/threads.py`; the sweep also reports what it closed).

`world-upd` already opens threads generously (see its own prompt); what it never did is
close any of them. Every dog run before this module existed for the life of the run held
nothing but open threads, growing without bound, picked by probes only as any other node.
This module is the mechanical half of a thread's life after it opens: normalizing an
extractor's ops before they reach the graph, and sweeping stale ones closed — no LLM call
of its own, so it stays cheap to run every round regardless of whether `--threads` is on.
The other half, actually advancing/resolving a thread against new material, is an LLM
call (`data/prompts/dog/thread-upd.md`), dispatched from `dogloop.DogEngine.step` because
it needs the run's own agent — this module only shapes and applies what comes back.

Every write below is an ordinary `{'op': 'thread', ...}` through `WorldGraph.apply()` — no
private state, so replay/fork/resume see thread closures exactly the way they see any other
graph write. `ThreadKeeper` itself is instantiated once per run and holds no graph reference.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

THREAD_DECAY = 0.95   # per story-tick decay of a thread's own `pressure`, for staleness only —
                      # distinct from WorldGraph.SALIENCE_DECAY, which governs retrieval, not
                      # this module's dormant/cap sweep
THREAD_FLOOR = 0.2    # pressure_eff below this closes a thread as dormant, unresolved by
                      # anyone — the mechanical half of "the pool stays tidy" when the LLM
                      # never gets around to resolving it
THREAD_MERGE = 0.85   # CLIP/SigLIP2 text cosine at or above which a freshly proposed thread
                      # is folded into an existing open one instead of opening a duplicate
THREAD_CAP_DEFAULT = 24


class ThreadKeeper:
    """Mechanical thread-pool hygiene: a grounding guard and a dedupe fold on the way in
    (`filter_ops`), a pressure/dormancy/cap sweep on the way out (`sweep`), and the
    pressure-weighted draw a probe aiming at an open thread uses (`pick`/`involved_ids`).
    No method here calls an LLM or touches anything but `graph`/`index`/`rng` passed in."""

    def __init__(self, decay: float = THREAD_DECAY, floor: float = THREAD_FLOOR,
                merge: float = THREAD_MERGE, cap: int = THREAD_CAP_DEFAULT):
        self.decay = decay
        self.floor = floor
        self.merge = merge
        self.cap = cap

    def pressure_eff(self, node: Any, now: int) -> float:
        """Derived, never written back — `node.traits['pressure']` stays whatever the last
        write set it to; staleness is measured against `last_seen`, which `WorldGraph._touch`
        already bumps on every resolve/ensure touching this node."""
        pressure = float(node.traits.get('pressure', 0.5) or 0.5)
        age = max(0, int(now) - int(node.last_seen))
        return pressure * (self.decay ** age)

    def open_threads(self, graph: Any) -> List[Any]:
        return [n for n in graph.nodes.values()
                if n.kind == 'thread' and n.traits.get('state') != 'resolved']

    def pick(self, graph: Any, rng: Any, now: int) -> Optional[str]:
        """Pressure-weighted open-thread draw — the `stim.breed.draw_thread` idea, revived
        for the v2 probe archive (see `exploration.ProbeArchive._thread_probe`)."""
        opens = self.open_threads(graph)
        if not opens:
            return None
        weights = [max(0.02, self.pressure_eff(n, now)) for n in opens]
        return rng.choices(opens, weights=weights, k=1)[0].id

    def involved_ids(self, graph: Any, tid: str) -> List[str]:
        """The thread's own grounded `part_of` subjects — see `WorldGraph._op_thread`,
        which asserts `subject part_of thread` for every name in an incoming op's
        `involves`. This is deliberately the only way a thread-aimed probe reaches
        component ids: the thread node's own name/gist text never becomes a query."""
        return [a.subject for a in graph.assertions.values()
                if a.open and a.pred == 'part_of' and a.object == tid]

    def filter_ops(self, ops: List[Dict[str, Any]], graph: Any, index: Any = None
                   ) -> List[Dict[str, Any]]:
        """Runs on every extraction batch (world-upd and thread-upd alike) before
        `graph.apply()`. Non-thread ops pass through untouched. A thread op:

        - opening (`state == 'open'`): dropped unless at least one of its `involves`
          names resolves to an existing element, or to a non-thread node this same
          batch is also proposing — an open thread with nothing concrete behind it is
          exactly the "hallucinated knowledge" `world-upd` is told to avoid, enforced
          here mechanically since prompting alone did not hold dog7's 62-for-62 record.
          A survivor is folded into an existing open thread (renamed onto it, so it
          becomes a touch rather than a duplicate) when semantic search finds one at or
          above `self.merge`.
        - advancing (`state == 'complicated'`) or closing (`state == 'resolved'`): the
          LLM's own `pressure` number, if any, is ignored — pressure is mechanical here,
          not model-estimated, so a fluent-sounding thread cannot buy itself a higher
          floor than a plainly-stated one.
        """
        out: List[Dict[str, Any]] = []
        batch_names = {str(op['name']).strip().lower() for op in ops
                       if op.get('op') == 'node' and op.get('kind') != 'thread' and op.get('name')}
        for op in ops:
            if op.get('op') != 'thread':
                out.append(op)
                continue
            state = op.get('state', 'open')
            if state == 'open':
                involves = [str(x).strip().lower() for x in (op.get('involves') or [])]
                grounded = any((graph.resolve(x) is not None) or x in batch_names for x in involves)
                if not grounded:
                    continue   # dropped: nothing concrete behind this open thread
                merged_name = self._merge_target(op, graph, index)
                op = {**op, 'pressure': 0.6}
                if merged_name:
                    op['name'] = merged_name
                    op['state'] = 'complicated'
            elif state == 'complicated':
                existing_id = graph.resolve(op.get('name'), 'thread')
                base = (graph.nodes[existing_id].traits.get('pressure', 0.6)
                       if existing_id in (graph.nodes or {}) else 0.6)
                op = {**op, 'pressure': min(1.0, float(base) + 0.15)}
            # 'resolved' passes through as given — its pressure no longer matters.
            out.append(op)
        return out

    def _merge_target(self, op: Dict[str, Any], graph: Any, index: Any) -> Optional[str]:
        if index is None:
            return None
        text = ' '.join(p for p in (op.get('name'), op.get('gist')) if p)
        if not text:
            return None
        try:
            qv = index.encode(text)
            hits = index.search(qv, prefix='node:', k=8)
        except Exception:
            return None
        for h in hits:
            if h.get('similarity', 0.0) < self.merge:
                continue
            nid = h['metadata'].get('node_id')
            n = graph.nodes.get(nid) if nid else None
            if n is not None and n.kind == 'thread' and n.traits.get('state') != 'resolved':
                return n.name
        return None

    def effective_open_names(self, graph: Any, now: int, top: int = 6) -> Optional[List[str]]:
        """Dog-scoped fix for `WorldGraph.open_threads()`'s own top-N (raw pressure,
        no recency — see `world/store.py`): re-ranks the SAME candidate pool by
        `pressure_eff` before taking the top `top`, so an old high-pressure thread
        that has gone stale actually loses its slot to something more current. This
        never touches `open_threads()`/`build_view()` themselves — every other mode
        keeps their exact byte-for-byte behaviour — a caller (`Digester.feed`,
        `stim.narrative.narrative_payload`) splices the result over `view['open']`
        after calling `build_view()` normally. Returns `None` when `self.decay >= 1.0`
        (no decay configured), so a caller can skip the splice and keep
        `build_view()`'s own selection exactly as given."""
        if self.decay >= 1.0:
            return None
        threads = self.open_threads(graph)
        ranked = sorted(threads, key=lambda n: -self.pressure_eff(n, now))
        return [t.name for t in ranked[:top]]

    def sweep(self, graph: Any, now: int) -> Dict[str, int]:
        """End-of-round hygiene: close anything dormant (mechanically, no LLM opinion
        needed), then, if still over `self.cap`, close the lowest-pressure overflow.
        This is what keeps the pool tidy even on a run where the LLM never once
        returns a `resolved` op of its own."""
        closed = {'dormant': 0, 'cap': 0, 'detail': []}       # detail: [(thread id, 'dormant'|'cap')]
        keep = []
        for n in self.open_threads(graph):
            if self.pressure_eff(n, now) < self.floor:
                graph.apply([{'op': 'thread', 'name': n.name, 'state': 'resolved'}],
                           at=now, by='threads:dormant')
                closed['dormant'] += 1
                closed['detail'].append((n.id, 'dormant'))
            else:
                keep.append(n)
        if len(keep) > self.cap:
            keep.sort(key=lambda n: self.pressure_eff(n, now))
            for n in keep[:len(keep) - self.cap]:
                graph.apply([{'op': 'thread', 'name': n.name, 'state': 'resolved'}],
                           at=now, by='threads:cap')
                closed['cap'] += 1
                closed['detail'].append((n.id, 'cap'))
        return closed

    def table(self, graph: Any, now: int) -> List[Dict[str, Any]]:
        """Every thread (open and closed) as plain data, for the round log and the UI: the state
        of the question at this moment. `involves` are the concept ids the thread is grounded on."""
        rows = []
        for n in graph.nodes.values():
            if n.kind != 'thread':
                continue
            rows.append({'id': n.id, 'name': n.name, 'gist': n.gist, 'state': n.traits.get('state', 'open'),
                         'pressure': round(float(n.traits.get('pressure', 0.5) or 0.5), 3),
                         'pressure_eff': round(self.pressure_eff(n, now), 3),
                         'involves': self.involved_ids(graph, n.id), 'first': n.first_seen, 'last': n.last_seen})
        return rows
