"""The stomach: a fetched page becomes graph material through one extraction pass.

A parcel is handed to an extractor (an LLM reading it against what the graph already holds),
and whatever ops come back are applied to the world graph. A parcel that yields no new nodes or
assertions is simply dropped: not "is this interesting" but "did this turn out to be about
anything". Trimmed from assembly's `stim/digest.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional

from .web import Parcel
from .world.ops import edge_role, parse_ops
from .world.store import WorldGraph
from .world.view import build_view

# (text, known_view) -> raw ops (list, dict-with-'ops', or a JSON string of either -
# world.ops.parse_ops accepts all three).
Extractor = Callable[[str, Dict[str, Any]], Awaitable[Any]]


@dataclass
class DigestResult:
    parcel: Parcel
    summary: Dict[str, int]
    kept: bool
    affected_nodes: List[str] = field(default_factory=list)
    ops: Optional[List[Dict[str, Any]]] = None
    error: str = ''


class Digester:
    def __init__(self, graph: WorldGraph, extractor: Extractor, budget: int = 2400, semantic_index=None,
                 op_filter=None, thread_ranker=None, orphan_focus: int = 0):
        self.op_filter = op_filter              # e.g. ThreadKeeper.filter_ops: drops ungrounded open threads
        self.thread_ranker = thread_ranker      # ThreadKeeper: ranks the `open` list by decayed pressure
        # How many nearest concepts with no relation edge are added to the extractor's focus, so
        # `known.present` shows them and the page can be related to them. 0 = off.
        self.orphan_focus = orphan_focus
        self.graph = graph
        self.extractor = extractor
        self.budget = budget
        self.semantic_index = semantic_index
        self.last_retrieval = []

    def _nearest_orphans(self, vector, skip) -> List[str]:
        """The nearest described concepts that no open *relation* claim touches: scaffold-only or
        isolated, so the walk never reaches them and nothing ever relates them to new material."""
        linked = set()
        for a in self.graph.open_assertions():
            if edge_role(a.pred) == 'relation':
                linked.update((a.subject, a.object))
        found: List[str] = []
        try:
            hits = self.semantic_index.search(vector, prefix='node:', k=60)
        except ValueError:
            return found
        for h in hits:
            nid = h['metadata'].get('node_id')
            n = self.graph.nodes.get(nid) if nid else None
            if n is None or not n.gist or n.kind == 'thread' or nid in linked or nid in skip or nid in found:
                continue
            found.append(nid)
            if len(found) >= self.orphan_focus:
                break
        return found

    async def feed(self, parcel: Parcel, at: int, query_vector=None) -> DigestResult:
        self.last_retrieval = []
        if not parcel.text.strip():
            return DigestResult(parcel=parcel, summary={}, kept=False, ops=[])
        focus = None
        if self.semantic_index is not None:
            self.semantic_index.sync_graph(self.graph)
            vector = query_vector if query_vector is not None else self.semantic_index.encode(parcel.text)
            try:
                self.last_retrieval = self.semantic_index.retrieve_nodes(vector, self.graph, k=6)
            except ValueError as e:
                # A non-finite embedding must cost only this page's neighbour retrieval, never the run.
                print(f'!! retrieval skipped ({e})')
                self.last_retrieval = []
            focus = [h['metadata']['node_id'] for h in self.last_retrieval]
            if self.orphan_focus > 0:
                focus += self._nearest_orphans(vector, set(focus))
        known = build_view(self.graph, focus=focus, budget=self.budget)
        if self.thread_ranker is not None:
            override = self.thread_ranker.effective_open_names(self.graph, at)
            if override is not None:
                known['open'] = override
        try:
            raw_ops = await self.extractor(parcel.text, known)
        except Exception as e:
            msg = f'!! extraction failed ({e}); parcel discarded ({parcel.origin})'
            print(msg.encode('ascii', 'backslashreplace').decode('ascii'))
            return DigestResult(parcel=parcel, summary={}, kept=False, ops=[], error=str(e)[:300])

        # Parsed once and handed to apply() already parsed, so the proposed ops survive
        # for the round log, not just the counts.
        ops = parse_ops(raw_ops)
        if self.op_filter is not None:
            ops = self.op_filter(ops, self.graph, self.semantic_index)
        summary = self.graph.apply(ops, at=at, by=parcel.origin)
        gained = summary.get('nodes', 0) + summary.get('assertions', 0) + summary.get('threads', 0)
        affected = set()
        for op in ops:
            for key in ('name', 'subject', 'object'):
                if op.get(key):
                    nid = self.graph.resolve(op[key])
                    if nid:
                        affected.add(nid)
        if self.semantic_index is not None:
            self.semantic_index.sync_graph(self.graph)
        return DigestResult(parcel=parcel, summary=summary, kept=gained > 0, ops=ops,
                            affected_nodes=sorted(affected))
