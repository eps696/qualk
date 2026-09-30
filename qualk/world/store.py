"""
WorldGraph — the store: apply ops, resolve names to elements, persist.

Owns its own two files and touches StoryState not at all. That is deliberate and
load-bearing, not fastidiousness: StoryState.merge_data() silently drops any
top-level key absent from a mode's initial schema (base.py:203-205), and infers
list identity from any key whose name contains 'name' or 'number'
(base.py:207-212) — an `id` field is not recognised, so routing assertions through
it would duplicate every edge on every merge. Going around it also keeps a growing
graph out of log.json, which is rewritten whole on every save.

The shape mirrors Blackboard (kernel/bboard.py): an append-only `world.jsonl` of
everything that was ever said to the graph, plus `world.json` as the materialised
projection, fully rebuildable by replaying the log.

Dependency-free by the same reasoning as kernel/signals.py — stdlib difflib, no
embeddings. Swapping in a semantic matcher later means replacing `_best_match()`
and nothing else.
"""

from __future__ import annotations

import json
import os
from difflib import SequenceMatcher
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from .model import (Assertion, Node, FRAME_WORLD, KIND_THREAD, KINDS,
                    SHOWN_RANK, coerce_kind, coerce_num, coerce_str, norm_name)
from .ops import (LITERAL_PREDS, OP_ASSERT, OP_NODE, OP_RETRACT, OP_SHOW,
                  OP_THREAD, parse_ops)
from .conflict import KIND_FUNCTIONAL, find_conflicts

# A false merge is unrecoverable — two elements become one and the prose that
# distinguished them is already written. A duplicate is merely visible and
# repairable. So this sits well above util.fuzzy_find's 0.65 default.
RESOLVE_FUZZ = 0.80

SALIENCE_BUMP  = 0.35   # asymptotic: repeated mention approaches 1.0, never jumps to it
SALIENCE_DECAY = 0.93   # per step of story time, for anything not mentioned
SALIENCE_FLOOR = 0.05


class WorldGraph:
    """Elements and the claims that hold between them."""

    def __init__(self, out_dir: str, verbose: bool = False, *,
                fuzzy_names: bool = True,
                resolve_fuzz: float = RESOLVE_FUZZ,
                allow_gist_revision: bool = False,
                on_fuzzy_merge: Optional[Callable[[str, 'Node', float], None]] = None):
        self.out_dir = out_dir
        self.verbose = verbose
        self.fuzzy_names = fuzzy_names
        # Per-instance override of the module default — dog runs with fuzzy_names=True
        # again (see its own comment further down and lib/dog.py::load_dog_world) but
        # at a stricter bar than prose narration needs, since a name a caption mints
        # ("Main skull at circular frame") tends to paraphrase its own prior mention
        # ("...in circular frame") rather than coin a wholly new spelling.
        self.resolve_fuzz = resolve_fuzz
        # Both default off/None so chat/conspir/flow/ingest and both selftests
        # are byte-identical — same "off by default, a mode opts in" stance
        # `apply()`'s `gate=` already takes.
        #
        # `allow_gist_revision`: `_op_node` normally never overwrites a node's
        # first-written gist (see its own comment) — right for a narrative,
        # where the first line to establish something is usually the one that
        # matters, wrong for a mode that re-probes the same region and needs
        # what it re-reads to actually update. Dog opts in.
        #
        # `on_fuzzy_merge(ref, node, score)`: called from `resolve()` whenever
        # a *non-exact* match fires — how much incoming material is being
        # silently absorbed into an existing node is otherwise invisible.
        self.allow_gist_revision = allow_gist_revision
        self.on_fuzzy_merge = on_fuzzy_merge
        self.nodes: Dict[str, Node] = {}
        self.assertions: Dict[str, Assertion] = {}
        self._seq = 0
        self._at = 0           # highest story time seen, for decay steps
        self._conflicts: List[Any] = []   # from the most recent apply(); see last_conflicts()
        self.json_path  = os.path.join(out_dir, 'world.json')
        self.jsonl_path = os.path.join(out_dir, 'world.jsonl')

    # --- resolution ---

    def _best_match(self, name: str, kind: Optional[str] = None) -> Tuple[Optional[Node], float]:
        """Closest existing element by name or alias.

        Deliberately argmax rather than first-over-threshold (which is what
        util.fuzzy_find does): with several candidates above the bar, the *best*
        one is the only defensible merge, and aliases mean each node has several
        surface forms to score against.
        """
        target = norm_name(name)
        if not target:
            return None, 0.0
        best, score = None, 0.0
        for n in self.nodes.values():
            if kind and n.kind != kind:
                continue
            for cand in n.names():
                r = SequenceMatcher(None, target, norm_name(cand)).ratio()
                if r > score:
                    best, score = n, r
        return best, score

    def resolve(self, name: str, kind: Optional[str] = None) -> Optional[str]:
        """Map a name (or an id) to an existing element id. None if unknown."""
        ref = coerce_str(name, 300)
        if not ref:
            return None
        if ref in self.nodes:
            return ref
        target = norm_name(ref)
        # Exact on name or alias, preferring the requested kind.
        exact = [n for n in self.nodes.values() if target in [norm_name(x) for x in n.names()]]
        if exact:
            same = [n for n in exact if not kind or n.kind == kind]
            return (same or exact)[0].id
        # A caller may still disable fuzzy matching outright (fuzzy_names=False) —
        # not dog's current stance, but kept for a mode/tool that wants exact-only.
        if not self.fuzzy_names:
            return None
        node, score = self._best_match(ref, kind)
        if node and score >= self.resolve_fuzz:
            if self.on_fuzzy_merge:
                self.on_fuzzy_merge(ref, node, score)
            return node.id
        # A kind-filtered search that found nothing may still match across kinds:
        # the writer's guess at a kind is far less reliable than the name it used.
        if kind:
            node, score = self._best_match(ref, None)
            if node and score >= self.resolve_fuzz:
                if self.on_fuzzy_merge:
                    self.on_fuzzy_merge(ref, node, score)
                return node.id
        return None

    def ensure(self, name: str, kind: Any = None, at: int = 0, **kw) -> str:
        """Resolve, or create a stub. Stubs are the normal case — an element
        mentioned once should cost one line and nothing more until it matters."""
        kind = coerce_kind(kind) if kind else None
        found = self.resolve(name, kind)
        if found:
            self._touch(found, at)
            return found
        node = Node.make(kind or 'motif', name, at=at, **kw)
        while node.id in self.nodes:   # distinct names, colliding slugs
            node.id += '-2'
        self.nodes[node.id] = node
        if self.verbose:
            print(f'   + {node.id}')
        return node.id

    def _touch(self, node_id: str, at: int) -> None:
        n = self.nodes.get(node_id)
        if not n:
            return
        n.last_seen = max(n.last_seen, int(at))
        n.salience = min(1.0, n.salience + SALIENCE_BUMP * (1.0 - n.salience))

    def _decay(self, steps: int) -> None:
        """Unmentioned material sinks out of retrieval rather than being deleted.
        Runs inside apply() on story-time advance, so it stays part of the
        deterministic replay instead of depending on who remembered to call it."""
        if steps <= 0:
            return
        factor = SALIENCE_DECAY ** min(steps, 50)
        for n in self.nodes.values():
            n.salience = max(SALIENCE_FLOOR, n.salience * factor)

    # --- writing ---

    def apply(self, payload: Any, at: int = 0, by: str = '', log: bool = True,
              gate: bool = False) -> Dict[str, int]:
        """Apply a batch of ops. Returns a small summary for the caller to report.

        `at` is story time (fragment/tick number) and `by` is provenance — the
        thing that established these claims, e.g. 'frag:11'.

        `gate` turns on the contradiction check in world/conflict.py: a new claim on
        a single-valued predicate closes the standing one instead of standing beside
        it, and opposed claims are counted so they are visible. Off by default so
        existing runs are byte-identical; the `world-gate` aux step turns it on.
        """
        ops = parse_ops(payload)
        at = int(at or 0)
        if at > self._at:
            self._decay(at - self._at)
            self._at = at

        summary = {'nodes': 0, 'assertions': 0, 'restated': 0, 'superseded': 0,
                   'retracted': 0, 'threads': 0, 'shown': 0, 'ops': len(ops),
                   'conflicts': 0, 'gated': 0}
        self._conflicts: List[Any] = []

        for op in ops:
            kind = op['op']
            try:
                if   kind == OP_NODE:    self._op_node(op, at, by, summary)
                elif kind == OP_ASSERT:  self._op_assert(op, at, by, summary, gate=gate)
                elif kind == OP_RETRACT: self._op_retract(op, at, summary)
                elif kind == OP_THREAD:  self._op_thread(op, at, by, summary)
                elif kind == OP_SHOW:    self._op_show(op, summary)
            except Exception as e:
                # One bad op must not cost the rest of the batch.
                print(f'!! world op failed ({kind}): {e}')

        if log and ops:
            self._append_log(ops, at, by)
        return summary

    def last_conflicts(self) -> List[Any]:
        """Conflicts seen during the most recent apply(), for the caller to report."""
        return list(getattr(self, '_conflicts', []))

    def _op_node(self, op: Dict[str, Any], at: int, by: str, summary: Dict[str, int]) -> None:
        before = len(self.nodes)
        nid = self.ensure(op['name'], op.get('kind'), at=at,
                          gist=op.get('gist'), detail=op.get('detail'),
                          aka=op.get('aka'), traits=op.get('traits'),
                          shown=op.get('shown'), salience=op.get('salience'), by=by)
        if len(self.nodes) > before:
            summary['nodes'] += 1
        n = self.nodes[nid]
        # Enrich an existing stub rather than overwrite it: the first gist written
        # is usually the one the prose actually established. `allow_gist_revision`
        # (default off) is the opt-in exception — a mode that re-probes the same
        # region needs its gist to actually update, not freeze at first digestion.
        if op.get('gist') and (not n.gist or self.allow_gist_revision):
            n.gist = op['gist']
        if op.get('detail'):
            n.detail = op['detail']
        if op.get('traits'):
            n.traits.update(op['traits'])
        for a in op.get('aka') or []:
            if norm_name(a) != norm_name(n.name) and a not in n.aka:
                n.aka.append(a)
        if op.get('shown') and SHOWN_RANK.get(op['shown'], 0) > SHOWN_RANK.get(n.shown, 0):
            n.shown = op['shown']

    def _op_assert(self, op: Dict[str, Any], at: int, by: str, summary: Dict[str, int],
                   gate: bool = False) -> None:
        subj = self.ensure(op['subject'], op.get('subject_kind'), at=at, by=by)
        pred = op['pred']
        if pred in LITERAL_PREDS:
            obj = coerce_str(op['object'], 400)     # a quality, not an element
        else:
            obj = self.ensure(op['object'], op.get('object_kind'), at=at, by=by)
        frame = op.get('frame') or FRAME_WORLD
        if frame != FRAME_WORLD:
            frame = self.resolve(frame) or frame    # a standpoint is a node, or a bare label

        if gate:
            # Checked against resolved ids, after ensure() — the same names the
            # standing claims are stored under, so 'the annex' and 'annex' compare
            # as one element rather than two.
            for c in find_conflicts(self.open_assertions(), subj, pred, obj, frame):
                self._conflicts.append(c)
                summary['conflicts'] += 1
                if c.kind == KIND_FUNCTIONAL and c.standing.open:
                    # Close, never delete: 'what held at frag 4' stays answerable.
                    c.standing.until = at
                    summary['gated'] += 1

        new = Assertion(id='', subject=subj, pred=pred, object=obj, frame=frame,
                        conf=op.get('conf', 0.7), valence=op.get('valence', 0.0),
                        intensity=op.get('intensity', 0.5), since=at, by=by,
                        shown=op.get('shown', 'latent'))

        standing = [a for a in self.assertions.values() if a.open and a.key() == new.key()]
        if standing:
            prev = standing[-1]
            if prev.same_values(new):
                # Prose restating something already true. Record nothing; just let
                # the disclosure level catch up if it was said out loud this time.
                if SHOWN_RANK.get(new.shown, 0) > SHOWN_RANK.get(prev.shown, 0):
                    prev.shown = new.shown
                summary['restated'] += 1
                return
            prev.until = at            # closed, never deleted
            summary['superseded'] += 1

        self._seq += 1
        new.id = f'a{self._seq:05d}'
        self.assertions[new.id] = new
        summary['assertions'] += 1

    def _op_retract(self, op: Dict[str, Any], at: int, summary: Dict[str, int]) -> None:
        a = self.assertions.get(op['id'])
        if a and a.open:
            a.until = at
            summary['retracted'] += 1

    def _op_thread(self, op: Dict[str, Any], at: int, by: str, summary: Dict[str, int]) -> None:
        before = len(self.nodes)
        tid = self.ensure(op['name'], KIND_THREAD, at=at, gist=op.get('gist'), by=by)
        if len(self.nodes) > before:
            summary['threads'] += 1
        n = self.nodes[tid]
        n.traits['state'] = op.get('state', 'open')
        n.traits['pressure'] = coerce_num(op.get('pressure'), 0.6)
        if op.get('gist') and not n.gist:
            n.gist = op['gist']
        for other in op.get('involves') or []:
            oid = self.ensure(other, at=at, by=by)
            self._op_assert({'op': OP_ASSERT, 'subject': oid, 'pred': 'part_of',
                             'object': tid, 'conf': 0.8, 'valence': 0.0,
                             'intensity': 0.5}, at, by, summary)

    def _op_show(self, op: Dict[str, Any], summary: Dict[str, int]) -> None:
        ref, shown = op['ref'], op['shown']
        target = self.assertions.get(ref)
        if target is None:
            nid = self.resolve(ref)
            target = self.nodes.get(nid) if nid else None
        if target is not None and SHOWN_RANK.get(shown, 0) > SHOWN_RANK.get(target.shown, 0):
            target.shown = shown
            summary['shown'] += 1

    # --- reading ---

    def open_assertions(self) -> List[Assertion]:
        return [a for a in self.assertions.values() if a.open]

    def assertions_of(self, node_id: str, open_only: bool = True) -> List[Assertion]:
        return [a for a in self.assertions.values()
                if (a.open or not open_only) and (a.subject == node_id or a.object == node_id)]

    def open_threads(self) -> List[Node]:
        ts = [n for n in self.nodes.values()
              if n.kind == KIND_THREAD and n.traits.get('state') != 'resolved']
        return sorted(ts, key=lambda n: -coerce_num(n.traits.get('pressure'), 0.5))

    def name_of(self, ref: str) -> str:
        n = self.nodes.get(ref)
        return n.name if n else ref

    def stats(self) -> Dict[str, Any]:
        by_kind = {k: 0 for k in KINDS}
        for n in self.nodes.values():
            by_kind[n.kind] = by_kind.get(n.kind, 0) + 1
        return {'nodes': len(self.nodes), 'by_kind': by_kind,
                'assertions': len(self.assertions),
                'open': len(self.open_assertions()),
                'threads_open': len(self.open_threads()), 'at': self._at}

    # --- persistence ---

    def _append_log(self, ops: List[Dict[str, Any]], at: int, by: str) -> None:
        os.makedirs(self.out_dir, exist_ok=True)
        try:
            with open(self.jsonl_path, 'a', encoding='utf-8') as f:
                for op in ops:
                    f.write(json.dumps({'at': at, 'by': by, **op}, ensure_ascii=False, default=str) + '\n')
        except OSError as e:
            print(f'!! world log write failed: {e}')

    def to_dict(self) -> Dict[str, Any]:
        return {'at': self._at, 'seq': self._seq,
                **({'fuzzy_names': False} if not self.fuzzy_names else {}),
                'nodes': {k: v.to_dict() for k, v in self.nodes.items()},
                'assertions': {k: v.to_dict() for k, v in self.assertions.items()}}

    def save(self) -> None:
        """Atomic tmp+replace, as atools/knowledge.py:122-125 does. StoryState.save
        rewrites in place and torn reads are a known consequence
        (ui/common.py:117-119 swallows them) — no reason to repeat that here.

        `os.replace` retries a few times on `PermissionError`: on Windows, a
        concurrent plain `open()` of `world.json` (e.g. the browser UI's own
        `ui/dog/runs.py::graph_provenance()`, polled every ~2s) holds a handle
        without `FILE_SHARE_DELETE`, which blocks a rename of that same path
        for the handle's lifetime — a few milliseconds in practice. This
        crashed a live run outright (`WinError 5`) the first time it was hit;
        the graph write itself was never the problem, an in-flight read was."""
        os.makedirs(self.out_dir, exist_ok=True)
        tmp = self.json_path + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False, default=str)
        import time
        for attempt in range(5):
            try:
                os.replace(tmp, self.json_path)
                return
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(0.05 * (attempt + 1))

    def load(self) -> 'WorldGraph':
        if not os.path.isfile(self.json_path):
            return self
        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                d = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            print(f'!! world load failed: {e}')
            return self
        self._at = int(d.get('at') or 0)
        # Identity policy belongs to the graph, including tool-created readers/writers.
        self.fuzzy_names = self.fuzzy_names and d.get('fuzzy_names', True)
        self._seq = int(d.get('seq') or 0)
        self.nodes = {k: Node.from_dict(v) for k, v in (d.get('nodes') or {}).items()}
        self.assertions = {k: Assertion.from_dict(v) for k, v in (d.get('assertions') or {}).items()}
        return self

    def logged_ops(self) -> Iterable[Dict[str, Any]]:
        if not os.path.isfile(self.jsonl_path):
            return
        with open(self.jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue   # tolerate a torn final line from a killed run

    @classmethod
    def replay(cls, out_dir: str, verbose: bool = False,
               until: Optional[int] = None, fuzzy_names: Optional[bool] = None) -> 'WorldGraph':
        """Rebuild purely from the op log. The check that the log is the real
        record and world.json only a projection of it.

        `until` stops at a story time — every op carries the fragment number it was
        established at, so 'the graph as it stood at frame 5' is a filter over the
        log rather than a reconstruction. That is what lib/fork.py cuts a branch's
        knowledge on.
        """
        if fuzzy_names is None:
            fuzzy_names = cls(out_dir).load().fuzzy_names
        g = cls(out_dir, verbose=verbose, fuzzy_names=fuzzy_names)
        for entry in g.logged_ops():
            at = int(entry.get('at') or 0)
            if until is not None and at > int(until):
                continue
            by = entry.get('by') or ''
            g.apply([entry], at=at, by=by, log=False)
        return g
