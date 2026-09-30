"""
Contradiction detection over incoming ops, used as a gate rather than a metric.

DOME builds a temporal-conflict analyzer that groups claims by five rule patterns
and then uses it only to *score itself afterwards* — the graph it generated is
never actually constrained by it. That "built a good checker, never wired it in"
shape recurs across the literature, and `lib/world/` had the same gap: `_op_assert`
supersedes a claim only when the new one has an identical key (same subject, pred,
object *and* frame), so the one pattern that genuinely corrupts a graph slips
through — same subject, same predicate, a **different** object.

    frag 4:  the annex is in the eastern yard
    frag 7:  the annex is in the flooded basement

Both stay open. Every later `build_view()` then projects both, and the model is
told two incompatible things about where the annex is.

Two kinds are separated here, because only one of them is safely resolvable
without a judgement call:

- **functional** — a predicate that can hold at most one open value per
  (subject, frame). Location is the clear case. The newer claim wins and the older
  is closed, which is exactly the supersede semantics `Assertion` already has.
- **polarity** — opposed predicates standing at once between the same pair
  ('X conceals Y' and 'X reveals Y'). Reported, never auto-resolved: in a narrative
  graph that is frequently the *point* rather than an error, and quietly closing one
  side would delete meaning. Adjudicating these is a judgement call and therefore an
  LLM's job, which is a separate opt-in step and not this module.

Everything here is deterministic and costs no LLM call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from .model import Assertion, FRAME_WORLD
from .ops import LITERAL_PREDS

# At most one open value per (subject, frame). Kept deliberately tiny: a predicate
# wrongly listed here silently deletes standing claims, which is worse than the
# contradiction it was meant to catch. 'in' and 'at' are the two where a second
# open value is incoherent rather than merely unusual.
#
# 'part_of', 'has' and 'is_a' are NOT functional — an element belongs to several
# threads at once, and that is how threading works. 'attr' is not either: qualities
# accumulate, and treating them as single-valued would let one adjective erase
# every other.
FUNCTIONAL_PREDS = frozenset({'in', 'at'})

# Predicates that cannot both stand between the same pair from the same standpoint.
OPPOSITE_PREDS = {
    'conceals': 'reveals',
    'trusts': 'distrusts',
    'desires': 'fears',
    'draws_toward': 'opposes',
}
# symmetric lookup
_OPPOSITES = {**OPPOSITE_PREDS, **{v: k for k, v in OPPOSITE_PREDS.items()}}

KIND_FUNCTIONAL = 'functional'
KIND_POLARITY = 'polarity'


@dataclass
class Conflict:
    kind: str                  # KIND_FUNCTIONAL | KIND_POLARITY
    standing: Assertion        # what the graph already holds
    incoming: Dict[str, Any]   # the resolved op about to be applied
    note: str = ''

    def describe(self, graph=None) -> str:
        name = (lambda r: graph.name_of(r)) if graph is not None else (lambda r: r)
        s, p = name(self.standing.subject), self.standing.pred
        return (f'{self.kind}: "{s} {p} {name(self.standing.object)}" '
                f'vs incoming "{s} {self.incoming.get("pred")} {name(self.incoming.get("object"))}"'
                + (f' — {self.note}' if self.note else ''))


def find_conflicts(open_assertions: Iterable[Assertion], subject: str, pred: str,
                   obj: Any, frame: str = FRAME_WORLD) -> List[Conflict]:
    """Standing claims that a new (subject, pred, object, frame) would contradict.

    Takes the resolved ids rather than a raw op, because resolution is the store's
    job and doing it twice risks the two disagreeing. Identical claims are not
    conflicts — those are restatements and `_op_assert` already handles them.
    """
    out: List[Conflict] = []
    incoming = {'subject': subject, 'pred': pred, 'object': obj, 'frame': frame}
    opposite = _OPPOSITES.get(pred)
    for a in open_assertions:
        if a.subject != subject or a.frame != frame:
            continue
        if a.pred == pred and a.object != obj and pred in FUNCTIONAL_PREDS:
            out.append(Conflict(KIND_FUNCTIONAL, a, incoming,
                                f'{pred!r} holds one value at a time'))
        elif opposite and a.pred == opposite and a.object == obj and pred not in LITERAL_PREDS:
            out.append(Conflict(KIND_POLARITY, a, incoming,
                                f'{a.pred!r} and {pred!r} are opposed'))
    return out


def summarize(conflicts: Iterable[Conflict], graph=None) -> str:
    """One readable block for a run's log. Empty string when there is nothing to say,
    so a caller can print it unconditionally."""
    items = list(conflicts)
    if not items:
        return ''
    lines = [f'.. world gate: {len(items)} contradiction(s)']
    lines += [f'   - {c.describe(graph)}' for c in items]
    return '\n'.join(lines)
