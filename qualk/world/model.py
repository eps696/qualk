"""
Node and Assertion — the two records the world graph is made of.

One node shape for every kind, with kind-specific extras in `traits`, so nothing
downstream has to branch on type. One assertion shape for hard and soft material
alike: a location and a resonance are both a directed, graded relation between two
elements, and what separates them is the predicate, not the container.

`frame` is the standpoint a claim holds from. 'world' means unconditional; any
other node id means the claim holds only from there. That standpoint is
deliberately not assumed to be a character — a lens, a phase or a medium is as
valid a vantage as an agent, and an abstract narrative uses those and no agents
at all.

Every field arriving from an LLM passes through the coercers here first. Nothing
in this codebase validates model output (see agt_llm.py:98-101, which checks only
that a key exists and holds dict-like items), so these records treat every
incoming value as untrusted rather than assuming a schema was honoured.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

# Kinds are narrative functions, not ontological categories: the question is what
# an element *does* in the machinery, not what sort of thing it is. An agent may
# be a person, an institution, a weather system or a process; a site may be a room,
# a medium or an interval. Closed set of six, kept small on purpose — open-ended
# typing goes in `traits`.
KIND_AGENT  = 'agent'    # acts, wants, exerts pressure
KIND_SITE   = 'site'     # locates or contains
KIND_OBJECT = 'object'   # persists and can be referred to, without acting
KIND_EVENT  = 'event'    # happens; reified so causes and participants can attach
KIND_MOTIF  = 'motif'    # recurs and accrues meaning: a claim, theme, signal, shape
KIND_THREAD = 'thread'   # unfinished business generating forward pressure

KINDS = (KIND_AGENT, KIND_SITE, KIND_OBJECT, KIND_EVENT, KIND_MOTIF, KIND_THREAD)

# Disclosure. Only 'established' — what actually reached the page — is a hard
# constraint. Latent material stays freely revisable, which is what keeps the graph
# a narrative substrate rather than a world that must be simulated.
SHOWN_LATENT      = 'latent'
SHOWN_HINTED      = 'hinted'
SHOWN_ESTABLISHED = 'established'

SHOWN = (SHOWN_LATENT, SHOWN_HINTED, SHOWN_ESTABLISHED)
SHOWN_RANK = {SHOWN_LATENT: 0, SHOWN_HINTED: 1, SHOWN_ESTABLISHED: 2}

FRAME_WORLD = 'world'

_SLUG_STRIP = re.compile(r"[^a-z0-9Ѐ-ӿ]+")
_ARTICLE = re.compile(r"^(the|a|an)\s+", re.I)


def slugify(text: str, limit: int = 48) -> str:
    """Stable, readable id component. Keeps Cyrillic — this repo runs ru as well as en."""
    s = _SLUG_STRIP.sub('-', str(text or '').strip().lower()).strip('-')
    return s[:limit].strip('-') or 'unnamed'


def node_id(kind: str, name: str) -> str:
    return f'{coerce_kind(kind)}:{slugify(name)}'


def norm_name(name: str) -> str:
    """Comparison form for matching. Leading articles are dropped because prose
    alternates 'the doubled doorway' / 'doubled doorway' freely and they are the
    same element; internal words are left alone so 'the doorway' stays distinct."""
    return _ARTICLE.sub('', str(name or '').strip().lower()).strip()


# --- coercion of untrusted values ---

def coerce_num(x: Any, default: float = 0.0, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp anything to a float in range. Models emit '0.8', 80, None and 'high'."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    if v != v:  # NaN
        return default
    return max(lo, min(hi, v))


def coerce_kind(kind: Any) -> str:
    k = str(kind or '').strip().lower()
    if k in KINDS:
        return k
    # Common near-misses. Not fuzzy-matched: a wrong kind is cheap to live with,
    # but silently reinterpreting an unknown one would hide a prompt problem.
    return {'actor': KIND_AGENT, 'character': KIND_AGENT, 'person': KIND_AGENT,
            'place': KIND_SITE, 'location': KIND_SITE, 'setting': KIND_SITE,
            'thing': KIND_OBJECT, 'item': KIND_OBJECT, 'artifact': KIND_OBJECT,
            'idea': KIND_MOTIF, 'concept': KIND_MOTIF, 'theme': KIND_MOTIF,
            'signal': KIND_MOTIF, 'pattern': KIND_MOTIF, 'claim': KIND_MOTIF,
            'question': KIND_THREAD, 'tension': KIND_THREAD,
            }.get(k, KIND_MOTIF)  # motif is the safe default: it assumes least


def coerce_shown(shown: Any, default: str = SHOWN_LATENT) -> str:
    s = str(shown or '').strip().lower()
    return s if s in SHOWN else default


def coerce_str(x: Any, limit: int = 0) -> str:
    if x is None:
        return ''
    s = x if isinstance(x, str) else (json.dumps(x, ensure_ascii=False) if isinstance(x, (dict, list)) else str(x))
    s = s.strip()
    return s[:limit] if limit and len(s) > limit else s


def coerce_list(x: Any) -> List[str]:
    if not x:
        return []
    if isinstance(x, str):
        return [x.strip()] if x.strip() else []
    if isinstance(x, (list, tuple)):
        return [coerce_str(v) for v in x if coerce_str(v)]
    return []


def coerce_dict(x: Any) -> Dict[str, Any]:
    return dict(x) if isinstance(x, dict) else {}


@dataclass
class Node:
    """One element of the narrative.

    `gist` is the one-line form and is always cheap enough to inject; `detail` is
    filled only when an element becomes focal. That split is how the graph
    elaborates on demand instead of being authored up front — an element referred
    to once stays a stub forever, and costs a line.
    """
    id: str
    kind: str
    name: str
    aka: List[str] = field(default_factory=list)
    gist: str = ''
    detail: str = ''
    traits: Dict[str, Any] = field(default_factory=dict)
    shown: str = SHOWN_LATENT
    salience: float = 0.5
    first_seen: int = 0
    last_seen: int = 0
    by: str = ''   # provenance of the node's own first creation, e.g. 'frag:11' or
                   # 'folder:_in/x.txt#L1-4' — mirrors Assertion.by, but set once at
                   # creation (first-write-wins, same stance as gist below) rather
                   # than per-claim, since what's being recorded is "why does this
                   # element exist at all", a fact about the node, not about any one
                   # assertion touching it. Added for lib/stim's anti-inbreeding
                   # filter (breed._intake_born), which needs to ask that question
                   # of nodes with no assertions yet — the common case for a stub.

    @classmethod
    def make(cls, kind: Any, name: Any, at: int = 0, **kw) -> 'Node':
        kind = coerce_kind(kind)
        name = coerce_str(name, 200) or 'unnamed'
        return cls(
            id         = kw.get('id') or node_id(kind, name),
            kind       = kind,
            name       = name,
            aka        = coerce_list(kw.get('aka')),
            gist       = coerce_str(kw.get('gist'), 400),
            detail     = coerce_str(kw.get('detail')),
            traits     = coerce_dict(kw.get('traits')),
            shown      = coerce_shown(kw.get('shown')),
            salience   = coerce_num(kw.get('salience'), 0.5),
            first_seen = int(at), last_seen = int(at),
            by         = coerce_str(kw.get('by'), 120),
        )

    def names(self) -> List[str]:
        return [self.name] + list(self.aka)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'Node':
        known = {k: d.get(k) for k in cls.__dataclass_fields__ if k in d}
        return cls(**{**{'id': '', 'kind': KIND_MOTIF, 'name': ''}, **known})


@dataclass
class Assertion:
    """One claim, holding from one standpoint over one stretch of story time.

    Assertions are never mutated or deleted — superseding one closes it with
    `until` and adds its replacement. That is what makes "what held in chapter 3"
    answerable, and what gives every claim in the graph an audit trail back to the
    fragment that produced it.
    """
    id: str
    subject: str
    pred: str
    object: str
    frame: str = FRAME_WORLD
    conf: float = 0.7
    valence: float = 0.0      # signed: -1 hostile/concealing .. +1 warm/revealing
    intensity: float = 0.5    # graded: how strongly it holds
    since: int = 0
    until: Optional[int] = None
    by: str = ''
    shown: str = SHOWN_LATENT
    # Cosine of the two endpoints' embeddings, filled by dog's semantic index the first
    # time either endpoint's text is embedded. Derived, not part of the claim: it is not
    # in key() or same_values(), and is left out of to_dict() while unset so every other
    # mode's world.json stays byte-identical.
    affinity: Optional[float] = None

    @property
    def open(self) -> bool:
        return self.until is None

    def key(self) -> tuple:
        """Identity for superseding: same claim from the same standpoint."""
        return (self.subject, self.pred, self.object, self.frame)

    def same_values(self, other: 'Assertion', eps: float = 0.15) -> bool:
        """Whether a restatement carries no new information. Restating a standing
        fact is the common case — prose repeats itself — and re-recording it every
        time would bloat the log without adding anything."""
        return (abs(self.conf - other.conf) < eps
                and abs(self.valence - other.valence) < eps
                and abs(self.intensity - other.intensity) < eps)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if d.get('affinity') is None:
            d.pop('affinity', None)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'Assertion':
        known = {k: d.get(k) for k in cls.__dataclass_fields__ if k in d}
        return cls(**{**{'id': '', 'subject': '', 'pred': '', 'object': ''}, **known})
