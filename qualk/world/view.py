"""
Projection — turning the graph into the small amount of it a prompt should see.

This is where "only to the degree needed" stops being a slogan. The graph may
hold hundreds of elements; a generation step should receive the handful that bear
on what it is about, rendered as sentences rather than triples because the result
goes into the single JSON blob that agt_llm.py:146-160 hands the model, and prose
survives that better than a relation dump.

The output keys are `present` / `where` / `holds` / `open`, not `cast` / `place` /
`facts`. A run may legitimately have no agents in it at all — conspir's are motifs
and threads from beginning to end — and a section labelled `cast` would push the
model toward inventing characters to fill it.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

from .model import (Assertion, KIND_SITE, KIND_THREAD, Node, FRAME_WORLD,
                    SHOWN_ESTABLISHED)
from .ops import LITERAL_PREDS, PRED_FAMILIES, SIGNED_PREDS
from .store import WorldGraph

# What an element leans toward or against — ID-RAG's values/goals/preferences tier.
STANCE_PREDS = frozenset(PRED_FAMILIES['stance'])

BUDGET_CHARS = 1200     # a character proxy, deliberately, rather than a tokenizer dep
HOP2_SALIENCE = 0.55    # a second hop only through material that still matters
ESTABLISHED_BOOST = 1.6 # what reached the page is a hard constraint; it must not be crowded out
THREAD_BOOST = 1.25

IDENTITY_BUDGET = 600   # smaller than the world view on purpose — see build_identity_view


def budget_for(a: Any, default: int = BUDGET_CHARS) -> int:
    """How much projection this run wants, from `-wbud`.

    Not a constant, because the right answer inverts with model size. ID-RAG measured
    a small model converging 58% faster on a *targeted* identity slice while a full
    dump barely helped it, and the same comparison reversing on a large one (full
    injection 41% faster). So breadth is a per-backbone setting; `-wbud` is the dial,
    and each call site keeps its own default when the flag is unset.
    """
    want = getattr(a, 'world_budget', None) if a is not None else None
    try:
        return max(0, int(want)) if want is not None else int(default)
    except (TypeError, ValueError):
        return int(default)


def _readable(pred: str) -> str:
    return pred.replace('_', ' ')


def _qualify(a: Assertion) -> str:
    if a.pred not in SIGNED_PREDS:
        return ''
    if a.intensity >= 0.75:
        return ' (strongly)'
    if a.intensity <= 0.30:
        return ' (faintly)'
    return ''


def render_assertion(graph: WorldGraph, a: Assertion) -> str:
    """One claim as a sentence."""
    subj = graph.name_of(a.subject)
    obj = a.object if a.pred in LITERAL_PREDS else graph.name_of(a.object)
    obj_node = graph.nodes.get(a.object)
    if a.pred == 'attr':
        core = f'{subj} is {obj}'
    elif a.pred == 'part_of' and obj_node is not None and obj_node.kind == KIND_THREAD:
        # 'part_of' is how an element attaches to a line of forward pressure, but
        # rendered literally it reads as containment. The relation is aboutness.
        core = f'{subj} bears on the open question of {obj}'
    else:
        core = f'{subj} {_readable(a.pred)} {obj}'
    core += _qualify(a)
    if a.frame and a.frame != FRAME_WORLD:
        # A claim that holds only from somewhere reads as a reading, not a fact.
        core = f'from {graph.name_of(a.frame)}: {core}'
    return core[0].upper() + core[1:] if core else core


def _recency(a: Assertion, now: int) -> float:
    age = max(0, int(now) - int(a.since or 0))
    return 1.0 / (1.0 + 0.06 * age)


# Per-element retrieval weighting, as exponents on the existing factors. 1.0
# everywhere reproduces the previous scoring exactly, which is why they are
# exponents rather than multipliers.
#
# Park's retrieval combines recency, importance and relevance at equal weight, and
# is explicit that this is "a simplifying default, not a principled choice" — two
# agents given identical memory streams but different weightings would plausibly
# read as different characters. That makes the weights a free source of *structural*
# divergence rather than cosmetic prose variation: a grudge-holder is high salience
# and low recency decay, a flighty one the reverse. Costs nothing and differentiates
# what each element actually notices, not just how it phrases things.
DEFAULT_WEIGHTS = {'recency': 1.0, 'salience': 1.0, 'established': 1.0, 'thread': 1.0}


def coerce_weights(raw: Any) -> Dict[str, float]:
    """Untrusted weights (a persona record an LLM wrote) to a usable set.

    Clamped to 0..3: a negative exponent inverts the factor's meaning and a large
    one collapses the ranking onto a single term, and neither is a thing a persona
    schema should be able to express by accident.
    """
    out = dict(DEFAULT_WEIGHTS)
    if not isinstance(raw, dict):
        return out
    for k in DEFAULT_WEIGHTS:
        if k in raw:
            try:
                out[k] = max(0.0, min(3.0, float(raw[k])))
            except (TypeError, ValueError):
                pass
    return out


def _score(graph: WorldGraph, a: Assertion, focus: Set[str], now: int,
           weights: Optional[Dict[str, float]] = None) -> float:
    w = weights or DEFAULT_WEIGHTS
    subj = graph.nodes.get(a.subject)
    sal = subj.salience if subj else 0.4
    dist = 1.0 if (a.subject in focus or a.object in focus) else 0.55
    s = max(0.05, a.conf) * (_recency(a, now) ** w['recency']) * (max(0.1, sal) ** w['salience']) * dist
    if a.shown == SHOWN_ESTABLISHED:
        s *= ESTABLISHED_BOOST ** w['established']
    obj = graph.nodes.get(a.object)
    if obj is not None and obj.kind == KIND_THREAD:
        s *= THREAD_BOOST ** w['thread']
    return s


def resolve_focus(graph: WorldGraph, focus: Any) -> List[str]:
    """Names or ids to element ids, dropping what the graph has never heard of.
    Focus is not assumed to contain an agent — it is whatever the next step is
    about: a speaker, a site, the motifs live in the last frame, an open thread."""
    if not focus:
        return []
    if isinstance(focus, str):
        focus = [focus]
    out: List[str] = []
    for f in focus:
        nid = graph.resolve(f) if not isinstance(f, Node) else f.id
        if nid and nid not in out:
            out.append(nid)
    return out


def build_view(graph: WorldGraph, focus: Any = None, budget: int = BUDGET_CHARS,
               now: Optional[int] = None, weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """The graph as a generation step should see it.

    `weights` re-weights retrieval for whoever is looking — see DEFAULT_WEIGHTS.
    None is the shared, neutral view.
    """
    now = graph.stats()['at'] if now is None else int(now)
    focal = resolve_focus(graph, focus)
    if not focal:
        # No focus given: fall back to whatever currently matters most, so the
        # view is never empty just because the caller had nothing to name.
        focal = [n.id for n in sorted(graph.nodes.values(), key=lambda n: -n.salience)[:4]]
    focus_set = set(focal)

    # Hop 1: everything standing that touches focus. Hop 2: only through material
    # still salient enough to be worth the tokens.
    standing = graph.open_assertions()
    hop1 = [a for a in standing if a.subject in focus_set or a.object in focus_set]
    hop1_ids = {a.id for a in hop1}
    reached: Set[str] = set()
    for a in hop1:
        for end in (a.subject, a.object):
            if end in graph.nodes and end not in focus_set:
                reached.add(end)
    hop2 = []
    for a in standing:
        if a.id in hop1_ids:
            continue
        # A quality of something already on screen ('the annex is brittle') comes
        # in regardless of salience: it is one short line, it describes an element
        # the step is already being told about, and it is exactly the grounding
        # detail the graph exists to supply. Gating it behind decay made it drop
        # out at precisely the point the element had been around long enough to
        # have accumulated qualities.
        if a.pred in LITERAL_PREDS and a.subject in reached:
            hop2.append(a)
            continue
        for end in (a.subject, a.object):
            n = graph.nodes.get(end)
            if end in reached and n and n.salience >= HOP2_SALIENCE:
                hop2.append(a)
                break

    ranked = sorted(hop1 + hop2, key=lambda a: -_score(graph, a, focus_set, now, weights))

    holds: List[str] = []
    used = 0
    seen_nodes: Set[str] = set(focal)
    for a in ranked:
        line = render_assertion(graph, a)
        if used + len(line) > budget:
            break
        holds.append(line)
        used += len(line)
        for end in (a.subject, a.object):
            if end in graph.nodes:
                seen_nodes.add(end)

    # `where` is the most salient site in play, if any. A run with no sites simply
    # has none — nothing downstream should require one.
    sites = [graph.nodes[n] for n in seen_nodes
             if n in graph.nodes and graph.nodes[n].kind == KIND_SITE]
    where_node = max(sites, key=lambda n: n.salience) if sites else None

    present: List[Dict[str, Any]] = []
    for nid in sorted(seen_nodes, key=lambda n: -(graph.nodes[n].salience if n in graph.nodes else 0)):
        n = graph.nodes.get(nid)
        if not n or n.kind == KIND_THREAD or (where_node and n.id == where_node.id):
            continue
        item = {'name': n.name, 'kind': n.kind, 'gist': n.gist}
        # Only focal elements are worth their long form.
        if n.id in focus_set and n.detail:
            item['detail'] = n.detail
        present.append(item)

    threads = graph.open_threads()
    view: Dict[str, Any] = {
        'present': present,
        'holds': holds,
        'open': [t.name for t in threads[:6]],
    }
    if where_node:
        view['where'] = {'name': where_node.name, 'gist': where_node.gist}
    return view


def build_identity_view(graph: WorldGraph, ref: Any, budget: int = IDENTITY_BUDGET,
                        now: Optional[int] = None) -> Dict[str, Any]:
    """What one element holds *about itself*, kept apart from the episodic stream.

    The problem this solves is measured, not theoretical: when a persona's identity
    is re-inferred each turn from a growing record of events, recent trivia competes
    with foundational traits and the self-model drifts — ID-RAG reports recall
    plateauing at 0.51–0.56 and trending *down* over a run under exactly that
    arrangement, with agents becoming impressionable to each other and hallucinating
    about themselves. A separate, small, typed store fixes it, and their working
    graphs were ~16 nodes / 15 edges per character: this is a retrieval discipline,
    not a data-collection problem.

    No new store is needed here. `Assertion.frame` already carries the standpoint a
    claim holds from (model.py's docstring is explicit that a standpoint need not be
    an agent), and it is 'world' for everything today. So identity is a *filtered
    read* of the graph already being written:

      `is`      attributive claims about it            -> traits
      `wants`   its stances toward other elements      -> values, goals, drives
      `holds`   claims framed by it rather than world  -> beliefs, its own readings

    Returns {} for an element the graph has never heard of, so a caller can inject
    unconditionally without branching.
    """
    now = graph.stats()['at'] if now is None else int(now)
    nid = graph.resolve(ref) if not isinstance(ref, Node) else ref.id
    node = graph.nodes.get(nid) if nid else None
    if node is None:
        return {}

    is_lines: List[str] = []
    wants_lines: List[str] = []
    holds_lines: List[str] = []
    w = coerce_weights(node.traits.get('retrieval'))
    for a in sorted(graph.open_assertions(), key=lambda x: -_score(graph, x, {nid}, now, w)):
        if a.frame not in (FRAME_WORLD, nid) and a.subject != nid:
            continue
        if a.frame == nid and a.subject != nid:
            # A reading this element holds about something else — its belief, not its trait.
            holds_lines.append(render_assertion(graph, a))
        elif a.subject == nid and a.pred in LITERAL_PREDS:
            is_lines.append(render_assertion(graph, a))
        elif a.subject == nid and a.pred in STANCE_PREDS:
            wants_lines.append(render_assertion(graph, a))

    # Budget is spent in order of what protects identity best: what it *is* first,
    # then what it wants, then what it believes. A trait crowded out by a belief is
    # the drift this function exists to prevent.
    out: Dict[str, Any] = {'name': node.name, 'kind': node.kind}
    if node.gist:
        out['gist'] = node.gist
    used = len(node.gist or '')
    for key, lines in (('is', is_lines), ('wants', wants_lines), ('holds', holds_lines)):
        kept: List[str] = []
        for line in lines:
            if used + len(line) > budget:
                break
            kept.append(line)
            used += len(line)
        if kept:
            out[key] = kept
    return out


def render_text(view: Dict[str, Any]) -> str:
    """Flat form, for eyeballing a graph from the terminal."""
    out: List[str] = []
    if view.get('where'):
        out.append(f"where: {view['where']['name']} — {view['where'].get('gist', '')}".rstrip(' —'))
    if view.get('present'):
        out.append('present:')
        for p in view['present']:
            out.append(f"  · [{p['kind']}] {p['name']}" + (f" — {p['gist']}" if p.get('gist') else ''))
    if view.get('holds'):
        out.append('holds:')
        out.extend(f'  · {h}' for h in view['holds'])
    if view.get('open'):
        out.append('open:')
        out.extend(f'  · {o}' for o in view['open'])
    return '\n'.join(out)
