"""
The op vocabulary — what a writer (LLM or tool) is allowed to say to the graph.

Both write paths speak this: the `world-upd` extraction pass emits a list of these,
and the `world_note` tool takes the same list as a JSON string. Keeping one
vocabulary means the store has a single entry point to get right.

Predicates are free strings. The core vocabulary below guides without gating —
an unrecognised predicate is recorded as given, because a narrative engine that
rejected 'unspools_into' would be refusing the thing it exists to capture. What
normalisation does is collapse near-misses ('mistrusts' -> 'distrusts') so the
same relation does not fragment across eight spellings.

The families matter more than the individual terms. What the store cares about is
a predicate's *shape* — directed, optionally signed, optionally graded — not
whether its label sounds human. 'agent fears agent' and 'motif resists reading'
are the same shape, and the formal/semiotic family exists because a vocabulary
built around people would have omitted exactly the relations an abstract
narrative runs on.
"""

from __future__ import annotations

import json
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional

from .model import (coerce_dict, coerce_list, coerce_num, coerce_shown,
                    coerce_str, FRAME_WORLD)

PRED_FAMILIES: Dict[str, tuple] = {
    # what is where, what belongs to what
    'structural': ('is_a', 'in', 'part_of', 'has', 'at'),
    # what follows from what
    'causal':     ('caused', 'enables', 'precedes', 'follows'),
    # a directed, signed, graded leaning. Works for an agent toward an agent and
    # for a form toward a form; the humanness of the label is incidental.
    'stance':     ('desires', 'fears', 'trusts', 'distrusts', 'resists',
                   'draws_toward', 'opposes'),
    # how forms relate to forms. The family conspir runs on: its whole loop is
    # "what is hidden there?", which is conceals/reveals over recurring motifs.
    'formal':     ('echoes', 'rhymes_with', 'transforms_into', 'displaces',
                   'conceals', 'reveals'),
    # propositional as well as personal
    'epistemic':  ('knows', 'told', 'implies', 'contradicts'),
    # qualities, moods, registers — object is a literal, not a node
    'attributive': ('attr',),
}

CORE_PREDS = tuple(p for fam in PRED_FAMILIES.values() for p in fam)
FAMILY_OF = {p: fam for fam, preds in PRED_FAMILIES.items() for p in preds}

# Predicates whose valence carries meaning, so the renderer can qualify them.
SIGNED_PREDS = frozenset(PRED_FAMILIES['stance']) | frozenset(PRED_FAMILIES['formal'])

# Unary: the object is a literal string, never resolved to a node.
LITERAL_PREDS = frozenset(PRED_FAMILIES['attributive'])

# Edge roles. A *scaffold* edge places or describes (containment, possession, property,
# type); it is useful to `build_view` but says nothing about how one element bears on
# another, and its conf/intensity are defaults, so a walk over the graph must not use it.
# A *relation* edge is any other claim between two elements. Extraction invents many
# predicates outside CORE_PREDS, so beyond the two core families the split is a
# name-pattern rule on the underscore-separated words of the predicate.
SCAFFOLD_FAMILIES = frozenset({'structural', 'attributive'})
SCAFFOLD_HEADS = frozenset({
    'has', 'have', 'contain', 'contains', 'include', 'includes', 'including',
    'on', 'in', 'inside', 'above', 'below', 'under', 'next', 'near',
    'date', 'cover', 'covers', 'wear', 'wears', 'located', 'is', 'are', 'attr',
    'size', 'color', 'colour', 'material', 'shape', 'type', 'name', 'label',
    'number', 'count', 'version', 'belongs',
})
SCAFFOLD_TAILS = frozenset({'id', 'attr'})


def edge_role(pred: str) -> str:
    """'scaffold' or 'relation' for a normalized predicate."""
    family = FAMILY_OF.get(pred)
    if family:
        return 'scaffold' if family in SCAFFOLD_FAMILIES else 'relation'
    words = [w for w in str(pred).lower().split('_') if w]
    if words and (words[0] in SCAFFOLD_HEADS or words[-1] in SCAFFOLD_TAILS):
        return 'scaffold'
    return 'relation'


# Spellings seen often enough to be worth collapsing exactly rather than by
# similarity, either because the fuzzy score is too low or because the surface
# forms diverge more than the meaning does.
PRED_ALIAS = {
    'mistrusts': 'distrusts', 'suspects': 'distrusts', 'doubts': 'distrusts',
    'wants': 'desires', 'seeks': 'desires', 'craves': 'desires',
    'afraid_of': 'fears', 'dreads': 'fears',
    'located_in': 'in', 'inside': 'in', 'within': 'in', 'is_in': 'in',
    'located_at': 'at', 'happens_at': 'at', 'set_in': 'at',
    'belongs_to': 'part_of', 'component_of': 'part_of',
    'owns': 'has', 'holds': 'has', 'carries': 'has',
    'type_of': 'is_a', 'kind_of': 'is_a', 'instance_of': 'is_a',
    'causes': 'caused', 'led_to': 'caused', 'results_in': 'caused',
    'before': 'precedes', 'after': 'follows',
    'hides': 'conceals', 'hidden_by': 'conceals', 'masks': 'conceals',
    'exposes': 'reveals', 'uncovers': 'reveals', 'discloses': 'reveals',
    'mirrors': 'echoes', 'recalls': 'echoes', 'repeats': 'echoes',
    'becomes': 'transforms_into', 'turns_into': 'transforms_into',
    'replaces': 'displaces', 'supplants': 'displaces',
    'attracted_to': 'draws_toward', 'pulled_toward': 'draws_toward',
    'against': 'opposes', 'undercuts': 'opposes', 'resists_reading': 'resists',
    'aware_of': 'knows', 'knows_of': 'knows', 'knows_about': 'knows',
    'said_to': 'told', 'informed': 'told',
    'suggests': 'implies', 'entails': 'implies',
    'contradicted_by': 'contradicts', 'conflicts_with': 'contradicts',
    'is': 'attr', 'attribute': 'attr', 'quality': 'attr', 'mood': 'attr',
    'state': 'attr', 'feels': 'attr',
}

PRED_FUZZ = 0.86  # deliberately high: collapsing two genuinely different
                  # relations is worse than carrying a synonym pair


def normalize_pred(pred: Any) -> str:
    """Lowercase, snake-case, then collapse onto a core term when clearly the same."""
    p = str(pred or '').strip().lower()
    p = ''.join(ch if (ch.isalnum() or ch == '_') else '_' for ch in p.replace('-', '_').replace(' ', '_'))
    while '__' in p:
        p = p.replace('__', '_')
    p = p.strip('_')
    if not p:
        return 'attr'
    if p in CORE_PREDS:
        return p
    if p in PRED_ALIAS:
        return PRED_ALIAS[p]
    best, score = None, 0.0
    for core in CORE_PREDS:
        r = SequenceMatcher(None, p, core).ratio()
        if r > score:
            best, score = core, r
    return best if score >= PRED_FUZZ else p


# --- ops ---

OP_NODE    = 'node'      # introduce or enrich an element
OP_ASSERT  = 'assert'    # state a relation
OP_RETRACT = 'retract'   # close a standing assertion (never deletes)
OP_THREAD  = 'thread'    # open / advance / resolve a line of forward pressure
OP_SHOW    = 'show'      # move something along latent -> hinted -> established

OPS = (OP_NODE, OP_ASSERT, OP_RETRACT, OP_THREAD, OP_SHOW)

_OP_ALIAS = {'entity': OP_NODE, 'add_node': OP_NODE, 'introduce': OP_NODE,
             'relation': OP_ASSERT, 'edge': OP_ASSERT, 'add_edge': OP_ASSERT,
             'assertion': OP_ASSERT, 'state': OP_ASSERT,
             'close': OP_RETRACT, 'end': OP_RETRACT,
             'tension': OP_THREAD, 'question': OP_THREAD,
             'disclose': OP_SHOW, 'reveal_to_audience': OP_SHOW}

THREAD_STATES = ('open', 'complicated', 'resolved')


def coerce_op(raw: Any) -> Optional[Dict[str, Any]]:
    """Normalise one op from untrusted output. Returns None if unusable.

    Everything here is defensive on purpose: nothing upstream validates model
    output, so an op arrives as whatever the model felt like emitting.
    """
    if not isinstance(raw, dict):
        return None
    op = str(raw.get('op') or raw.get('type') or '').strip().lower()
    op = _OP_ALIAS.get(op, op)
    if op not in OPS:
        # An op with a subject and a predicate is an assertion whatever it called
        # itself; one with a name and a kind is a node. Recovering these is worth
        # it because a single wrong 'op' value would otherwise drop real content.
        if raw.get('subject') and (raw.get('pred') or raw.get('predicate')):
            op = OP_ASSERT
        elif raw.get('name'):
            op = OP_NODE
        else:
            return None

    out: Dict[str, Any] = {'op': op}

    if op == OP_NODE:
        name = coerce_str(raw.get('name') or raw.get('id'), 200)
        if not name:
            return None
        out.update(name=name, kind=raw.get('kind'), gist=coerce_str(raw.get('gist'), 400),
                   detail=coerce_str(raw.get('detail')), aka=coerce_list(raw.get('aka')),
                   traits=coerce_dict(raw.get('traits')))
        if raw.get('shown') is not None:
            out['shown'] = coerce_shown(raw.get('shown'))
        if raw.get('salience') is not None:
            out['salience'] = coerce_num(raw.get('salience'), 0.5)

    elif op == OP_ASSERT:
        subject = coerce_str(raw.get('subject') or raw.get('s'), 200)
        obj     = coerce_str(raw.get('object') if raw.get('object') is not None else raw.get('o'), 400)
        pred    = normalize_pred(raw.get('pred') or raw.get('predicate') or raw.get('p'))
        if not subject or not obj:
            return None
        out.update(subject=subject, pred=pred, object=obj,
                   frame=coerce_str(raw.get('frame'), 120) or FRAME_WORLD,
                   conf=coerce_num(raw.get('conf'), 0.7),
                   valence=coerce_num(raw.get('valence'), 0.0, -1.0, 1.0),
                   intensity=coerce_num(raw.get('intensity'), 0.5),
                   subject_kind=raw.get('subject_kind'), object_kind=raw.get('object_kind'))
        if raw.get('shown') is not None:
            out['shown'] = coerce_shown(raw.get('shown'))

    elif op == OP_RETRACT:
        ref = coerce_str(raw.get('id') or raw.get('ref'), 120)
        if not ref:
            return None
        out['id'] = ref

    elif op == OP_THREAD:
        name = coerce_str(raw.get('name') or raw.get('question'), 300)
        if not name:
            return None
        state = str(raw.get('state') or 'open').strip().lower()
        out.update(name=name, gist=coerce_str(raw.get('gist'), 400),
                   state=state if state in THREAD_STATES else 'open',
                   pressure=coerce_num(raw.get('pressure'), 0.6),
                   involves=coerce_list(raw.get('involves')))

    elif op == OP_SHOW:
        ref = coerce_str(raw.get('ref') or raw.get('id') or raw.get('name'), 300)
        if not ref:
            return None
        out.update(ref=ref, shown=coerce_shown(raw.get('shown'), 'established'))

    return out


def parse_ops(payload: Any) -> List[Dict[str, Any]]:
    """Accept whatever a writer hands over — a list, a {'ops': [...]} envelope, or
    a JSON string of either — and return clean ops. Unusable entries are dropped
    silently rather than failing the batch: one malformed op should not cost the
    other nine.
    """
    if payload is None:
        return []
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except (json.JSONDecodeError, ValueError):
            return []
    if isinstance(payload, dict):
        payload = payload.get('ops') or payload.get('world_ops') or payload.get('items') or []
    if not isinstance(payload, (list, tuple)):
        return []
    out = []
    for raw in payload:
        op = coerce_op(raw)
        if op:
            out.append(op)
    return out
