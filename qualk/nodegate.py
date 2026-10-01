"""
Node gate (extracted from assembly's `stim/nodegate.py`; the engine composes it into `Digester.op_filter`).

`world-upd` was told a detail is not an element and every element needs a gist, and neither
held: `WorldGraph._op_assert` calls `ensure()` on whatever names a claim mentions, so any
name — a figure, a date, a place named once — became a node with nothing but its name
(dog8: 229 of 564 nodes; a probe built from one is a bare word, and a bare word is a poor web
query and a poor image prompt). Enforced here, mechanically, before `graph.apply`:

- a `node` op without a gist is dropped, unless the node already exists with one;
- a claim is dropped unless both ends are existing gisted nodes or gisted nodes declared in
  the same batch (a literal-valued predicate's object is plain text and is not an end);
- threads and every other op pass through untouched (a thread's name is its question).

`last` holds the drop counts of the latest batch, for the round card.
"""
from typing import Any, Dict, List

from .world.model import norm_name
from .world.ops import LITERAL_PREDS


class NodeGate:
    def __init__(self):
        self.last: Dict[str, int] = {'ops': 0, 'nodes': 0, 'claims': 0}

    def filter_ops(self, ops: List[Dict[str, Any]], graph: Any, index: Any = None
                   ) -> List[Dict[str, Any]]:
        declared = {norm_name(op['name']) for op in ops
                    if op.get('op') == 'node' and op.get('kind') != 'thread'
                    and op.get('name') and str(op.get('gist') or '').strip()}

        def ground(name) -> bool:
            nid = graph.resolve(name)
            if nid is not None and graph.nodes[nid].gist:
                return True
            return norm_name(str(name)) in declared

        out: List[Dict[str, Any]] = []
        nodes = claims = 0
        for op in ops:
            kind = op.get('op')
            if kind == 'node' and op.get('kind') != 'thread':
                if not str(op.get('gist') or '').strip() and not ground(op.get('name')):
                    nodes += 1
                    continue
            elif kind == 'assert':
                ends = [op.get('subject')]
                if op.get('pred') not in LITERAL_PREDS:
                    ends.append(op.get('object'))
                if not all(ground(e) for e in ends):
                    claims += 1
                    continue
            out.append(op)
        self.last = {'ops': len(ops), 'nodes': nodes, 'claims': claims}
        return out
