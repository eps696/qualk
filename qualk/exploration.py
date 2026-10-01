"""Channel-neutral probes and bounded outcome-driven parent selection."""
from dataclasses import dataclass, field, asdict
import math


@dataclass
class Probe:
    id: str
    born_tick: int
    components: list[str]
    render: str
    origin: str = 'spawn'
    parents: list[str] = field(default_factory=list)
    selection: str = 'fresh exploration'
    fitness: float = 0.0
    uses: int = 0
    outcomes: dict = field(default_factory=dict)
    walk: dict = field(default_factory=dict)
    thread: str = ''   # open-thread node id this probe was aimed at, or ''


def score_observation(novelty, content=1.0, fatigue=0.0):
    """One new observation contributes once. No salience or shape multiplier."""
    novelty = min(1.0, max(0.0, float(novelty)))
    content = min(1.0, max(0.0, float(content)))
    fatigue = max(0.0, float(fatigue))
    return dict(novelty=novelty, content=content, fatigue=fatigue,
                score=novelty * content / (1.0 + fatigue))


# Below this `1 - max cosine` an observation is a restatement of one already seen (the same
# page re-fetched, or a near-copy) and is worth exactly nothing; above it the raw distance is
# the novelty, uncalibrated, so its number means the same thing in every run. The scale comes
# from the embedder (see stim/embed.py: CLIP squeezed web text into 0.04, jina-clip-v2 does not).
DUPLICATE_DISTANCE = 0.05
NO_HISTORY_NOVELTY = 0.5   # nothing to compare against is neither novel nor stale — 1.0 here made
                           # the very first observation a runaway (narrated four times in a row)


def observation_novelty(distance):
    """`distance` = `1 - max cosine` against recent observations, or None with no history."""
    if distance is None:
        return NO_HISTORY_NOVELTY
    distance = min(1.0, max(0.0, float(distance)))
    return 0.0 if distance < DUPLICATE_DISTANCE else distance


# A concept that appears in one of this many latest probes is not picked again as a walk target:
# without it a hub of the small early relation graph sits in almost every window and every
# probe re-queries the same topic (dog7: 23% of picks), which keeps observation novelty low.
TARGET_COOLDOWN = 24
# How hard a node's visit count weighs against drawing it again: weight = 1 / (1 + visits) ** VISIT_POWER.
VISIT_POWER = 2


class ProbeArchive:
    """Small recency-bounded archive; outcome fitness biases reuse, unless explicitly
    told not to (see `explore` below).

    `explore` is not "how much new information enters" — a fresh pick still draws
    from the same graph a mutate/cross pick does (see `next()`'s own docstring); new
    information only ever enters through whichever external channel (web/visual)
    fires afterward, independent of this setting either way. What it actually
    controls is *which graph region* a probe aims at: fresh spreads queries across
    under-visited nodes by inverse visit count, exploit re-aims at a previously
    fitness-scored probe's own node combination. A `0.2` floor used to be forced
    regardless of the caller's own setting; removed so a caller that explicitly
    wants pure exploitation (`explore=0`) actually gets it."""
    def __init__(self, rng, explore=.4, capacity=48):
        self.rng = rng
        self.explore = max(0., min(1.0, explore))
        self.capacity = capacity
        self.probes = []
        self.visits = {}
        self.sequence = 0

    def reward(self, probe_id, novelty=None, attention=None, field_weight=None):
        p = next((p for p in self.probes if p.id == probe_id), None)
        if p is None:
            return
        for name, value in [('novelty', novelty), ('attention', attention)]:
            if value is not None:
                p.outcomes[name] = min(1., max(0., value))
        if field_weight is not None:
            p.outcomes['field_weight'] = max(
                p.outcomes.get('field_weight', 0.), max(0., float(field_weight)))
        # Bounded components and usage pressure stop an early winner owning every parent slot.
        p.fitness = (.5 * p.outcomes.get('novelty', 0.) +
                     .25 * p.outcomes.get('attention', 0.) +
                     .25 * (1. - 1. / (1. + p.outcomes.get('field_weight', 0.))))

    def _recent_components(self):
        return {c for p in self.probes[-TARGET_COOLDOWN:] for c in p.components}

    def _parent(self, candidates):
        return self.rng.choices(candidates, weights=[
            (.1 + p.fitness) / math.sqrt(1 + p.uses) for p in candidates], k=1)[0]

    def next(self, graph, index, at, born_of_intake=True, quantum_walk=None, seed_override=None,
             threads=None, thread_aim=0.0, thread_focus=None):
        """The next probe: fresh (seed by inverse visit count + walk target), mutated (one
        component of a fitness-chosen parent replaced by a walk pick) or crossed. `seed_override`
        (a node id) forces a fresh probe seeded from that concept, its target still walk-chosen.

        `threads` (a `ThreadKeeper`) switches thread behaviour on: thread nodes leave the ordinary
        draw, and with probability `thread_aim` - or always, for a steered `thread_focus` (a thread
        id) - the probe is aimed at an open thread (see `_thread_probe`)."""
        ids = [n.id for n in graph.nodes.values()
               if (n.kind != 'thread' or n.traits.get('state') != 'resolved')
               and (not born_of_intake or not (n.by or '').startswith('frag:'))]
        if threads is not None:
            ids = [i for i in ids if graph.nodes[i].kind != 'thread']
        # A probe is rendered from its components' gists (a web query): a node with only a name is
        # never worth drawing. Unfiltered only when too few have one.
        gisted = [i for i in ids if graph.nodes[i].gist]
        if len(gisted) >= 2:
            ids = gisted
        if not ids:
            return None
        walk_ids = [i for i in ids if graph.nodes[i].kind != 'thread']
        def draw(exclude=(), candidates=None):
            choices = candidates if candidates is not None else ids
            options = [i for i in choices if i not in exclude] or choices
            return self.rng.choices(options, weights=[1 / (1 + self.visits.get(i, 0)) ** VISIT_POWER
                                                     for i in options], k=1)[0]
        if threads is not None and seed_override is None and (
                thread_focus or (thread_aim > 0 and self.rng.random() < thread_aim)):
            probe = self._thread_probe(graph, at, threads, walk_ids, draw, quantum_walk, tid=thread_focus)
            if probe is not None:
                return probe
            # No open, groundable thread this round - fall through to ordinary selection.
        parents = []
        walk_trace = {}
        live = [p for p in self.probes if all(c in ids for c in p.components)]
        origin, reason = 'spawn', 'fresh: inverse visit count'
        if seed_override is not None and seed_override not in walk_ids:
            seed_override = None
        if live and seed_override is None and self.rng.random() >= self.explore:
            a = self._parent(live)
            parents = [a]
            others = [p for p in live if p.id != a.id and p.components != a.components]
            if others and self.rng.random() < .5:
                # Favour semantically distinct mates; raw cosine remains inspectable in the index.
                av = index.put('probe:' + a.id, text=a.render)
                weights = [max(.05, 1 - sum(x*y for x,y in zip(av,index.put('probe:' + p.id, text=p.render))))
                           for p in others]
                b = self.rng.choices(others, weights=weights, k=1)[0]
                parents.append(b)
                comps = list(dict.fromkeys([self.rng.choice(a.components), self.rng.choice(b.components)]))
                origin, reason = 'cross', 'fitness parent + semantic diversity mate'
                if quantum_walk is not None:
                    walk_trace = {'skipped': 'crossover'}
            else:
                comps = list(a.components)
                slot = self.rng.randrange(len(comps))
                old = comps[slot]
                replacement = None
                if quantum_walk is not None:
                    if old in walk_ids:
                        replacement, walk_trace = quantum_walk.select(
                            graph, old, walk_ids, self.rng,
                            exclude=set(comps) | self._recent_components(), round_idx=at)
                    else:
                        walk_trace = {'fallback': 'ineligible_seed', 'seed_node': old}
                if replacement is not None:
                    comps[slot] = replacement
                    origin, reason = 'mutate', 'fitness parent + graph walk replacement'
                else:
                    hits = index.search(index.encode(graph.nodes[old].gist or graph.nodes[old].name), k=8)
                    near = [h['metadata']['node_id'] for h in hits
                            if h['metadata'].get('node_id') in ids and h['metadata']['node_id'] not in comps]
                    comps[slot] = self.rng.choice(near) if near else draw(comps)
                    origin, reason = 'mutate', 'fitness parent + semantic neighbour replacement'
        else:
            # A walk only has an arena where a relation cluster exists, so its seed is drawn
            # (still by inverse visit count) from those nodes; a walk mode without the hook
            # accepts every node.
            seedable = walk_ids
            if quantum_walk is not None and hasattr(quantum_walk, 'seed_candidates'):
                allowed = quantum_walk.seed_candidates(graph, walk_ids)
                seedable = [i for i in walk_ids if i in allowed]
            if seed_override is not None:
                seedable = [seed_override]
                reason = 'pinned seed'
            if quantum_walk is not None and seedable:
                comps = [seed_override if seed_override is not None else draw(candidates=seedable)]
                if len(walk_ids) > 1:
                    target, walk_trace = quantum_walk.select(
                        graph, comps[0], walk_ids, self.rng,
                        exclude=set(comps) | self._recent_components(), round_idx=at)
                    if target is not None:
                        comps.append(target)
                        reason = ('pinned seed + graph walk target' if seed_override is not None
                                  else 'fresh: inverse visits seed + graph walk target')
                    elif len(ids) > 1 and self.rng.random() < .5:
                        comps.append(draw(comps))
                else:
                    walk_trace = {'fallback': 'one_nonthread_node', 'seed_node': comps[0]}
                    if len(ids) > 1 and self.rng.random() < .5:
                        comps.append(draw(comps))
            else:
                comps = [seed_override if seed_override is not None else draw()]
                if quantum_walk is not None:
                    walk_trace = {'fallback': 'no_seedable_cluster' if walk_ids
                                  else 'no_nonthread_nodes'}
                if len(ids) > 1 and self.rng.random() < .5:
                    comps.append(draw(comps))
        # The same component set is the same query, the same top result and observation
        # novelty 0 (dog1 p12 = p13, dog2 p5 = p7 = p13: crossover alone can only ever
        # reproduce its parents' halves). A set already in the archive is re-drawn.
        taken = {frozenset(p.components) for p in self.probes}
        if frozenset(comps) in taken and seed_override is None:      # an explicit pin is never re-drawn
            alt = None
            if origin == 'cross':
                pairs = [list(dict.fromkeys([x, y])) for x in parents[0].components
                         for y in parents[1].components]
                self.rng.shuffle(pairs)
                alt = next((c for c in pairs if frozenset(c) not in taken), None)
            if alt is None:
                for _ in range(8):
                    c = [draw()]
                    if len(ids) > 1 and self.rng.random() < .5:
                        c.append(draw(c))
                    if frozenset(c) not in taken:
                        alt = c
                        break
            if alt is not None:
                comps = alt
                if origin != 'cross' or frozenset(comps) not in {frozenset(x) for x in pairs}:
                    origin, parents, walk_trace = 'spawn', [], {}
                reason += ' [re-drawn: component set already probed]'
        for p in parents:
            p.uses += 1
        for c in comps:
            self.visits[c] = self.visits.get(c, 0) + 1
        self.sequence += 1
        render = ' / '.join(graph.nodes[c].gist or graph.nodes[c].name for c in comps)
        p = Probe(f'p{self.sequence:04d}', at, comps, render, origin,
                  [p.id for p in parents], reason, walk=walk_trace)
        self.probes.append(p)
        self.probes = self.probes[-self.capacity:]
        return p

    def _thread_probe(self, graph, at, threads, walk_ids, draw, quantum_walk, tid=None):
        """A probe aimed at an open thread: the pressure-weighted pick, or the steered `tid`. Its
        components are the thread's own grounded `part_of` elements (concrete concepts the thread was
        recorded as involving) - never the thread's own LLM-written name/gist text, which never
        reaches a search query. The walk then chooses the second concept from the whole graph."""
        steered = tid is not None
        if steered:
            node = graph.nodes.get(tid)
            if node is None or node.kind != 'thread' or node.traits.get('state') == 'resolved':
                return None
        else:
            tid = threads.pick(graph, self.rng, at)
        if tid is None:
            return None
        involved = [i for i in threads.involved_ids(graph, tid) if i in walk_ids]
        if not involved:
            return None
        comps = [draw(candidates=involved)]
        walk_trace = {}
        if quantum_walk is not None and len(walk_ids) > 1:
            target, walk_trace = quantum_walk.select(
                graph, comps[0], walk_ids, self.rng,
                exclude=set(comps) | self._recent_components(), round_idx=at)
            if target is not None:
                comps.append(target)
        if len(comps) == 1 and len(involved) > 1:
            comps.append(draw(exclude=comps, candidates=involved))
        for c in comps:
            self.visits[c] = self.visits.get(c, 0) + 1
        self.sequence += 1
        render = ' / '.join(graph.nodes[c].gist or graph.nodes[c].name for c in comps)
        p = Probe(f'p{self.sequence:04d}', at, comps, render, 'draw', [],
                  'thread: steered focus' if steered else 'thread: pressure-weighted open question',
                  walk=walk_trace, thread=tid)
        self.probes.append(p)
        self.probes = self.probes[-self.capacity:]
        return p

    def snapshot(self):
        # A probe's walk trace is already logged once, with the round that made it; copying
        # it into every snapshot made the archive ~half trace.
        return dict(probes=[{k: v for k, v in asdict(p).items() if k != 'walk'}
                            for p in self.probes], visits=self.visits,
                    sequence=self.sequence, explore=self.explore, rng_state=self.rng.getstate())

    def restore(self, data):
        self.probes = [Probe(**p) for p in data.get('probes', [])]
        for probe in self.probes:
            probe.outcomes.pop('narrated', None)
            self.reward(probe.id)
        self.visits = dict(data.get('visits', {}))
        self.sequence = data.get('sequence', 0)
        if data.get('rng_state'):
            state = data['rng_state']
            self.rng.setstate((state[0], tuple(state[1]), state[2]))
