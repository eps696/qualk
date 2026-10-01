"""Optional graph walk for dog probes: a quantum circuit and its classical control.

Both walks run on the same window of the active dog graph and the same couplings, so they
differ only in their dynamics: `DiffusionProbeWalk` is an ordinary random walker (needs no
Qiskit), `QuantumProbeWalk` is the coherent version as a Qiskit circuit. Only *relation* edges
of the world graph are coupled (never scaffold edges such as `part_of`), weighted by
confidence and the edge's CLIP affinity. The circuit moves one excitation between node
qubits; no operation in this module updates the world graph. Qiskit is imported only when
the quantum option is constructed. See docs/quantum-probes.md.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np


# CLIP cosines between related concepts sit in roughly [0.6, 1] (random pairs average 0.62),
# so an edge's affinity is rescaled from [floor, 1] to [0, 1]. Calibrated for the default
# CLIP backend; another embedding backend needs its own floor.
AFFINITY_FLOOR = 0.6
HOSTILE_VALENCE = -0.3      # at or below this an edge couples with a negative sign


def _open_assertions(graph):
    assertions = graph.assertions.values() if isinstance(graph.assertions, dict) else graph.assertions
    return [a for a in assertions if a.until is None and a.frame == 'world']


def _weight(assertion, floor=None):
    """conf x (0.4 + 0.6 x affinity), signed by valence. Unknown affinity is neutral."""
    floor = AFFINITY_FLOOR if floor is None else floor      # read at call time: set per embedder
    confidence = float(assertion.conf)
    if not math.isfinite(confidence):
        return None
    affinity = getattr(assertion, 'affinity', None)
    scaled = 0.5 if affinity is None else max(0., min(1., (float(affinity) - floor) / (1. - floor)))
    weight = max(0., min(1., confidence)) * (0.4 + 0.6 * scaled)
    valence = float(getattr(assertion, 'valence', 0.) or 0.)
    return -weight if valence <= HOSTILE_VALENCE else weight


def relation_edges(graph, eligible_ids):
    """{(a, b): edge} for the meaningful edges the walk may use: open, world-frame *relation*
    assertions (never scaffold - see world.ops.edge_role) between two eligible non-thread
    nodes. Per unordered pair the strongest claim wins. Nothing here writes to the graph."""
    from .world.ops import edge_role
    eligible = {node_id for node_id in eligible_ids
                if node_id in graph.nodes and graph.nodes[node_id].kind != 'thread'}
    pairs = {}
    for assertion in _open_assertions(graph):
        u, v = assertion.subject, assertion.object
        if u not in eligible or v not in eligible or u == v or edge_role(assertion.pred) != 'relation':
            continue
        weight = _weight(assertion)
        if not weight:
            continue
        pair = tuple(sorted((u, v)))
        previous = pairs.get(pair)
        if previous is None or abs(weight) > abs(previous['weight']):
            pairs[pair] = {'a': pair[0], 'b': pair[1], 'weight': weight,
                           'conf': float(assertion.conf),
                           'affinity': getattr(assertion, 'affinity', None),
                           'valence': float(getattr(assertion, 'valence', 0.) or 0.),
                           'predicates': [assertion.pred]}
        elif abs(weight) == abs(previous['weight']) and assertion.pred not in previous['predicates']:
            previous['predicates'].append(assertion.pred)
    return pairs


def seedable_nodes(graph, eligible_ids, min_component=3):
    """Nodes whose relation cluster has at least `min_component` concepts - the only places
    a walk has an arena. Union-find over the relation edges."""
    parent = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in relation_edges(graph, eligible_ids):
        parent[find(a)] = find(b)
    sizes = {}
    for node in list(parent):
        root = find(node)
        sizes[root] = sizes.get(root, 0) + 1
    return {node for node in parent if sizes[find(node)] >= min_component}


def project_window(graph, seed_id, eligible_ids, max_nodes=12):
    """A deterministic connected window: from the seed, repeatedly take the frontier neighbour
    reached by the strongest relation edge (ties by id), up to `max_nodes`. The couplings are
    every relation edge among the chosen nodes."""
    eligible = set(eligible_ids)
    if seed_id not in eligible or seed_id not in graph.nodes or graph.nodes[seed_id].kind == 'thread':
        return {'nodes': [], 'edges': []}
    pairs = relation_edges(graph, eligible)
    chosen = [seed_id]
    seen = {seed_id}
    while len(chosen) < max_nodes:
        frontier = []
        for edge in pairs.values():
            a, b = edge['a'], edge['b']
            if (a in seen) != (b in seen):
                destination = b if a in seen else a
                frontier.append((-abs(edge['weight']), destination))
        if not frontier:
            break
        _, destination = min(frontier)
        chosen.append(destination)
        seen.add(destination)
    positions = {node_id: i for i, node_id in enumerate(chosen)}
    edges = [{**edge, 'u': positions[edge['a']], 'v': positions[edge['b']],
              'predicates': sorted(edge['predicates'])}
             for edge in pairs.values() if edge['a'] in seen and edge['b'] in seen]
    edges.sort(key=lambda e: (min(e['u'], e['v']), max(e['u'], e['v']), e['a'], e['b']))
    return {'nodes': chosen, 'edges': edges}


def coupling_matrix(window):
    n = len(window['nodes'])
    A = np.zeros((n, n))
    for edge in window['edges']:
        A[edge['u'], edge['v']] = A[edge['v'], edge['u']] = edge['weight']
    return A


def quantum_distribution(A, time):
    """Exact one-excitation continuous-time quantum walk |exp(-iAt) e_0|^2 from node 0."""
    values, vectors = np.linalg.eigh(np.asarray(A, float))
    amplitudes = vectors @ (np.exp(-1j * time * values) * vectors[0, :])
    return [float(p) for p in np.abs(amplitudes) ** 2]


def trotter_distribution(window, time, steps):
    """What the symmetric-Trotter circuit outputs, exactly and without Qiskit: the same
    edge-by-edge partial swaps applied to the one-excitation amplitudes. Lets a backend that
    only returns samples (Atlas) report its circuit error against the ideal circuit."""
    n = len(window['nodes'])
    psi = np.zeros(n, complex)
    psi[0] = 1.
    edges = window['edges']
    for _ in range(steps):
        for edge in edges + edges[::-1]:
            theta = 0.5 * time * edge['weight'] / steps
            u, v = edge['u'], edge['v']
            a, b = psi[u], psi[v]
            psi[u] = np.cos(theta) * a - 1j * np.sin(theta) * b
            psi[v] = np.cos(theta) * b - 1j * np.sin(theta) * a
    return [float(p) for p in np.abs(psi) ** 2]


def diffusion_distribution(A, time):
    """The classical counterpart: a random walker on the same graph, exp(-tL) e_0 with
    L = D - |A|. Probabilities only, no phases; edge signs cannot matter to it."""
    weights = np.abs(np.asarray(A, float))
    laplacian = np.diag(weights.sum(axis=1)) - weights
    values, vectors = np.linalg.eigh(laplacian)
    probabilities = vectors @ (np.exp(-time * values) * vectors[0, :])
    return [float(p) for p in probabilities]


def parse_shot(bitstring, n):
    """Node index of a one-hot shot, or None if it does not hold exactly one excitation
    (hardware error). Qiskit prints qubit n-1 first."""
    if bitstring.count('1') != 1:
        return None
    return n - 1 - bitstring.index('1')


class _GraphWalk:
    """Window, couplings, eligibility and fallbacks shared by the quantum walk and its
    classical control, so the two differ only in their dynamics."""
    method = ''

    def __init__(self, max_nodes=12, time=2.0):
        if not isinstance(max_nodes, int) or not 2 <= max_nodes <= 24:
            raise ValueError('quantum_nodes must be an integer in [2, 24]')
        if not math.isfinite(time) or time <= 0:
            raise ValueError('quantum_time must be positive and finite')
        self.max_nodes = max_nodes
        self.time = float(time)

    def seed_candidates(self, graph, eligible_ids):
        """Where a walk has an arena at all: nodes in a relation cluster of >= 3 concepts."""
        return seedable_nodes(graph, eligible_ids)

    def _settings(self):
        return {}

    def _pick(self, window, A, options, rng, round_idx):
        raise NotImplementedError

    def select(self, graph, seed_id, eligible_ids, rng, exclude=(), round_idx=None):
        window = project_window(graph, seed_id, eligible_ids, self.max_nodes)
        ids, edges = window['nodes'], window['edges']
        trace = {'method': self.method, 'seed_node': seed_id, 'nodes': ids, 'edges': edges,
                 'time': self.time, **self._settings()}
        if not ids:
            return None, {**trace, 'fallback': 'ineligible_seed'}
        if not edges:
            return None, {**trace, 'fallback': 'isolated_seed'}
        options = [i for i, node_id in enumerate(ids) if i != 0 and node_id not in exclude]
        if not options:
            return None, {**trace, 'fallback': 'no_eligible_target'}

        A = coupling_matrix(window)
        classical = diffusion_distribution(A, self.time)
        trace['forced'] = len(options) == 1
        trace['classical_probabilities'] = classical
        trace['quantum_probabilities'] = quantum_distribution(A, self.time)
        choice, extra = self._pick(window, A, options, rng, round_idx)
        trace.update(extra)
        trace['total_variation_distance'] = 0.5 * sum(
            abs(a - b) for a, b in zip(trace['quantum_probabilities'], classical))
        if choice is None:
            return None, {**trace, 'fallback': 'no_eligible_measurements'}
        trace['target'] = ids[choice]
        return ids[choice], trace


class DiffusionProbeWalk(_GraphWalk):
    """The classical control: a random walker on exactly the window and couplings the quantum
    walk uses, for the same time. Needs no Qiskit."""
    method = 'diffusion'

    def _pick(self, window, A, options, rng, round_idx):
        weights = [max(0., p) for p in diffusion_distribution(A, self.time)]
        weights = [weights[i] for i in options]
        if not any(weights):
            return None, {}
        return rng.choices(options, weights=weights, k=1)[0], {}


class QuantumProbeWalk(_GraphWalk):
    """Finite-shot Qiskit simulation of a weighted, coherent XY graph walk. One qubit per
    concept, one excitation; `RXX`/`RYY` pairs in a symmetric (second-order) Trotter schedule
    realise exp(-iAt) on the one-excitation subspace."""
    method = 'qiskit-statevector'
    TROTTER_WARN = 0.1
    ATLAS_MAX_FAILURES = 3      # consecutive Atlas failures before it is skipped for the rest of the run
    # A 2**n Qiskit statevector is wasteful (and, per dog7's crash, risky) past this width - the
    # one-excitation subspace has an exact closed form (trotter_distribution), checked to match
    # the real circuit exactly (TrotterNumpyTests), so the local fallback uses that instead above
    # this size. Atlas is asked for its matrix_product_state method at the same width, since its
    # default 'automatic' method runs out of memory there too (measured: fails above 24 qubits on
    # this window density; matrix_product_state carried a 24-qubit window in ~17s).
    LOCAL_STATEVECTOR_MAX = 16
    ATLAS_WIDE_BACKEND = 'matrix_product_state'

    def __init__(self, max_nodes=12, steps=5, time=2.0, shots=1024, trace_dir=None,
                 backend='atlas', atlas=None, atlas_strict=None):
        super().__init__(max_nodes=max_nodes, time=time)
        if not isinstance(steps, int) or not 1 <= steps <= 20:
            raise ValueError('quantum_steps must be an integer in [1, 20]')
        if not isinstance(shots, int) or not 1 <= shots <= 65536:
            raise ValueError('quantum_shots must be an integer in [1, 65536]')
        if backend not in ('atlas', 'qpu', 'qiskit'):
            raise ValueError('quantum_backend must be atlas, qpu or qiskit')
        try:
            from qiskit import QuantumCircuit, qasm2, qasm3
            from qiskit.primitives import StatevectorSampler
            from qiskit.quantum_info import Statevector
        except ImportError as exc:
            raise RuntimeError('Quantum probe walk requires Qiskit; classical mode does not') from exc
        self.QuantumCircuit = QuantumCircuit
        self.StatevectorSampler = StatevectorSampler
        self.Statevector = Statevector
        self.qasm2 = qasm2
        self.qasm3 = qasm3
        self.steps = steps
        self.shots = shots
        self.trace_dir = Path(trace_dir) if trace_dir is not None else None
        # Atlas (Moth's platform) runs the circuit when it can; local Qiskit is the fallback and
        # is also what builds the circuit, so it stays a requirement either way.
        self.atlas = None
        if backend in ('atlas', 'qpu'):
            self.atlas = atlas
            if self.atlas is None:
                from .atlas import AtlasClient
                self.atlas = AtlasClient.from_env(backend)
            if self.atlas is None and backend == 'qpu':
                raise ValueError('quantum backend qpu needs MOTH_API_KEY and MOTH_QPU_PROVIDER (and usually '
                                 'MOTH_QPU_BACKEND) to be set; the walk will not silently use a simulator')
            if self.atlas is None:
                print('.. MOTH_API_KEY is not set: the quantum walk runs on local Qiskit '
                      '(set it, e.g. via env.bat, to use Moth Atlas)')
        self.atlas_failures = 0
        self.atlas_disabled = False
        # Strict: an Atlas failure stops the run instead of quietly substituting local Qiskit (for a real
        # QPU run, a silent simulator pick would be a lie). Default from MOTH_ATLAS_STRICT.
        if atlas_strict is None and backend == 'qpu':
            atlas_strict = True                  # the real device never falls back to a simulator
        if atlas_strict is None:
            atlas_strict = os.environ.get('MOTH_ATLAS_STRICT', '').strip().lower() in ('1', 'true', 'yes', 'on')
        self.atlas_strict = bool(atlas_strict) and self.atlas is not None

    def _settings(self):
        return {'steps': self.steps, 'shots': self.shots}

    def _circuit(self, window):
        circuit = self.QuantumCircuit(len(window['nodes']))
        circuit.x(0)
        edges = window['edges']
        for _ in range(self.steps):
            for edge in edges + edges[::-1]:
                theta = 0.5 * self.time * edge['weight'] / self.steps
                circuit.rxx(theta, edge['u'], edge['v'])
                circuit.ryy(theta, edge['u'], edge['v'])
        return circuit

    def circuit_probabilities(self, window):
        """Exact per-node probabilities of the Trotter circuit."""
        n = len(window['nodes'])
        probabilities = self.Statevector.from_instruction(self._circuit(window)).probabilities()
        per_node = [float(probabilities[1 << i]) for i in range(n)]
        if not math.isclose(sum(per_node), 1., abs_tol=1e-9):
            raise AssertionError('Quantum walk left the one-excitation subspace')
        return per_node

    def _pick(self, window, A, options, rng, round_idx):
        """Atlas when it is configured and healthy, otherwise (or on any Atlas failure) local Qiskit."""
        failure = None
        if self.atlas is not None and not self.atlas_disabled:
            from .atlas import AtlasError
            try:
                choice, extra = self._pick_atlas(window, A, options, rng, round_idx)
                self.atlas_failures = 0
                return choice, extra
            except AtlasError as exc:
                failure = str(exc)
                if self.atlas_strict:
                    raise RuntimeError(f'Atlas run failed and MOTH_ATLAS_STRICT forbids the local fallback: {failure}') from exc
                self.atlas_failures += 1
                if self.atlas_failures >= self.ATLAS_MAX_FAILURES:
                    self.atlas_disabled = True
                    print(f'.. Atlas failed {self.atlas_failures} times in a row ({failure}); '
                          f'using local Qiskit for the rest of this run')
                else:
                    print(f'.. Atlas walk failed ({failure}); falling back to local Qiskit for this pick')
        choice, extra = self._pick_local(window, A, options, rng, round_idx)
        if failure is not None:
            extra['atlas_error'] = failure
        if self.atlas_disabled:
            extra['atlas_disabled'] = True
        return choice, extra

    def _pick_atlas(self, window, A, options, rng, round_idx):
        """Run the circuit on Moth's Atlas platform (its `tomography-api-v2` engine returns the
        Z-basis shots of the whole register) and pick from the returned counts. The pick draws from
        the eligible concepts' counts, which is what "the first eligible shot" amounts to."""
        from .atlas import AtlasError
        n, edges = len(window['nodes']), window['edges']
        qasm = self.qasm2.dumps(self._circuit(window))
        # Respect an explicit MOTH_ATLAS_BACKEND; only 'automatic' (the client default) is
        # widened past LOCAL_STATEVECTOR_MAX, where Atlas's own automatic method runs out of
        # memory on this platform (measured).
        backend_name = self.atlas.backend_name
        if backend_name == 'automatic' and n > self.LOCAL_STATEVECTOR_MAX:
            backend_name = self.ATLAS_WIDE_BACKEND
        shot_counts, meta = self.atlas.z_counts(qasm, n, self.shots, backend_name=backend_name)
        counts, rejected = [0] * n, 0
        for bitstring, count in shot_counts.items():
            node = parse_shot(bitstring, n) if len(bitstring) == n else None
            if node is None:
                rejected += count       # not a one-hot outcome: an error, not a measurement of a concept
            else:
                counts[node] += count
        total = sum(counts)
        if total == 0:
            raise AtlasError('Atlas returned no valid one-excitation measurements')
        measured = [c / total for c in counts]
        exact = quantum_distribution(A, self.time)
        ideal = trotter_distribution(window, self.time, self.steps)
        trotter_error = 0.5 * sum(abs(a - b) for a, b in zip(ideal, exact))
        weights = [counts[i] for i in options]
        choice = rng.choices(options, weights=weights, k=1)[0] if any(weights) else None
        extra = {'method': 'atlas-%s' % meta['provider'], 'quantum_probabilities': measured,
                 'exact_quantum_probabilities': exact, 'trotter_error': trotter_error,
                 'trotter_warning': trotter_error > self.TROTTER_WARN,
                 'sampling_error': 0.5 * sum(abs(a - b) for a, b in zip(measured, ideal)),
                 'sampled_counts': counts, 'rejected_shots': rejected,
                 'two_qubit_gate_count': 4 * len(edges) * self.steps,
                 'atlas_job_id': meta['job_id'], 'atlas_seconds': meta['seconds'],
                 'atlas_provider': meta['provider'], 'atlas_backend': backend_name}
        if self.trace_dir is not None and round_idx is not None:
            self.trace_dir.mkdir(parents=True, exist_ok=True)
            qasm_path = self.trace_dir / f'round-{round_idx:05d}.qasm'
            qasm_path.write_text(qasm, encoding='utf-8')     # the exact OpenQASM 2 that was submitted
            extra['qasm_path'] = '%s/%s' % (self.trace_dir.name, qasm_path.name)
        return choice, extra

    def _pick_local(self, window, A, options, rng, round_idx):
        if len(window['nodes']) > self.LOCAL_STATEVECTOR_MAX:
            return self._pick_local_wide(window, A, options, rng, round_idx)
        ids, edges = window['nodes'], window['edges']
        n = len(ids)
        circuit = self._circuit(window)
        circuit_p = self.circuit_probabilities(window)
        exact = quantum_distribution(A, self.time)
        trotter_error = 0.5 * sum(abs(a - b) for a, b in zip(circuit_p, exact))
        sampled = circuit.copy()
        sampled.measure_all()
        sampler_seed = rng.randrange(2 ** 32)
        shots = self.StatevectorSampler(seed=sampler_seed).run(
            [sampled], shots=self.shots).result()[0].data.meas.get_bitstrings()
        counts, rejected, choice = [0] * n, 0, None
        eligible = set(options)
        for bitstring in shots:
            node = parse_shot(bitstring, n)
            if node is None:
                rejected += 1          # a hardware error, not a measurement of any concept
                continue
            counts[node] += 1
            if choice is None and node in eligible:
                choice = node           # the first eligible shot decides
        extra = {'sampler_seed': sampler_seed, 'quantum_probabilities': circuit_p,
                 'exact_quantum_probabilities': exact, 'trotter_error': trotter_error,
                 'trotter_warning': trotter_error > self.TROTTER_WARN,
                 'sampled_counts': counts, 'rejected_shots': rejected,
                 'two_qubit_gate_count': 4 * len(edges) * self.steps}
        if self.trace_dir is not None and round_idx is not None:
            self.trace_dir.mkdir(parents=True, exist_ok=True)
            qasm_path = self.trace_dir / f'round-{round_idx:05d}.qasm'
            qasm_path.write_text(self.qasm3.dumps(sampled), encoding='utf-8')
            extra['qasm_path'] = '%s/%s' % (self.trace_dir.name, qasm_path.name)
        return choice, extra

    def _pick_local_wide(self, window, A, options, rng, round_idx):
        """Beyond LOCAL_STATEVECTOR_MAX qubits: skip building a 2**n Qiskit statevector and use
        the one-excitation subspace's exact closed form directly (`trotter_distribution` - checked
        to equal the real circuit's output exactly at sizes where both run, TrotterNumpyTests).
        Sampling from it is exact too: the state never leaves the one-excitation subspace, so
        every real measurement would be one-hot with precisely these probabilities."""
        ids, edges = window['nodes'], window['edges']
        n = len(ids)
        probabilities = trotter_distribution(window, self.time, self.steps)
        exact = quantum_distribution(A, self.time)
        trotter_error = 0.5 * sum(abs(a - b) for a, b in zip(probabilities, exact))
        picks = rng.choices(range(n), weights=probabilities, k=self.shots)
        counts = [0] * n
        for node in picks:
            counts[node] += 1
        eligible = set(options)
        choice = next((node for node in picks if node in eligible), None)
        extra = {'method': 'numpy-trotter', 'quantum_probabilities': probabilities,
                 'exact_quantum_probabilities': exact, 'trotter_error': trotter_error,
                 'trotter_warning': trotter_error > self.TROTTER_WARN,
                 'sampled_counts': counts, 'rejected_shots': 0,
                 'two_qubit_gate_count': 4 * len(edges) * self.steps}
        if self.trace_dir is not None and round_idx is not None:
            self.trace_dir.mkdir(parents=True, exist_ok=True)
            qasm_path = self.trace_dir / f'round-{round_idx:05d}.qasm'
            qasm_path.write_text(self.qasm3.dumps(self._circuit(window)), encoding='utf-8')
            extra['qasm_path'] = '%s/%s' % (self.trace_dir.name, qasm_path.name)
        return choice, extra
