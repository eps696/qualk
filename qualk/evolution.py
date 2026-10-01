"""How a probe's walk unfolds before it is measured.

A quantum walker has no path: between its start and the measurement it is a superposition, and
the measurement yields only an endpoint. What can be shown honestly is the probability of finding
the walker on each concept as the evolution time grows, next to the same quantity for the classical
diffusion control on the identical window and couplings. This module computes those series from the
window a walk recorded (its nodes and weighted edges); it uses the exact ideal evolution
`exp(-iAt)` (the circuit approximates it, and reports its error separately).
"""

from __future__ import annotations

from typing import Any, Dict, List

from .quantum_walk import coupling_matrix, diffusion_distribution, quantum_distribution


def evolution_series(window: Dict[str, Any], time: float, points: int = 16) -> Dict[str, List]:
    """Per-concept probabilities at `points` evenly spaced times in [0, time], for both walks.

    `window` is {'nodes': [...], 'edges': [...]} as recorded in a walk trace (node 0 is the seed).
    Returns {'times': [t_k], 'quantum': [[P_i(t_k)]], 'classical': [[P_i(t_k)]]}, rounded to 4 places.
    """
    if points < 2:
        raise ValueError('points must be at least 2')
    A = coupling_matrix(window)
    times = [time * k / (points - 1) for k in range(points)]
    return {'times': [round(t, 4) for t in times],
            'quantum': [[round(p, 4) for p in quantum_distribution(A, t)] for t in times],
            'classical': [[round(max(0., p), 4) for p in diffusion_distribution(A, t)] for t in times]}
