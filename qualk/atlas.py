"""Client for Moth's Atlas platform (https://api.mothquantum.com), used to run dog's quantum-walk
circuit there instead of on a local simulator.

Atlas is a catalogue of prebuilt engines, not a generic "run any circuit" service, and none of
them is a quantum walk (checked against its OpenAPI spec, v0.41.0). The one engine that runs a
caller's own circuit is `tomography-api-v2`: it takes OpenQASM 2, runs it on a provider (`aer`
= Moth's emulator; real-QPU access needs the account feature `run_quantum`; this engine has no `mode` switch, so the
hardware is chosen with `provider_name` / `backend_name`; see `from_env('qpu')`) and returns the
measurements in the X, Y and Z bases. The Z-basis counts of the whole register are exactly the
shots of our one-excitation walk, so that is what is used. `graph-v1` (QuantumGraph) was tried
as an alternative candidate selector and was worse: it designs a state from correlation
targets, saturates and does not honour edge signs on a mixed window (see docs/quantum-probes.md).

Contract used (all `Authorization: Bearer <MOTH_API_KEY>`):
    POST /api/v1/engines/tomography-api-v2/process   -> 202 {job_id, status}
    GET  /api/v1/jobs/{id}/status                    -> queued | processing | completed | failed
    GET  /api/v1/jobs/{id}/result                    -> {result: {measurements: {'ZZ..Z': {counts}}}}

Only circuit numbers travel; no concept names or graph text leave the machine. The key is read
from the environment and never stored, logged or put in an error message.
"""

from __future__ import annotations

import os
import time
from typing import Dict, Optional, Tuple

import requests

DEFAULT_URL = 'https://api.mothquantum.com'
ENGINE = 'tomography-api-v2'
_FAILED = {'failed', 'error', 'cancelled', 'canceled', 'terminated', 'timed_out'}


def _seconds(text, default: float) -> float:
    try:
        return max(5., float(text)) if text else default
    except ValueError:
        return default


class AtlasError(RuntimeError):
    """Any way the Atlas run can fail (network, HTTP error, failed or slow job, odd result)."""


class AtlasClient:
    def __init__(self, api_key: str, base_url: str = DEFAULT_URL, provider: str = 'aer',
                 poll_seconds: float = 1.0, timeout_seconds: float = 90.0,
                 backend_name: str = 'automatic'):
        self._key = api_key
        self.base_url = base_url.rstrip('/')
        self.provider = provider
        # Within provider `aer` this is the Aer simulation method: 'automatic', 'statevector',
        # 'density_matrix', 'stabilizer', 'matrix_product_state', ... ('matrix_product_state' scales
        # a one-excitation walk far past what a local statevector holds).
        self.backend_name = backend_name
        self.poll_seconds = poll_seconds
        self.timeout_seconds = timeout_seconds

    def __repr__(self) -> str:
        return f'AtlasClient({self.base_url!r}, provider={self.provider!r}, backend={self.backend_name!r})'

    @classmethod
    def from_env(cls, target: str = 'atlas') -> Optional['AtlasClient']:
        """A client from MOTH_API_KEY, or None. target 'atlas' is the Atlas emulator (MOTH_ATLAS_PROVIDER,
        default aer, MOTH_ATLAS_BACKEND, MOTH_ATLAS_TIMEOUT); target 'qpu' is the real device
        (MOTH_QPU_PROVIDER, which must be set, MOTH_QPU_BACKEND, MOTH_QPU_TIMEOUT)."""
        key = os.environ.get('MOTH_API_KEY')
        if not key:
            return None
        if target == 'qpu':
            # The real device: provider/backend names come from Moth for the account, so there is no
            # default; a QPU queue takes minutes, hence the long timeout.
            provider = os.environ.get('MOTH_QPU_PROVIDER')
            if not provider:
                return None
            timeout = _seconds(os.environ.get('MOTH_QPU_TIMEOUT'), 900.)
            return cls(key, base_url=os.environ.get('MOTH_API_URL') or DEFAULT_URL, provider=provider,
                       backend_name=os.environ.get('MOTH_QPU_BACKEND') or 'automatic',
                       timeout_seconds=timeout, poll_seconds=3.0)
        timeout = _seconds(os.environ.get('MOTH_ATLAS_TIMEOUT'), 90.)
        return cls(key, base_url=os.environ.get('MOTH_API_URL') or DEFAULT_URL,
                   provider=os.environ.get('MOTH_ATLAS_PROVIDER') or 'aer',
                   backend_name=os.environ.get('MOTH_ATLAS_BACKEND') or 'automatic',
                   timeout_seconds=timeout, poll_seconds=1.0 if timeout <= 120 else 3.0)

    def _request(self, method: str, path: str, **kwargs):
        try:
            response = requests.request(method, self.base_url + path, timeout=30,
                                        headers={'Authorization': 'Bearer ' + self._key}, **kwargs)
        except requests.RequestException as exc:
            raise AtlasError(f'{method} {path}: {type(exc).__name__}') from None
        if response.status_code >= 400:
            detail = ''
            try:
                body = response.json()
                detail = str(body.get('detail') or body.get('title') or '')[:200]
            except ValueError:
                pass
            raise AtlasError(f'{method} {path} -> HTTP {response.status_code} {detail}'.strip())
        try:
            return response.json()
        except ValueError:
            raise AtlasError(f'{method} {path}: response was not JSON') from None

    def z_counts(self, qasm2: str, n_qubits: int, shots: int,
                 backend_name: Optional[str] = None) -> Tuple[Dict[str, int], Dict[str, object]]:
        """Run an OpenQASM 2 circuit and return its Z-basis measurement counts (Qiskit bitstring
        order: qubit n-1 leftmost) plus {job_id, seconds, provider}. `backend_name` overrides the
        client's default for this one call (the caller may need a wider-circuit method than the
        one it was configured with)."""
        started = time.time()
        params = {'circuit_qasm': qasm2, 'provider_name': self.provider,
                  'backend_name': backend_name if backend_name is not None else self.backend_name,
                  'shots': shots,
                  'qubit_list': list(range(n_qubits)), 'qubit_pair_list': [],
                  'single_tomography': True, 'double_tomography': False,
                  'mutual_information': False, 'classical_mutual_information': False}
        job = self._request('POST', f'/api/v1/engines/{ENGINE}/process', json={'params': params}).get('job_id')
        if not job:
            raise AtlasError('submit returned no job_id')
        while True:
            status = self._request('GET', f'/api/v1/jobs/{job}/status')
            state = str(status.get('status') or '').lower()
            if state == 'completed':
                break
            if state in _FAILED:
                error = status.get('error')
                message = error.get('message') if isinstance(error, dict) else error
                raise AtlasError(f'job {job} {state}: {str(message or "")[:200]}'.strip())
            if time.time() - started > self.timeout_seconds:
                raise AtlasError(f'job {job} timed out after {self.timeout_seconds:g}s')
            time.sleep(self.poll_seconds)
        result = self._request('GET', f'/api/v1/jobs/{job}/result').get('result') or {}
        measurement = (result.get('measurements') or {}).get('Z' * n_qubits)
        if not measurement or not measurement.get('counts'):
            raise AtlasError(f'job {job} returned no Z-basis counts')
        counts = {str(k): int(v) for k, v in measurement['counts'].items()}
        return counts, {'job_id': job, 'seconds': round(time.time() - started, 2), 'provider': self.provider}
