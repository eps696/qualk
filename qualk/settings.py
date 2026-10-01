"""Provider settings shared by the CLI and the web app: environment variables backed by a `.env`.

Secret values are write-only: the app can say a key *is set*, never what it is (not even a suffix).
Every message that leaves this module is scrubbed of the secrets currently in the environment.
"""

from __future__ import annotations

import os
import tempfile
import threading
from typing import Any, Dict, List

SECRET_KEYS = ('TAVILY_API_KEY', 'SERPER_API_KEY', 'BRAVE_API_KEY', 'QUALK_LLM_KEY', 'MOTH_API_KEY')
PLAIN_KEYS = ('SEARCH_PROVIDER_ORDER', 'QUALK_LLM_URL', 'QUALK_LLM_MODEL', 'QUALK_EMBED_MODEL',
              'MOTH_ATLAS_PROVIDER', 'MOTH_ATLAS_BACKEND', 'MOTH_ATLAS_TIMEOUT', 'MOTH_ATLAS_STRICT',
              'MOTH_QPU_PROVIDER', 'MOTH_QPU_BACKEND', 'MOTH_QPU_TIMEOUT')
KEYS = SECRET_KEYS + PLAIN_KEYS
LABELS = {
    'TAVILY_API_KEY': 'Tavily search key', 'SERPER_API_KEY': 'Serper search key',
    'BRAVE_API_KEY': 'Brave search key', 'SEARCH_PROVIDER_ORDER': 'Search provider order',
    'QUALK_LLM_URL': 'LLM base URL', 'QUALK_LLM_KEY': 'LLM API key', 'QUALK_LLM_MODEL': 'LLM model',
    'QUALK_EMBED_MODEL': 'Embedding model', 'MOTH_API_KEY': 'Moth Atlas key',
    'MOTH_ATLAS_PROVIDER': 'Atlas provider', 'MOTH_ATLAS_BACKEND': 'Atlas backend',
    'MOTH_ATLAS_TIMEOUT': 'Atlas timeout (s)', 'MOTH_ATLAS_STRICT': 'Atlas strict (no fallback)',
    'MOTH_QPU_PROVIDER': 'QPU provider', 'MOTH_QPU_BACKEND': 'QPU backend', 'MOTH_QPU_TIMEOUT': 'QPU timeout (s)',
}
PLACEHOLDERS = {'SEARCH_PROVIDER_ORDER': 'tavily,serper,brave', 'QUALK_LLM_URL': 'http://localhost:1234/v1',
                'QUALK_LLM_MODEL': 'gpt-oss-20b', 'QUALK_EMBED_MODEL': 'BAAI/bge-small-en-v1.5',
                'MOTH_ATLAS_PROVIDER': 'aer (emulator)', 'MOTH_ATLAS_BACKEND': 'automatic',
                'MOTH_ATLAS_TIMEOUT': '90 (QPU: 900+)', 'MOTH_ATLAS_STRICT': '1 = fail instead of using local Qiskit',
                'MOTH_QPU_PROVIDER': 'as given by Moth', 'MOTH_QPU_BACKEND': 'automatic', 'MOTH_QPU_TIMEOUT': '900'}
MAX_VALUE = 512


def scrub(text: Any) -> str:
    """`text` with every secret value currently in the environment replaced by ***."""
    out = str(text)
    for key in SECRET_KEYS:
        value = os.environ.get(key)
        if value and len(value) >= 4:
            out = out.replace(value, '***')
    return out


def _read_lines(path: str) -> List[str]:
    if not os.path.isfile(path):
        return []
    with open(path, encoding='utf-8') as f:
        return f.read().splitlines()


def load_env(path: str = '.env') -> None:
    """Load `.env` into the environment without overriding variables that are already set."""
    for line in _read_lines(path):
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            key, value = line.split('=', 1)
            value = value.strip().strip('"\'')
            if value:
                os.environ.setdefault(key.strip(), value)


class SettingsStore:
    def __init__(self, env_path: str = '.env'):
        self.env_path = env_path
        self._lock = threading.Lock()
        load_env(env_path)

    def view(self) -> Dict[str, Dict[str, Any]]:
        """Per key: label, whether it is secret, whether it is set, and (plain keys only) its value."""
        out = {}
        for key in KEYS:
            value = os.environ.get(key, '')
            item = {'label': LABELS[key], 'secret': key in SECRET_KEYS, 'set': bool(value),
                    'placeholder': PLACEHOLDERS.get(key, '')}
            if key not in SECRET_KEYS:
                item['value'] = value
            out[key] = item
        return out

    def providers(self) -> Dict[str, bool]:
        """Which capabilities are configured (the status pills)."""
        env = os.environ.get
        return {'search': any(env(k) for k in ('TAVILY_API_KEY', 'SERPER_API_KEY', 'BRAVE_API_KEY')),
                'llm': bool(env('QUALK_LLM_URL') or env('QUALK_LLM_KEY') or env('QUALK_LLM_MODEL')),
                'atlas': bool(env('MOTH_API_KEY')),
                'qpu': bool(env('MOTH_API_KEY') and env('MOTH_QPU_PROVIDER'))}

    def update(self, changes: Dict[str, Any], clear=()) -> List[str]:
        """Apply `changes` (an empty string for a secret means "keep it") and unset the keys in
        `clear`. Returns the keys that changed. Writes `.env` atomically."""
        applied = {}
        for key, value in (changes or {}).items():
            if key not in KEYS:
                raise ValueError(f'unknown setting {key!r}')
            if not isinstance(value, str):
                raise ValueError(f'{key} must be a string')
            value = value.strip()
            if any(c in value for c in '\r\n"\0') or len(value) > MAX_VALUE:
                raise ValueError(f'{key}: invalid value')
            if value == '' or value == os.environ.get(key):
                continue          # blank keeps the current value; clear is explicit
            applied[key] = value
        for key in clear or ():
            if key not in KEYS:
                raise ValueError(f'unknown setting {key!r}')
            if os.environ.get(key):
                applied[key] = ''
        with self._lock:
            lines = _read_lines(self.env_path)
            for key, value in applied.items():
                new = f'{key}={value}' if value else None
                idx = next((i for i, l in enumerate(lines) if l.split('=', 1)[0].strip() == key), None)
                if idx is None:
                    if new:
                        lines.append(new)
                elif new:
                    lines[idx] = new
                else:
                    del lines[idx]
                if value:
                    os.environ[key] = value
                else:
                    os.environ.pop(key, None)
            if applied:
                directory = os.path.dirname(os.path.abspath(self.env_path))
                fd, tmp = tempfile.mkstemp(dir=directory, prefix='.env.', suffix='.tmp')
                with os.fdopen(fd, 'w', encoding='utf-8', newline='\n') as f:
                    f.write('\n'.join(lines) + '\n')
                try:
                    os.chmod(tmp, 0o600)
                except OSError:
                    pass
                os.replace(tmp, self.env_path)
        if 'SEARCH_PROVIDER_ORDER' in applied:
            from . import web
            web.set_provider_order(applied['SEARCH_PROVIDER_ORDER'])
        return sorted(applied)


# --- connection tests -------------------------------------------------------------------------

async def check_search() -> Dict[str, Any]:
    from . import web
    if not web.search_available():
        return {'ok': False, 'detail': 'no search key set'}
    try:
        results = await web.search('quantum walk', count=1)
    except Exception as e:
        return {'ok': False, 'detail': scrub(e)[:300]}
    return {'ok': bool(results), 'detail': f'{len(results)} result(s)'}


async def check_llm() -> Dict[str, Any]:
    from openai import AsyncOpenAI
    client = AsyncOpenAI(base_url=os.environ.get('QUALK_LLM_URL') or 'http://localhost:1234/v1',
                         api_key=os.environ.get('QUALK_LLM_KEY') or 'lm-studio', timeout=30.0)
    model = os.environ.get('QUALK_LLM_MODEL') or 'gpt-oss-20b'
    try:
        reply = await client.chat.completions.create(
            model=model, max_tokens=8, messages=[{'role': 'user', 'content': 'Reply with the word ok.'}])
    except Exception as e:
        return {'ok': False, 'detail': f'{type(e).__name__}: {scrub(e)}'[:300]}
    return {'ok': True, 'detail': f'{model} answered ({len(reply.choices)} choice)'}


def check_atlas(target: str = 'atlas') -> Dict[str, Any]:
    """Validates the key against /api/v1/me and reports the account's features (QPU needs `run_quantum`)
    and the provider/backend that runs would use. No job is submitted, so nothing is billed."""
    from .atlas import AtlasClient
    client = AtlasClient.from_env(target)
    if client is None:
        return {'ok': False, 'detail': 'MOTH_API_KEY is not set' if not os.environ.get('MOTH_API_KEY')
                else 'MOTH_QPU_PROVIDER is not set'}
    import requests
    try:
        response = requests.get(client.base_url + '/api/v1/me', timeout=10,
                                headers={'Authorization': 'Bearer ' + os.environ['MOTH_API_KEY']})
    except requests.RequestException as e:
        return {'ok': False, 'detail': type(e).__name__}
    if response.status_code in (401, 403):
        return {'ok': False, 'detail': f'the key was rejected (HTTP {response.status_code})'}
    if response.status_code >= 400:
        return {'ok': False, 'detail': f'{client.base_url} answered HTTP {response.status_code}'}
    try:
        features = [str(f) for f in (response.json().get('features') or [])]
    except ValueError:
        features = []
    qpu = 'run_quantum' in features
    return {'ok': True,
            'detail': f'key accepted; provider={client.provider}, backend={client.backend_name}, '
                      f'timeout={client.timeout_seconds:g}s; account features: {", ".join(features) or "none"}; '
                      f'QPU feature (run_quantum): {"yes" if qpu else "NOT granted to this key"}'}



def check_embed(embedder_factory) -> Dict[str, Any]:
    try:
        embedder = embedder_factory()
        vector = embedder.encode_text(['hello world'])[0]
    except Exception as e:
        return {'ok': False, 'detail': f'{type(e).__name__}: {scrub(e)}'[:300]}
    return {'ok': True, 'detail': f'{getattr(embedder, "model_name", "embedder")}, {len(vector)} dimensions'}
