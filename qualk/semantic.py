"""Small persistent cosine index for dog-mode graph and probe embeddings.

The graph remains authoritative. This database is a rebuildable derived cache.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
from typing import Any, Dict, List, Optional, Sequence


def _vector(values: Any) -> List[float]:
    if hasattr(values, 'detach'):
        values = values.detach().float().cpu().tolist()
    result = [float(value) for value in values]
    if not result or not all(math.isfinite(value) for value in result):
        raise ValueError('semantic vector must be nonempty and finite')
    length = math.sqrt(sum(value * value for value in result))
    if length == 0:
        raise ValueError('semantic vector must have nonzero length')
    return [value / length for value in result]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine of two finite vectors, clamped for floating-point roundoff."""
    if len(a) != len(b):
        raise ValueError('semantic vector dimensions differ')
    va, vb = _vector(a), _vector(b)
    return max(-1.0, min(1.0, sum(x * y for x, y in zip(va, vb))))


class SemanticIndex:
    """Disk-backed exact vector search with content and model invalidation."""

    def __init__(self, path: str, embedder: Any):
        self.path = os.fspath(path)
        self.embedder = embedder
        fingerprint = embedder.fingerprint
        if not fingerprint:
            raise ValueError('embedder must provide a nonempty model fingerprint')
        parent = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(parent, exist_ok=True)
        self._db = sqlite3.connect(self.path)
        self._db.execute('CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
        self._db.execute('''CREATE TABLE IF NOT EXISTS vectors (
            key TEXT PRIMARY KEY, content_hash TEXT NOT NULL,
            vector TEXT NOT NULL, metadata TEXT NOT NULL)''')
        previous = self._db.execute("SELECT value FROM meta WHERE key='fingerprint'").fetchone()
        with self._db:
            if previous is None or previous[0] != fingerprint:
                self._db.execute('DELETE FROM vectors')
            self._db.execute("INSERT OR REPLACE INTO meta(key, value) VALUES('fingerprint', ?)",
                             (fingerprint,))
        row = self._db.execute('SELECT vector FROM vectors LIMIT 1').fetchone()
        self._dimension = len(json.loads(row[0])) if row else None

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> 'SemanticIndex':
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    def _validate_dimension(self, vector: Sequence[float]) -> None:
        if self._dimension is not None and len(vector) != self._dimension:
            raise ValueError(f'semantic vector dimension {len(vector)} != {self._dimension}')

    def encode(self, text: str) -> List[float]:
        if not isinstance(text, str):
            raise TypeError('semantic text must be a string')
        vector = _vector(self.embedder.encode_text([text])[0])
        self._validate_dimension(vector)
        return vector

    def put(self, key: str, text: str = '', vector: Any = None,
            metadata: Optional[Dict[str, Any]] = None) -> List[float]:
        return self._put(key, text, vector, metadata)[0]

    def _put(self, key: str, text: str = '', vector: Any = None,
             metadata: Optional[Dict[str, Any]] = None):
        """`put`, also reporting whether the stored vector is new or was replaced."""
        if not key:
            raise ValueError('semantic key must be nonempty')
        if vector is None and not text:
            raise ValueError('put requires text or vector')
        raw = text if vector is None else json.dumps(_vector(vector), separators=(',', ':'))
        digest = hashlib.sha256(raw.encode('utf-8')).hexdigest()
        row = self._db.execute('SELECT content_hash, vector, metadata FROM vectors WHERE key=?',
                               (key,)).fetchone()
        if row and row[0] == digest:
            result = json.loads(row[1])
            if metadata is not None and metadata != json.loads(row[2]):
                with self._db:
                    self._db.execute('UPDATE vectors SET metadata=? WHERE key=?',
                                     (json.dumps(metadata, ensure_ascii=False), key))
            return result, False
        result = self.encode(text) if vector is None else _vector(vector)
        self._validate_dimension(result)
        with self._db:
            self._db.execute('INSERT OR REPLACE INTO vectors VALUES (?, ?, ?, ?)',
                             (key, digest, json.dumps(result),
                              json.dumps(metadata or {}, ensure_ascii=False)))
        self._dimension = len(result)
        return result, True

    def get(self, key: str) -> Optional[List[float]]:
        row = self._db.execute('SELECT vector FROM vectors WHERE key=?', (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def search(self, vector: Sequence[float], prefix: str = 'node:',
               k: int = 6, recent: Optional[int] = None) -> List[Dict[str, Any]]:
        query = _vector(vector)
        self._validate_dimension(query)
        if k <= 0:
            return []
        if recent is not None:
            if recent <= 0:
                return []
            # Limit by insertion order before ranking by similarity; otherwise
            # the nearest of every observation ever seen drives novelty toward
            # zero on a long run, even when the current neighbourhood changed.
            rows = self._db.execute(
                'SELECT key, vector, metadata FROM vectors WHERE key LIKE ? '
                'ORDER BY rowid DESC LIMIT ?', (prefix + '%', recent)).fetchall()
        else:
            rows = self._db.execute('SELECT key, vector, metadata FROM vectors').fetchall()
        hits = [{'key': key, 'similarity': cosine(query, json.loads(encoded)),
                 'metadata': json.loads(metadata)}
                for key, encoded, metadata in rows if key.startswith(prefix)]
        hits.sort(key=lambda item: (-item['similarity'], item['key']))
        return hits[:k]

    def sync_graph(self, graph: Any) -> int:
        """Embed new or edited nodes, drop vanished ones, and give every open edge its
        `affinity` (cosine of the endpoint vectors). Nothing is recomputed for an edge
        whose endpoints did not change. Returns how many affinities were (re)computed."""
        from .world.ops import LITERAL_PREDS
        keys = set()
        changed = set()
        for node in graph.nodes.values():
            key = 'node:' + node.id
            text = ' '.join(part for part in (node.name, node.gist) if part)
            if text:
                keys.add(key)
                if self._put(key, text, metadata={'node_id': node.id})[1]:
                    changed.add(node.id)
        stale = [(key,) for (key,) in self._db.execute('SELECT key FROM vectors')
                 if key.startswith('node:') and key not in keys]
        if stale:
            with self._db:
                self._db.executemany('DELETE FROM vectors WHERE key=?', stale)
        updated = 0
        for a in getattr(graph, 'assertions', {}).values():
            if not a.open or a.pred in LITERAL_PREDS:
                continue
            if a.affinity is not None and a.subject not in changed and a.object not in changed:
                continue
            left, right = self.get('node:' + a.subject), self.get('node:' + a.object)
            if left is None or right is None:
                continue
            a.affinity = round(cosine(left, right), 4)
            updated += 1
        return updated

    def retrieve_nodes(self, vector, graph, k=6):
        """Text and observed-image representations vote for the same graph identity.

        Every `asset:` row an earlier visual stimulus wrote shares *that one
        stimulus's own image vector* across all its affected nodes (see
        `kernel/dogloop.py::DogEngine.step`), and image-to-image similarity runs far
        hotter than image-to-text under CLIP/SigLIP2 — so a plain merge-and-sort let
        a single prior image's entire affected-node set fill every slot, all tied at
        one identical score: never actually "the graph's nearest neighbours," just
        "whichever one earlier picture looks similar." At most one hit per source
        `stim_id` is kept (text `node:` rows carry no `stim_id` and are unaffected),
        so a strong single image match still wins its own slot honestly, but can no
        longer flood the rest of the list — the next slots go to the next-best
        *distinct* source, text or another image."""
        hits = (self.search(vector, prefix='node:', k=k) +
               self.search(vector, prefix='asset:', k=max(k, len(graph.nodes))))
        hits.sort(key=lambda h: (-h['similarity'], h['key']))
        selected, seen_nodes, seen_stims = [], set(), set()
        for hit in hits:
            nid = hit['metadata'].get('node_id')
            if nid not in graph.nodes or nid in seen_nodes:
                continue
            stim_id = hit['metadata'].get('stim_id')   # only asset: rows carry this
            if stim_id is not None and stim_id in seen_stims:
                continue
            selected.append(hit)
            seen_nodes.add(nid)
            if stim_id is not None:
                seen_stims.add(stim_id)
            if len(selected) >= k:
                break
        return selected
