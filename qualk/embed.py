"""Text embedder for edge affinity and observation novelty (sentence-transformers, CPU is fine).

The walk couples two concepts by `conf x (0.4 + 0.6 x a)`, where `a` is the cosine of their
embeddings rescaled from [AFFINITY_FLOOR, 1] to [0, 1]; the floor is the cosine that unrelated
concepts reach, so it belongs to the embedder. `calibrate.py` measures it; `configure()` applies
the measured values to the walk (`AFFINITY_FLOOR`) and to observation novelty
(`DUPLICATE_DISTANCE`).
"""

from __future__ import annotations

import hashlib
import os

DEFAULT_MODEL = 'BAAI/bge-small-en-v1.5'
# model -> (affinity floor, duplicate distance); measured by `python -m qualk.calibrate`
CALIBRATION = {
    # measured with the built-in concept set: unrelated pairs mean 0.553, related 0.722
    'BAAI/bge-small-en-v1.5': (0.55, 0.05),
}


class STEmbedder:
    backend = 'sentence-transformers'

    def __init__(self, model=None):
        from sentence_transformers import SentenceTransformer
        self.model_name = model or os.environ.get('QUALK_EMBED_MODEL') or DEFAULT_MODEL
        self._model = SentenceTransformer(self.model_name)
        self.fingerprint = hashlib.sha256(f'st:{self.model_name}'.encode()).hexdigest()[:16]

    def encode_text(self, texts):
        return self._model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False).tolist()


def configure(embedder):
    """Apply the embedder's calibrated floor/duplicate distance to the walk and novelty scale."""
    from . import exploration, quantum_walk
    floor, duplicate = CALIBRATION.get(getattr(embedder, 'model_name', ''), (None, None))
    if floor is not None:
        quantum_walk.AFFINITY_FLOOR = floor
        exploration.DUPLICATE_DISTANCE = duplicate
    return quantum_walk.AFFINITY_FLOOR, exploration.DUPLICATE_DISTANCE
