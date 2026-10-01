"""Offline stand-ins for the embedder, the web and the LLM.

Used by the tests and by `python app.py --simulate` (a keyless UI demo). Everything they produce
is synthetic: the app shows a SIMULATED banner whenever they are active.
"""

from __future__ import annotations

import asyncio
import hashlib


class HashEmbedder:
    """Deterministic 16-d vectors from word hashes: texts sharing words have similar vectors."""
    fingerprint = 'fixture:hash16'
    backend = 'fixture'
    model_name = 'fixture'

    def encode_text(self, texts):
        out = []
        for text in texts:
            v = [0.] * 16
            for w in text.lower().split():
                h = int(hashlib.md5(w.encode()).hexdigest(), 16)
                v[h % 16] += 1.
                v[(h >> 8) % 16] += .5
            out.append(v if any(v) else [1.] + [0.] * 15)
        return out


class FakeWeb:
    """A `WebSource` lookalike returning one synthetic page per query."""

    def __init__(self, delay: float = 0.0, pages=None, mirror_every: int = 0):
        self.seen_urls, self.last_attempt, self.n = set(), {}, 0
        self.queries = []
        self.delay = delay
        self.pages = list(pages) if pages is not None else None      # scripted texts, served in order
        self.mirror_every = mirror_every        # every k-th search first returns a copy of the previous page
        self.calls, self.last_text = 0, None

    @staticmethod
    def available() -> bool:
        return True

    async def harvest(self, query, n=1, accept=None):
        from .web import Parcel
        if self.delay:
            await asyncio.sleep(self.delay)
        self.queries.append(query)
        self.calls += 1
        self.last_attempt = {'query': query, 'status': 'ok'}
        skipped = []
        mirror = bool(self.mirror_every and self.last_text and self.calls % self.mirror_every == 0)
        # a search returns several results: try them in order until one is accepted
        for _ in range(3):
            self.n += 1
            if mirror:
                text, mirror = self.last_text, False          # a mirror site serving the page just read
            elif self.pages is not None:
                if not self.pages:
                    break
                text = self.pages.pop(0)
            else:
                text = f'page {self.n} about topic{self.n} and topic{self.n + 1}: {query[:80]}'
            parcel = Parcel(text=text, origin=f'web:https://example.org/{self.n}', source='web')
            veto = accept(parcel) if accept is not None else None
            if veto:
                skipped.append({'source': parcel.origin, **veto})
                continue
            self.last_text = text
            if skipped:
                self.last_attempt['skipped'] = skipped
            return [parcel]
        if skipped:
            self.last_attempt.update(skipped=skipped, status='all-duplicate')
        else:
            self.last_attempt['status'] = 'no-results'
        return []


def chain_extractor():
    """Each page adds two concepts linked to the previous ones: a growing relation cluster."""
    state = {'i': 0}

    async def extract(text, known):
        i = state['i'] = state['i'] + 1
        names = [f'concept {i}', f'concept {i + 1}']
        ops = [{'op': 'node', 'kind': 'motif', 'name': n, 'gist': f'{n} gist'} for n in names]
        ops.append({'op': 'assert', 'subject': names[0], 'pred': 'causes', 'object': names[1], 'conf': .9})
        if i > 1:
            ops.append({'op': 'assert', 'subject': f'concept {i - 1}', 'pred': 'enables',
                        'object': names[0], 'conf': .8})
        return {'ops': ops}
    return extract


def ring_extractor(delay: float = 0.0):
    """Richer synthetic graph: triples closed into loops, one hostile edge each, plus echoes."""
    state = {'i': 0}

    async def extract(text, known):
        if delay:
            await asyncio.sleep(delay)
        i = state['i'] = state['i'] + 1
        a, b, c = f'idea {i}', f'idea {i + 1}', f'idea {i + 2}'
        ops = [{'op': 'node', 'kind': 'motif', 'name': n, 'gist': f'{n}: {text[:40]}'} for n in (a, b, c)]
        ops += [{'op': 'assert', 'subject': a, 'pred': 'causes', 'object': b, 'conf': .9},
                {'op': 'assert', 'subject': b, 'pred': 'enables', 'object': c, 'conf': .8},
                {'op': 'assert', 'subject': c, 'pred': 'opposes', 'object': a, 'conf': .7, 'valence': -.6}]
        if i > 1:
            ops.append({'op': 'assert', 'subject': f'idea {i - 1}', 'pred': 'echoes', 'object': b, 'conf': .8})
        return {'ops': ops}
    return extract


QUESTIONS = [
    ('Which timing anomaly accounts for {a}?', 'The anomaly behind {a} has no accepted explanation yet.'),
    ('Does {a} survive without {c}?', 'Whether {a} depends on {c} is untested.'),
    ('What limits how far {a} can propagate?', 'The ceiling on {a} is unknown.'),
    ('Who first recorded {a}, and when?', 'The earliest record of {a} is disputed.'),
    ('Is {c} a cause or a symptom of {a}?', 'The direction of influence between {c} and {a} is unclear.'),
    ('How reliable is the evidence for {a}?', 'The sources for {a} have not been cross-checked.'),
]


def question_extractor(delay: float = 0.0):
    """`ring_extractor` plus open questions: every page but the first raises one, with its own
    wording (so the keeper does not fold them together), grounded on two of the new concepts."""
    inner = ring_extractor(delay)
    state = {'i': 0}

    async def extract(text, known):
        out = await inner(text, known)
        i = state['i'] = state['i'] + 1
        title, gist = QUESTIONS[(i - 1) % len(QUESTIONS)]
        a, c = f'idea {i}', f'idea {i + 2}'
        out['ops'].append({'op': 'thread', 'name': title.format(a=a, c=c), 'state': 'open', 'gist': gist.format(a=a, c=c),
                           'pressure': 0.7, 'involves': [a, c]})
        return out
    return extract


def fake_thread_extractor(delay: float = 0.0):
    """Moves the first candidate forward when it is offered, and settles it on the third offer."""
    seen = {}

    async def extract(text, candidates):
        if delay:
            await asyncio.sleep(delay)
        if not candidates:
            return {'ops': []}
        cand = candidates[0]
        n = seen[cand['name']] = seen.get(cand['name'], 0) + 1
        if n == 1 or n == 2:
            return {'ops': [{'op': 'thread', 'name': cand['name'], 'state': 'complicated',
                             'gist': f"Partly explained (evidence {n}): {text[:60]}"}]}
        return {'ops': [{'op': 'thread', 'name': cand['name'], 'state': 'resolved',
                         'gist': f"Answered by: {text[:60]}"}]}
    return extract


def sloppy(extractor, every: int = 3):
    """Wrap an extractor so that every k-th page it also proposes what the node gate exists to stop:
    a concept without a description, and a claim about a figure that was never established."""
    state = {'i': 0}

    async def extract(text, known):
        out = await extractor(text, known)
        state['i'] += 1
        if state['i'] % every == 0:
            i = state['i']
            out['ops'] += [{'op': 'node', 'kind': 'motif', 'name': f'figure {i}'},
                           {'op': 'assert', 'subject': f'idea {i}', 'pred': 'precedes', 'object': f'the year {1900 + i}', 'conf': .6}]
        return out
    return extract
