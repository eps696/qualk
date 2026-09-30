"""Measure an embedder's cosine scale so the walk's coupling is calibrated to it.

    python -m qualk.calibrate [--model BAAI/bge-small-en-v1.5]

The walk rescales an edge's affinity from [AFFINITY_FLOOR, 1] to [0, 1]. The floor should sit
where *unrelated* concepts land, so that only real semantic closeness raises a coupling. This
embeds `name: gist` texts of built-in related and unrelated concept pairs (or the nodes and
edges of a finished run with `--run`) and reports the distributions.
"""

import argparse
import itertools
import json
import os
import statistics

CONCEPTS = {
    'astronomy': ['radio telescope: an antenna array that collects radio waves from space',
                  'pulsar: a rapidly rotating neutron star emitting beams of radiation',
                  'hydrogen line: the 21 cm emission of neutral hydrogen used to map galaxies',
                  'signal search: scanning the sky for narrowband artificial transmissions'],
    'medieval manuscripts': ['illuminated codex: a handwritten book decorated with gold and colour',
                             'scribe: a monk who copied texts in a scriptorium',
                             'vellum: prepared calfskin used as a writing surface',
                             'cipher script: an unreadable writing system in an old manuscript'],
    'cooking': ['sourdough starter: a fermented culture of flour and water used to leaven bread',
                'oven spring: the final rise of dough in the first minutes of baking',
                'maillard reaction: browning that gives crust its flavour',
                'proofing basket: a cloth-lined bowl that shapes dough while it rises'],
    'finance': ['central bank: the institution that sets interest rates and issues currency',
                'yield curve: interest rates plotted against bond maturity',
                'inflation expectations: what households predict about future price rises',
                'liquidity crisis: banks unable to meet short-term obligations'],
}


def scan(texts_by_topic, encode):
    topics = list(texts_by_topic)
    vectors = {t: encode(texts_by_topic[t]) for t in topics}
    related, unrelated = [], []
    for t in topics:
        for a, b in itertools.combinations(vectors[t], 2):
            related.append(sum(x * y for x, y in zip(a, b)))
    for t1, t2 in itertools.combinations(topics, 2):
        for a in vectors[t1]:
            for b in vectors[t2]:
                unrelated.append(sum(x * y for x, y in zip(a, b)))
    return related, unrelated


def summary(values):
    values = sorted(values)
    return {'mean': round(statistics.mean(values), 3), 'p10': round(values[len(values) // 10], 3),
            'p50': round(values[len(values) // 2], 3), 'p90': round(values[(len(values) * 9) // 10], 3),
            'max': round(values[-1], 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=None)
    a = ap.parse_args()
    from .embed import STEmbedder
    emb = STEmbedder(a.model)
    encode = lambda texts: emb.encode_text(texts)
    related, unrelated = scan(CONCEPTS, encode)
    print('model:', emb.model_name)
    print('related pairs (same topic):   ', summary(related))
    print('unrelated pairs (other topic):', summary(unrelated))
    floor = round(statistics.mean(unrelated), 2)
    dup = 0.05
    print(f'suggested AFFINITY_FLOOR = {floor}   (mean of unrelated pairs, as CLIP's 0.62 average -> 0.6 in assembly)')
    print(f'suggested DUPLICATE_DISTANCE = {dup}')
    print(json.dumps({emb.model_name: [floor, dup]}))


if __name__ == '__main__':
    main()
