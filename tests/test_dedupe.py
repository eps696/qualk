"""Near-duplicate pages are skipped before they cost an LLM call."""
import asyncio
import json
import os
import tempfile
import unittest
from unittest.mock import patch

os.environ.pop("MOTH_API_KEY", None)

from qualk.fakes import FakeWeb, HashEmbedder, chain_extractor
from qualk.web import Parcel, WebSource

A = ' '.join(f'alpha{i} common{i % 3} river stone' for i in range(30))
A_COPY = A + ' footer'                                   # the same page with a little boilerplate
B = ' '.join(f'zulu{i} quartz{i % 5} desert lantern' for i in range(30))


def make(pages, dedupe=0.05, seed_text='seed text'):
    from qualk.engine import Engine
    from qualk.semantic import SemanticIndex
    root = tempfile.mkdtemp(prefix='qualk_dedupe_')
    index = SemanticIndex(os.path.join(root, 'semantic.sqlite'), HashEmbedder())
    web = FakeWeb(pages=pages)
    engine = Engine(root, chain_extractor(), index, web, walk=None, explore=1.0, seed=3, dedupe=dedupe)
    asyncio.run(engine.seed([Parcel(seed_text, 'seed:test', 'seed')]))
    return engine, index, web


class EngineDedupeTests(unittest.TestCase):
    def test_a_duplicate_falls_through_to_the_next_result(self):
        engine, index, web = make([A, A_COPY, B])
        first = asyncio.run(engine.step())
        second = asyncio.run(engine.step())
        index.close()
        self.assertTrue(first['source'].endswith('/1'))
        self.assertEqual(second['harvest']['status'], 'ok')
        skipped = second['harvest']['skipped']
        self.assertEqual(len(skipped), 1)
        self.assertEqual(skipped[0]['reason'], 'near-duplicate')
        self.assertLess(skipped[0]['distance'], 0.05)
        self.assertNotEqual(second['source'], skipped[0]['source'])
        self.assertTrue(second['kept'], 'the distinct page was digested')

    def test_only_duplicates_means_nothing_is_digested_and_scores_zero(self):
        engine, index, web = make([A, A_COPY, A, A_COPY])
        asyncio.run(engine.step())
        nodes = len(engine.graph.nodes)
        record = asyncio.run(engine.step())
        index.close()
        self.assertEqual(record['harvest']['status'], 'all-duplicate')
        self.assertEqual(record['source'], '')
        self.assertFalse(record['kept'])
        self.assertEqual(len(engine.graph.nodes), nodes, 'no LLM extraction, no graph change')
        self.assertEqual(engine.archive.probes[-1].outcomes.get('novelty'), 0.0)

    def test_dedupe_zero_disables_the_filter(self):
        engine, index, web = make([A, A_COPY], dedupe=0.0)
        asyncio.run(engine.step())
        second = asyncio.run(engine.step())
        index.close()
        self.assertNotIn('skipped', second['harvest'])
        self.assertTrue(second['source'])

    def test_a_page_equal_to_the_seed_is_a_duplicate(self):
        engine, index, web = make([A, B], seed_text=A)
        record = asyncio.run(engine.step())
        index.close()
        self.assertEqual(len(record['harvest']['skipped']), 1)
        self.assertEqual(record['harvest']['skipped'][0]['of'], 'seed:test')

    def test_injected_text_is_never_filtered(self):
        engine, index, web = make([A])
        asyncio.run(engine.step())
        record = asyncio.run(engine.inject(Parcel(A, 'inject:text', 'inject')))
        index.close()
        self.assertEqual(record['kind'], 'inject')

    def test_the_threshold_can_change_between_rounds(self):
        engine, index, web = make([A, A_COPY, B], dedupe=0.05)
        asyncio.run(engine.step())
        engine.set_dedupe(0.0)
        second = asyncio.run(engine.step())
        index.close()
        self.assertNotIn('skipped', second['harvest'])


class WebSourceAcceptTests(unittest.TestCase):
    def run_harvest(self, texts, accept):
        results = [{'title': f't{i}', 'url': f'https://example.org/{i}', 'snippet': 's'} for i in range(len(texts))]

        async def fake_search(query, count=5):
            return results

        async def fake_fetch(url, max_length=4000):
            return texts[int(url.rsplit('/', 1)[1])]

        source = WebSource()
        with patch('qualk.web.search', fake_search), patch('qualk.web.fetch_text', fake_fetch), \
                patch('qualk.web.search_available', lambda: True):
            parcels = asyncio.run(source.harvest('query', accept=accept))
        return source, parcels

    def test_vetoed_pages_are_skipped_remembered_and_reported(self):
        source, parcels = self.run_harvest(['dup one', 'dup two', 'fresh'],
                                           lambda p: {'reason': 'near-duplicate', 'distance': 0.01, 'of': 'x'} if 'dup' in p.text else None)
        self.assertEqual([p.text for p in parcels], ['fresh'])
        self.assertEqual(source.last_attempt['status'], 'ok')
        self.assertEqual(len(source.last_attempt['skipped']), 2)
        self.assertEqual(source.seen_urls, {'https://example.org/0', 'https://example.org/1', 'https://example.org/2'})

    def test_everything_vetoed_is_all_duplicate_not_a_snippet_fallback(self):
        source, parcels = self.run_harvest(['a', 'b'], lambda p: {'reason': 'near-duplicate', 'distance': 0.0, 'of': 'x'})
        self.assertEqual(parcels, [])
        self.assertEqual(source.last_attempt['status'], 'all-duplicate')

    def test_without_accept_behaviour_is_unchanged(self):
        source, parcels = self.run_harvest(['first', 'second'], None)
        self.assertEqual([p.text for p in parcels], ['first'])
        self.assertNotIn('skipped', source.last_attempt)


class ConfigTests(unittest.TestCase):
    def test_dedupe_is_validated_and_live(self):
        from qualk.runner import RunConfig
        self.assertEqual(RunConfig().dedupe, 0.10)
        for bad in (-0.1, 0.6):
            with self.assertRaises(ValueError):
                RunConfig.from_dict({'dedupe': bad})
        self.assertEqual(RunConfig().merged({'dedupe': 0.2}).dedupe, 0.2)


if __name__ == '__main__':
    unittest.main()
