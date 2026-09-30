"""Web probes: a probe's concepts become a search query, the top fresh page becomes a Parcel.

Trimmed from assembly's `atools/web.py` + `stim/intake.py`. Search goes through Brave, Tavily or
Serper (first one with a key that answers wins; `SEARCH_PROVIDER_ORDER` or `set_provider_order`
picks the order); pages are fetched with aiohttp and reduced to markdown by trafilatura.
"""

from __future__ import annotations

import html
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import aiohttp

USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36")

# --- search providers -------------------------------------------------------------------------

_PROVIDERS = {
    'brave': {
        'env_key': 'BRAVE_API_KEY', 'url': 'https://api.search.brave.com/res/v1/web/search', 'method': 'get',
        'build_request': lambda q, n, key: dict(params={'q': q, 'count': n},
                                                headers={'Accept': 'application/json', 'X-Subscription-Token': key}),
        'parse': lambda data: data.get('web', {}).get('results', []),
        'fields': ('title', 'url', 'description'),
    },
    'tavily': {
        'env_key': 'TAVILY_API_KEY', 'url': 'https://api.tavily.com/search', 'method': 'post',
        'build_request': lambda q, n, key: dict(json={'api_key': key, 'query': q, 'max_results': n}),
        'parse': lambda data: data.get('results', []),
        'fields': ('title', 'url', 'content'),
    },
    'serper': {
        'env_key': 'SERPER_API_KEY', 'url': 'https://google.serper.dev/search', 'method': 'post',
        'build_request': lambda q, n, key: dict(headers={'X-API-KEY': key, 'Content-Type': 'application/json'},
                                                json={'q': q, 'num': n}),
        'parse': lambda data: data.get('organic', []),
        'fields': ('title', 'link', 'snippet'),
    },
}
_ORDER = ['tavily', 'serper', 'brave']       # Brave last: its 400-char / 50-word query cap is the tightest
_MAX_QUERY = {'brave': (400, 50), 'tavily': (400, None)}


def set_provider_order(order) -> None:
    """`order`: comma-separated string or list of provider names; unknown names are ignored."""
    global _ORDER
    names = [n.strip().lower() for n in (order.split(',') if isinstance(order, str) else (order or []))]
    names = [n for n in dict.fromkeys(names) if n in _PROVIDERS]
    _ORDER = names + [n for n in ['tavily', 'serper', 'brave'] if n not in names]


if os.environ.get('SEARCH_PROVIDER_ORDER'):
    set_provider_order(os.environ['SEARCH_PROVIDER_ORDER'])


def _fit_query(query: str, provider: str) -> str:
    chars, words = _MAX_QUERY.get(provider, (None, None))
    if words:
        query = ' '.join(query.split()[:words])
    if chars and len(query) > chars:
        query = query[:chars].rsplit(' ', 1)[0]
    return query


def search_available() -> bool:
    return any(os.environ.get(cfg['env_key']) for cfg in _PROVIDERS.values())


async def search(query: str, count: int = 5) -> List[Dict[str, str]]:
    """[{title, url, snippet}] from the first configured provider that answers."""
    n = min(max(count, 1), 10)
    errors, attempted = [], 0
    for name in _ORDER:
        cfg = _PROVIDERS[name]
        key = os.environ.get(cfg['env_key'])
        if not key:
            continue
        attempted += 1
        try:
            kwargs = cfg['build_request'](_fit_query(query, name), n, key)
            async with aiohttp.ClientSession() as session:
                async with getattr(session, cfg['method'])(
                        cfg['url'], **kwargs, timeout=aiohttp.ClientTimeout(total=20.0)) as response:
                    if response.status >= 400:
                        raise RuntimeError(f'{name} API error: {response.status} {response.reason}')
                    data = await response.json()
            title_f, url_f, desc_f = cfg['fields']
            return [{'title': item.get(title_f, '') or '', 'url': item.get(url_f, '') or '',
                     'snippet': item.get(desc_f, '') or ''}
                    for item in cfg['parse'](data)[:n] if item.get(url_f)]
        except Exception as e:
            errors.append(f'{name}: {type(e).__name__} {str(e)[:120]}'.strip())
    if not attempted:
        raise RuntimeError('No search API key configured (set TAVILY_API_KEY, SERPER_API_KEY or BRAVE_API_KEY)')
    raise RuntimeError('All configured search providers failed: ' + '; '.join(errors))


# --- fetching ---------------------------------------------------------------------------------

_BOILERPLATE_LINES = re.compile(
    r'^\s*(skip to (main )?content|an official website of the united states government|'
    r"here'?s how you know|official websites use \.gov|secure \.gov websites use https|"
    r'donate|archives?|partners?|subscribe|sign in|log ?in|menu|navigation|'
    r'cookie(s| policy| settings)?|accept cookies|we use cookies)\s*$', re.I)
_NAV_LIST_ITEM = re.compile(r'^-\s+(.*\S)\s*$')


def _normalize(text: str) -> str:
    text = re.sub(r'[ \t]+', ' ', text)
    return re.sub(r'\n{3,}', '\n\n', text).strip()


def _strip_tags(text: str) -> str:
    text = re.sub(r'<script[\s\S]*?</script>', '', text, flags=re.I)
    text = re.sub(r'<style[\s\S]*?</style>', '', text, flags=re.I)
    return html.unescape(re.sub(r'<[^>]+>', ' ', text)).strip()


def _leading_nav_trim(text: str, max_items: int = 8, max_words: int = 6) -> str:
    """Drop a leading run of short markdown list items (a breadcrumb or nav menu)."""
    lines, i = text.split('\n'), 0
    while i < len(lines) and i < max_items:
        stripped = lines[i].strip()
        if not stripped:
            i += 1
            continue
        m = _NAV_LIST_ITEM.match(stripped)
        if m and len(m.group(1).split()) <= max_words:
            i += 1
            continue
        break
    return '\n'.join(lines[i:])


def clean_page_text(text: str) -> str:
    kept = [line for line in text.split('\n') if not _BOILERPLATE_LINES.match(line)]
    return _normalize(_leading_nav_trim('\n'.join(kept)))


def extract_text(html_text: str) -> str:
    """Article text of an HTML page: trafilatura, else a crude tag strip."""
    body, title = '', ''
    try:
        import trafilatura
        doc = trafilatura.extract(html_text, output_format='markdown', include_comments=False,
                                  include_tables=True, with_metadata=True, favor_recall=True) or ''
        m = re.match(r'^---\n(.*?)\n---\n(.*)$', doc, re.S)
        if m:
            tm = re.search(r'^title:\s*"?(.*?)"?\s*$', m.group(1), re.M)
            title, body = (tm.group(1) if tm else ''), m.group(2).strip()
        else:
            body = doc.strip()
    except Exception:
        body = ''
    if len(body) < 200:
        body = _normalize(_strip_tags(html_text))
    body = clean_page_text(body)
    return f'# {title}\n\n{body}' if title else body


async def fetch_text(url: str, max_length: int = 4000) -> Optional[str]:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers={'User-Agent': USER_AGENT},
                                   timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status >= 400:
                    return None
                raw = await response.text()
                kind = response.headers.get('content-type', '')
        if 'html' in kind or '<html' in raw[:200].lower() or not kind:
            raw = extract_text(raw)
        return raw.strip()[:max_length] or None
    except Exception:
        return None


# --- probes -----------------------------------------------------------------------------------

QUERY_LIMIT = 400        # Brave and Tavily reject longer queries


@dataclass
class Parcel:
    """One harvested scrap of raw text, with the provenance to trace it back."""
    text: str
    origin: str      # 'web:https://...', 'web:search:<query>', 'seed:<what>'
    source: str      # 'web' | 'seed'


def compose_query_terms(pairs: Sequence[Tuple[str, str]], limit: int = QUERY_LIMIT) -> List[str]:
    """One term per concept, `name: gist`, the budget split evenly across concepts. A name is
    never cut; a gist is cut only when its share runs out - at a sentence end if one fits, else
    at a word boundary. A concept without a gist is just its name."""
    pairs = [(str(n).strip(), str(g or '').strip()) for n, g in pairs if str(n or '').strip()]
    if not pairs:
        return []
    share = max(1, (limit - (len(pairs) - 1)) // len(pairs))
    terms = []
    for name, gist in pairs:
        if not gist or gist.lower() == name.lower():
            terms.append(name)
            continue
        room = share - len(name) - 2
        if room < 12:
            terms.append(name)
            continue
        if len(gist) > room:
            cut = gist[:room]
            end = max(cut.rfind('. '), cut.rfind('; '))
            gist = cut[:end + 1] if end >= room // 2 else cut.rsplit(' ', 1)[0]
        terms.append(f'{name}: {gist.rstrip(" ,;:")}')
    return terms


class WebSource:
    """Search a probe's query, fetch the first page not digested before."""

    def __init__(self):
        self.last_attempt: Dict[str, Any] = {}
        self.seen_urls: set = set()      # pages already digested; the same top hit twice scores nothing

    @staticmethod
    def available() -> bool:
        return search_available()

    async def harvest(self, query: str, n: int = 1, accept=None) -> List[Parcel]:
        """Fetch up to `n` pages for `query`. `accept(parcel)` may veto a fetched page by returning a
        dict describing why (for example a near-duplicate of something already read); the page is
        then remembered as seen, recorded in `last_attempt['skipped']`, and the next result is tried."""
        query = ' '.join(str(query).split())
        self.last_attempt = {'query': query, 'status': 'unavailable'}
        if not query:
            self.last_attempt['status'] = 'empty-query'
            return []
        try:
            results = await search(query, count=5)
        except Exception as e:
            self.last_attempt.update(status='search-error', message=str(e)[:300])
            return []
        self.last_attempt['results'] = len(results)
        if not results:
            self.last_attempt['status'] = 'no-results'
            return []
        fresh = [r for r in results if r['url'] not in self.seen_urls]
        self.last_attempt['already_seen'] = len(results) - len(fresh)
        if not fresh:
            self.last_attempt['status'] = 'all-seen'
            return []
        out: List[Parcel] = []
        skipped: List[Dict[str, Any]] = []
        for r in fresh:
            text = await fetch_text(r['url'])
            if not text:
                self.last_attempt['fetch_failures'] = self.last_attempt.get('fetch_failures', 0) + 1
                continue
            self.seen_urls.add(r['url'])
            parcel = Parcel(text=text, origin=f"web:{r['url']}", source='web')
            veto = accept(parcel) if accept is not None else None
            if veto:
                skipped.append({'source': parcel.origin, **veto})
                continue
            out.append(parcel)
            if len(out) >= n:
                break
        if skipped:
            self.last_attempt['skipped'] = skipped
        if out:
            self.last_attempt['status'] = 'ok'
        elif skipped:
            self.last_attempt['status'] = 'all-duplicate'
        else:                                   # nothing fetchable: fall back to the search snippets
            snippets = '\n'.join(f"{r['title']}: {r['snippet']}" for r in fresh if r['snippet'])
            if snippets:
                self.last_attempt['status'] = 'snippet-only'
                out.append(Parcel(text=snippets[:4000], origin=f'web:search:{query}', source='web'))
            else:
                self.last_attempt['status'] = 'fetch-failed'
        return out
