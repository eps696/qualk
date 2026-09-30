"""The LLM passes: an OpenAI-compatible chat model reads a page and answers with graph ops.

- `llm_extractor`: prompts/world-upd.md - concepts, relations and open questions (threads) from a page.
- `thread_extractor`: prompts/thread-upd.md - whether a page advances or resolves already-open threads.

Configuration (environment, see .env.example):
  QUALK_LLM_URL    base URL, default http://localhost:1234/v1 (LM Studio)
  QUALK_LLM_KEY    API key, default 'lm-studio'
  QUALK_LLM_MODEL  model name, default 'gpt-oss-20b'
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

PROMPTS = Path(__file__).with_name('prompts')


def parse_json(text):
    """The JSON object in an LLM reply: first '{' to last '}', trailing commas tolerated."""
    if not text:
        return None
    strip_commas = lambda x: re.sub(r',(\s*[}\]])', r'\1', x)
    start, end = text.find('{'), text.rfind('}') + 1
    if start == -1 or end <= start:
        return None
    body = ''.join(c for c in text[start:end] if ord(c) >= 32 or c in '\n\r')
    try:
        return json.loads(strip_commas(body), strict=False)
    except json.JSONDecodeError:
        return None


def _caller(prompt_file, second_key, url=None, key=None, model=None, tries=3, timeout=300.0, temperature=0.6):
    """An async (text, payload) -> {"ops": [...]} function: `prompt_file` is the system prompt and
    the user message is {"at": 0, "text": text, <second_key>: payload} as JSON."""
    from openai import AsyncOpenAI
    client = AsyncOpenAI(base_url=url or os.environ.get('QUALK_LLM_URL') or 'http://localhost:1234/v1',
                         api_key=key or os.environ.get('QUALK_LLM_KEY') or 'lm-studio', timeout=timeout)
    model = model or os.environ.get('QUALK_LLM_MODEL') or 'gpt-oss-20b'
    system = (PROMPTS / prompt_file).read_text(encoding='utf-8')

    async def call(text, payload):
        last = 'no reply'
        for _ in range(tries):
            reply = await client.chat.completions.create(
                model=model, temperature=temperature,
                messages=[{'role': 'system', 'content': system},
                          {'role': 'user', 'content': json.dumps({'at': 0, 'text': text, second_key: payload},
                                                                 ensure_ascii=False)}])
            parsed = parse_json(reply.choices[0].message.content or '')
            if isinstance(parsed, dict) and 'ops' in parsed:
                return parsed
            last = 'reply had no parsable {"ops": [...]}'
        raise RuntimeError(last)

    return call


def llm_extractor(**kw):
    """The world extractor: (text, known view) -> ops that add concepts, relations and open questions."""
    return _caller('world-upd.md', 'known', **kw)


def thread_extractor(**kw):
    """The thread pass: (text, candidate open threads) -> ops that advance or resolve them."""
    return _caller('thread-upd.md', 'candidates', **kw)
