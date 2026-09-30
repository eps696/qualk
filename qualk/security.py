"""Access control and outbound-fetch safety for the web app.

- `TokenAuth`: one shared access token (Bearer header, an HttpOnly cookie, or `?token=` for the
  WebSocket). Required whenever the server listens on a non-loopback address.
- `fetch_public_text`: fetch a user-supplied URL for injection without letting the server be used
  to reach its own network. The address check runs inside the connector's resolver, i.e. on the
  exact IP that is about to be connected to, for the first request and for every redirect.
"""

from __future__ import annotations

import hmac
import ipaddress
import secrets
import socket
from typing import Optional
from urllib.parse import urlparse

import aiohttp
from aiohttp.abc import AbstractResolver

COOKIE = 'qualk_token'
MAX_BYTES = 2_000_000


def is_loopback_host(host: str) -> bool:
    if host in ('localhost', ''):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def new_token() -> str:
    return secrets.token_urlsafe(24)


class TokenAuth:
    def __init__(self, token: Optional[str]):
        self.token = token or None       # None: open (loopback only)

    @property
    def required(self) -> bool:
        return self.token is not None

    def verify(self, candidate: Optional[str]) -> bool:
        if not self.required:
            return True
        return bool(candidate) and hmac.compare_digest(str(candidate).encode(), self.token.encode())

    def from_request(self, request) -> bool:
        """`request` is a Starlette Request or WebSocket."""
        if not self.required:
            return True
        header = request.headers.get('authorization', '')
        if header.lower().startswith('bearer ') and self.verify(header[7:].strip()):
            return True
        if self.verify(request.cookies.get(COOKIE)):
            return True
        if request.scope.get('type') == 'websocket':
            return self.verify(request.query_params.get('token'))
        return False


def public_ip(address: str) -> bool:
    try:
        ip = ipaddress.ip_address(address)
    except ValueError:
        return False
    if getattr(ip, 'ipv4_mapped', None):
        ip = ip.ipv4_mapped
    return not (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast
                or ip.is_reserved or ip.is_unspecified)


class PublicOnlyResolver(AbstractResolver):
    """Resolve names, keeping only public addresses; if none is left the connection fails."""

    def __init__(self, inner: Optional[AbstractResolver] = None):
        if inner is None:
            from aiohttp.resolver import DefaultResolver
            inner = DefaultResolver()
        self.inner = inner

    async def resolve(self, host, port=0, family=socket.AF_INET):
        infos = await self.inner.resolve(host, port, family)
        allowed = [info for info in infos if public_ip(info['host'])]
        if not allowed:
            raise OSError(f'{host}: refusing to connect to a non-public address')
        return allowed

    async def close(self):
        await self.inner.close()


def check_url(url: str) -> str:
    """Scheme/host sanity before any network access; returns the cleaned URL."""
    url = (url or '').strip()
    parsed = urlparse(url)
    if parsed.scheme not in ('http', 'https') or not parsed.hostname:
        raise ValueError('only http(s) URLs are allowed')
    if parsed.username or parsed.password:
        raise ValueError('URLs with credentials are not allowed')
    try:
        literal = ipaddress.ip_address(parsed.hostname)
    except ValueError:
        return url
    if not public_ip(str(literal)):
        raise ValueError('that address is not public')
    return url


async def fetch_public_text(url: str, max_length: int = 4000, resolver: Optional[AbstractResolver] = None) -> str:
    """Article text of a public web page. Raises ValueError/OSError/aiohttp errors on refusal or failure."""
    from .web import USER_AGENT, extract_text
    url = check_url(url)
    connector = aiohttp.TCPConnector(resolver=resolver or PublicOnlyResolver())
    async with aiohttp.ClientSession(connector=connector) as session:
        async with session.get(url, headers={'User-Agent': USER_AGENT}, max_redirects=3,
                               timeout=aiohttp.ClientTimeout(total=30)) as response:
            if response.status >= 400:
                raise ValueError(f'the page answered HTTP {response.status}')
            raw = await response.content.read(MAX_BYTES)
            kind = response.headers.get('content-type', '')
            text = raw.decode(response.get_encoding() or 'utf-8', errors='replace')
    if 'html' in kind or '<html' in text[:200].lower() or not kind:
        text = extract_text(text)
    text = text.strip()[:max_length]
    if not text:
        raise ValueError('no readable text on that page')
    return text
