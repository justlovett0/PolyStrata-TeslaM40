"""Built-in DuckDuckGo search and safe webpage reading for the web chat.

The model sees two ordinary function tools.  Search and page downloads happen in
this local server process; no chat history is sent, only the query or URL chosen
by the model.  The implementation uses the standard library so every normal
PolyStrata install has the feature without another package or API key.
"""
from __future__ import annotations

import html
import ipaddress
import json
import re
import socket
import zlib
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) PolyStrata-WebSearch/1.0"
MAX_DOWNLOAD = 2 * 1024 * 1024


class WebSearchError(RuntimeError):
    pass


def _decompress(raw: bytes, encoding: str, limit: int) -> bytes:
    """Decode common HTTP compression while retaining the response size cap."""
    encoding = encoding.split(",", 1)[0].strip().lower()
    if not encoding or encoding == "identity":
        return raw
    modes = {"gzip": 16 + zlib.MAX_WBITS, "x-gzip": 16 + zlib.MAX_WBITS,
             "deflate": zlib.MAX_WBITS}
    if encoding not in modes:
        raise WebSearchError(f"unsupported content encoding {encoding!r}")
    try:
        decoder = zlib.decompressobj(modes[encoding])
        decoded = decoder.decompress(raw, limit + 1)
        if len(decoded) <= limit:
            decoded += decoder.flush(limit + 1 - len(decoded))
    except zlib.error as e:
        # A few servers label raw DEFLATE streams as ordinary deflate.
        if encoding != "deflate":
            raise WebSearchError(f"could not decode {encoding} response: {e}") from None
        try:
            decoder = zlib.decompressobj(-zlib.MAX_WBITS)
            decoded = decoder.decompress(raw, limit + 1)
            if len(decoded) <= limit:
                decoded += decoder.flush(limit + 1 - len(decoded))
        except zlib.error as raw_error:
            raise WebSearchError(f"could not decode deflate response: {raw_error}") from None
    if len(decoded) > limit or decoder.unconsumed_tail:
        raise WebSearchError(f"the decompressed page is larger than {limit // 1048576} MiB")
    return decoded


def _public_url(url: str) -> str:
    """Validate an outbound URL and reject local/private network targets."""
    try:
        p = urllib.parse.urlsplit(str(url).strip())
    except ValueError as e:
        raise WebSearchError(f"invalid URL: {e}") from None
    if p.scheme not in ("http", "https") or not p.hostname or p.username or p.password:
        raise WebSearchError("only public http:// or https:// URLs are allowed")
    try:
        infos = socket.getaddrinfo(p.hostname, p.port or (443 if p.scheme == "https" else 80), type=socket.SOCK_STREAM)
    except OSError as e:
        raise WebSearchError(f"could not resolve {p.hostname}: {e}") from None
    for info in infos:
        ip = ipaddress.ip_address(info[4][0].split("%", 1)[0])
        if not ip.is_global:
            raise WebSearchError(f"refusing private or local address {ip}")
    return urllib.parse.urlunsplit(p)


class _SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return super().redirect_request(req, fp, code, msg, headers, _public_url(newurl))


def _download(url: str, timeout: float, data: bytes | None = None, cancel=None,
              limit: int = MAX_DOWNLOAD) -> tuple[str, str, str]:
    """Return (final URL, content type, decoded body), bounded and SSRF checked."""
    url = _public_url(url)
    req = urllib.request.Request(url, data=data, headers={
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.2",
        "Accept-Encoding": "gzip, deflate",
        "Accept-Language": "en-US,en;q=0.8",
    })
    try:
        with urllib.request.build_opener(_SafeRedirect()).open(req, timeout=max(1.0, timeout)) as r:
            final = _public_url(r.geturl())
            ctype = (r.headers.get_content_type() or "").lower()
            allowed = ctype.startswith("text/") or ctype in {
                "application/xhtml+xml", "application/xml", "application/json", "application/rss+xml"
            }
            if not allowed:
                raise WebSearchError(f"the URL returned {ctype or 'a non-text file'}")
            chunks, size = [], 0
            while True:
                if cancel is not None and cancel.is_set():
                    raise WebSearchError("stopped")
                part = r.read(min(65536, limit + 1 - size))
                if not part:
                    break
                chunks.append(part)
                size += len(part)
                if size > limit:
                    raise WebSearchError(f"the page is larger than {limit // 1048576} MiB")
            raw = _decompress(b"".join(chunks), r.headers.get("Content-Encoding", ""), limit)
            charset = r.headers.get_content_charset() or "utf-8"
            try:
                body = raw.decode(charset, "replace")
            except LookupError:
                body = raw.decode("utf-8", "replace")
            return final, ctype, body
    except WebSearchError:
        raise
    except urllib.error.HTTPError as e:
        raise WebSearchError(f"HTTP {e.code} from {urllib.parse.urlsplit(url).hostname}") from None
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        why = getattr(e, "reason", e)
        raise WebSearchError(f"could not download {url}: {why}") from None


def _result_url(url: str) -> str:
    """Turn DuckDuckGo's redirect link into its destination URL."""
    url = html.unescape(url).strip()
    if url.startswith("//"):
        url = "https:" + url
    try:
        p = urllib.parse.urlsplit(url)
        if p.hostname and p.hostname.lower().endswith("duckduckgo.com"):
            target = urllib.parse.parse_qs(p.query).get("uddg", [None])[0]
            if target:
                return urllib.parse.unquote(target)
    except ValueError:
        pass
    return url


class _SearchParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.results = []
        self._link = None
        self._link_text = []
        self._snippet = None
        self._snippet_text = []

    @staticmethod
    def _classes(attrs):
        return set(dict(attrs).get("class", "").split())

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        classes = self._classes(attrs)
        if tag == "a" and ("result__a" in classes or "result-link" in classes or "uddg=" in a.get("href", "")):
            self._link = a.get("href", "")
            self._link_text = []
        if any("result__snippet" == c or "result-snippet" == c for c in classes):
            self._snippet = tag
            self._snippet_text = []

    def handle_data(self, data):
        if self._link is not None:
            self._link_text.append(data)
        if self._snippet is not None:
            self._snippet_text.append(data)

    def handle_endtag(self, tag):
        if tag == "a" and self._link is not None:
            title = " ".join("".join(self._link_text).split())
            url = _result_url(self._link)
            if title and url.startswith(("http://", "https://")):
                self.results.append({"title": title, "url": url, "snippet": ""})
            self._link = None
        if tag == self._snippet and self._snippet is not None:
            snippet = " ".join("".join(self._snippet_text).split())
            if self.results and snippet:
                self.results[-1]["snippet"] = snippet
            self._snippet = None


def _parse_results(body: str) -> list[dict]:
    p = _SearchParser()
    p.feed(body)
    seen, out = set(), []
    for result in p.results:
        url = result["url"]
        if url in seen:
            continue
        seen.add(url)
        out.append(result)
    return out


class _PageParser(HTMLParser):
    BREAKS = {"article", "blockquote", "br", "div", "h1", "h2", "h3", "h4", "li", "p", "section", "tr"}
    SKIP = {"aside", "canvas", "footer", "form", "header", "nav", "noscript", "script", "style", "svg"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.title, self._title = [], False
        self.parts, self._skip = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip += 1
        if tag == "title" and not self._skip:
            self._title = True
        if tag in self.BREAKS and not self._skip:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag == "title":
            self._title = False
        if tag in self.SKIP and self._skip:
            self._skip -= 1
        if tag in self.BREAKS and not self._skip:
            self.parts.append("\n")

    def handle_data(self, data):
        if self._skip:
            return
        if self._title:
            self.title.append(data)
        self.parts.append(data)

    def text(self) -> tuple[str, str]:
        title = " ".join("".join(self.title).split())
        lines = [" ".join(line.split()) for line in "".join(self.parts).splitlines()]
        return title, "\n".join(line for line in lines if line)


def _page_text(body: str, content_type: str) -> tuple[str, str]:
    if content_type in ("text/html", "application/xhtml+xml") or "<html" in body[:1000].lower():
        p = _PageParser()
        p.feed(body)
        return p.text()
    return "", "\n".join(line.strip() for line in body.splitlines() if line.strip())


def search_duckduckgo(query: str, count: int = 6, timeout: float = 20, cancel=None) -> list[dict]:
    query = " ".join(str(query).split())
    if not query:
        raise WebSearchError("query is empty")
    if len(query) > 500:
        raise WebSearchError("query is longer than 500 characters")
    count = max(1, min(int(count), 10))
    attempts = [
        ("https://html.duckduckgo.com/html/", urllib.parse.urlencode({"q": query}).encode()),
        ("https://lite.duckduckgo.com/lite/?" + urllib.parse.urlencode({"q": query}), None),
    ]
    errors = []
    for url, data in attempts:
        try:
            _, _, body = _download(url, timeout, data=data, cancel=cancel)
            results = _parse_results(body)
            if results:
                return results[:count]
            errors.append("DuckDuckGo returned no readable results")
        except WebSearchError as e:
            errors.append(str(e))
    raise WebSearchError(errors[-1] if errors else "DuckDuckGo search failed")


def _search_text(query: str, results: list[dict]) -> str:
    lines = [f'DuckDuckGo results for "{query}":']
    for i, r in enumerate(results, 1):
        lines += [f"\n[{i}] {r['title']}", f"URL: {r['url']}"]
        if r.get("snippet"):
            lines.append("Snippet: " + r["snippet"])
    lines.append("\nUse open_webpage on useful results before making detailed claims, and cite their URLs in the answer.")
    return "\n".join(lines)


class WebSearchServer:
    """An in-process server with the small interface McpHub expects."""
    builtin = True
    kind = "built-in"

    def __init__(self, name: str = "web search"):
        self.name, self.status, self.error = name, "ready", None
        self.info = {"name": "DuckDuckGo Web Search", "version": "1"}
        self.tools = [
            {"name": "web_search", "description": "Search the live web with DuckDuckGo. Use this for current or externally verifiable information. Only the query is sent to DuckDuckGo.",
             "inputSchema": {"type": "object", "properties": {
                 "query": {"type": "string", "description": "A concise web search query"},
                 "max_results": {"type": "integer", "minimum": 1, "maximum": 10, "default": 6}},
                 "required": ["query"]}},
            {"name": "open_webpage", "description": "Read the text of a public webpage from a web_search result. Use it to verify details and obtain the source URL before answering.",
             "inputSchema": {"type": "object", "properties": {
                 "url": {"type": "string", "description": "The public http or https URL to read"},
                 "max_chars": {"type": "integer", "minimum": 2000, "maximum": 20000, "default": 12000}},
                 "required": ["url"]}},
        ]

    def start(self):
        self.status = "ready"

    def alive(self):
        return True

    def close(self):
        pass

    def call(self, tool: str, arguments: dict, timeout: float, cancel=None) -> dict:
        try:
            if tool == "web_search":
                query = str(arguments.get("query") or "").strip()
                results = search_duckduckgo(query, arguments.get("max_results", 6), timeout, cancel)
                text = _search_text(query, results)
            elif tool == "open_webpage":
                url = str(arguments.get("url") or "").strip()
                cap = max(2000, min(int(arguments.get("max_chars", 12000)), 20000))
                final, ctype, body = _download(url, timeout, cancel=cancel)
                title, page = _page_text(body, ctype)
                if not page:
                    raise WebSearchError("the page contained no readable text")
                if len(page) > cap:
                    page = page[:cap].rsplit(" ", 1)[0] + f"\n\n[page text cut at {cap:,} characters]"
                text = f"Title: {title or '(untitled)'}\nSource: {final}\n\n{page}"
            else:
                raise WebSearchError(f"unknown web tool {tool!r}")
            return {"content": [{"type": "text", "text": text}]}
        except (TypeError, ValueError) as e:
            return {"isError": True, "content": [{"type": "text", "text": f"invalid arguments: {e}"}]}
        except WebSearchError as e:
            return {"isError": True, "content": [{"type": "text", "text": str(e)}]}

