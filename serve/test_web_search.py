"""Tests for the built-in, dependency-free web search provider."""
from __future__ import annotations

import threading
import unittest
import zlib
from unittest import mock

from serve.mcp import McpHub, hub_from_config
from serve.web_search import (WebSearchError, WebSearchServer, _decompress, _page_text, _parse_results, _public_url,
                              _result_url)


class Parsing(unittest.TestCase):
    def test_gzip_response_is_bounded_and_decoded(self):
        source = b"<html><title>Compressed</title><p>Readable page</p></html>"
        compressor = zlib.compressobj(wbits=16 + zlib.MAX_WBITS)
        packed = compressor.compress(source) + compressor.flush()
        self.assertEqual(_decompress(packed, "gzip", 1024), source)
        with self.assertRaises(WebSearchError):
            _decompress(packed, "gzip", 10)

    def test_duckduckgo_results_and_redirects(self):
        body = """
        <div class="result">
          <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fnews%3Fa%3D1">Example News</a>
          <a class="result__snippet">The current example result.</a>
        </div>
        <div><a class="result__a" href="https://example.org/other">Other result</a>
          <div class="result__snippet">Another snippet.</div></div>
        """
        results = _parse_results(body)
        self.assertEqual(results[0], {"title": "Example News", "url": "https://example.com/news?a=1",
                                      "snippet": "The current example result."})
        self.assertEqual(results[1]["url"], "https://example.org/other")
        self.assertEqual(_result_url("https://duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.net%2Fx"),
                         "https://example.net/x")

    def test_page_text_ignores_navigation_and_scripts(self):
        title, text = _page_text("""<html><head><title>Useful page</title><script>secret()</script></head>
          <body><nav>menus</nav><article><h1>Heading</h1><p>Useful facts here.</p></article>
          <footer>tracking links</footer></body></html>""", "text/html")
        self.assertEqual(title, "Useful page")
        self.assertIn("Heading\nUseful facts here.", text)
        self.assertNotIn("secret", text)
        self.assertNotIn("menus", text)
        self.assertNotIn("tracking", text)


class Safety(unittest.TestCase):
    @mock.patch("serve.web_search.socket.getaddrinfo")
    def test_public_url_accepts_public_and_rejects_private(self, resolve):
        resolve.return_value = [(2, 1, 6, "", ("93.184.216.34", 443))]
        self.assertEqual(_public_url("https://example.com/a"), "https://example.com/a")
        resolve.return_value = [(2, 1, 6, "", ("127.0.0.1", 80))]
        with self.assertRaises(WebSearchError):
            _public_url("http://localhost/private")
        with self.assertRaises(WebSearchError):
            _public_url("file:///etc/passwd")


class Provider(unittest.TestCase):
    def test_hub_exposes_direct_builtin_names(self):
        hub = McpHub({"configured": {"command": "not-started"}}, web_search=True)
        names = {t["name"] for t in hub.template_tools(builtin=True)}
        self.assertEqual(names, {"web_search", "open_webpage"})
        self.assertTrue(all("configured__" not in n for n in names))
        status = hub.status()
        self.assertEqual((status["web_tools"], status["external_tools"]), (2, 0))
        self.assertIsNotNone(hub_from_config({}, web_search=True))

    @mock.patch("serve.web_search.search_duckduckgo")
    def test_search_result_is_mcp_text(self, search):
        search.return_value = [{"title": "Example", "url": "https://example.com/", "snippet": "Fresh fact"}]
        server = WebSearchServer()
        result = server.call("web_search", {"query": "fresh example", "max_results": 3}, 5, threading.Event())
        self.assertFalse(result.get("isError", False))
        text = result["content"][0]["text"]
        self.assertIn('DuckDuckGo results for "fresh example"', text)
        self.assertIn("https://example.com/", text)
        search.assert_called_once()


if __name__ == "__main__":
    unittest.main()
