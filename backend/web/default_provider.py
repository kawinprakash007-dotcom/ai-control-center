import html
import re
from typing import List
from urllib.parse import unquote, urlparse
import requests

from core.interfaces.web_interface import WebProviderInterface, WebProviderError
from core.models.web import SearchResult, FetchResult


class DefaultWebProvider(WebProviderInterface):
    """
    Lightweight, deterministic default implementation of WebProviderInterface.
    Uses public HTTP search endpoints and standard HTTP fetching without browser automation.
    """

    DEFAULT_USER_AGENT = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36 AI-Control-Center/1.0"
    )
    SEARCH_ENDPOINT = "https://html.duckduckgo.com/html/"
    MAX_FETCH_CONTENT_LENGTH = 4000

    def __init__(self, user_agent: str | None = None):
        self.user_agent = user_agent or self.DEFAULT_USER_AGENT

    def search(
        self,
        query: str,
        max_results: int = 5,
        timeout_seconds: float = 10.0,
    ) -> List[SearchResult]:
        if not query or not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string.")
        if max_results <= 0:
            return []
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be a positive number.")

        q = query.strip()
        headers = {"User-Agent": self.user_agent}
        data = {"q": q}

        try:
            response = requests.post(
                self.SEARCH_ENDPOINT,
                data=data,
                headers=headers,
                timeout=timeout_seconds,
            )
            response.raise_for_status()
        except requests.RequestException as e:
            raise WebProviderError(f"Web search failed: {e}") from e

        html_text = response.text
        titles = re.findall(
            r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
            html_text,
            re.DOTALL,
        )
        snippets = re.findall(
            r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>',
            html_text,
            re.DOTALL,
        )

        results: List[SearchResult] = []
        for i in range(min(max_results, len(titles))):
            raw_url, raw_title = titles[i]
            uddg_match = re.search(r"uddg=([^&]+)", raw_url)
            actual_url = unquote(uddg_match.group(1)) if uddg_match else raw_url

            clean_title = html.unescape(re.sub(r"<[^>]+>", "", raw_title).strip())
            raw_snippet = snippets[i] if i < len(snippets) else ""
            clean_snippet = html.unescape(re.sub(r"<[^>]+>", "", raw_snippet).strip())
            source = urlparse(actual_url).netloc

            results.append(
                SearchResult(
                    title=clean_title,
                    url=actual_url,
                    snippet=clean_snippet,
                    source=source,
                )
            )

        return results

    def fetch(
        self,
        url: str,
        timeout_seconds: float = 10.0,
    ) -> FetchResult:
        if not url or not isinstance(url, str) or not url.strip():
            raise ValueError("url must be a non-empty string.")
        u = url.strip()
        if not (u.startswith("http://") or u.startswith("https://")):
            raise ValueError("url must start with 'http://' or 'https://'.")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be a positive number.")

        headers = {"User-Agent": self.user_agent}

        try:
            response = requests.get(
                u,
                headers=headers,
                timeout=timeout_seconds,
            )
            response.raise_for_status()
        except requests.RequestException as e:
            raise WebProviderError(f"Web fetch failed: {e}") from e

        final_url = response.url or u
        domain = urlparse(final_url).netloc
        resp_text = response.text

        title_match = re.search(
            r"<title[^>]*>(.*?)</title>", resp_text, re.IGNORECASE | re.DOTALL
        )
        title = (
            html.unescape(re.sub(r"<[^>]+>", "", title_match.group(1)).strip())
            if title_match
            else domain
        )

        # Clean HTML content
        cleaned = re.sub(
            r"<(script|style|noscript)[^>]*>.*?</\1>",
            "",
            resp_text,
            flags=re.DOTALL | re.IGNORECASE,
        )
        cleaned = re.sub(r"<[^>]+>", " ", cleaned)
        cleaned = html.unescape(cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        if len(cleaned) > self.MAX_FETCH_CONTENT_LENGTH:
            cleaned = cleaned[: self.MAX_FETCH_CONTENT_LENGTH] + " ... [truncated]"

        return FetchResult(
            url=final_url,
            title=title,
            content=cleaned,
            status_code=response.status_code,
            source=domain,
        )
