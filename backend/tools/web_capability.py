from typing import Optional, Union, Dict, Any

from core.interfaces.web_interface import WebProviderInterface
from core.models.task import Task


class WebCapability:
    """
    Adapter between AI Control Center's Tool protocol and WebProviderInterface.
    Executes web intelligence tasks: SEARCH and FETCH.
    """

    def __init__(self, provider: Optional[WebProviderInterface] = None):
        """
        Initialize WebCapability with an injected or lazily-retrieved WebProvider.
        """
        if provider is not None:
            self.provider = provider
        else:
            try:
                from web.default_provider import DefaultWebProvider
                self.provider = DefaultWebProvider()
            except ImportError:
                self.provider = None

    def __call__(self, task: Optional[Union[Task, str, Dict[str, Any]]] = None) -> str:
        """
        Execute web operation for the given task.

        Args:
            task: Task instance, dict of parameters, or raw query/URL string.

        Returns:
            Deterministic response string representing search results or fetched content.
        """
        params: Dict[str, Any] = {}
        action_from_str = ""
        query_from_str = ""
        url_from_str = ""

        if isinstance(task, str):
            raw = task.strip()
            if raw.startswith("http://") or raw.startswith("https://"):
                action_from_str = "fetch"
                url_from_str = raw
            else:
                action_from_str = "search"
                query_from_str = raw
        elif isinstance(task, dict):
            params = task
        elif task is not None and hasattr(task, "parameters") and isinstance(task.parameters, dict):
            params = task.parameters

        provider = params.get("web_provider") or self.provider
        if provider is None:
            raise RuntimeError("Web provider is unavailable.")

        action = str(params.get("action") or action_from_str).strip().lower()
        query = str(params.get("query") or query_from_str).strip()
        url = str(params.get("url") or url_from_str).strip()

        try:
            max_results = int(params.get("max_results") or 5)
        except (ValueError, TypeError):
            max_results = 5

        try:
            timeout_seconds = float(params.get("timeout_seconds") or 10.0)
        except (ValueError, TypeError):
            timeout_seconds = 10.0

        if not action:
            raise ValueError("No web action specified. Expected 'search' or 'fetch'.")

        if action not in ("search", "fetch"):
            raise ValueError(f"Unsupported web action: '{action}'. Expected 'search' or 'fetch'.")

        # ------------------------------------------------------------------
        # 1. SEARCH
        # ------------------------------------------------------------------
        if action == "search":
            if not query:
                raise ValueError("Missing query for web search.")

            results = provider.search(
                query=query,
                max_results=max_results,
                timeout_seconds=timeout_seconds,
            )

            if not results:
                return f"No web search results found for: '{query}'."

            lines = [f"Web search results for '{query}':\n"]
            for idx, r in enumerate(results, start=1):
                lines.append(f"{idx}. {r.title}")
                lines.append(f"   URL: {r.url}")
                if r.snippet:
                    lines.append(f"   Snippet: {r.snippet}")

            return "\n".join(lines)

        # ------------------------------------------------------------------
        # 2. FETCH
        # ------------------------------------------------------------------
        if action == "fetch":
            if not url:
                raise ValueError("Missing URL for web fetch.")

            fetch_res = provider.fetch(
                url=url,
                timeout_seconds=timeout_seconds,
            )

            lines = [
                f"URL: {fetch_res.url}",
                f"Title: {fetch_res.title}",
            ]
            if fetch_res.source:
                lines.append(f"Source: {fetch_res.source}")
            lines.append("")
            lines.append(fetch_res.content)

            return "\n".join(lines)

        raise ValueError(f"Unsupported web action: '{action}'.")
