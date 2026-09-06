from typing import Optional, Union, Dict, Any

from core.interfaces.web_interface import WebProviderInterface
from core.models.task import Task
from core.models.web import (
    EvidenceSet,
    ResearchResult,
    create_evidence_from_search,
    create_evidence_from_fetch,
)
from web.research_coordinator import WebResearchCoordinator
from web.research_agent import ResearchAgent
from core.models.research import ResearchLimits


class WebCapability:
    """
    Adapter between AI Control Center's Tool protocol and WebProviderInterface.
    Executes web intelligence tasks: SEARCH, FETCH, and bounded RESEARCH.
    Produces structured, source-backed EvidenceSet records for downstream reasoning.
    """

    def __init__(self, provider: Optional[WebProviderInterface] = None):
        """
        Initialize WebCapability with an injected or lazily-retrieved WebProvider.
        """
        self.last_evidence: Optional[EvidenceSet] = None
        self.last_research_result: Optional[ResearchResult] = None
        if provider is not None:
            self.provider = provider
        else:
            try:
                from web.default_provider import DefaultWebProvider
                self.provider = DefaultWebProvider()
            except ImportError:
                self.provider = None

    def get_evidence(self) -> Optional[EvidenceSet]:
        """
        Retrieve the most recently generated evidence set, if any.
        """
        return self.last_evidence

    def get_research_result(self) -> Optional[ResearchResult]:
        """
        Retrieve the most recently generated research result, if any.
        """
        return self.last_research_result

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
            elif raw.lower().startswith("research:") or raw.lower().startswith("research "):
                action_from_str = "research"
                query_from_str = raw.split(":", 1)[-1].strip() if raw.lower().startswith("research:") else raw[9:].strip()
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
            raise ValueError("No web action specified. Expected 'search', 'fetch', or 'research'.")

        if action not in ("search", "fetch", "research"):
            raise ValueError(f"Unsupported web action: '{action}'. Expected 'search', 'fetch', or 'research'.")

        self.last_evidence = None
        self.last_research_result = None

        # ------------------------------------------------------------------
        # 1. SEARCH
        # ------------------------------------------------------------------
        if action == "search":
            if not query:
                raise ValueError("Missing query for web search.")

            try:
                results = provider.search(
                    query=query,
                    max_results=max_results,
                    timeout_seconds=timeout_seconds,
                )
            except Exception:
                self.last_evidence = None
                raise

            if not results:
                self.last_evidence = EvidenceSet.from_items([], query=query, max_items=max_results)
                if isinstance(params, dict):
                    params["evidence"] = self.last_evidence
                return f"No web search results found for: '{query}'."

            evidence_items = [create_evidence_from_search(r) for r in results]
            self.last_evidence = EvidenceSet.from_items(
                evidence_items,
                query=query,
                max_items=max_results,
            )
            if isinstance(params, dict):
                params["evidence"] = self.last_evidence

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

            try:
                fetch_res = provider.fetch(
                    url=url,
                    timeout_seconds=timeout_seconds,
                )
            except Exception:
                self.last_evidence = None
                raise

            fetch_item = create_evidence_from_fetch(fetch_res)
            self.last_evidence = EvidenceSet.from_items(
                [fetch_item],
                query=url,
                max_items=1,
            )
            if isinstance(params, dict):
                params["evidence"] = self.last_evidence

            lines = [
                f"URL: {fetch_res.url}",
                f"Title: {fetch_res.title}",
            ]
            if fetch_res.source:
                lines.append(f"Source: {fetch_res.source}")
            lines.append("")
            lines.append(fetch_res.content)

            return "\n".join(lines)

        # ------------------------------------------------------------------
        # 3. RESEARCH
        # ------------------------------------------------------------------
        if action == "research":
            objective = query or str(params.get("objective") or "")
            if not objective:
                raise ValueError("Missing objective or query for web research.")

            try:
                max_iterations = int(params.get("max_iterations") or 3)
            except (ValueError, TypeError):
                max_iterations = 3

            try:
                max_searches = int(params.get("max_searches") or 3)
            except (ValueError, TypeError):
                max_searches = 3

            try:
                max_fetches = int(params.get("max_fetches") or 2)
            except (ValueError, TypeError):
                max_fetches = 2

            try:
                min_evidence = int(params.get("min_evidence") or 1)
            except (ValueError, TypeError):
                min_evidence = 1

            max_evidence = max_results if max_results > 5 else 10
            if "max_evidence" in params:
                try:
                    max_evidence = int(params["max_evidence"])
                except (ValueError, TypeError):
                    pass

            queries = params.get("queries")
            if queries is not None and not isinstance(queries, list):
                queries = None

            reasoner = params.get("reasoner")
            synthesizer = params.get("synthesizer")

            if reasoner is not None:
                agent = ResearchAgent(
                    provider=provider,
                    reasoner=reasoner,
                    synthesizer=synthesizer,
                )
                limits = ResearchLimits(
                    max_iterations=max_iterations,
                    max_searches=max_searches,
                    max_fetches=max_fetches,
                    max_evidence=max_evidence,
                    min_evidence=min_evidence,
                    timeout_seconds=timeout_seconds,
                )
                try:
                    research_res = agent.run(
                        objective=objective,
                        limits=limits,
                        raise_on_error=True,
                    )
                except Exception:
                    self.last_evidence = None
                    self.last_research_result = None
                    raise
            else:
                coordinator = WebResearchCoordinator(
                    provider=provider,
                    synthesizer=synthesizer,
                )
                try:
                    research_res = coordinator.research(
                        objective=objective,
                        max_iterations=max_iterations,
                        max_searches=max_searches,
                        max_fetches=max_fetches,
                        max_evidence=max_evidence,
                        min_evidence=min_evidence,
                        timeout_seconds=timeout_seconds,
                        queries=queries,
                        raise_on_error=True,
                    )
                except Exception:
                    self.last_evidence = None
                    self.last_research_result = None
                    raise

            self.last_evidence = research_res.evidence
            self.last_research_result = research_res
            if isinstance(params, dict):
                params["evidence"] = research_res.evidence
                params["research_result"] = research_res

            return research_res.output

        raise ValueError(f"Unsupported web action: '{action}'.")
