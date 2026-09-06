import re
from typing import Optional, List, Dict, Any, Callable, Set

from core.interfaces.web_interface import WebProviderInterface, WebProviderError
from core.models.web import (
    EvidenceItem,
    EvidenceSet,
    Citation,
    CitationSet,
    ResearchState,
    ResearchResult,
    create_evidence_from_search,
    create_evidence_from_fetch,
    canonicalize_url,
)


class WebResearchCoordinator:
    """
    Bounded multi-step web research coordinator.
    Executes an iterative, strictly bounded loop of search and fetch operations
    using a WebProviderInterface, accumulates and deduplicates evidence,
    generates deterministic citations, and synthesizes structured findings.

    Hard architectural boundaries enforced:
    - MAX_ITERATIONS (default 3)
    - MAX_SEARCHES (default 3)
    - MAX_FETCHES (default 2)
    - MAX_EVIDENCE_ITEMS (default 10)
    - MIN_EVIDENCE_ITEMS (default 1)
    - TIMEOUT_SECONDS (default 10.0)

    Never loops infinitely, never spawns recursive cycles, and strictly
    preserves source provenance and factual URL grounding.
    """

    def __init__(
        self,
        provider: Optional[WebProviderInterface] = None,
        synthesizer: Optional[Callable[[str, EvidenceSet, CitationSet], str]] = None,
        reasoner: Optional[Any] = None,
    ):
        """
        Initialize the research coordinator.

        Args:
            provider: WebProviderInterface instance. If omitted, lazily imports DefaultWebProvider.
            synthesizer: Optional synthesis function for testing or future LLM integration.
            reasoner: Optional ResearchReasonerInterface instance for autonomous reasoning.
        """
        if provider is not None:
            self.provider = provider
        else:
            try:
                from web.default_provider import DefaultWebProvider
                self.provider = DefaultWebProvider()
            except ImportError:
                self.provider = None

        self.synthesizer = synthesizer
        self.reasoner = reasoner
        self.last_error: Optional[Exception] = None

    def generate_query_facets(self, objective: str) -> List[str]:
        """
        Deterministically decompose a research objective into 2-3 focused query facets.

        Example:
            Objective: 'Research the latest Raspberry Pi 5 AI performance and compare several sources.'
            Core topic: 'Raspberry Pi 5 AI performance'
            Facets:
                1. 'Raspberry Pi 5 AI performance'
                2. 'Raspberry Pi 5 AI performance benchmark comparison'
                3. 'Raspberry Pi 5 AI performance specifications overview'
        """
        cleaned = objective.strip()

        # Strip leading conversational wrappers
        leading_patterns = [
            r"^(?:please\s+)?research\s+(?:the\s+)?(?:latest\s+)?(?:about\s+|on\s+)?",
            r"^(?:please\s+)?research\s+",
            r"^(?:please\s+)?investigate\s+(?:the\s+)?(?:latest\s+)?(?:about\s+|on\s+)?",
            r"^(?:please\s+)?investigate\s+",
            r"^(?:please\s+)?compare\s+(?:several\s+)?sources\s+(?:for|about|on)\s+",
            r"^(?:please\s+)?deep\s+search\s+(?:for|about|on)\s+",
            r"^(?:please\s+)?find\s+out\s+about\s+",
        ]
        for pat in leading_patterns:
            m = re.match(pat, cleaned, re.IGNORECASE)
            if m:
                cleaned = cleaned[m.end():].strip()
                break

        # Strip trailing conversational wrappers
        trailing_patterns = [
            r"\s+and\s+compare\s+(?:several\s+)?sources[.!?]?$",
            r"\s+comparing\s+(?:several\s+)?sources[.!?]?$",
            r"\s+across\s+(?:several|multiple)\s+sources[.!?]?$",
            r"\s+from\s+(?:several|multiple)\s+sources[.!?]?$",
            r"[.!?]+$",
        ]
        for pat in trailing_patterns:
            cleaned = re.sub(pat, "", cleaned, flags=re.IGNORECASE).strip()

        core_topic = cleaned if cleaned else objective.strip()

        facets = [
            core_topic,
            f"{core_topic} benchmark comparison",
            f"{core_topic} specifications overview",
        ]

        # Return unique non-empty facets
        seen = set()
        unique_facets = []
        for f in facets:
            fn = f.strip().lower()
            if fn and fn not in seen:
                seen.add(fn)
                unique_facets.append(f.strip())

        return unique_facets

    def research(
        self,
        objective: str,
        max_iterations: int = 3,
        max_searches: int = 3,
        max_fetches: int = 2,
        max_evidence: int = 10,
        min_evidence: int = 1,
        timeout_seconds: float = 10.0,
        queries: Optional[List[str]] = None,
        raise_on_error: bool = False,
    ) -> ResearchResult:
        """
        Execute a bounded multi-step research loop.

        Args:
            objective: User query or research goal.
            max_iterations: Hard upper bound on research loop cycles (capped at 10).
            max_searches: Hard upper bound on search requests (capped at 10).
            max_fetches: Hard upper bound on fetch requests (capped at 5).
            max_evidence: Maximum number of EvidenceItems to retain (capped at 50).
            min_evidence: Minimum evidence items required for full success.
            timeout_seconds: Timeout per provider operation.
            queries: Optional explicit queries overriding default facet decomposition.
            raise_on_error: Whether to re-raise provider errors immediately.

        Returns:
            ResearchResult containing accumulated EvidenceSet, CitationSet,
            ResearchState, and formatted synthesis with sources.
        """
        self.last_error = None
        clean_objective = objective.strip()
        if not clean_objective:
            raise ValueError("Research objective cannot be empty.")

        if self.provider is None:
            err = RuntimeError("Web provider is unavailable.")
            self.last_error = err
            if raise_on_error:
                raise err
            empty_evidence = EvidenceSet(query=clean_objective)
            empty_citations = CitationSet()
            state = ResearchState(
                objective=clean_objective,
                completed=True,
                stop_reason="STOP_FAILURE",
            )
            return ResearchResult(
                objective=clean_objective,
                evidence=empty_evidence,
                citations=empty_citations,
                status="failed",
                output=f"Research failed for: '{clean_objective}'. Web provider is unavailable.",
                state=state,
            )

        # Enforce defensive caps
        effective_max_iterations = max(1, min(max_iterations, 10))
        effective_max_searches = max(1, min(max_searches, 10))
        effective_max_fetches = max(0, min(max_fetches, 5))
        effective_max_evidence = max(1, min(max_evidence, 50))
        effective_min_evidence = max(1, min_evidence)

        if self.reasoner is not None:
            from web.research_agent import ResearchAgent
            from core.models.research import ResearchLimits
            agent = ResearchAgent(
                provider=self.provider,
                reasoner=self.reasoner,
                synthesizer=self.synthesizer,
            )
            limits = ResearchLimits(
                max_iterations=effective_max_iterations,
                max_searches=effective_max_searches,
                max_fetches=effective_max_fetches,
                max_evidence=effective_max_evidence,
                min_evidence=effective_min_evidence,
                timeout_seconds=timeout_seconds,
            )
            res = agent.run(clean_objective, limits=limits, raise_on_error=raise_on_error)
            self.last_error = agent.last_error
            return res

        # Decompose query facets or use provided queries
        if queries and len(queries) > 0:
            query_facets = list(queries)
        else:
            query_facets = self.generate_query_facets(clean_objective)

        accumulated_items: List[EvidenceItem] = []
        seen_canonical_urls: Set[str] = set()
        fetched_urls: Set[str] = set()

        searches_used = 0
        fetches_used = 0
        iteration = 0
        stop_reason = ""
        completed = False
        query_index = 0

        while iteration < effective_max_iterations and not completed:
            iteration += 1

            # -------------------------------------------------------------
            # STEP 1: Search Execution
            # -------------------------------------------------------------
            if searches_used < effective_max_searches and query_index < len(query_facets):
                current_query = query_facets[query_index]
                query_index += 1

                try:
                    search_results = self.provider.search(
                        query=current_query,
                        max_results=min(5, effective_max_evidence - len(accumulated_items)),
                        timeout_seconds=timeout_seconds,
                    )
                    searches_used += 1
                except Exception as e:
                    self.last_error = e
                    if raise_on_error and len(accumulated_items) == 0:
                        raise
                    stop_reason = "STOP_FAILURE"
                    completed = True
                    break

                for sr in search_results:
                    c_url = canonicalize_url(sr.url) or sr.url
                    if c_url not in seen_canonical_urls:
                        item = create_evidence_from_search(sr)
                        seen_canonical_urls.add(c_url)
                        accumulated_items.append(item)
                        if len(accumulated_items) >= effective_max_evidence:
                            break

            # -------------------------------------------------------------
            # STEP 2: Fetch Execution (Deepening)
            # -------------------------------------------------------------
            if not completed and fetches_used < effective_max_fetches and len(accumulated_items) > 0:
                url_to_fetch = None
                for ev in accumulated_items:
                    if ev.url and ev.url not in fetched_urls:
                        url_to_fetch = ev.url
                        break

                if url_to_fetch:
                    fetched_urls.add(url_to_fetch)
                    try:
                        fetch_res = self.provider.fetch(
                            url=url_to_fetch,
                            timeout_seconds=timeout_seconds,
                        )
                        fetches_used += 1
                        fetch_item = create_evidence_from_fetch(fetch_res)

                        # Replace search snippet with richer fetch content
                        c_fetch_url = canonicalize_url(fetch_res.url) or fetch_res.url
                        replaced = False
                        for idx, existing in enumerate(accumulated_items):
                            c_exist = canonicalize_url(existing.url) or existing.url
                            if c_exist == c_fetch_url:
                                accumulated_items[idx] = fetch_item
                                replaced = True
                                break

                        if not replaced and len(accumulated_items) < effective_max_evidence:
                            seen_canonical_urls.add(c_fetch_url)
                            accumulated_items.append(fetch_item)

                    except Exception as e:
                        self.last_error = e
                        # A single fetch failure does not invalidate gathered search evidence
                        if len(accumulated_items) == 0:
                            if raise_on_error:
                                raise
                            stop_reason = "STOP_FAILURE"
                            completed = True
                            break

            # -------------------------------------------------------------
            # STEP 3: Stopping Rules Evaluation
            # -------------------------------------------------------------
            evidence_count = len(accumulated_items)

            if evidence_count >= effective_max_evidence:
                stop_reason = "STOP_LIMIT"
                completed = True
            elif searches_used >= effective_max_searches and fetches_used >= effective_max_fetches:
                stop_reason = "STOP_SUCCESS" if evidence_count >= effective_min_evidence else "STOP_LIMIT"
                completed = True
            elif query_index >= len(query_facets) and (fetches_used >= effective_max_fetches or len(fetched_urls) >= len(accumulated_items)):
                stop_reason = "STOP_SUCCESS" if evidence_count >= effective_min_evidence else "STOP_LIMIT"
                completed = True
            elif query_index >= len(query_facets) and evidence_count >= effective_min_evidence:
                stop_reason = "STOP_SUCCESS"
                completed = True
            elif iteration >= effective_max_iterations:
                stop_reason = "STOP_LIMIT"
                completed = True

        if not stop_reason:
            if len(accumulated_items) >= effective_min_evidence:
                stop_reason = "STOP_SUCCESS"
            elif self.last_error is not None:
                stop_reason = "STOP_FAILURE"
            else:
                stop_reason = "STOP_LIMIT"

        evidence_set = EvidenceSet.from_items(
            accumulated_items,
            query=clean_objective,
            max_items=effective_max_evidence,
        )
        citation_set = CitationSet.from_evidence_set(evidence_set)

        if len(evidence_set) == 0:
            status = "failed"
        elif stop_reason == "STOP_SUCCESS":
            status = "completed"
        elif stop_reason == "STOP_LIMIT":
            status = "completed" if len(evidence_set) >= effective_min_evidence else "partial"
        elif stop_reason == "STOP_FAILURE":
            status = "failed"
        else:
            status = "failed" if len(evidence_set) < effective_min_evidence else "partial"

        if status == "failed" and raise_on_error and self.last_error is not None:
            raise self.last_error

        output = self._synthesize(clean_objective, evidence_set, citation_set, status, stop_reason)

        final_state = ResearchState(
            objective=clean_objective,
            iteration=iteration,
            searches_used=searches_used,
            fetches_used=fetches_used,
            evidence_count=len(evidence_set),
            completed=True,
            stop_reason=stop_reason,
        )

        return ResearchResult(
            objective=clean_objective,
            evidence=evidence_set,
            citations=citation_set,
            status=status,
            output=output,
            state=final_state,
        )

    def _synthesize(
        self,
        objective: str,
        evidence: EvidenceSet,
        citations: CitationSet,
        status: str,
        stop_reason: str,
    ) -> str:
        """
        Synthesize evidence and citations into a user-facing response deterministically.
        """
        if self.synthesizer is not None:
            return self.synthesizer(objective, evidence, citations)

        if evidence.is_empty:
            if status == "failed":
                return f"Research failed for: '{objective}'. Unable to gather required evidence ({stop_reason})."
            return f"No research findings found for: '{objective}'."

        lines: List[str] = []
        if status == "partial":
            lines.append("[Note: Research concluded with partial results due to execution limits.]\n")

        lines.append(f"Research findings for '{objective}':\n")
        lines.append("Key Findings:")
        for idx, item in enumerate(evidence.items, start=1):
            citation = citations[idx - 1] if idx - 1 < len(citations) else None
            marker = citation.render_marker() if citation else f"[{idx}]"
            content_snippet = (item.content or "").strip()
            # Bound snippet display
            if len(content_snippet) > 200:
                content_snippet = content_snippet[:197] + "..."
            lines.append(f"- {item.title}: {content_snippet} {marker}")

        lines.append("")
        lines.append(citations.render_sources_block())

        return "\n".join(lines)
