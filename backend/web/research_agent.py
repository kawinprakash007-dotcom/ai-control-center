import re
from typing import Optional, List, Dict, Any, Callable, Set, Tuple, Union

from core.interfaces.web_interface import WebProviderInterface, WebProviderError
from core.interfaces.research_interface import (
    ResearchReasonerInterface,
    ResearchSynthesizerInterface,
)
from core.models.research import (
    AgentAction,
    AgentActionType,
    ResearchObservation,
    ResearchLimits,
    EvidenceAssessment,
    ResearchGap,
    Contradiction,
)
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
    validate_citation_references,
)
from web.evidence_evaluator import EvidenceEvaluator


class ActionValidator:
    """
    Runtime-owned validator enforcing safety, parameter constraints,
    and hard resource bounds for proposed agent actions.

    The model can never bypass or expand these rules.
    """

    @staticmethod
    def validate(
        action: Any,
        state: ResearchState,
        limits: ResearchLimits,
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate proposed action against schema, semantics, and runtime limits.

        Returns:
            (is_valid, error_message_if_invalid)
        """
        if not isinstance(action, AgentAction):
            return False, f"Action must be an AgentAction instance, got {type(action).__name__}."

        # Verify action type is in allowed whitelist
        raw_type = action.action_type
        if isinstance(raw_type, AgentActionType):
            atype = raw_type
        elif isinstance(raw_type, str):
            try:
                atype = AgentActionType(raw_type.strip().lower())
            except ValueError:
                return (
                    False,
                    f"Unsupported action type: '{raw_type}'. Allowed actions: search, fetch, finish.",
                )
        else:
            return False, f"Invalid action_type representation: {raw_type}."

        # Check action parameter dictionary
        if not isinstance(action.parameters, dict):
            return False, f"Action parameters must be a dictionary, got {type(action.parameters).__name__}."

        # -------------------------------------------------------------
        # 1. SEARCH validation
        # -------------------------------------------------------------
        if atype == AgentActionType.SEARCH:
            query = action.parameters.get("query")
            if not query or not isinstance(query, str) or not query.strip():
                return False, "Search action requires a non-empty 'query' string parameter."

            if state.searches_used >= limits.max_searches:
                return False, f"Maximum search limit reached ({limits.max_searches})."

            if state.evidence_count >= limits.max_evidence:
                return False, f"Maximum evidence capacity reached ({limits.max_evidence})."

            return True, None

        # -------------------------------------------------------------
        # 2. FETCH validation
        # -------------------------------------------------------------
        if atype == AgentActionType.FETCH:
            url = action.parameters.get("url")
            if not url or not isinstance(url, str) or not url.strip():
                return False, "Fetch action requires a non-empty 'url' parameter."

            clean_url = url.strip().lower()
            if not (clean_url.startswith("http://") or clean_url.startswith("https://")):
                return False, f"Fetch action requires a valid HTTP/HTTPS URL, got '{url}'."

            if state.fetches_used >= limits.max_fetches:
                return False, f"Maximum fetch limit reached ({limits.max_fetches})."

            return True, None

        # -------------------------------------------------------------
        # 3. FINISH validation
        # -------------------------------------------------------------
        if atype == AgentActionType.FINISH:
            return True, None

        return False, f"Unsupported action type: '{atype}'."


class DeterministicResearchReasoner(ResearchReasonerInterface):
    """
    Model-neutral deterministic baseline reasoner.
    Generates structured actions (SEARCH, FETCH, FINISH) without calling any external LLM.
    Used as the safe default when no AI model is injected.
    """

    def __init__(self, query_facets: Optional[List[str]] = None):
        self.query_facets = query_facets
        self._facet_index = 0
        self._fetched_urls: Set[str] = set()

    def generate_facets(self, objective: str) -> List[str]:
        """Decompose objective into deterministic query facets."""
        cleaned = objective.strip()
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
        seen = set()
        unique = []
        for f in facets:
            fn = f.strip().lower()
            if fn and fn not in seen:
                seen.add(fn)
                unique.append(f.strip())
        return unique

    def decide_next_action(self, state: ResearchState) -> AgentAction:
        """Propose next structured action based on current state."""
        if self.query_facets is None:
            self.query_facets = self.generate_facets(state.objective)

        # Step A: Perform search if facets remain
        if self._facet_index < len(self.query_facets):
            query = self.query_facets[self._facet_index]
            self._facet_index += 1
            return AgentAction(
                action_type=AgentActionType.SEARCH,
                parameters={"query": query, "max_results": 5},
                reason=f"Gather evidence for facet '{query}'.",
            )

        # Step B: Fetch detailed content for first un-fetched evidence item
        for item in state.evidence:
            if item.url and item.url not in self._fetched_urls:
                self._fetched_urls.add(item.url)
                return AgentAction(
                    action_type=AgentActionType.FETCH,
                    parameters={"url": item.url},
                    reason=f"Deepen evidence by fetching content from {item.url}.",
                )

        # Step C: Complete research
        return AgentAction(
            action_type=AgentActionType.FINISH,
            parameters={},
            reason="All query facets and relevant evidence fetches completed.",
        )


class AdaptiveResearchReasoner(ResearchReasonerInterface):
    """
    Model-neutral adaptive research reasoner (Phase 3.1).
    Evaluates evidence coverage and research gaps to dynamically generate the next
    adaptive query, investigate contradictions, or deepen evidence via fetch.
    """

    def __init__(self):
        self._fetched_urls: Set[str] = set()
        self._issued_queries: Set[str] = set()

    def decide_next_action(self, state: ResearchState) -> AgentAction:
        clean_obj = state.objective.strip()

        # Step 1: Initial query if no evidence collected yet
        if not state.evidence:
            query = self._build_initial_query(clean_obj)
            self._issued_queries.add(query.lower())
            return AgentAction(
                action_type=AgentActionType.SEARCH,
                parameters={"query": query, "max_results": 5},
                reason=f"Initial evidence collection for '{query}'.",
            )

        # Step 2: Adaptive query targeting identified research gaps
        if state.gaps:
            sorted_gaps = sorted(state.gaps, key=lambda g: g.priority)
            for gap in sorted_gaps:
                gap_query = self._build_gap_query(clean_obj, gap.topic)
                if gap_query.lower() not in self._issued_queries:
                    self._issued_queries.add(gap_query.lower())
                    return AgentAction(
                        action_type=AgentActionType.SEARCH,
                        parameters={"query": gap_query, "max_results": 5},
                        reason=f"Adaptive search to cover missing aspect '{gap.topic}': {gap.reason}",
                    )

        # Step 3: Deepen understanding of contradictory claims if detected
        if state.contradictions and state.searches_used < 4:
            unresolved = [c for c in state.contradictions if not c.resolved]
            if unresolved:
                contra = unresolved[0]
                contra_query = f"official {contra.topic} specifications"
                if contra_query.lower() not in self._issued_queries:
                    self._issued_queries.add(contra_query.lower())
                    return AgentAction(
                        action_type=AgentActionType.SEARCH,
                        parameters={"query": contra_query, "max_results": 3},
                        reason=f"Targeted search to clarify contradiction on '{contra.topic}'.",
                    )

        # Step 4: Fetch detailed content for highest-quality un-fetched evidence item
        unfetched = [
            item for item in state.evidence
            if item.url and item.url not in self._fetched_urls and item.metadata.get("type") != "fetch_result"
        ]
        if unfetched:
            assessment_map = {a.evidence_id: (a.quality_score + a.relevance_score) for a in state.assessments}
            unfetched.sort(key=lambda item: assessment_map.get(item.id, 0.5), reverse=True)
            chosen = unfetched[0]
            self._fetched_urls.add(chosen.url)
            return AgentAction(
                action_type=AgentActionType.FETCH,
                parameters={"url": chosen.url},
                reason=f"Fetch rich content from authoritative source {chosen.url}.",
            )

        # Step 5: If all gaps and fetches addressed, or sufficient evidence gathered -> FINISH
        return AgentAction(
            action_type=AgentActionType.FINISH,
            parameters={},
            reason="Research objective covered with sufficient verified evidence.",
        )

    def _build_initial_query(self, objective: str) -> str:
        cleaned = objective.strip()
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

        trailing_patterns = [
            r"\s+and\s+compare\s+(?:several\s+)?sources[.!?]?$",
            r"\s+comparing\s+(?:several\s+)?sources[.!?]?$",
            r"\s+across\s+(?:several|multiple)\s+sources[.!?]?$",
            r"\s+from\s+(?:several|multiple)\s+sources[.!?]?$",
            r"[.!?]+$",
        ]
        for pat in trailing_patterns:
            cleaned = re.sub(pat, "", cleaned, flags=re.IGNORECASE).strip()

        compare_match = re.search(r"^(?:compare\s+)?(.+?)\s+(?:vs\.?|versus|and|with)\s+(.+?)(?:\s+(?:for|on|in)\s+|$)", cleaned, re.IGNORECASE)
        if compare_match:
            side_a = compare_match.group(1).strip()
            return f"{side_a} AI benchmark" if "ai" in cleaned.lower() else side_a

        return cleaned if cleaned else objective.strip()

    def _build_gap_query(self, objective: str, gap_topic: str) -> str:
        gap_clean = gap_topic.strip()
        if any(w in objective.lower() for w in ("ai", "benchmark", "performance")):
            if not any(w in gap_clean.lower() for w in ("benchmark", "performance", "specs", "power", "price")):
                return f"{gap_clean} AI benchmark"
        return f"{gap_clean} benchmark overview" if len(gap_clean.split()) <= 2 else gap_clean


class ResearchAgent:
    """
    Autonomous Research Agent (Phase 3.0).
    Executes a bounded, reasoning-driven research loop.

    Key Architectural Principles:
    - Model-Neutral: Operates against ResearchReasonerInterface; never binds to Gemini or Ollama directly.
    - Safety First: The model proposes AgentAction; the runtime validates, limits, and executes.
    - Provenance-Backed: Accumulates structured EvidenceItem records, builds CitationSet deterministically.
    - Non-Recursive: Never spawns subagents, never loops infinitely, capped by ResearchLimits.
    """

    def __init__(
        self,
        provider: Optional[WebProviderInterface] = None,
        reasoner: Optional[ResearchReasonerInterface] = None,
        synthesizer: Optional[Union[ResearchSynthesizerInterface, Callable[[str, EvidenceSet, CitationSet], str]]] = None,
        limits: Optional[ResearchLimits] = None,
        evaluator: Optional[EvidenceEvaluator] = None,
    ):
        if provider is not None:
            self.provider = provider
        else:
            try:
                from web.default_provider import DefaultWebProvider
                self.provider = DefaultWebProvider()
            except ImportError:
                self.provider = None

        self.evaluator = evaluator if evaluator is not None else EvidenceEvaluator()
        self.reasoner = reasoner if reasoner is not None else AdaptiveResearchReasoner()
        self.synthesizer = synthesizer
        self.limits = (limits or ResearchLimits()).enforce_caps()
        self.validator = ActionValidator()
        self.last_error: Optional[Exception] = None

    def run(
        self,
        objective: str,
        limits: Optional[ResearchLimits] = None,
        raise_on_error: bool = False,
    ) -> ResearchResult:
        """
        Execute the autonomous research loop for the given objective.

        Args:
            objective: Target research topic or query.
            limits: Optional override of runtime limits (defensively capped).
            raise_on_error: Whether unrecoverable provider failures should re-raise.

        Returns:
            Structured ResearchResult containing EvidenceSet, CitationSet,
            ResearchState, and synthesized output with verified citations.
        """
        self.last_error = None
        clean_objective = objective.strip()
        if not clean_objective:
            raise ValueError("Research objective cannot be empty.")

        effective_limits = (limits or self.limits).enforce_caps()

        if self.provider is None:
            err = RuntimeError("Web provider is unavailable.")
            self.last_error = err
            if raise_on_error:
                raise err
            empty_evidence = EvidenceSet(query=clean_objective)
            empty_citations = CitationSet()
            final_state = ResearchState(
                objective=clean_objective,
                iteration=0,
                completed=True,
                stop_reason="STOP_FAILURE",
                status="failed",
                last_error=str(err),
            )
            return ResearchResult(
                objective=clean_objective,
                evidence=empty_evidence,
                citations=empty_citations,
                status="failed",
                output=f"Research failed for: '{clean_objective}'. Web provider is unavailable.",
                state=final_state,
            )

        accumulated_items: List[EvidenceItem] = []
        seen_canonical_urls: Set[str] = set()
        observations: List[ResearchObservation] = []

        searches_used = 0
        fetches_used = 0
        iteration = 0
        invalid_actions_count = 0
        stop_reason = ""
        completed = False
        last_action: Optional[AgentAction] = None
        last_error_str: Optional[str] = None

        while iteration < effective_limits.max_iterations and not completed:
            iteration += 1

            # -------------------------------------------------------------
            # 1. Evidence Evaluation & Gap Analysis
            # -------------------------------------------------------------
            assessments, gaps, contradictions = self.evaluator.evaluate(
                accumulated_items, clean_objective
            )

            current_state = ResearchState(
                objective=clean_objective,
                iteration=iteration,
                searches_used=searches_used,
                fetches_used=fetches_used,
                evidence_count=len(accumulated_items),
                completed=False,
                stop_reason="",
                evidence=tuple(accumulated_items),
                observations=tuple(observations),
                last_action=last_action,
                last_error=last_error_str,
                status="in_progress",
                assessments=assessments,
                gaps=gaps,
                contradictions=contradictions,
            )

            # -------------------------------------------------------------
            # 2. Reasoner Proposes Action
            # -------------------------------------------------------------
            try:
                proposed_action = self.reasoner.decide_next_action(current_state)
                last_action = proposed_action
            except Exception as e:
                self.last_error = e
                last_error_str = f"Reasoner failure: {str(e)}"
                stop_reason = "STOP_FAILURE"
                completed = True
                break

            # -------------------------------------------------------------
            # 3. Runtime Validates Action
            # -------------------------------------------------------------
            is_valid, validation_err = self.validator.validate(
                proposed_action, current_state, effective_limits
            )

            if not is_valid:
                invalid_actions_count += 1
                last_error_str = validation_err
                obs = ResearchObservation(
                    action_type=(
                        proposed_action.action_type.value
                        if isinstance(proposed_action.action_type, AgentActionType)
                        else str(getattr(proposed_action, "action_type", "unknown"))
                    ),
                    success=False,
                    summary=f"Action validation failed: {validation_err}",
                    error=validation_err,
                )
                observations.append(obs)

                if invalid_actions_count >= effective_limits.max_invalid_actions:
                    stop_reason = "STOP_INVALID_ACTION"
                    completed = True
                continue

            # -------------------------------------------------------------
            # 4. Action Execution
            # -------------------------------------------------------------
            action_type = (
                proposed_action.action_type
                if isinstance(proposed_action.action_type, AgentActionType)
                else AgentActionType(str(proposed_action.action_type).lower())
            )

            # --- FINISH ACTION ---
            if action_type == AgentActionType.FINISH:
                if len(accumulated_items) >= effective_limits.min_evidence:
                    stop_reason = "STOP_SUCCESS"
                else:
                    stop_reason = "STOP_INSUFFICIENT_EVIDENCE"
                completed = True
                obs = ResearchObservation(
                    action_type="finish",
                    success=True,
                    summary=f"Research finished by agent ({proposed_action.reason}).",
                    data={"evidence_count": len(accumulated_items)},
                )
                observations.append(obs)
                break

            # --- SEARCH ACTION ---
            if action_type == AgentActionType.SEARCH:
                query = str(proposed_action.parameters.get("query", "")).strip()
                max_res = min(
                    5,
                    max(1, int(proposed_action.parameters.get("max_results", 5))),
                    effective_limits.max_evidence - len(accumulated_items),
                )

                try:
                    results = self.provider.search(
                        query=query,
                        max_results=max_res,
                        timeout_seconds=effective_limits.timeout_seconds,
                    )
                    searches_used += 1

                    new_items_count = 0
                    bounded_titles = []
                    bounded_urls = []
                    for sr in results:
                        c_url = canonicalize_url(sr.url) or sr.url
                        if c_url not in seen_canonical_urls:
                            item = create_evidence_from_search(sr)
                            seen_canonical_urls.add(c_url)
                            accumulated_items.append(item)
                            new_items_count += 1
                            bounded_titles.append(sr.title[:100])
                            bounded_urls.append(sr.url)
                            if len(accumulated_items) >= effective_limits.max_evidence:
                                break

                    obs = ResearchObservation(
                        action_type="search",
                        success=True,
                        summary=f"Search '{query}' returned {len(results)} results ({new_items_count} new).",
                        data={
                            "query": query,
                            "count": len(results),
                            "new_evidence": new_items_count,
                            "titles": bounded_titles[:5],
                            "urls": bounded_urls[:5],
                        },
                    )
                    observations.append(obs)

                except Exception as e:
                    self.last_error = e
                    last_error_str = str(e)
                    obs = ResearchObservation(
                        action_type="search",
                        success=False,
                        summary=f"Search for '{query}' failed: {e}",
                        error=str(e),
                    )
                    observations.append(obs)
                    if len(accumulated_items) == 0 and raise_on_error:
                        raise
                    if len(accumulated_items) == 0:
                        stop_reason = "STOP_FAILURE"
                        completed = True
                        break

            # --- FETCH ACTION ---
            elif action_type == AgentActionType.FETCH:
                url = str(proposed_action.parameters.get("url", "")).strip()
                try:
                    fetch_res = self.provider.fetch(
                        url=url,
                        timeout_seconds=effective_limits.timeout_seconds,
                    )
                    fetches_used += 1
                    fetch_item = create_evidence_from_fetch(fetch_res)

                    c_fetch_url = canonicalize_url(fetch_res.url) or fetch_res.url
                    replaced = False
                    for idx, existing in enumerate(accumulated_items):
                        c_exist = canonicalize_url(existing.url) or existing.url
                        if c_exist == c_fetch_url:
                            accumulated_items[idx] = fetch_item
                            replaced = True
                            break

                    if not replaced and len(accumulated_items) < effective_limits.max_evidence:
                        seen_canonical_urls.add(c_fetch_url)
                        accumulated_items.append(fetch_item)

                    content_preview = (fetch_res.content or "")[:500]
                    obs = ResearchObservation(
                        action_type="fetch",
                        success=True,
                        summary=f"Fetched {url} ({len(fetch_res.content or '')} chars, status {fetch_res.status_code}).",
                        data={
                            "url": url,
                            "title": fetch_res.title[:100],
                            "status_code": fetch_res.status_code,
                            "content_preview": content_preview,
                        },
                    )
                    observations.append(obs)

                except Exception as e:
                    self.last_error = e
                    last_error_str = str(e)
                    obs = ResearchObservation(
                        action_type="fetch",
                        success=False,
                        summary=f"Fetch from '{url}' failed: {e}",
                        error=str(e),
                    )
                    observations.append(obs)
                    if len(accumulated_items) == 0:
                        if raise_on_error:
                            raise
                        stop_reason = "STOP_FAILURE"
                        completed = True
                        break

            # -------------------------------------------------------------
            # 5. Check Runtime Capacity Stops
            # -------------------------------------------------------------
            if len(accumulated_items) >= effective_limits.max_evidence:
                stop_reason = "STOP_LIMIT"
                completed = True
            elif searches_used >= effective_limits.max_searches and fetches_used >= effective_limits.max_fetches:
                stop_reason = (
                    "STOP_SUCCESS"
                    if len(accumulated_items) >= effective_limits.min_evidence
                    else "STOP_LIMIT"
                )
                completed = True
            elif iteration >= effective_limits.max_iterations:
                stop_reason = "STOP_LIMIT"
                completed = True

        # Fallback stop reason determination
        if not stop_reason:
            if len(accumulated_items) >= effective_limits.min_evidence:
                stop_reason = "STOP_SUCCESS"
            elif self.last_error is not None:
                stop_reason = "STOP_FAILURE"
            else:
                stop_reason = "STOP_LIMIT"

        # -------------------------------------------------------------
        # 6. Evidence & Citation Sets Construction
        # -------------------------------------------------------------
        evidence_set = EvidenceSet.from_items(
            accumulated_items,
            query=clean_objective,
            max_items=effective_limits.max_evidence,
        )
        citation_set = CitationSet.from_evidence_set(evidence_set)

        # Citation validation: verify every citation maps to true evidence
        is_cit_valid, cit_errors = validate_citation_references(citation_set, evidence_set)
        if not is_cit_valid:
            # Defensive guard: invalid citations are logged and cleaned
            valid_evidence_ids = {e.id for e in evidence_set.items}
            cleaned_citations = [c for c in citation_set.citations if c.evidence_id in valid_evidence_ids]
            citation_set = CitationSet(citations=tuple(cleaned_citations))

        # -------------------------------------------------------------
        # 7. Status Resolution
        # -------------------------------------------------------------
        if len(evidence_set) == 0:
            status = "failed"
        elif stop_reason in ("STOP_SUCCESS", "STOP_LIMIT") and len(evidence_set) >= effective_limits.min_evidence:
            status = "completed"
        elif stop_reason == "STOP_FAILURE":
            status = "failed"
        else:
            status = "partial"

        if status == "failed" and raise_on_error and self.last_error is not None:
            raise self.last_error

        # -------------------------------------------------------------
        # 8. Final Evaluation & Synthesis
        # -------------------------------------------------------------
        final_assessments, final_gaps, final_contradictions = self.evaluator.evaluate(
            evidence_set.items, clean_objective
        )

        final_state = ResearchState(
            objective=clean_objective,
            iteration=iteration,
            searches_used=searches_used,
            fetches_used=fetches_used,
            evidence_count=len(evidence_set),
            completed=True,
            stop_reason=stop_reason,
            evidence=tuple(evidence_set.items),
            observations=tuple(observations),
            last_action=last_action,
            last_error=last_error_str,
            status=status,
            assessments=final_assessments,
            gaps=final_gaps,
            contradictions=final_contradictions,
        )

        output = self._synthesize(
            clean_objective, evidence_set, citation_set, final_state, status, stop_reason
        )

        return ResearchResult(
            objective=clean_objective,
            evidence=evidence_set,
            citations=citation_set,
            status=status,
            output=output,
            state=final_state,
            assessments=final_assessments,
            gaps=final_gaps,
            contradictions=final_contradictions,
        )

    def _synthesize(
        self,
        objective: str,
        evidence: EvidenceSet,
        citations: CitationSet,
        state: ResearchState,
        status: str,
        stop_reason: str,
    ) -> str:
        """
        Synthesize final user-facing text from verified evidence.
        Priority:
        1. Injected synthesizer callback / interface
        2. Reasoner.synthesize hook (if provided)
        3. Deterministic synthesis fallback with quality ranking and contradiction awareness
        """
        # 1. Injected synthesizer
        if self.synthesizer is not None:
            if isinstance(self.synthesizer, ResearchSynthesizerInterface):
                return self.synthesizer.synthesize(objective, evidence, citations)
            elif callable(self.synthesizer):
                return self.synthesizer(objective, evidence, citations)

        # 2. Reasoner hook
        if hasattr(self.reasoner, "synthesize") and callable(self.reasoner.synthesize):
            try:
                res_output = self.reasoner.synthesize(state, citations)
                if res_output:
                    return res_output
            except Exception:
                pass

        # 3. Deterministic synthesis fallback
        if evidence.is_empty:
            if status == "failed":
                return f"Research failed for: '{objective}'. Unable to gather required evidence ({stop_reason})."
            return f"No research findings found for: '{objective}'."

        # Order evidence items by composite quality & relevance
        assessment_map = {
            a.evidence_id: (a.quality_score + a.relevance_score)
            for a in state.assessments
        }
        sorted_items = sorted(
            evidence.items,
            key=lambda it: assessment_map.get(it.id, 0.5),
            reverse=True,
        )

        lines: List[str] = []
        if status == "partial":
            lines.append("[Note: Research concluded with partial results due to execution limits.]\n")

        lines.append(f"Research findings for '{objective}':\n")
        lines.append("Key Findings:")
        for idx, item in enumerate(sorted_items, start=1):
            citation = citations.get_by_evidence_id(item.id)
            marker = citation.render_marker() if citation else f"[{idx}]"
            content_snippet = (item.content or "").strip()
            if len(content_snippet) > 200:
                content_snippet = content_snippet[:197] + "..."
            lines.append(f"- {item.title}: {content_snippet} {marker}")

        # Phase 3.1: Explicitly surface unresolved contradictions
        if state.contradictions:
            unresolved = [c for c in state.contradictions if not c.resolved]
            if unresolved:
                lines.append("")
                lines.append("Contradictions & Discrepancies:")
                for c in unresolved:
                    lines.append(f"- {c.topic}:")
                    for claim in c.conflicting_claims:
                        lines.append(f"  * {claim}")

        lines.append("")
        lines.append(citations.render_sources_block())

        return "\n".join(lines)
