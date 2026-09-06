"""Tests for Phase 3.1 Adaptive Research Reasoner & Loop."""

import pytest
from typing import List, Optional
from datetime import datetime, timezone

from core.interfaces.web_interface import WebProviderInterface
from core.models.research import (
    AgentAction,
    AgentActionType,
    EvidenceAssessment,
    ResearchGap,
    Contradiction,
    ResearchLimits,
)
from core.models.web import (
    SearchResult,
    FetchResult,
    EvidenceItem,
    EvidenceSet,
    ResearchState,
    ResearchResult,
)
from web.research_agent import (
    ResearchAgent,
    AdaptiveResearchReasoner,
    DeterministicResearchReasoner,
)
from web.evidence_evaluator import EvidenceEvaluator
from tools.web_capability import WebCapability


class FakeWebProvider(WebProviderInterface):
    """Controlled fake web provider for adaptive research loop tests."""

    def __init__(self, search_responses=None, fetch_responses=None):
        self.search_responses = search_responses or {}
        self.fetch_responses = fetch_responses or {}
        self.searches_called: List[str] = []
        self.fetches_called: List[str] = []

    def search(self, query: str, max_results: int = 5, timeout_seconds: float = 10.0) -> List[SearchResult]:
        self.searches_called.append(query)
        if query in self.search_responses:
            return self.search_responses[query]
        # Default mock search result
        return [
            SearchResult(
                title=f"Result for {query}",
                url=f"https://example.org/{abs(hash(query)) % 1000}",
                snippet=f"Information snippet about {query} with metrics.",
            )
        ]

    def fetch(self, url: str, timeout_seconds: float = 10.0) -> FetchResult:
        self.fetches_called.append(url)
        if url in self.fetch_responses:
            return self.fetch_responses[url]
        return FetchResult(
            url=url,
            title=f"Fetched Title for {url}",
            content=f"Detailed full page content fetched from {url}. Contains comprehensive benchmark data.",
            status_code=200,
        )


class TestAdaptiveQueryGeneration:
    """Test adaptive query generation based on gaps and contradictions."""

    def test_initial_query_comparative_objective(self):
        """Initial query targets comparative aspect."""
        reasoner = AdaptiveResearchReasoner()
        state = ResearchState(
            objective="Compare PostgreSQL vs MySQL for high-concurrency read workloads"
        )
        action = reasoner.decide_next_action(state)
        assert action.action_type == AgentActionType.SEARCH
        assert "postgresql" in action.parameters["query"].lower()

    def test_gap_directed_query_generation(self):
        """When gaps exist, next query directly addresses highest-priority gap."""
        reasoner = AdaptiveResearchReasoner()
        item = EvidenceItem(
            id="ev-rpi",
            title="Benchmarks",
            url="https://example.com/rpi5-bench",
            domain="example.com",
            content="Raspberry Pi 5 runs basic models at 5 FPS.",
            retrieved_at=datetime.now(timezone.utc).isoformat(),
        )
        # Inject gaps with priority 1 (highest) and 2
        gaps = (
            ResearchGap(
                topic="NPU accelerator",
                reason="Missing dedicated NPU details",
                priority=1,
            ),
            ResearchGap(
                topic="software ecosystem",
                reason="Missing runtime framework details",
                priority=2,
            ),
        )
        state = ResearchState(
            objective="Raspberry Pi 5 AI performance",
            evidence=(item,),
            gaps=gaps,
        )

        action = reasoner.decide_next_action(state)
        assert action.action_type == AgentActionType.SEARCH
        # Highest priority gap (priority 1) topic should be addressed
        assert "npu" in action.parameters["query"].lower()

    def test_contradiction_directed_query_generation(self):
        """When unresolved contradiction exists and fetches used, targets clarifying search."""
        reasoner = AdaptiveResearchReasoner()
        reasoner._fetched_urls.add("https://source1.com/a")
        reasoner._fetched_urls.add("https://source2.com/b")

        item1 = EvidenceItem(
            id="ev-1",
            title="Source 1",
            url="https://source1.com/a",
            domain="source1.com",
            content="Model A latency is 10 ms.",
            retrieved_at=datetime.now(timezone.utc).isoformat(),
            metadata={"type": "fetch_result"},
        )
        item2 = EvidenceItem(
            id="ev-2",
            title="Source 2",
            url="https://source2.com/b",
            domain="source2.com",
            content="Model A latency is 50 ms.",
            retrieved_at=datetime.now(timezone.utc).isoformat(),
            metadata={"type": "fetch_result"},
        )
        contra = Contradiction(
            topic="inference latency",
            evidence_ids=(item1.id, item2.id),
            conflicting_claims=("10 ms", "50 ms"),
            resolved=False,
        )
        state = ResearchState(
            objective="Model A latency",
            evidence=(item1, item2),
            contradictions=(contra,),
            fetches_used=2,
            searches_used=1,
        )

        action = reasoner.decide_next_action(state)
        assert action.action_type == AgentActionType.SEARCH
        assert "inference latency" in action.parameters["query"].lower() or "specifications" in action.parameters["query"].lower()

    def test_adaptive_fetch_selection(self):
        """Fetch selects highest-quality un-fetched URL."""
        reasoner = AdaptiveResearchReasoner()
        item1 = EvidenceItem(
            id="ev-1",
            title="Tips",
            url="https://low-quality-blog.com/tips",
            domain="low-quality-blog.com",
            content="Blog post snippet.",
            retrieved_at=datetime.now(timezone.utc).isoformat(),
        )
        item2 = EvidenceItem(
            id="ev-2",
            title="Python Official Docs",
            url="https://docs.python.org/3/whatsnew/3.12.html",
            domain="docs.python.org",
            content="Official documentation overview.",
            retrieved_at=datetime.now(timezone.utc).isoformat(),
        )
        assessments = (
            EvidenceAssessment(
                evidence_id=item1.id,
                quality_score=0.3,
                relevance_score=0.4,
            ),
            EvidenceAssessment(
                evidence_id=item2.id,
                quality_score=0.9,
                relevance_score=0.9,
            ),
        )
        state = ResearchState(
            objective="Python 3.12 optimizations",
            evidence=(item1, item2),
            assessments=assessments,
        )

        action = reasoner.decide_next_action(state)
        assert action.action_type == AgentActionType.FETCH
        assert action.parameters["url"] == "https://docs.python.org/3/whatsnew/3.12.html"

    def test_stopping_intelligence_sufficient_coverage(self):
        """When all gaps addressed and evidence items fetched or reviewed, FINISH is proposed."""
        reasoner = AdaptiveResearchReasoner()
        item = EvidenceItem(
            id="ev-guide",
            title="Guide",
            url="https://example.org/guide",
            domain="example.org",
            content="Comprehensive guide.",
            retrieved_at=datetime.now(timezone.utc).isoformat(),
            metadata={"type": "fetch_result"},
        )
        reasoner._fetched_urls.add("https://example.org/guide")
        state = ResearchState(
            objective="Topic guide",
            evidence=(item,),
            gaps=(),
            contradictions=(),
        )

        action = reasoner.decide_next_action(state)
        assert action.action_type == AgentActionType.FINISH


class TestResearchAgentAdaptiveExecution:
    """Test ResearchAgent using AdaptiveResearchReasoner and EvidenceEvaluator."""

    def test_full_adaptive_loop_execution(self):
        """Full execution populates assessments, gaps, and completes with valid result."""
        search_res = {
            "postgresql AI benchmark": [
                SearchResult(
                    title="PostgreSQL Official",
                    url="https://postgresql.org/about",
                    snippet="PostgreSQL supports MVCC, JSONB, and advanced query planning.",
                ),
                SearchResult(
                    title="Database Comparison Benchmark",
                    url="https://benchmark.org/pg-vs-mysql",
                    snippet="Comparative benchmark reveals PostgreSQL throughput and latency.",
                ),
            ]
        }
        provider = FakeWebProvider(search_responses=search_res)
        agent = ResearchAgent(
            provider=provider,
            reasoner=AdaptiveResearchReasoner(),
            limits=ResearchLimits(max_iterations=3, max_evidence=5),
        )

        result = agent.run("Compare PostgreSQL vs MySQL performance")

        assert isinstance(result, ResearchResult)
        assert result.state.iteration >= 1
        assert len(result.evidence) >= 1
        assert len(result.assessments) > 0
        assert all(isinstance(a, EvidenceAssessment) for a in result.assessments)

    def test_unresolved_contradiction_reported_in_synthesis(self):
        """Contradictions are formatted into the final synthesis output."""
        contradicting_search = {
            "system x throughput": [
                SearchResult(
                    title="Source A Benchmarks",
                    url="https://source-a.org/bench",
                    snippet="System X throughput is 1000 requests per second under peak load.",
                ),
                SearchResult(
                    title="Source B Benchmarks",
                    url="https://source-b.org/bench",
                    snippet="System X throughput is 5000 requests per second under peak load.",
                ),
            ]
        }
        provider = FakeWebProvider(search_responses=contradicting_search)
        agent = ResearchAgent(
            provider=provider,
            reasoner=AdaptiveResearchReasoner(),
            limits=ResearchLimits(max_iterations=2, max_evidence=5),
        )

        result = agent.run("system x throughput")

        assert len(result.contradictions) >= 1
        assert "Contradictions & Discrepancies" in result.output
        assert "1000 requests per second" in result.output
        assert "5000 requests per second" in result.output

    def test_web_capability_research_action_compatibility(self):
        """WebCapability execute(action='research') integrates smoothly with Phase 3.1."""
        provider = FakeWebProvider()
        capability = WebCapability(provider=provider)
        response_text = capability({
            "action": "research",
            "query": "microcontroller benchmarks",
            "max_iterations": 2,
        })
        assert isinstance(response_text, str)
        assert len(response_text) > 0
        res = capability.get_research_result()
        assert res is not None
        assert isinstance(res, ResearchResult)

    def test_runtime_limits_override_adaptive_reasoner(self):
        """Runtime hard bounds terminate the loop even if reasoner still wants to search."""
        provider = FakeWebProvider()
        agent = ResearchAgent(
            provider=provider,
            reasoner=AdaptiveResearchReasoner(),
            limits=ResearchLimits(max_iterations=2),
        )

        result = agent.run("endless research topic")
        assert result.state.iteration <= 2
        assert result.state.completed is True

    def test_model_neutrality_preserved(self):
        """Ensure no vendor model names appear in reasoner code or docstrings."""
        import inspect
        reasoner_source = inspect.getsource(AdaptiveResearchReasoner)
        evaluator_source = inspect.getsource(EvidenceEvaluator)

        for disallowed in ["gemini", "claude", "ollama", "openai", "gpt-4"]:
            assert disallowed not in reasoner_source.lower(), f"Disallowed string '{disallowed}' in reasoner"
            assert disallowed not in evaluator_source.lower(), f"Disallowed string '{disallowed}' in evaluator"
