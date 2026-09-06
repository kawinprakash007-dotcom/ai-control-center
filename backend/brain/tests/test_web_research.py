import pytest
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any

from core.interfaces.web_interface import WebProviderInterface, WebProviderError
from core.models.web import (
    SearchResult,
    FetchResult,
    EvidenceItem,
    EvidenceSet,
    Citation,
    CitationSet,
    ResearchState,
    ResearchResult,
    create_evidence_from_search,
    create_evidence_from_fetch,
)
from core.models.request import Request
from core.models.decision import Decision, CapabilityType, ExecutionMode
from core.models.plan import Plan
from core.models.task import Task
from core.models.result import Result
from web.research_coordinator import WebResearchCoordinator
from tools.web_capability import WebCapability
from tools.executor import Executor
from brain.decision_engine import StandardDecisionEngine
from brain.planning import StandardPlanner
from brain.verification import StandardVerifier
from brain.pipeline import StandardPipeline


# ============================================================================
# FAKE WEB PROVIDER FOR CONTROLLED RESEARCH TESTING
# ============================================================================

class MultiFacetFakeWebProvider(WebProviderInterface):
    """
    Fake web provider that yields distinct search results for different query facets
    and provides canned page fetch content.
    """

    def __init__(self, should_fail: bool = False, fail_on_fetch: bool = False):
        self.should_fail = should_fail
        self.fail_on_fetch = fail_on_fetch
        self.searches_called: List[str] = []
        self.fetches_called: List[str] = []

    def search(
        self, query: str, max_results: int = 5, timeout_seconds: float = 10.0
    ) -> List[SearchResult]:
        if self.should_fail:
            raise WebProviderError("Provider search failure: network unreachable.")

        self.searches_called.append(query)
        q_lower = query.lower()

        if "benchmark" in q_lower:
            return [
                SearchResult(
                    title="Raspberry Pi 5 AI Benchmarks 2026",
                    url="https://benchmarks.example.org/rpi5-ai",
                    snippet="Detailed benchmark scores: Hailo-8L reaches 13 TOPS.",
                    source="benchmarks.example.org",
                ),
                SearchResult(
                    title="Edge AI Benchmark Comparison",
                    url="https://edgeai.org/rpi5-vs-jetson",
                    snippet="Comparing RPi 5 AI Kit against Jetson Nano.",
                    source="edgeai.org",
                ),
            ]
        elif "specifications" in q_lower or "spec" in q_lower:
            return [
                SearchResult(
                    title="Raspberry Pi 5 Hardware Specifications",
                    url="https://datasheets.raspberrypi.com/rpi5/spec.pdf",
                    snippet="Official hardware specs: PCIe 2.0 interface for AI HAT+.",
                    source="datasheets.raspberrypi.com",
                ),
            ]
        else:
            return [
                SearchResult(
                    title="Raspberry Pi 5 AI Kit Official Announcement",
                    url="https://www.raspberrypi.com/news/raspberry-pi-ai-kit-available-now/",
                    snippet="Raspberry Pi introduces the M.2 HAT+ and Hailo AI module.",
                    source="raspberrypi.com",
                ),
                SearchResult(
                    title="Raspberry Pi 5 Documentation",
                    url="https://www.raspberrypi.com/documentation/computers/raspberry-pi-5.html",
                    snippet="Comprehensive documentation and getting started guide.",
                    source="raspberrypi.com",
                ),
            ]

    def fetch(self, url: str, timeout_seconds: float = 10.0) -> FetchResult:
        if self.should_fail or self.fail_on_fetch:
            raise WebProviderError(f"Provider fetch failure for url: {url}")

        self.fetches_called.append(url)
        return FetchResult(
            url=url,
            title=f"Page Title for {url}",
            content=f"Deep fetched text content from {url}. Full detailed paragraphs explaining architecture.",
            status_code=200,
            source="fetched.example.com",
        )


# ============================================================================
# PHASE 2.9.6 CITATION TESTS
# ============================================================================

def test_citation_references_real_evidence_item():
    """1. Citation strictly references an actual EvidenceItem and preserves ID and URL."""
    item = EvidenceItem(
        id="ev-12345",
        title="Raspberry Pi Documentation",
        url="https://www.raspberrypi.com/docs/",
        domain="raspberrypi.com",
        content="Documentation content",
        retrieved_at="2026-09-06T12:00:00Z",
    )
    evidence_set = EvidenceSet(items=(item,), query="test")
    citations = CitationSet.from_evidence_set(evidence_set)

    assert len(citations) == 1
    c = citations[0]
    assert c.index == 1
    assert c.evidence_id == "ev-12345"
    assert c.url == "https://www.raspberrypi.com/docs/"
    assert c.title == "Raspberry Pi Documentation"
    assert c.domain == "raspberrypi.com"


def test_citation_never_invents_url():
    """2. Citation never invents a URL; all citations map directly to evidence items."""
    item1 = EvidenceItem(
        id="ev-1",
        title="Source 1",
        url="https://source1.org/info",
        domain="source1.org",
        content="Snippet 1",
        retrieved_at="2026-09-06T12:00:00Z",
    )
    item2 = EvidenceItem(
        id="ev-2",
        title="Source 2",
        url="https://source2.org/info",
        domain="source2.org",
        content="Snippet 2",
        retrieved_at="2026-09-06T12:00:00Z",
    )
    evidence_set = EvidenceSet(items=(item1, item2), query="test")
    citations = CitationSet.from_evidence_set(evidence_set)

    evidence_urls = {item.url for item in evidence_set.items}
    for c in citations:
        assert c.url in evidence_urls


def test_deterministic_citation_numbering():
    """3. Citation indices are deterministic, 1-based, and sequential."""
    items = [
        EvidenceItem(
            id=f"ev-{i}",
            title=f"Source {i}",
            url=f"https://source{i}.org",
            domain=f"source{i}.org",
            content=f"Content {i}",
            retrieved_at="2026-09-06T12:00:00Z",
        )
        for i in range(1, 4)
    ]
    evidence_set = EvidenceSet(items=tuple(items))
    citations = CitationSet.from_evidence_set(evidence_set)

    assert [c.index for c in citations] == [1, 2, 3]


def test_citation_formatting_and_rendering():
    """4 & 5. Citation renders inline marker and references deterministically preserving title, domain, URL."""
    c = Citation(
        index=1,
        evidence_id="ev-abc",
        url="https://docs.python.org/3/",
        title="Python 3 Documentation",
        domain="docs.python.org",
    )
    assert c.render_marker() == "[1]"
    ref = c.render_reference()
    assert "[1] Python 3 Documentation" in ref
    assert "https://docs.python.org/3/" in ref

    c_set = CitationSet(citations=(c,))
    sources_block = c_set.render_sources_block()
    assert sources_block.startswith("Sources:\n")
    assert "[1] Python 3 Documentation" in sources_block


# ============================================================================
# PHASE 2.9.7 BOUNDED RESEARCH LOOP TESTS
# ============================================================================

def test_query_facet_generation_deterministic():
    """Deterministic decomposition of complex user research queries."""
    coordinator = WebResearchCoordinator()
    facets = coordinator.generate_query_facets(
        "Research the latest Raspberry Pi 5 AI performance and compare several sources."
    )
    assert len(facets) == 3
    assert facets[0] == "Raspberry Pi 5 AI performance"
    assert "benchmark comparison" in facets[1]
    assert "specifications overview" in facets[2]


def test_research_objective_accepted_and_state_initialized():
    """7. Research coordinator accepts objective and populates state."""
    provider = MultiFacetFakeWebProvider()
    coordinator = WebResearchCoordinator(provider=provider)
    result = coordinator.research("Raspberry Pi 5 AI", max_iterations=1, max_searches=1, max_fetches=0)

    assert result.objective == "Raspberry Pi 5 AI"
    assert result.state is not None
    assert result.state.objective == "Raspberry Pi 5 AI"
    assert result.state.completed is True
    assert result.state.searches_used == 1
    assert result.state.fetches_used == 0


def test_one_step_research():
    """8. Bounded execution with single search step completes normally."""
    provider = MultiFacetFakeWebProvider()
    coordinator = WebResearchCoordinator(provider=provider)
    result = coordinator.research("Raspberry Pi 5 AI", max_iterations=1, max_searches=1, max_fetches=0)

    assert result.status == "completed"
    assert len(result.evidence) >= 1
    assert len(result.citations) == len(result.evidence)
    assert "[1]" in result.output
    assert "Sources:" in result.output


def test_multi_step_search():
    """9 & 17. Multi-step research searches multiple facets across iterations and accumulates evidence."""
    provider = MultiFacetFakeWebProvider()
    coordinator = WebResearchCoordinator(provider=provider)
    result = coordinator.research(
        "Raspberry Pi 5 AI performance",
        max_iterations=3,
        max_searches=3,
        max_fetches=0,
    )

    assert result.status == "completed"
    assert result.state.searches_used >= 2
    # Multiple queries were issued
    assert len(provider.searches_called) >= 2
    # Accumulated evidence from multiple searches
    assert len(result.evidence) >= 3


def test_search_and_fetch_workflow():
    """10. Search + Fetch deepens content by fetching top search URLs."""
    provider = MultiFacetFakeWebProvider()
    coordinator = WebResearchCoordinator(provider=provider)
    result = coordinator.research(
        "Raspberry Pi 5 AI",
        max_iterations=2,
        max_searches=1,
        max_fetches=1,
    )

    assert result.status == "completed"
    assert result.state.fetches_used == 1
    assert len(provider.fetches_called) == 1
    # Check that fetched URL had content updated
    fetched_url = provider.fetches_called[0]
    matching_ev = next(item for item in result.evidence if item.url == fetched_url)
    assert "Deep fetched text content" in matching_ev.content
    assert matching_ev.metadata.get("type") == "fetch_result"


def test_bounded_iteration_stopping():
    """11. Research stops strictly when max_iterations is reached."""
    provider = MultiFacetFakeWebProvider()
    coordinator = WebResearchCoordinator(provider=provider)
    result = coordinator.research("Raspberry Pi 5", max_iterations=1, max_searches=5, max_fetches=5)

    assert result.state.iteration == 1
    assert result.state.completed is True


def test_max_search_limit():
    """12. Searches used strictly respects max_searches limit."""
    provider = MultiFacetFakeWebProvider()
    coordinator = WebResearchCoordinator(provider=provider)
    result = coordinator.research("Raspberry Pi 5 AI", max_iterations=5, max_searches=2, max_fetches=0)

    assert result.state.searches_used <= 2
    assert len(provider.searches_called) <= 2


def test_max_fetch_limit():
    """13. Fetches used strictly respects max_fetches limit."""
    provider = MultiFacetFakeWebProvider()
    coordinator = WebResearchCoordinator(provider=provider)
    result = coordinator.research("Raspberry Pi 5 AI", max_iterations=5, max_searches=3, max_fetches=1)

    assert result.state.fetches_used <= 1
    assert len(provider.fetches_called) <= 1


def test_max_evidence_limit():
    """14. Evidence items accumulated strictly respects max_evidence limit."""
    provider = MultiFacetFakeWebProvider()
    coordinator = WebResearchCoordinator(provider=provider)
    result = coordinator.research(
        "Raspberry Pi 5 AI",
        max_iterations=3,
        max_searches=3,
        max_fetches=0,
        max_evidence=2,
    )

    assert len(result.evidence) <= 2
    assert len(result.citations) <= 2


def test_duplicate_evidence_elimination():
    """6. Duplicate URLs from multiple search facets are eliminated."""
    class DuplicateProvider(WebProviderInterface):
        def search(self, query: str, max_results: int = 5, timeout_seconds: float = 10.0) -> List[SearchResult]:
            return [
                SearchResult(title="Dup Title", url="https://example.org/dup/", snippet="Snippet 1"),
                SearchResult(title="Dup Title WWW", url="https://www.example.org/dup", snippet="Snippet 2"),
            ]
        def fetch(self, url: str, timeout_seconds: float = 10.0) -> FetchResult:
            return FetchResult(url=url, title="T", content="C", status_code=200)

    provider = DuplicateProvider()
    coordinator = WebResearchCoordinator(provider=provider)
    result = coordinator.research("Test Dup", max_iterations=2, max_searches=2, max_fetches=0)

    # Canonical URL deduplication should leave only 1 evidence item
    assert len(result.evidence) == 1


def test_provider_failure_handling():
    """15. Provider failure causes stop with STOP_FAILURE status."""
    failing_provider = MultiFacetFakeWebProvider(should_fail=True)
    coordinator = WebResearchCoordinator(provider=failing_provider)

    # Graceful handling when raise_on_error=False
    result = coordinator.research("Raspberry Pi 5", raise_on_error=False)
    assert result.status == "failed"
    assert result.state.stop_reason == "STOP_FAILURE"
    assert "Research failed" in result.output

    # Raising when raise_on_error=True
    with pytest.raises(WebProviderError):
        coordinator.research("Raspberry Pi 5", raise_on_error=True)


def test_incomplete_research_handling():
    """16. When stopping with zero evidence, status is failed."""
    class EmptyProvider(WebProviderInterface):
        def search(self, query: str, max_results: int = 5, timeout_seconds: float = 10.0) -> List[SearchResult]:
            return []
        def fetch(self, url: str, timeout_seconds: float = 10.0) -> FetchResult:
            return FetchResult(url=url, title="", content="", status_code=404)

    coordinator = WebResearchCoordinator(provider=EmptyProvider())
    result = coordinator.research("Obscure Nonexistent Topic", min_evidence=1)
    assert result.status == "failed"
    assert len(result.evidence) == 0


def test_session_and_context_isolation():
    """18. Multiple research invocations remain completely isolated."""
    provider = MultiFacetFakeWebProvider()
    coordinator = WebResearchCoordinator(provider=provider)

    res1 = coordinator.research("Topic A", max_iterations=1, max_searches=1, max_fetches=0)
    res2 = coordinator.research("Topic B", max_iterations=1, max_searches=1, max_fetches=0)

    assert res1.objective == "Topic A"
    assert res2.objective == "Topic B"
    assert res1.state is not res2.state


def test_research_reuses_web_capability_and_provider():
    """21 & 22. WebCapability delegates action='research' to WebResearchCoordinator reusing same provider."""
    provider = MultiFacetFakeWebProvider()
    capability = WebCapability(provider=provider)

    out = capability({"action": "research", "query": "Raspberry Pi 5 AI Kit"})
    assert isinstance(out, str)
    assert "Raspberry Pi 5 AI Kit" in out
    assert "Sources:" in out

    # Check that capability tracked research result and evidence
    res = capability.get_research_result()
    assert res is not None
    assert isinstance(res, ResearchResult)
    assert res.status == "completed"
    assert capability.get_evidence() is res.evidence


def test_custom_synthesizer_injection():
    """Custom synthesizer callback injection."""
    provider = MultiFacetFakeWebProvider()
    custom_called = False

    def mock_synthesizer(objective: str, ev: EvidenceSet, cit: CitationSet) -> str:
        nonlocal custom_called
        custom_called = True
        return f"CUSTOM_SYNTHESIS for {objective} with {len(ev)} items"

    coordinator = WebResearchCoordinator(provider=provider, synthesizer=mock_synthesizer)
    result = coordinator.research("Test Custom Synth", max_iterations=1, max_searches=1, max_fetches=0)

    assert custom_called is True
    assert "CUSTOM_SYNTHESIS for Test Custom Synth" in result.output


def test_research_state_serialization():
    """30. ResearchState and ResearchResult to_dict serialization."""
    state = ResearchState(
        objective="Pi 5",
        iteration=2,
        searches_used=2,
        fetches_used=1,
        evidence_count=3,
        completed=True,
        stop_reason="STOP_SUCCESS",
    )
    d = state.to_dict()
    assert d["objective"] == "Pi 5"
    assert d["searches_used"] == 2
    assert d["stop_reason"] == "STOP_SUCCESS"

    result = ResearchResult(
        objective="Pi 5",
        evidence=EvidenceSet(),
        citations=CitationSet(),
        status="completed",
        output="Result text",
        state=state,
    )
    rd = result.to_dict()
    assert rd["status"] == "completed"
    assert rd["state"]["completed"] is True


# ============================================================================
# PIPELINE & DECISION ENGINE INTEGRATION TESTS
# ============================================================================

def test_decision_engine_detects_research_intent():
    """StandardDecisionEngine recognizes research queries and sets web_research goal."""
    engine = StandardDecisionEngine()
    req = Request(
        id="req-1",
        session_id="s1",
        timestamp=datetime.now(timezone.utc),
        original_text="Research the latest Raspberry Pi 5 AI performance and compare several sources.",
        normalized_text="research the latest raspberry pi 5 ai performance and compare several sources",
    )
    decision = engine.decide(req)

    assert decision.primary_goal == "web_research"
    assert CapabilityType.WEB in decision.required_capabilities
    assert decision.routing_hints.get("action") == "research"
    assert "Raspberry Pi 5 AI performance" in decision.routing_hints.get("query", "")


def test_planner_creates_web_research_task():
    """StandardPlanner maps web_research goal to Task with action 'Web Research'."""
    planner = StandardPlanner()
    decision = Decision(
        request_id="req-1",
        primary_goal="web_research",
        required_capabilities=[CapabilityType.WEB],
        execution_mode=ExecutionMode.SINGLE_STEP,
        confidence=0.90,
        reasoning="Research identified",
        routing_hints={"action": "research", "query": "Raspberry Pi 5 AI"},
    )
    plan = planner.plan(decision)

    assert len(plan.steps) == 1
    task = plan.steps[0]
    assert task.type == "web"
    assert task.action == "Web Research"
    assert task.parameters.get("action") == "research"
    assert task.parameters.get("query") == "Raspberry Pi 5 AI"


def test_executor_transports_research_result():
    """27. Executor attaches ResearchResult in Result.data."""
    provider = MultiFacetFakeWebProvider()
    capability = WebCapability(provider=provider)
    executor = Executor()

    task = Task(
        id=1,
        type="web",
        action="Web Research",
        tool="web",
        parameters={"action": "research", "query": "Raspberry Pi 5 AI"},
        status="pending",
    )

    res = executor.execute(capability, task)
    assert res.success is True
    assert isinstance(res.data, ResearchResult)
    assert res.data.status == "completed"
    assert len(res.data.citations) >= 1


def test_pipeline_executes_web_research_end_to_end():
    """End-to-end StandardPipeline execution of research directive."""
    provider = MultiFacetFakeWebProvider()
    pipeline = StandardPipeline(web_provider=provider)

    response = pipeline.run("Research the latest Raspberry Pi 5 AI performance and compare several sources.")

    assert isinstance(response, str)
    assert "Raspberry Pi 5 AI performance" in response
    assert "[1]" in response
    assert "Sources:" in response


def test_no_sqlite_dependency():
    """23. Research coordinator operates without importing or requiring SQLite."""
    import sys
    coord_mod = sys.modules.get("web.research_coordinator")
    assert coord_mod is not None
    assert not hasattr(coord_mod, "sqlite3")


def test_no_memory_dependency():
    """24. Research coordinator operates independently of MemoryServiceInterface or store."""
    provider = MultiFacetFakeWebProvider()
    coordinator = WebResearchCoordinator(provider=provider)
    res = coordinator.research("Raspberry Pi 5", max_iterations=1, max_searches=1, max_fetches=0)
    assert res.status == "completed"


def test_no_direct_ollama_dependency():
    """25. Deterministic synthesis operates without Ollama or LLM calls."""
    provider = MultiFacetFakeWebProvider()
    coordinator = WebResearchCoordinator(provider=provider)
    res = coordinator.research("Raspberry Pi 5", max_iterations=1, max_searches=1, max_fetches=0)
    assert "ollama" not in str(res.output).lower()
    assert "[1]" in res.output
