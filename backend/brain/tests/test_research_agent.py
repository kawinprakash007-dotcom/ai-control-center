import pytest
from typing import List, Optional, Dict, Any

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
)
from core.models.web import (
    SearchResult,
    FetchResult,
    EvidenceItem,
    EvidenceSet,
    Citation,
    CitationSet,
    ResearchState,
    ResearchResult,
    validate_citation_references,
)
from web.research_agent import (
    ResearchAgent,
    ActionValidator,
    DeterministicResearchReasoner,
)
from tools.web_capability import WebCapability


# ============================================================================
# FAKES FOR RESEARCH AGENT TESTING
# ============================================================================

class FakeWebProvider(WebProviderInterface):
    """Controlled fake web provider for testing the autonomous research agent."""

    def __init__(self, should_fail: bool = False, fail_on_fetch: bool = False):
        self.should_fail = should_fail
        self.fail_on_fetch = fail_on_fetch
        self.searches_called: List[str] = []
        self.fetches_called: List[str] = []

    def search(
        self, query: str, max_results: int = 5, timeout_seconds: float = 10.0
    ) -> List[SearchResult]:
        if self.should_fail:
            raise WebProviderError("Provider search error: network unreachable.")

        self.searches_called.append(query)
        q = query.lower()

        if "jetson" in q:
            return [
                SearchResult(
                    title="Jetson Orin Nano AI Performance",
                    url="https://nvidia.example.org/orin-nano",
                    snippet="Jetson Orin Nano delivers up to 40 TOPS INT8 computation.",
                    source="nvidia.example.org",
                )
            ]
        elif "benchmark" in q:
            return [
                SearchResult(
                    title="RPi 5 AI Benchmark Scores",
                    url="https://benchmarks.example.org/rpi5-ai",
                    snippet="Hailo-8L reaches 13 TOPS at 2.5W power draw.",
                    source="benchmarks.example.org",
                ),
                SearchResult(
                    title="Edge AI Benchmark Shootout",
                    url="https://edgeai.example.org/shootout",
                    snippet="Comparative shootout between Raspberry Pi 5 AI Kit and Jetson.",
                    source="edgeai.example.org",
                ),
            ]
        else:
            return [
                SearchResult(
                    title="Raspberry Pi 5 AI Kit Overview",
                    url="https://raspberrypi.example.org/ai-kit",
                    snippet="Official announcement of Raspberry Pi 5 AI Kit.",
                    source="raspberrypi.example.org",
                )
            ]

    def fetch(self, url: str, timeout_seconds: float = 10.0) -> FetchResult:
        if self.should_fail or self.fail_on_fetch:
            raise WebProviderError(f"Provider fetch error for url: {url}")

        self.fetches_called.append(url)
        return FetchResult(
            url=url,
            title=f"Detailed Page: {url}",
            content=f"Detailed fetched content from {url}. Full architectural analysis and deep benchmark numbers.",
            status_code=200,
            source="fetched.example.org",
        )


class MockSequenceReasoner(ResearchReasonerInterface):
    """
    Fake reasoner returning a pre-programmed sequence of AgentActions.
    Simulates multi-turn autonomous reasoning.
    """

    def __init__(self, actions: List[AgentAction]):
        self.actions = list(actions)
        self.decide_calls = 0
        self.received_states: List[ResearchState] = []

    def decide_next_action(self, state: ResearchState) -> AgentAction:
        self.received_states.append(state)
        if self.decide_calls < len(self.actions):
            act = self.actions[self.decide_calls]
            self.decide_calls += 1
            return act
        return AgentAction(
            action_type=AgentActionType.FINISH,
            parameters={},
            reason="Exhausted pre-programmed actions.",
        )


# ============================================================================
# PHASE 3.0 ACTION AND VALIDATION TESTS (1 - 13)
# ============================================================================

def test_valid_search_action():
    """1. Valid SEARCH action instantiates and validates correctly."""
    action = AgentAction(
        action_type=AgentActionType.SEARCH,
        parameters={"query": "Raspberry Pi 5 AI benchmark", "max_results": 5},
        reason="Need benchmark evidence",
    )
    state = ResearchState(objective="Raspberry Pi 5 AI")
    limits = ResearchLimits()
    is_valid, err = ActionValidator.validate(action, state, limits)

    assert is_valid is True
    assert err is None
    assert action.action_type == AgentActionType.SEARCH
    assert action.parameters["query"] == "Raspberry Pi 5 AI benchmark"


def test_valid_fetch_action():
    """2. Valid FETCH action instantiates and validates correctly."""
    action = AgentAction(
        action_type=AgentActionType.FETCH,
        parameters={"url": "https://example.com/article"},
        reason="Inspect detailed methodology",
    )
    state = ResearchState(objective="Test")
    limits = ResearchLimits()
    is_valid, err = ActionValidator.validate(action, state, limits)

    assert is_valid is True
    assert err is None
    assert action.parameters["url"] == "https://example.com/article"


def test_valid_finish_action():
    """3. Valid FINISH action instantiates and validates correctly."""
    action = AgentAction(
        action_type=AgentActionType.FINISH,
        parameters={},
        reason="Sufficient evidence collected",
    )
    state = ResearchState(objective="Test")
    limits = ResearchLimits()
    is_valid, err = ActionValidator.validate(action, state, limits)

    assert is_valid is True
    assert err is None


def test_malformed_action_rejected():
    """4. Non-AgentAction objects are rejected by ActionValidator."""
    state = ResearchState(objective="Test")
    limits = ResearchLimits()
    is_valid, err = ActionValidator.validate("not an agent action", state, limits)

    assert is_valid is False
    assert "must be an AgentAction instance" in err


def test_invalid_action_type_rejected():
    """5. Unsupported action types (e.g. bash, code_execution) are strictly rejected."""
    dangerous_action = AgentAction(
        action_type="execute_bash",
        parameters={"command": "rm -rf /"},
        reason="Malicious code execution attempt",
    )
    state = ResearchState(objective="Test")
    limits = ResearchLimits()
    is_valid, err = ActionValidator.validate(dangerous_action, state, limits)

    assert is_valid is False
    assert "Unsupported action type" in err
    assert "Allowed actions: search, fetch, finish" in err


def test_missing_query_for_search_rejected():
    """6. Search action without query parameter is rejected."""
    action = AgentAction(
        action_type=AgentActionType.SEARCH,
        parameters={},
        reason="Missing query",
    )
    state = ResearchState(objective="Test")
    limits = ResearchLimits()
    is_valid, err = ActionValidator.validate(action, state, limits)

    assert is_valid is False
    assert "requires a non-empty 'query'" in err


def test_missing_url_for_fetch_rejected():
    """7. Fetch action without valid HTTP/HTTPS URL is rejected."""
    action1 = AgentAction(
        action_type=AgentActionType.FETCH,
        parameters={},
        reason="Missing url",
    )
    action2 = AgentAction(
        action_type=AgentActionType.FETCH,
        parameters={"url": "ftp://malicious.org/file"},
        reason="Non-HTTP URL",
    )
    state = ResearchState(objective="Test")
    limits = ResearchLimits()

    is_valid1, err1 = ActionValidator.validate(action1, state, limits)
    is_valid2, err2 = ActionValidator.validate(action2, state, limits)

    assert is_valid1 is False
    assert "requires a non-empty 'url'" in err1
    assert is_valid2 is False
    assert "requires a valid HTTP/HTTPS URL" in err2


def test_action_validation_runtime_owned():
    """8. Action validation is owned by the runtime and enforces parameters."""
    action = AgentAction.from_dict({
        "action_type": "SEARCH",
        "parameters": {"query": "valid query"},
        "reason": "Test",
    })
    state = ResearchState(objective="Test")
    limits = ResearchLimits()
    is_valid, err = ActionValidator.validate(action, state, limits)
    assert is_valid is True


def test_max_iterations_limit():
    """9. Loop halts when max_iterations is reached."""
    provider = FakeWebProvider()
    infinite_search_reasoner = MockSequenceReasoner([
        AgentAction(AgentActionType.SEARCH, {"query": f"Query {i}"})
        for i in range(20)
    ])
    agent = ResearchAgent(
        provider=provider,
        reasoner=infinite_search_reasoner,
        limits=ResearchLimits(max_iterations=3),
    )
    result = agent.run("Test Loop Limit")

    assert result.state.iteration == 3
    assert result.state.completed is True
    assert result.state.stop_reason == "STOP_LIMIT"


def test_max_searches_limit():
    """10. Searches used strictly respects max_searches limit."""
    provider = FakeWebProvider()
    searches = [
        AgentAction(AgentActionType.SEARCH, {"query": f"Query {i}"})
        for i in range(10)
    ]
    reasoner = MockSequenceReasoner(searches)
    agent = ResearchAgent(
        provider=provider,
        reasoner=reasoner,
        limits=ResearchLimits(max_iterations=10, max_searches=2),
    )
    result = agent.run("Test Max Searches")

    assert result.state.searches_used <= 2
    assert len(provider.searches_called) <= 2


def test_max_fetches_limit():
    """11. Fetches used strictly respects max_fetches limit."""
    provider = FakeWebProvider()
    actions = [
        AgentAction(AgentActionType.SEARCH, {"query": "benchmarks"}),
        AgentAction(AgentActionType.FETCH, {"url": "https://benchmarks.example.org/rpi5-ai"}),
        AgentAction(AgentActionType.FETCH, {"url": "https://edgeai.example.org/shootout"}),
        AgentAction(AgentActionType.FETCH, {"url": "https://third.example.org"}),
    ]
    reasoner = MockSequenceReasoner(actions)
    agent = ResearchAgent(
        provider=provider,
        reasoner=reasoner,
        limits=ResearchLimits(max_iterations=10, max_fetches=1),
    )
    result = agent.run("Test Max Fetches")

    assert result.state.fetches_used <= 1
    assert len(provider.fetches_called) <= 1


def test_max_evidence_limit():
    """12. Evidence items accumulated strictly respects max_evidence limit."""
    provider = FakeWebProvider()
    actions = [
        AgentAction(AgentActionType.SEARCH, {"query": "benchmark 1"}),
        AgentAction(AgentActionType.SEARCH, {"query": "benchmark 2"}),
    ]
    reasoner = MockSequenceReasoner(actions)
    agent = ResearchAgent(
        provider=provider,
        reasoner=reasoner,
        limits=ResearchLimits(max_iterations=5, max_evidence=1),
    )
    result = agent.run("Test Max Evidence")

    assert len(result.evidence) <= 1
    assert len(result.citations) <= 1


def test_repeated_invalid_actions_termination():
    """13. Repeated invalid actions cause safe termination with STOP_INVALID_ACTION."""
    provider = FakeWebProvider()
    invalid_actions = [
        AgentAction("invalid_1", {}),
        AgentAction("invalid_2", {}),
        AgentAction("invalid_3", {}),
        AgentAction("invalid_4", {}),
    ]
    reasoner = MockSequenceReasoner(invalid_actions)
    agent = ResearchAgent(
        provider=provider,
        reasoner=reasoner,
        limits=ResearchLimits(max_invalid_actions=3),
    )
    result = agent.run("Test Repeated Invalid Actions")

    assert result.state.stop_reason == "STOP_INVALID_ACTION"
    assert result.status == "failed"
    assert result.state.completed is True


# ============================================================================
# PHASE 3.0 REASONING, EXECUTION, AND FEEDBACK TESTS (14 - 24)
# ============================================================================

def test_reasoner_injection():
    """14. Injected reasoner implementation is invoked by ResearchAgent."""
    provider = FakeWebProvider()
    actions = [
        AgentAction(AgentActionType.SEARCH, {"query": "Raspberry Pi 5 AI"}),
        AgentAction(AgentActionType.FINISH, {}),
    ]
    reasoner = MockSequenceReasoner(actions)
    agent = ResearchAgent(provider=provider, reasoner=reasoner)
    result = agent.run("Raspberry Pi 5 AI")

    assert reasoner.decide_calls == 2
    assert result.status == "completed"


def test_synthesizer_injection():
    """15. Custom synthesizer callback/interface is properly invoked with evidence."""
    provider = FakeWebProvider()
    actions = [
        AgentAction(AgentActionType.SEARCH, {"query": "Raspberry Pi 5 AI"}),
        AgentAction(AgentActionType.FINISH, {}),
    ]
    reasoner = MockSequenceReasoner(actions)

    called_synth = False
    def my_synth(obj: str, ev: EvidenceSet, cit: CitationSet) -> str:
        nonlocal called_synth
        called_synth = True
        return f"CUSTOM: {obj} ({len(ev)} items)"

    agent = ResearchAgent(provider=provider, reasoner=reasoner, synthesizer=my_synth)
    result = agent.run("Raspberry Pi 5 AI")

    assert called_synth is True
    assert "CUSTOM: Raspberry Pi 5 AI" in result.output


def test_model_neutral_reasoning_boundary():
    """16. Reasoner interface does not import or depend on Gemini, Ollama, OpenAI, or SQLite."""
    import inspect
    import core.interfaces.research_interface as ri
    source = inspect.getsource(ri)

    assert "gemini" not in source.lower()
    assert "ollama" not in source.lower()
    assert "openai" not in source.lower()
    assert "anthropic" not in source.lower()
    assert "sqlite" not in source.lower()


def test_search_execution():
    """17. SEARCH action executes via provider and accumulates evidence."""
    provider = FakeWebProvider()
    actions = [
        AgentAction(AgentActionType.SEARCH, {"query": "benchmark comparison"}),
        AgentAction(AgentActionType.FINISH, {}),
    ]
    agent = ResearchAgent(provider=provider, reasoner=MockSequenceReasoner(actions))
    result = agent.run("Test Search")

    assert len(provider.searches_called) == 1
    assert provider.searches_called[0] == "benchmark comparison"
    assert len(result.evidence) >= 1


def test_fetch_execution():
    """18. FETCH action executes via provider and enriches evidence."""
    provider = FakeWebProvider()
    actions = [
        AgentAction(AgentActionType.SEARCH, {"query": "RPi 5"}),
        AgentAction(AgentActionType.FETCH, {"url": "https://raspberrypi.example.org/ai-kit"}),
        AgentAction(AgentActionType.FINISH, {}),
    ]
    agent = ResearchAgent(provider=provider, reasoner=MockSequenceReasoner(actions))
    result = agent.run("Test Fetch")

    assert len(provider.fetches_called) == 1
    assert "https://raspberrypi.example.org/ai-kit" in provider.fetches_called
    # Verified that content was replaced with deep fetched content
    fetched_item = next(i for i in result.evidence if i.url == "https://raspberrypi.example.org/ai-kit")
    assert "Detailed fetched content" in fetched_item.content


def test_search_then_fetch_sequence():
    """19. SEARCH -> FETCH -> FINISH autonomous sequence operates cleanly."""
    provider = FakeWebProvider()
    actions = [
        AgentAction(AgentActionType.SEARCH, {"query": "benchmarks", "max_results": 2}),
        AgentAction(AgentActionType.FETCH, {"url": "https://benchmarks.example.org/rpi5-ai"}),
        AgentAction(AgentActionType.FINISH, {}, reason="Complete evidence assembled"),
    ]
    agent = ResearchAgent(provider=provider, reasoner=MockSequenceReasoner(actions))
    result = agent.run("Compare benchmarks")

    assert result.status == "completed"
    assert result.state.searches_used == 1
    assert result.state.fetches_used == 1
    assert result.state.stop_reason == "STOP_SUCCESS"


def test_multi_iteration_reasoning():
    """20. Multi-iteration reasoning updates query dynamically based on feedback."""
    provider = FakeWebProvider()
    # Model generates intelligent follow-up query:
    # 1. search RPi 5 -> 2. search Jetson -> 3. finish
    actions = [
        AgentAction(AgentActionType.SEARCH, {"query": "Raspberry Pi 5 AI"}),
        AgentAction(AgentActionType.SEARCH, {"query": "Jetson Orin Nano AI"}),
        AgentAction(AgentActionType.FINISH, {}),
    ]
    agent = ResearchAgent(provider=provider, reasoner=MockSequenceReasoner(actions))
    result = agent.run("Compare Pi 5 vs Jetson")

    assert result.state.searches_used == 2
    assert "Raspberry Pi 5 AI" in provider.searches_called
    assert "Jetson Orin Nano AI" in provider.searches_called
    assert any("Jetson" in i.title for i in result.evidence)
    assert any("Raspberry" in i.title for i in result.evidence)


def test_evidence_accumulation():
    """21. Evidence items accumulate across multiple actions."""
    provider = FakeWebProvider()
    actions = [
        AgentAction(AgentActionType.SEARCH, {"query": "benchmark"}),
        AgentAction(AgentActionType.SEARCH, {"query": "jetson"}),
        AgentAction(AgentActionType.FINISH, {}),
    ]
    agent = ResearchAgent(provider=provider, reasoner=MockSequenceReasoner(actions))
    result = agent.run("Accumulate evidence")

    assert len(result.evidence) >= 3


def test_evidence_deduplication():
    """22. Evidence items with duplicate canonical URLs are deduplicated."""
    class DuplicateProvider(WebProviderInterface):
        def search(self, query: str, max_results: int = 5, timeout_seconds: float = 10.0) -> List[SearchResult]:
            return [
                SearchResult(title="Dup Title", url="https://example.org/dup/", snippet="Snippet 1"),
                SearchResult(title="Dup Title WWW", url="https://www.example.org/dup", snippet="Snippet 2"),
            ]
        def fetch(self, url: str, timeout_seconds: float = 10.0) -> FetchResult:
            return FetchResult(url=url, title="T", content="C", status_code=200)

    agent = ResearchAgent(
        provider=DuplicateProvider(),
        reasoner=MockSequenceReasoner([
            AgentAction(AgentActionType.SEARCH, {"query": "q1"}),
            AgentAction(AgentActionType.FINISH, {}),
        ]),
    )
    result = agent.run("Test Dup")
    assert len(result.evidence) == 1


def test_observation_generation():
    """23. Runtime creates structured ResearchObservation after every action."""
    provider = FakeWebProvider()
    actions = [
        AgentAction(AgentActionType.SEARCH, {"query": "benchmark"}),
        AgentAction(AgentActionType.FINISH, {}),
    ]
    agent = ResearchAgent(provider=provider, reasoner=MockSequenceReasoner(actions))
    result = agent.run("Test Observations")

    assert len(result.state.observations) == 2
    obs0 = result.state.observations[0]
    assert obs0.action_type == "search"
    assert obs0.success is True
    assert "count" in obs0.data

    obs1 = result.state.observations[1]
    assert obs1.action_type == "finish"
    assert obs1.success is True


def test_bounded_observation_content():
    """24. Observation contents are strictly bounded in length and capacity."""
    provider = FakeWebProvider()
    actions = [
        AgentAction(AgentActionType.SEARCH, {"query": "benchmark"}),
        AgentAction(AgentActionType.FETCH, {"url": "https://benchmarks.example.org/rpi5-ai"}),
        AgentAction(AgentActionType.FINISH, {}),
    ]
    agent = ResearchAgent(provider=provider, reasoner=MockSequenceReasoner(actions))
    result = agent.run("Test Bounded Obs")

    fetch_obs = result.state.observations[1]
    assert fetch_obs.action_type == "fetch"
    # Content preview bounded to <= 500 chars
    assert len(fetch_obs.data.get("content_preview", "")) <= 500


# ============================================================================
# PHASE 3.0 FAILURE, PARTIAL, AND COMPLETENESS TESTS (25 - 30)
# ============================================================================

def test_provider_failure_handling():
    """25. Provider failure creates error observation and halts safely."""
    failing_provider = FakeWebProvider(should_fail=True)
    actions = [
        AgentAction(AgentActionType.SEARCH, {"query": "benchmark"}),
    ]
    agent = ResearchAgent(provider=failing_provider, reasoner=MockSequenceReasoner(actions))
    result = agent.run("Test Failure", raise_on_error=False)

    assert result.status == "failed"
    assert result.state.stop_reason == "STOP_FAILURE"
    assert len(result.state.observations) >= 1
    assert result.state.observations[0].success is False


def test_partial_research_when_limits_reached():
    """26. When evidence is gathered but limits prevent completion, status is partial or completed."""
    provider = FakeWebProvider()
    actions = [
        AgentAction(AgentActionType.SEARCH, {"query": "benchmark"}),
        AgentAction(AgentActionType.SEARCH, {"query": "jetson"}),
    ]
    agent = ResearchAgent(
        provider=provider,
        reasoner=MockSequenceReasoner(actions),
        limits=ResearchLimits(max_iterations=1, min_evidence=1),
    )
    result = agent.run("Test Partial")

    assert result.state.iteration == 1
    assert len(result.evidence) >= 1
    assert result.state.completed is True


def test_zero_evidence_failure():
    """27. Zero evidence collected results in failed status."""
    class EmptyProvider(WebProviderInterface):
        def search(self, query: str, max_results: int = 5, timeout_seconds: float = 10.0) -> List[SearchResult]:
            return []
        def fetch(self, url: str, timeout_seconds: float = 10.0) -> FetchResult:
            return FetchResult(url=url, title="", content="", status_code=404)

    actions = [
        AgentAction(AgentActionType.SEARCH, {"query": "empty topic"}),
        AgentAction(AgentActionType.FINISH, {}),
    ]
    agent = ResearchAgent(
        provider=EmptyProvider(),
        reasoner=MockSequenceReasoner(actions),
        limits=ResearchLimits(min_evidence=1),
    )
    result = agent.run("Test Zero Evidence")

    assert result.status == "failed"
    assert len(result.evidence) == 0


def test_finish_with_sufficient_evidence():
    """28. FINISH action with evidence >= min_evidence sets STOP_SUCCESS and completed status."""
    provider = FakeWebProvider()
    actions = [
        AgentAction(AgentActionType.SEARCH, {"query": "RPi 5 AI"}),
        AgentAction(AgentActionType.FINISH, {}, reason="Done"),
    ]
    agent = ResearchAgent(
        provider=provider,
        reasoner=MockSequenceReasoner(actions),
        limits=ResearchLimits(min_evidence=1),
    )
    result = agent.run("Test Success Finish")

    assert result.status == "completed"
    assert result.state.stop_reason == "STOP_SUCCESS"


def test_no_infinite_loops():
    """29. Loop cannot run indefinitely; hard bounded by max_iterations."""
    provider = FakeWebProvider()
    # Reasoner never proposes FINISH
    always_search = MockSequenceReasoner([
        AgentAction(AgentActionType.SEARCH, {"query": "infinite"}) for _ in range(50)
    ])
    agent = ResearchAgent(
        provider=provider,
        reasoner=always_search,
        limits=ResearchLimits(max_iterations=4),
    )
    result = agent.run("Test No Infinite Loop")

    assert result.state.iteration == 4
    assert result.state.completed is True


def test_no_recursive_spawning():
    """30. Action type validation rejects attempts to recursively spawn subagents."""
    action = AgentAction(
        action_type="spawn_agent",
        parameters={"agent": "ResearchAgent", "objective": "recursive"},
        reason="Recursive spawn attempt",
    )
    is_valid, err = ActionValidator.validate(action, ResearchState(objective="test"), ResearchLimits())
    assert is_valid is False
    assert "Unsupported action type: 'spawn_agent'" in err


# ============================================================================
# PHASE 3.0 CITATION VALIDATION & SYNTHESIS BOUNDS TESTS (31 - 33)
# ============================================================================

def test_citation_references_real_evidence():
    """31. Every citation strictly corresponds to an accumulated EvidenceItem."""
    provider = FakeWebProvider()
    actions = [
        AgentAction(AgentActionType.SEARCH, {"query": "RPi 5 AI"}),
        AgentAction(AgentActionType.FINISH, {}),
    ]
    agent = ResearchAgent(provider=provider, reasoner=MockSequenceReasoner(actions))
    result = agent.run("Test Citations")

    evidence_ids = {e.id for e in result.evidence.items}
    for c in result.citations.citations:
        assert c.evidence_id in evidence_ids

    is_valid, errors = validate_citation_references(result.citations, result.evidence)
    assert is_valid is True
    assert len(errors) == 0


def test_invalid_citation_rejected():
    """32. Citations referencing non-existent evidence IDs are flagged and rejected."""
    item = EvidenceItem(
        id="ev-real",
        title="Real Source",
        url="https://real.org",
        domain="real.org",
        content="Real content",
        retrieved_at="2026-09-06T12:00:00Z",
    )
    evidence_set = EvidenceSet(items=(item,))

    fake_citation = Citation(
        index=1,
        evidence_id="ev-fabricated",
        url="https://fabricated.org",
        title="Fabricated Source",
        domain="fabricated.org",
    )
    cit_set = CitationSet(citations=(fake_citation,))

    is_valid, errors = validate_citation_references(cit_set, evidence_set)
    assert is_valid is False
    assert any("references non-existent evidence_id 'ev-fabricated'" in e for e in errors)


def test_synthesis_gets_bounded_evidence():
    """33. Synthesizer receives bounded EvidenceSet and CitationSet."""
    provider = FakeWebProvider()
    received_ev: Optional[EvidenceSet] = None
    received_cit: Optional[CitationSet] = None

    def capture_synth(obj: str, ev: EvidenceSet, cit: CitationSet) -> str:
        nonlocal received_ev, received_cit
        received_ev = ev
        received_cit = cit
        return "Synthesized answer"

    actions = [
        AgentAction(AgentActionType.SEARCH, {"query": "RPi 5"}),
        AgentAction(AgentActionType.FINISH, {}),
    ]
    agent = ResearchAgent(
        provider=provider,
        reasoner=MockSequenceReasoner(actions),
        synthesizer=capture_synth,
        limits=ResearchLimits(max_evidence=2),
    )
    result = agent.run("Test Synth Bounds")

    assert received_ev is not None
    assert received_cit is not None
    assert len(received_ev) <= 2
    assert len(received_cit) == len(received_ev)
    assert result.output == "Synthesized answer"


# ============================================================================
# PHASE 3.0 WEBCAPABILITY INTEGRATION TEST
# ============================================================================

def test_web_capability_with_reasoner():
    """WebCapability routes to ResearchAgent when reasoner is supplied in task parameters."""
    provider = FakeWebProvider()
    capability = WebCapability(provider=provider)

    custom_reasoner = MockSequenceReasoner([
        AgentAction(AgentActionType.SEARCH, {"query": "Raspberry Pi 5 AI"}),
        AgentAction(AgentActionType.FINISH, {}),
    ])

    task_params = {
        "action": "research",
        "query": "Raspberry Pi 5 AI",
        "reasoner": custom_reasoner,
    }
    out = capability(task_params)

    assert custom_reasoner.decide_calls == 2
    assert "Raspberry Pi 5 AI" in out
    assert capability.get_research_result() is not None
    assert capability.get_research_result().status == "completed"
