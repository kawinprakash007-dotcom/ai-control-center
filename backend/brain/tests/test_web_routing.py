import pytest
from datetime import datetime
from typing import List, Optional

from core.interfaces.web_interface import WebProviderInterface, WebProviderError
from core.models.request import Request
from core.models.decision import Decision, CapabilityType, ExecutionMode
from core.models.plan import Plan
from core.models.task import Task
from core.models.result import Result
from core.models.web import SearchResult, FetchResult
from core.models.pipeline import PipelineResult

from brain.decision_engine import StandardDecisionEngine
from brain.planning import StandardPlanner
from brain.execution import StandardExecutionEngine
from brain.verification import StandardVerifier
from brain.response import StandardResponseComposer
from brain.pipeline import StandardPipeline
from brain.router import Router
from tools.web_capability import WebCapability
from memory.sqlite_store import SQLiteMemoryStore


# ============================================================================
# FAKE WEB PROVIDER FOR DETERMINISTIC TESTING (ZERO NETWORK CALLS)
# ============================================================================

class FakeWebProvider(WebProviderInterface):
    """Deterministic, in-memory fake web provider for testing."""

    def __init__(
        self,
        canned_search_results: Optional[List[SearchResult]] = None,
        canned_fetch_result: Optional[FetchResult] = None,
        should_fail: bool = False,
        failure_message: str = "Simulated network failure",
    ):
        self.canned_search_results = canned_search_results or [
            SearchResult(
                title="Python 3.14 Release Notes",
                url="https://docs.python.org/3.14/whatsnew/",
                snippet="Detailed release notes for Python 3.14.",
            ),
            SearchResult(
                title="Python 3.14 Schedule",
                url="https://peps.python.org/pep-0745/",
                snippet="PEP 745 describes Python 3.14 release timeline.",
            ),
        ]
        self.canned_fetch_result = canned_fetch_result or FetchResult(
            url="https://example.com",
            title="Example Domain",
            source="https://example.com",
            content="Example domain content extracted successfully.",
        )
        self.should_fail = should_fail
        self.failure_message = failure_message
        self.searches_called: List[dict] = []
        self.fetches_called: List[dict] = []

    def search(
        self,
        query: str,
        max_results: int = 5,
        timeout_seconds: float = 10.0,
    ) -> List[SearchResult]:
        self.searches_called.append({
            "query": query,
            "max_results": max_results,
            "timeout_seconds": timeout_seconds,
        })
        if self.should_fail:
            raise WebProviderError(self.failure_message)
        return list(self.canned_search_results[:max_results])

    def fetch(
        self,
        url: str,
        timeout_seconds: float = 10.0,
    ) -> FetchResult:
        self.fetches_called.append({
            "url": url,
            "timeout_seconds": timeout_seconds,
        })
        if self.should_fail:
            raise WebProviderError(self.failure_message)
        return self.canned_fetch_result


def make_request(
    text: str,
    parameters: dict = None,
    constraints: dict = None,
    req_id: str = "req-web-001",
    session_id: str = "test-session",
) -> Request:
    return Request(
        id=req_id,
        original_text=text,
        normalized_text=text,
        session_id=session_id,
        timestamp=datetime.now(),
        parameters=parameters or {},
        constraints=constraints or {},
    )


# ============================================================================
# PHASE 2.9.3 — DECISION ENGINE & ROUTING RECOGNITION TESTS
# ============================================================================

def test_explicit_search_the_web_directive():
    """1. Explicit 'search the web' directive routes deterministically to WEB."""
    engine = StandardDecisionEngine()
    req = make_request("Search the web for Raspberry Pi 5 benchmarks.")
    dec = engine.decide(req)

    assert dec.primary_goal == "web_search"
    assert dec.required_capabilities == [CapabilityType.WEB]
    assert dec.execution_mode == ExecutionMode.SINGLE_STEP
    assert dec.routing_hints.get("action") == "search"
    assert dec.routing_hints.get("query") == "Raspberry Pi 5 benchmarks"


def test_look_up_online_directive():
    """2. 'look up online' directive routes deterministically to WEB."""
    engine = StandardDecisionEngine()
    req = make_request("look up Python 3.14 online")
    dec = engine.decide(req)

    assert dec.primary_goal == "web_search"
    assert dec.required_capabilities == [CapabilityType.WEB]
    assert dec.routing_hints.get("action") == "search"
    assert "Python 3.14" in dec.routing_hints.get("query", "")


def test_latest_signal():
    """3. 'latest' signal routes to WEB."""
    engine = StandardDecisionEngine()
    req = make_request("What's the latest NVIDIA GPU news?")
    dec = engine.decide(req)

    assert dec.primary_goal == "web_search"
    assert dec.required_capabilities == [CapabilityType.WEB]
    assert dec.routing_hints.get("action") == "search"
    assert "latest NVIDIA GPU news" in dec.routing_hints.get("query", "")


def test_current_signal():
    """4. 'current' signal routes to WEB."""
    engine = StandardDecisionEngine()
    req = make_request("What is the current price of RTX 5090?")
    dec = engine.decide(req)

    assert dec.primary_goal == "web_search"
    assert dec.required_capabilities == [CapabilityType.WEB]
    assert dec.routing_hints.get("action") == "search"
    assert "current price of RTX 5090" in dec.routing_hints.get("query", "")


def test_today_signal():
    """5. 'today' signal routes to WEB."""
    engine = StandardDecisionEngine()
    req = make_request("What happened today in tech?")
    dec = engine.decide(req)

    assert dec.primary_goal == "web_search"
    assert dec.required_capabilities == [CapabilityType.WEB]
    assert dec.routing_hints.get("action") == "search"


def test_news_signal():
    """6. 'news' signal routes to WEB."""
    engine = StandardDecisionEngine()
    req = make_request("world news")
    dec = engine.decide(req)

    assert dec.primary_goal == "web_search"
    assert dec.required_capabilities == [CapabilityType.WEB]
    assert dec.routing_hints.get("action") == "search"


def test_current_price_query():
    """7. Current price query routes to WEB."""
    engine = StandardDecisionEngine()
    req = make_request("price of bitcoin")
    dec = engine.decide(req)

    assert dec.primary_goal == "web_search"
    assert dec.required_capabilities == [CapabilityType.WEB]
    assert dec.routing_hints.get("action") == "search"


def test_weather_current_event_query():
    """8. Weather and current-event queries route to WEB."""
    engine = StandardDecisionEngine()
    req = make_request("weather in Tokyo")
    dec = engine.decide(req)

    assert dec.primary_goal == "web_search"
    assert dec.required_capabilities == [CapabilityType.WEB]
    assert dec.routing_hints.get("action") == "search"


def test_direct_url_web_fetch():
    """9. Direct URL in request triggers action='fetch'."""
    engine = StandardDecisionEngine()
    req1 = make_request("open https://example.com")
    dec1 = engine.decide(req1)
    assert dec1.primary_goal == "web_search"
    assert dec1.required_capabilities == [CapabilityType.WEB]
    assert dec1.routing_hints.get("action") == "fetch"
    assert dec1.routing_hints.get("url") == "https://example.com"

    req2 = make_request("https://docs.python.org/3/")
    dec2 = engine.decide(req2)
    assert dec2.primary_goal == "web_search"
    assert dec2.routing_hints.get("action") == "fetch"
    assert dec2.routing_hints.get("url") == "https://docs.python.org/3/"


def test_generic_factual_question_does_not_become_web():
    """10. Generic factual questions remain KNOWLEDGE or CHAT, not WEB."""
    engine = StandardDecisionEngine()
    req = make_request("What is backpropagation?")
    dec = engine.decide(req)

    assert CapabilityType.WEB not in dec.required_capabilities
    assert dec.primary_goal in ("retrieve_knowledge", "answer_chat")


def test_local_paper_question_does_not_become_web():
    """10b. Questions about local uploaded files remain KNOWLEDGE, not WEB."""
    engine = StandardDecisionEngine()
    req = make_request("What does my uploaded paper say about backpropagation?")
    dec = engine.decide(req)

    assert CapabilityType.WEB not in dec.required_capabilities
    assert dec.primary_goal == "retrieve_knowledge"


def test_memory_query_beats_web_and_knowledge():
    """11. Personal memory recall queries beat Web and Knowledge."""
    engine = StandardDecisionEngine()
    req1 = make_request("What is my favorite language?")
    dec1 = engine.decide(req1)
    assert dec1.required_capabilities == [CapabilityType.MEMORY]
    assert dec1.primary_goal == "manage_memory"

    req2 = make_request("What do you remember about my project?")
    dec2 = engine.decide(req2)
    assert dec2.required_capabilities == [CapabilityType.MEMORY]
    assert dec2.primary_goal == "manage_memory"


def test_normal_chat_remains_chat():
    """12. Normal conversational dialogue remains CHAT."""
    engine = StandardDecisionEngine()
    req1 = make_request("hello friend")
    dec1 = engine.decide(req1)
    assert dec1.required_capabilities == [CapabilityType.CHAT]
    assert dec1.primary_goal == "answer_chat"

    req2 = make_request("how are you today?")
    dec2 = engine.decide(req2)
    assert dec2.required_capabilities == [CapabilityType.CHAT]
    assert dec2.primary_goal == "answer_chat"


def test_query_extraction_preserves_entity_casing():
    """13. Extracted web search query preserves entity casing."""
    engine = StandardDecisionEngine()
    req = make_request("search the web for OpenAI, Python 3.14, and ESP32")
    dec = engine.decide(req)

    query = dec.routing_hints.get("query")
    assert "OpenAI" in query
    assert "Python 3.14" in query
    assert "ESP32" in query


# ============================================================================
# PHASE 2.9.3 — PLANNER PARAMETER PROPAGATION TESTS
# ============================================================================

def test_web_action_propagated_to_task():
    """14. Web action ('search') is correctly propagated to Task parameters."""
    planner = StandardPlanner()
    dec = Decision(
        request_id="req-1",
        primary_goal="web_search",
        required_capabilities=[CapabilityType.WEB],
        execution_mode=ExecutionMode.SINGLE_STEP,
        confidence=0.95,
        reasoning="Web test",
        routing_hints={"action": "search", "query": "Python news"},
    )
    plan = planner.plan(dec)
    assert len(plan.steps) == 1
    assert plan.steps[0].parameters["action"] == "search"
    assert plan.steps[0].action == "Web Search"


def test_web_query_propagated_to_task():
    """15. Web query is correctly propagated to Task parameters."""
    planner = StandardPlanner()
    dec = Decision(
        request_id="req-1",
        primary_goal="web_search",
        required_capabilities=[CapabilityType.WEB],
        execution_mode=ExecutionMode.SINGLE_STEP,
        confidence=0.95,
        reasoning="Web test",
        routing_hints={"action": "search", "query": "latest Raspberry Pi 5 benchmarks"},
    )
    plan = planner.plan(dec)
    assert plan.steps[0].parameters["query"] == "latest Raspberry Pi 5 benchmarks"


def test_url_propagated_to_task():
    """16. Web URL is correctly propagated to Task parameters."""
    planner = StandardPlanner()
    dec = Decision(
        request_id="req-1",
        primary_goal="web_search",
        required_capabilities=[CapabilityType.WEB],
        execution_mode=ExecutionMode.SINGLE_STEP,
        confidence=0.95,
        reasoning="Web test",
        routing_hints={"action": "fetch", "url": "https://example.com/api"},
    )
    plan = planner.plan(dec)
    assert plan.steps[0].parameters["action"] == "fetch"
    assert plan.steps[0].parameters["url"] == "https://example.com/api"
    assert plan.steps[0].action == "Web Fetch"


# ============================================================================
# PHASE 2.9.4 — EXECUTION, ROUTER, & VERIFICATION TESTS
# ============================================================================

def test_router_reaches_web_capability():
    """17-18. Router reaches WebCapability and returns Result."""
    fake_provider = FakeWebProvider()
    router = Router()

    task = Task(
        id=1,
        type="web",
        action="Web Search",
        tool="web",
        parameters={
            "action": "search",
            "query": "Python 3.14",
            "web_provider": fake_provider,
        },
        status="pending",
    )
    result = router.route(task)

    assert isinstance(result, Result)
    assert result.success is True
    assert "Python 3.14 Release Notes" in str(result.output)
    assert len(fake_provider.searches_called) == 1


def test_execution_engine_reaches_web_capability():
    """17b. StandardExecutionEngine executes web task through Router."""
    fake_provider = FakeWebProvider()
    engine = StandardExecutionEngine()

    plan = Plan(
        goal="web_search",
        steps=[
            Task(
                id=1,
                type="web",
                action="Web Search",
                tool="web",
                parameters={
                    "action": "search",
                    "query": "Python 3.14",
                    "web_provider": fake_provider,
                },
                status="pending",
            )
        ],
        status="pending",
    )
    results = engine.execute(plan)

    assert len(results) == 1
    assert results[0].success is True
    assert plan.steps[0].status == "completed"
    assert "Python 3.14 Release Notes" in str(results[0].output)


def test_successful_web_execution_verifies_correctly():
    """19. Successful Web execution verifies with verified=True."""
    verifier = StandardVerifier()
    task = Task(
        id=1,
        type="web",
        action="Web Search",
        tool="web",
        status="completed",
    )
    plan = Plan(goal="web_search", steps=[task], status="completed")
    results = [Result(success=True, message="Success", output="Found 2 results")]

    verification = verifier.verify(plan, results)
    assert verification.verified is True
    assert verification.status == "verified"


def test_failed_web_execution_verifies_as_failure():
    """20. Failed Web execution verifies as failure without false success."""
    verifier = StandardVerifier()
    task = Task(
        id=1,
        type="web",
        action="Web Search",
        tool="web",
        status="failed",
    )
    plan = Plan(goal="web_search", steps=[task], status="failed")
    results = [Result(success=False, message="Web search timed out", output=None)]

    verification = verifier.verify(plan, results)
    assert verification.verified is False
    assert verification.status == "failed"
    assert "Web search timed out" in verification.reason or "task 1" in verification.reason


# ============================================================================
# PHASE 2.9.4 — FULL PIPELINE INTEGRATION TESTS (END-TO-END)
# ============================================================================

def test_pipeline_executes_web_search_with_fake_provider(tmp_path):
    """21-23. Pipeline executes Web search end-to-end using injected fake provider."""
    fake_provider = FakeWebProvider()
    mem_store = SQLiteMemoryStore(db_path=str(tmp_path / "test_pipeline.db"))

    pipeline = StandardPipeline(
        memory_service=mem_store,
        web_provider=fake_provider,
    )

    result = pipeline.process("search the web for Raspberry Pi 5 benchmarks")

    assert isinstance(result, PipelineResult)
    assert result.verification.verified is True
    assert len(fake_provider.searches_called) == 1
    assert fake_provider.searches_called[0]["query"] == "Raspberry Pi 5 benchmarks"
    assert "Python 3.14 Release Notes" in result.response
    # Conversational turn should be persisted in memory
    history = mem_store.get_history(result.request.session_id, limit=10)
    assert len(history) == 2  # user message + assistant response


def test_pipeline_executes_web_fetch_with_fake_provider(tmp_path):
    """21b. Pipeline executes Web fetch end-to-end using injected fake provider."""
    fake_provider = FakeWebProvider()
    mem_store = SQLiteMemoryStore(db_path=str(tmp_path / "test_fetch.db"))

    pipeline = StandardPipeline(
        memory_service=mem_store,
        web_provider=fake_provider,
    )

    result = pipeline.process("open https://example.com")

    assert isinstance(result, PipelineResult)
    assert result.verification.verified is True
    assert len(fake_provider.fetches_called) == 1
    assert fake_provider.fetches_called[0]["url"] == "https://example.com"
    assert "Example domain content extracted successfully." in result.response


def test_pipeline_failed_web_execution_returns_failure_response(tmp_path):
    """20b. Failed Web execution in pipeline returns error response without false success."""
    failing_provider = FakeWebProvider(should_fail=True, failure_message="Connection refused")
    mem_store = SQLiteMemoryStore(db_path=str(tmp_path / "test_fail.db"))

    pipeline = StandardPipeline(
        memory_service=mem_store,
        web_provider=failing_provider,
    )

    result = pipeline.process("search the web for server status")

    assert result.verification.verified is False
    assert "Execution failed" in result.response
    assert "Connection refused" in result.response


def test_pipeline_no_chat_history_injected_into_web_step(tmp_path):
    """Pipeline does NOT pass conversational history or MemoryService into Web task parameters."""
    fake_provider = FakeWebProvider()
    mem_store = SQLiteMemoryStore(db_path=str(tmp_path / "test_clean_ctx.db"))

    pipeline = StandardPipeline(
        memory_service=mem_store,
        web_provider=fake_provider,
    )

    # First add some history
    pipeline.process("hello friend")

    # Now run a web search
    result = pipeline.process("What's the latest NVIDIA GPU news?")

    assert result.plan is not None
    assert len(result.plan.steps) == 1
    web_task = result.plan.steps[0]
    assert "history" not in web_task.parameters
    assert "memory_service" not in web_task.parameters
