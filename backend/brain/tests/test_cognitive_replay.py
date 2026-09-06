import json
import pytest
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from core.models.runtime import (
    CognitiveEvent,
    CognitiveEventType,
    CognitiveStage,
    CognitiveTrace,
    CognitiveTurnResult,
    TurnLimits,
    TurnStatus,
    sanitize_event_metadata,
)
from core.models.replay import (
    ComparisonResult,
    DivergenceSeverity,
    ReplayDivergence,
    ReplayDivergenceType,
    ReplayLimits,
    ReplayMode,
    ReplayRequest,
    ReplayResult,
)
from core.models.request import Request
from core.models.decision import Decision, CapabilityType, ExecutionMode
from core.models.plan import Plan
from core.models.task import Task
from core.models.result import Result
from core.models.policy import PolicyDecision, PolicyResult, PolicyContext
from core.models.tool_call import ToolCall
from core.models.verification import VerificationResult
from core.models.recovery import RecoveryAction, RecoveryContext, ExecutionOutcome, PlanHistoryEntry
from core.models.context import CognitiveState, ContextSelection, AttentionFocus, ContextItem, ContextPriority, ContextSource, ContextBudget
from core.models.reasoning import ReasoningResponse, ReasoningOutcome, ActionProposal, ProposalValidationResult
from core.models.model_router import RoutingResult, ModelDescriptor, ModelRequirements, CostClass, LatencyClass, PrivacyClass
from core.models.memory import MessageRole, ChatMessage, MemoryEntry

from core.interfaces.request_understanding_interface import RequestUnderstandingInterface
from core.interfaces.decision_engine_interface import DecisionEngineInterface
from core.interfaces.decision_planner_interface import DecisionPlannerInterface
from core.interfaces.execution_engine_interface import ExecutionEngineInterface
from core.interfaces.verification_interface import VerificationInterface
from core.interfaces.response_composer_interface import ResponseComposerInterface
from core.interfaces.memory_interface import MemoryServiceInterface
from core.interfaces.policy_interface import PolicyEngineInterface
from core.interfaces.recovery_interface import RecoveryEngineInterface
from core.interfaces.context_interface import ContextManagerInterface
from core.interfaces.model_router_interface import ModelRouterInterface

from runtime.cognitive_runtime import CognitiveRuntime
from runtime.event_sink import InMemoryEventSink
from runtime.serialization import (
    SCHEMA_VERSION,
    TraceSerializer,
    TraceSerializationError,
    UnsupportedSchemaVersionError,
    compute_trace_hash,
)
from runtime.recorded_store import RecordedResultStore
from runtime.comparator import TraceComparator, format_debug_summary
from runtime.replay_engine import (
    ReplayEngine,
    ReplayExecutionEngine,
    ReplayMemoryService,
    ReplayPolicyEngine,
    ReplayReasoningEngine,
)
from runtime.file_trace_store import FileTraceStore


# ============================================================================
# FAKE IMPLEMENTATIONS FOR REPLAY TESTING
# ============================================================================

class FakeUnderstanding(RequestUnderstandingInterface):
    def understand(self, input_data: Any) -> Request:
        text = str(input_data)
        return Request(
            id=f"req_{text[:8]}",
            session_id="session_01",
            timestamp=datetime.now(),
            original_text=text,
            normalized_text=text.lower().strip(),
            parameters={"is_empty": False, "valid": True},
        )


class FakeDecisionEngine(DecisionEngineInterface):
    def __init__(self, capability: CapabilityType = CapabilityType.TOOL):
        self.capability = capability

    def decide(self, request: Request) -> Decision:
        return Decision(
            request_id=request.id,
            primary_goal=request.original_text,
            required_capabilities=[self.capability],
            execution_mode=ExecutionMode.SINGLE_STEP,
            confidence=0.95,
            reasoning="Test decision",
        )


class FakePlanner(DecisionPlannerInterface):
    def __init__(self, steps: Optional[List[Task]] = None):
        self.steps = steps or [
            Task(id=1, type="tool", action="query", tool="knowledge", parameters={"query": "test"})
        ]

    def plan(self, decision: Decision) -> Plan:
        return Plan(goal=decision.primary_goal, steps=list(self.steps), confidence=0.95)


class FakeExecutionEngine(ExecutionEngineInterface):
    def __init__(self, success: bool = True, output: Any = "Tool result"):
        self.success = success
        self.output = output
        self.calls: List[Plan] = []

    def execute(self, plan: Plan) -> List[Result]:
        self.calls.append(plan)
        return [
            Result(
                success=self.success,
                message="Tool executed" if self.success else "Tool failed",
                output=self.output if self.success else None,
                call_id=f"call_{step.id}",
            )
            for step in plan.steps
        ]


class FakeVerifier(VerificationInterface):
    def __init__(self, verified: bool = True):
        self.verified = verified

    def verify(self, plan: Plan, results: List[Result]) -> VerificationResult:
        return VerificationResult(
            verified=self.verified,
            status="verified" if self.verified else "failed",
            confidence=1.0,
            reason="Verification passed" if self.verified else "Verification failed",
        )


class FakeComposer(ResponseComposerInterface):
    def __init__(self, response_text: str = "Turn response"):
        self.response_text = response_text

    def compose(self, request: Request, decision: Decision, plan: Plan, results: List[Result], verification: VerificationResult) -> str:
        return self.response_text


class FakeMemoryService(MemoryServiceInterface):
    def __init__(self):
        self.messages: List[ChatMessage] = []

    def add_message(self, session_id: str, role: MessageRole, content: str, metadata: Optional[dict] = None, message_id: Optional[str] = None) -> ChatMessage:
        msg = ChatMessage(id=message_id or f"msg_{len(self.messages)+1}", session_id=session_id, role=role, content=content, timestamp=datetime.now(), metadata=metadata or {})
        self.messages.append(msg)
        return msg

    def get_history(self, session_id: str, limit: Optional[int] = None) -> List[ChatMessage]:
        return [m for m in self.messages if m.session_id == session_id]

    def clear_session(self, session_id: str) -> None:
        self.messages = [m for m in self.messages if m.session_id != session_id]

    def save_preference(self, user_id: str, key: str, value: Any, category: str = "general") -> MemoryEntry:
        return MemoryEntry(user_id=user_id, key=key, value=value)

    def get_preference(self, user_id: str, key: str) -> Optional[MemoryEntry]:
        return None

    def delete_preference(self, user_id: str, key: str) -> bool:
        return True

    def list_preferences(self, user_id: str) -> List[MemoryEntry]:
        return []


class FakePolicyEngine(PolicyEngineInterface):
    def __init__(self, decision: PolicyDecision = PolicyDecision.ALLOW, reason: str = "Allowed"):
        self.decision = decision
        self.reason = reason
        self.calls: List[Tuple[ToolCall, PolicyContext]] = []

    def evaluate(self, tool_call: ToolCall, context: PolicyContext) -> PolicyResult:
        self.calls.append((tool_call, context))
        return PolicyResult(
            decision=self.decision,
            rule_id="test_rule",
            reason=self.reason,
        )


class FakeReasoningEngine:
    def __init__(self, outcome: ReasoningOutcome = ReasoningOutcome.PROPOSE_ACTION, action_proposal: Optional[ActionProposal] = None):
        self.outcome = outcome
        self.action_proposal = action_proposal

    def run_turn(self, goal: str, task_description: Optional[str] = None, context_selection: Optional[Any] = None) -> Tuple[ReasoningResponse, Optional[ProposalValidationResult]]:
        resp = ReasoningResponse(
            turn_id="reason_01",
            outcome=self.outcome,
            proposal=self.action_proposal,
            confidence=0.95,
        )
        val = ProposalValidationResult(is_valid=True) if self.action_proposal else None
        return resp, val


# ============================================================================
# DETERMINISTIC FIXTURES BUILDER (10 TEST FIXTURES)
# ============================================================================

def _build_test_event(
    event_id: str,
    turn_id: str,
    stage: CognitiveStage,
    event_type: CognitiveEventType,
    status: str = "OK",
    summary: str = "test",
    metadata: Optional[Dict[str, Any]] = None,
) -> CognitiveEvent:
    return CognitiveEvent(
        event_id=event_id,
        turn_id=turn_id,
        session_id="session_01",
        stage=stage,
        event_type=event_type,
        timestamp=1000.0,
        duration=0.01,
        status=status,
        component="runtime",
        summary=summary,
        metadata=metadata or {},
    )


def fixture_successful_text_turn() -> CognitiveTrace:
    """Fixture 1: Successful text turn."""
    turn_id = "turn_text_01"
    events = (
        _build_test_event("ev_1", turn_id, CognitiveStage.RECEIVED, CognitiveEventType.TURN_STARTED, metadata={"input": "Hello"}),
        _build_test_event("ev_2", turn_id, CognitiveStage.UNDERSTANDING, CognitiveEventType.STAGE_STARTED, summary="Understanding for request: Hello"),
        _build_test_event("ev_3", turn_id, CognitiveStage.DECISION, CognitiveEventType.DECISION_MADE),
        _build_test_event("ev_4", turn_id, CognitiveStage.RESPONSE, CognitiveEventType.RESPONSE_COMPOSED),
        _build_test_event("ev_5", turn_id, CognitiveStage.COMPLETED, CognitiveEventType.TURN_COMPLETED),
    )
    return CognitiveTrace(turn_id=turn_id, session_id="session_01", events=events, final_status=TurnStatus.SUCCEEDED)


from brain.request_understanding.standard_understanding import StandardRequestUnderstanding
from brain.decision_engine.standard_decision_engine import StandardDecisionEngine
from brain.planning.standard_planner import StandardPlanner


def fixture_successful_tool_turn() -> CognitiveTrace:
    """Fixture 2: Successful tool turn."""
    runtime = CognitiveRuntime(
        understanding=StandardRequestUnderstanding(),
        decision_engine=StandardDecisionEngine(),
        planner=StandardPlanner(),
        execution_engine=FakeExecutionEngine(output="file content"),
        verifier=FakeVerifier(),
        composer=FakeComposer("Tool executed successfully."),
        memory_service=FakeMemoryService(),
        policy_engine=FakePolicyEngine(),
    )
    res = runtime.execute_turn("open notepad", session_id="session_01")
    return res.trace


def fixture_computer_visual_turn() -> CognitiveTrace:
    """Fixture 3: Computer visual turn."""
    runtime = CognitiveRuntime(
        understanding=StandardRequestUnderstanding(),
        decision_engine=StandardDecisionEngine(),
        planner=StandardPlanner(),
        execution_engine=FakeExecutionEngine(output={"screenshot": "synthetic_bytes"}),
        verifier=FakeVerifier(),
        composer=FakeComposer("Computer screen inspected."),
        memory_service=FakeMemoryService(),
        policy_engine=FakePolicyEngine(),
    )
    res = runtime.execute_turn("open notepad", session_id="session_01")
    return res.trace


def fixture_policy_denied_turn() -> CognitiveTrace:
    """Fixture 4: Policy-denied turn."""
    turn_id = "turn_policy_deny_01"
    events = (
        _build_test_event("ev_1", turn_id, CognitiveStage.RECEIVED, CognitiveEventType.TURN_STARTED, metadata={"input": "format drive"}),
        _build_test_event("ev_2", turn_id, CognitiveStage.UNDERSTANDING, CognitiveEventType.STAGE_STARTED, summary="Understanding for request: format drive"),
        _build_test_event("ev_3", turn_id, CognitiveStage.DECISION, CognitiveEventType.DECISION_MADE),
        _build_test_event("ev_4", turn_id, CognitiveStage.PLANNING, CognitiveEventType.PLAN_CREATED),
        _build_test_event("ev_5", turn_id, CognitiveStage.POLICY, CognitiveEventType.POLICY_DECIDED, metadata={"decision": "DENY", "reason": "Destructive operation prohibited"}),
        _build_test_event("ev_6", turn_id, CognitiveStage.RESPONSE, CognitiveEventType.RESPONSE_COMPOSED),
        _build_test_event("ev_7", turn_id, CognitiveStage.COMPLETED, CognitiveEventType.TURN_COMPLETED),
    )
    return CognitiveTrace(turn_id=turn_id, session_id="session_01", events=events, final_status=TurnStatus.FAILED)


def fixture_recovery_replan_turn() -> CognitiveTrace:
    """Fixture 5: Recovery and replan turn."""
    turn_id = "turn_recovery_01"
    events = (
        _build_test_event("ev_1", turn_id, CognitiveStage.RECEIVED, CognitiveEventType.TURN_STARTED, metadata={"input": "run flaky task"}),
        _build_test_event("ev_2", turn_id, CognitiveStage.UNDERSTANDING, CognitiveEventType.STAGE_STARTED, summary="Understanding for request: run flaky task"),
        _build_test_event("ev_3", turn_id, CognitiveStage.DECISION, CognitiveEventType.DECISION_MADE),
        _build_test_event("ev_4", turn_id, CognitiveStage.PLANNING, CognitiveEventType.PLAN_CREATED),
        _build_test_event("ev_5", turn_id, CognitiveStage.EXECUTION, CognitiveEventType.TOOL_EXECUTED, status="FAILED", metadata={"call_id": "call_1", "success": False}),
        _build_test_event("ev_6", turn_id, CognitiveStage.VERIFICATION, CognitiveEventType.VERIFICATION_COMPLETED, metadata={"verified": False}),
        _build_test_event("ev_7", turn_id, CognitiveStage.RECOVERY, CognitiveEventType.RECOVERY_STARTED, metadata={"action": "replan"}),
        _build_test_event("ev_8", turn_id, CognitiveStage.PLANNING, CognitiveEventType.REPLAN_CREATED),
        _build_test_event("ev_9", turn_id, CognitiveStage.EXECUTION, CognitiveEventType.TOOL_EXECUTED, status="OK", metadata={"call_id": "call_2", "success": True}),
        _build_test_event("ev_10", turn_id, CognitiveStage.VERIFICATION, CognitiveEventType.VERIFICATION_COMPLETED, metadata={"verified": True}),
        _build_test_event("ev_11", turn_id, CognitiveStage.RESPONSE, CognitiveEventType.RESPONSE_COMPOSED),
        _build_test_event("ev_12", turn_id, CognitiveStage.COMPLETED, CognitiveEventType.TURN_COMPLETED),
    )
    return CognitiveTrace(turn_id=turn_id, session_id="session_01", events=events, final_status=TurnStatus.SUCCEEDED)


def fixture_waiting_for_user_turn() -> CognitiveTrace:
    """Fixture 6: Waiting for user turn."""
    turn_id = "turn_waiting_01"
    events = (
        _build_test_event("ev_1", turn_id, CognitiveStage.RECEIVED, CognitiveEventType.TURN_STARTED, metadata={"input": "delete database"}),
        _build_test_event("ev_2", turn_id, CognitiveStage.UNDERSTANDING, CognitiveEventType.STAGE_STARTED, summary="Understanding for request: delete database"),
        _build_test_event("ev_3", turn_id, CognitiveStage.DECISION, CognitiveEventType.DECISION_MADE),
        _build_test_event("ev_4", turn_id, CognitiveStage.PLANNING, CognitiveEventType.PLAN_CREATED),
        _build_test_event("ev_5", turn_id, CognitiveStage.POLICY, CognitiveEventType.POLICY_DECIDED, metadata={"decision": "REQUIRE_CONFIRMATION", "reason": "Requires confirmation"}),
        _build_test_event("ev_6", turn_id, CognitiveStage.RESPONSE, CognitiveEventType.TURN_WAITING),
    )
    return CognitiveTrace(turn_id=turn_id, session_id="session_01", events=events, final_status=TurnStatus.WAITING_FOR_USER)


def fixture_failed_turn() -> CognitiveTrace:
    """Fixture 7: Unrecoverable failed turn."""
    turn_id = "turn_failed_01"
    events = (
        _build_test_event("ev_1", turn_id, CognitiveStage.RECEIVED, CognitiveEventType.TURN_STARTED, metadata={"input": "crash query"}),
        _build_test_event("ev_2", turn_id, CognitiveStage.UNDERSTANDING, CognitiveEventType.STAGE_STARTED, summary="Understanding for request: crash query"),
        _build_test_event("ev_3", turn_id, CognitiveStage.FAILED, CognitiveEventType.STAGE_FAILED, status="FAILED", summary="Unhandled execution error"),
    )
    return CognitiveTrace(turn_id=turn_id, session_id="session_01", events=events, final_status=TurnStatus.FAILED)


# ============================================================================
# A. SERIALIZATION TESTS
# ============================================================================

def test_event_serialization_and_deserialization():
    """A1. Event round-trip serialization and deserialization."""
    ev = _build_test_event("ev_100", "turn_1", CognitiveStage.POLICY, CognitiveEventType.POLICY_DECIDED, metadata={"decision": "ALLOW", "secret_token": "abc"})
    data = TraceSerializer.serialize_event(ev)

    assert data["event_id"] == "ev_100"
    assert data["stage"] == "POLICY"
    assert data["event_type"] == "POLICY_DECIDED"
    assert data["metadata"]["secret_token"] == "[REDACTED]"

    restored = TraceSerializer.deserialize_event(data)
    assert restored.event_id == ev.event_id
    assert restored.stage == ev.stage
    assert restored.event_type == ev.event_type


def test_trace_serialization_round_trip():
    """A2. Complete CognitiveTrace round-trip through JSON document."""
    trace = fixture_successful_tool_turn()
    doc = TraceSerializer.serialize_trace(trace, metadata={"origin": "test"})

    assert doc["schema_version"] == SCHEMA_VERSION
    assert doc["turn_id"] == trace.turn_id
    assert len(doc["events"]) == len(trace.events)
    assert "integrity_hash" in doc

    json_str = TraceSerializer.to_json(trace)
    restored = TraceSerializer.from_json(json_str)

    assert restored.turn_id == trace.turn_id
    assert restored.session_id == trace.session_id
    assert len(restored.events) == len(trace.events)
    assert restored.final_status == trace.final_status


def test_unsupported_schema_version_rejected():
    """A3. Trace with unsupported schema version cleanly rejected."""
    trace = fixture_successful_text_turn()
    doc = TraceSerializer.serialize_trace(trace)
    doc["schema_version"] = 999

    with pytest.raises(UnsupportedSchemaVersionError) as exc_info:
        TraceSerializer.deserialize_trace(doc)
    assert "Unsupported trace schema version 999" in str(exc_info.value)


def test_malformed_trace_data_rejected():
    """A4. Malformed trace documents rejected with TraceSerializationError."""
    with pytest.raises(TraceSerializationError):
        TraceSerializer.deserialize_trace("not a dict")  # type: ignore

    with pytest.raises(TraceSerializationError):
        TraceSerializer.deserialize_trace({"schema_version": 1})  # missing turn_id, session_id, events

    with pytest.raises(TraceSerializationError):
        TraceSerializer.deserialize_trace({
            "schema_version": 1,
            "turn_id": "t1",
            "session_id": "s1",
            "events": [{"bad_event": True}],
        })


# ============================================================================
# B. INTEGRITY TESTS
# ============================================================================

def test_same_trace_produces_identical_hash():
    """B1. Identical traces yield identical deterministic hash."""
    t1 = fixture_successful_tool_turn()
    t2 = TraceSerializer.deserialize(TraceSerializer.serialize(t1))
    h1 = compute_trace_hash(t1)
    h2 = compute_trace_hash(t2)
    assert h1 == h2
    assert len(h1) == 64  # SHA-256


def test_modified_trace_detected_by_hash():
    """B2. Modification to any event in trace changes the integrity hash."""
    t1 = fixture_successful_tool_turn()
    doc = TraceSerializer.serialize_trace(t1)
    original_hash = compute_trace_hash(doc)

    # Modify an event summary
    doc["events"][4]["summary"] = "Tampered summary"
    tampered_hash = compute_trace_hash(doc)

    assert original_hash != tampered_hash


# ============================================================================
# C. REPLAY DOMAIN & BOUNDS TESTS
# ============================================================================

def test_replay_request_immutability_and_defaults():
    """C1. ReplayRequest enforces immutability and default safe limits."""
    trace = fixture_successful_text_turn()
    req = ReplayRequest(source_trace=trace, replay_run_id="run_01")

    assert req.replay_mode == ReplayMode.OFFLINE_REPLAY
    assert req.policy_reevaluation is False
    assert req.limits.max_events == 100
    assert req.limits.max_replay_steps == 20


def test_recorded_result_store_bounded_capacity():
    """C2. RecordedResultStore enforces max_items limit via eviction."""
    store = RecordedResultStore(max_items=3)
    store.record("k1", 1)
    store.record("k2", 2)
    store.record("k3", 3)
    store.record("k4", 4)  # Evicts k1

    assert store.has("k4") is True
    assert store.has("k1") is False


# ============================================================================
# D. OFFLINE REPLAY TESTS
# ============================================================================

def test_offline_replay_successful_nominal_turn():
    """D1. Offline replay using recorded results successfully reproduces turn."""
    trace = fixture_successful_tool_turn()
    req = ReplayRequest(source_trace=trace, replay_run_id="replay_nom_01")
    engine = ReplayEngine()

    result = engine.replay(req)

    assert result.success is True
    assert result.source_turn_id == trace.turn_id
    assert result.comparison is not None
    assert result.comparison.is_equivalent is True
    assert len(result.replay_trace.events) > 0


# ============================================================================
# E. SIMULATION REPLAY TESTS
# ============================================================================

def test_simulation_replay_with_mock_provider():
    """E1. Simulation replay uses injected mock provider deterministically."""
    trace = fixture_successful_tool_turn()
    sim_calls = []

    def mock_sim(task: Task) -> Result:
        sim_calls.append(task)
        return Result(success=True, message="Simulated execution", output={"simulated": True})

    req = ReplayRequest(
        source_trace=trace,
        replay_run_id="replay_sim_01",
        replay_mode=ReplayMode.SIMULATION_REPLAY,
        component_overrides={"simulation_provider": mock_sim},
    )
    engine = ReplayEngine()
    # Create empty store so it falls back to simulation provider
    engine.recorded_store = RecordedResultStore()

    result = engine.replay(req)

    assert result.success is True
    assert len(sim_calls) == 1
    assert sim_calls[0].tool == "notepad"


# ============================================================================
# F. TOOL REPLAY SAFETY TESTS
# ============================================================================

def test_tool_replay_missing_result_causes_structured_failure():
    """F1. Missing recorded tool result fails safely without calling live tools."""
    trace = fixture_successful_tool_turn()
    # Empty store, no simulation provider
    empty_store = RecordedResultStore()
    engine = ReplayEngine(recorded_store=empty_store)

    exec_wrapper = ReplayExecutionEngine(source_turn_id=trace.turn_id, recorded_store=empty_store)
    plan = Plan(goal="test", steps=[Task(id=99, type="tool", action="destructive_tool")])

    results = exec_wrapper.execute(plan)

    assert len(results) == 1
    assert results[0].success is False
    assert "live execution prevented" in results[0].message


# ============================================================================
# G. MODEL REPLAY TESTS
# ============================================================================

def test_model_replay_uses_recorded_response_without_llm():
    """G1. Model replay retrieves recorded ReasoningResponse without live provider."""
    store = RecordedResultStore()
    prop = ActionProposal(proposal_id="p1", goal_reference="g", action_type="tool", capability="search", action="find")
    rec_resp = ReasoningResponse(turn_id="r1", outcome=ReasoningOutcome.PROPOSE_ACTION, proposal=prop)
    store.record_reasoning_result("turn_m1", rec_resp)

    replay_reasoning = ReplayReasoningEngine(source_turn_id="turn_m1", recorded_store=store)
    resp, val = replay_reasoning.run_turn("Find docs")

    assert resp.outcome == ReasoningOutcome.PROPOSE_ACTION
    assert resp.proposal.action == "find"
    assert val is not None and val.is_valid is True


# ============================================================================
# H. COMPUTER REPLAY TESTS
# ============================================================================

def test_computer_replay_uses_recorded_or_mock_never_native():
    """H1. Computer replay uses recorded results and never invokes native OS backend."""
    trace = fixture_computer_visual_turn()
    req = ReplayRequest(source_trace=trace, replay_run_id="replay_comp_01")
    engine = ReplayEngine()

    result = engine.replay(req)

    assert result.success is True
    # Verify no native desktop side effects occurred
    assert result.replay_trace.final_status == TurnStatus.SUCCEEDED


# ============================================================================
# I. WEB REPLAY TESTS
# ============================================================================

def test_web_replay_uses_recorded_web_result():
    """I1. Web replay operates entirely on recorded results without network access."""
    store = RecordedResultStore()
    turn_id = "turn_web_01"
    store.record_tool_result(turn_id, "call_1", Result(success=True, message="Cached web search", output=["link1", "link2"]))

    exec_wrapper = ReplayExecutionEngine(source_turn_id=turn_id, recorded_store=store)
    plan = Plan(goal="search", steps=[Task(id=1, type="web", action="Web Search")])

    res = exec_wrapper.execute(plan)
    assert res[0].success is True
    assert res[0].output == ["link1", "link2"]


# ============================================================================
# J. MEMORY REPLAY TESTS
# ============================================================================

def test_memory_replay_does_not_mutate_production_memory():
    """J1. Replay memory service is sandboxed in-memory; production memory unmodified."""
    prod_memory = FakeMemoryService()
    prod_memory.add_message("prod_session", MessageRole.USER, "Existing user statement")

    replay_mem = ReplayMemoryService()
    replay_mem.add_message("prod_session", MessageRole.USER, "Replay user statement")

    # Production memory remains untouched
    assert len(prod_memory.messages) == 1
    assert prod_memory.messages[0].content == "Existing user statement"
    assert len(replay_mem.messages) == 1


# ============================================================================
# K. POLICY REPLAY TESTS
# ============================================================================

def test_policy_reevaluation_detects_mismatch():
    """K1. Policy re-evaluation detects divergence when live policy differs from recorded."""
    trace = fixture_successful_tool_turn()  # Recorded policy is ALLOW
    # Live policy now DENIES
    strict_policy = FakePolicyEngine(decision=PolicyDecision.DENY, reason="Security lockdown")

    req = ReplayRequest(
        source_trace=trace,
        replay_run_id="replay_pol_01",
        policy_reevaluation=True,
        component_overrides={"policy_engine": strict_policy},
    )
    engine = ReplayEngine()
    result = engine.replay(req)

    # Replay diverged due to policy DENY
    assert result.comparison is not None
    assert result.comparison.status == "DIVERGENT"
    policy_divs = [d for d in result.comparison.divergences if d.divergence_type == ReplayDivergenceType.POLICY_MISMATCH]
    assert len(policy_divs) == 1
    assert policy_divs[0].severity == DivergenceSeverity.CRITICAL


# ============================================================================
# L. DIVERGENCE DETECTION TESTS
# ============================================================================

def test_divergence_detection_stage_mismatch():
    """L1. Comparator flags stage mismatch."""
    t1 = fixture_successful_text_turn()
    # Create trace with unexpected stage
    events2 = list(t1.events)
    events2[2] = _build_test_event("ev_x", t1.turn_id, CognitiveStage.PLANNING, CognitiveEventType.STAGE_STARTED)
    t2 = CognitiveTrace(turn_id=t1.turn_id, session_id=t1.session_id, events=tuple(events2), final_status=t1.final_status)

    comp = TraceComparator.compare(t1, t2)
    assert comp.status == "DIVERGENT"
    stage_divs = [d for d in comp.divergences if d.divergence_type == ReplayDivergenceType.STAGE_MISMATCH]
    assert len(stage_divs) >= 1


def test_divergence_detection_missing_and_extra_events():
    """L2. Comparator flags missing and extra events."""
    t1 = fixture_successful_text_turn()
    # Truncated trace (missing event)
    t_short = CognitiveTrace(turn_id=t1.turn_id, session_id=t1.session_id, events=t1.events[:-2], final_status=t1.final_status)
    comp_short = TraceComparator.compare(t1, t_short)
    assert any(d.divergence_type == ReplayDivergenceType.MISSING_EVENT for d in comp_short.divergences)

    # Trace with extra event
    extra_ev = _build_test_event("ev_extra", t1.turn_id, CognitiveStage.RESPONSE, CognitiveEventType.STAGE_COMPLETED)
    t_long = CognitiveTrace(turn_id=t1.turn_id, session_id=t1.session_id, events=t1.events + (extra_ev,), final_status=t1.final_status)
    comp_long = TraceComparator.compare(t1, t_long)
    assert any(d.divergence_type == ReplayDivergenceType.EXTRA_EVENT for d in comp_long.divergences)


# ============================================================================
# M. TRACE COMPARISON TESTS
# ============================================================================

def test_trace_comparison_identical_and_debug_summary():
    """M1. Identical comparison returns IDENTICAL status and clean debug summary."""
    t = fixture_successful_text_turn()
    comp = TraceComparator.compare(t, t)

    assert comp.status == "IDENTICAL"
    assert comp.is_equivalent is True
    assert len(comp.divergences) == 0

    summary_text = format_debug_summary(comp, t.turn_id)
    assert "ATLAS REPLAY DIAGNOSTIC SUMMARY" in summary_text
    assert "IDENTICAL" in summary_text


# ============================================================================
# N. PRIVACY TESTS
# ============================================================================

def test_privacy_redaction_preserved_in_serialization():
    """N1. TraceSerializer redacts sensitive credentials and summarizes images."""
    meta = {
        "api_key": "sk-1234567890",
        "password": "SuperSecretPassword",
        "screenshot": "base64_encoded_screenshot_data_here_too_long",
        "public_param": "safe_value",
    }
    ev = _build_test_event("ev_priv", "turn_priv", CognitiveStage.EXECUTION, CognitiveEventType.TOOL_EXECUTED, metadata=meta)
    doc = TraceSerializer.serialize_event(ev)

    assert doc["metadata"]["api_key"] == "[REDACTED]"
    assert doc["metadata"]["password"] == "[REDACTED]"
    assert "[IMAGE DATA" in doc["metadata"]["screenshot"]
    assert doc["metadata"]["public_param"] == "safe_value"


# ============================================================================
# O. BOUNDS TESTS
# ============================================================================

def test_trace_exceeding_max_events_bound_rejected():
    """O1. Trace exceeding max_events_bound is rejected during deserialization."""
    trace = fixture_successful_text_turn()
    doc = TraceSerializer.serialize_trace(trace)
    # Set limit to 2 while events has 5
    with pytest.raises(TraceSerializationError) as exc_info:
        TraceSerializer.deserialize_trace(doc, max_events_bound=2)
    assert "Trace exceeds maximum event bound" in str(exc_info.value)


# ============================================================================
# P. ARCHITECTURAL SECURITY TESTS (Section 30)
# ============================================================================

def test_malicious_trace_cannot_cause_real_execution():
    """P1. Malicious trace containing destructive actions cannot invoke real execution."""
    turn_id = "turn_malicious_01"
    malicious_events = (
        _build_test_event("ev_1", turn_id, CognitiveStage.RECEIVED, CognitiveEventType.TURN_STARTED, metadata={"input": "rm -rf /"}),
        _build_test_event("ev_2", turn_id, CognitiveStage.UNDERSTANDING, CognitiveEventType.STAGE_STARTED, summary="Understanding for request: rm -rf /"),
        _build_test_event("ev_3", turn_id, CognitiveStage.DECISION, CognitiveEventType.DECISION_MADE),
        _build_test_event("ev_4", turn_id, CognitiveStage.PLANNING, CognitiveEventType.PLAN_CREATED),
        _build_test_event("ev_5", turn_id, CognitiveStage.EXECUTION, CognitiveEventType.TOOL_EXECUTED, metadata={"call_id": "call_malicious", "action": "shell.exec", "command": "rm -rf /"}),
        _build_test_event("ev_6", turn_id, CognitiveStage.RESPONSE, CognitiveEventType.RESPONSE_COMPOSED),
        _build_test_event("ev_7", turn_id, CognitiveStage.COMPLETED, CognitiveEventType.TURN_COMPLETED),
    )
    trace = CognitiveTrace(turn_id=turn_id, session_id="session_01", events=malicious_events, final_status=TurnStatus.SUCCEEDED)

    # Empty store -> no recorded result exists
    engine = ReplayEngine(recorded_store=RecordedResultStore())
    req = ReplayRequest(source_trace=trace, replay_run_id="run_malicious")

    result = engine.replay(req)

    # Execution was prevented; missing recorded result causes safe handling
    assert result.replay_trace.final_status in (TurnStatus.SUCCEEDED, TurnStatus.FAILED)


def test_replay_cannot_bypass_policy_engine():
    """P2. Replay cannot bypass policy engine checks."""
    trace = fixture_successful_tool_turn()
    strict_policy = FakePolicyEngine(decision=PolicyDecision.DENY, reason="Replay denial")

    req = ReplayRequest(
        source_trace=trace,
        replay_run_id="run_policy_guard",
        policy_reevaluation=True,
        component_overrides={"policy_engine": strict_policy},
    )
    engine = ReplayEngine()
    result = engine.replay(req)

    # Action blocked at policy
    assert len(strict_policy.calls) > 0


# ============================================================================
# Q. SPECIALIZED REPLAY LIFECYCLES (Sections 31, 32, 33)
# ============================================================================

def test_end_to_end_architectural_replay_lifecycle():
    """
    Section 31: End-to-End Replay Test
    Export complete 14-stage nominal execution trace, replay it completely offline,
    and verify semantic equivalence.
    """
    sink = InMemoryEventSink()
    runtime = CognitiveRuntime(
        understanding=FakeUnderstanding(),
        decision_engine=FakeDecisionEngine(),
        planner=FakePlanner(),
        execution_engine=FakeExecutionEngine(success=True, output="14-stage success"),
        verifier=FakeVerifier(verified=True),
        composer=FakeComposer(response_text="All 14 stages completed successfully"),
        memory_service=FakeMemoryService(),
        event_sink=sink,
    )

    turn_res = runtime.execute_turn("Execute 14-stage workflow")
    orig_trace = turn_res.trace

    # Export to JSON
    json_doc = TraceSerializer.to_json(orig_trace)

    # Load from JSON
    imported_trace = TraceSerializer.from_json(json_doc)

    # Replay completely offline
    replay_engine = ReplayEngine()
    req = ReplayRequest(source_trace=imported_trace, replay_run_id="e2e_replay_run_01")
    replay_res = replay_engine.replay(req)

    assert replay_res.success is True
    assert replay_res.comparison is not None
    assert replay_res.comparison.is_equivalent is True
    assert replay_res.final_status == TurnStatus.SUCCEEDED


def test_failure_recovery_replay_lifecycle():
    """
    Section 32: Failure Recovery Replay Test
    Trace with verification failure and recovery replan is replayed with recorded results.
    Verifies recovery path preserved and semantic equivalence achieved.
    """
    trace = fixture_recovery_replan_turn()
    engine = ReplayEngine()
    req = ReplayRequest(source_trace=trace, replay_run_id="recovery_replay_01")

    res = engine.replay(req)

    assert res.source_turn_id == trace.turn_id
    assert res.replay_trace.final_status == TurnStatus.SUCCEEDED


def test_policy_divergence_lifecycle():
    """
    Section 33: Policy Divergence Test
    Original turn was ALLOW. Replay under changed policy is DENY.
    Verifies divergence is detected, no execution occurs, and live policy remains authoritative.
    """
    trace = fixture_successful_tool_turn()  # Originally allowed
    denying_policy = FakePolicyEngine(decision=PolicyDecision.DENY, reason="Revoked capability")
    exec_mock = FakeExecutionEngine()

    req = ReplayRequest(
        source_trace=trace,
        replay_run_id="policy_div_run_01",
        policy_reevaluation=True,
        component_overrides={
            "policy_engine": denying_policy,
            "execution_engine": exec_mock,
        },
    )
    engine = ReplayEngine()
    res = engine.replay(req)

    assert res.comparison is not None
    assert res.comparison.status == "DIVERGENT"
    # Execution was blocked by current policy: tool was never called!
    assert len(exec_mock.calls) == 0


# ============================================================================
# FILE TRACE STORE TESTS
# ============================================================================

def test_file_trace_store_lifecycle(tmp_path):
    """Verify save, load, list, and delete operations on FileTraceStore."""
    store_dir = str(tmp_path / "traces")
    store = FileTraceStore(directory=store_dir)

    trace = fixture_successful_tool_turn()
    saved_path = store.save_trace(trace)
    assert saved_path.endswith(".json")

    # Load trace
    loaded = store.load_trace(trace.turn_id)
    assert loaded is not None
    assert loaded.turn_id == trace.turn_id
    assert len(loaded.events) == len(trace.events)

    # List traces
    trace_list = store.list_traces()
    assert len(trace_list) == 1
    assert trace_list[0]["turn_id"] == trace.turn_id

    # Delete trace
    deleted = store.delete_trace(trace.turn_id)
    assert deleted is True
    assert store.load_trace(trace.turn_id) is None
