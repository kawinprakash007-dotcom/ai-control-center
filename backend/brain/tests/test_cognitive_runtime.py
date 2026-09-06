import pytest
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

from core.models.runtime import (
    CognitiveStage,
    TurnStatus,
    CognitiveEventType,
    CognitiveEvent,
    CognitiveTrace,
    CognitiveTurn,
    CognitiveTurnResult,
    TurnLimits,
    sanitize_event_metadata,
)
from core.models.request import Request
from core.models.decision import Decision, CapabilityType, ExecutionMode
from core.models.plan import Plan
from core.models.task import Task
from core.models.result import Result
from core.models.verification import VerificationResult
from core.models.memory import MessageRole, MemoryEntry
from core.models.policy import PolicyDecision, PolicyResult, PolicyContext
from core.models.recovery import RecoveryAction, RecoveryContext, ExecutionOutcome, PlanHistoryEntry
from core.models.context import CognitiveState, ContextSelection, AttentionFocus, ContextItem, ContextPriority, ContextSource, ContextBudget
from core.models.reasoning import ReasoningResponse, ReasoningOutcome, ActionProposal, ProposalValidationResult
from core.models.model_router import RoutingResult, ModelDescriptor, ModelRequirements, CostClass, LatencyClass, PrivacyClass

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
from runtime.transitions import (
    VALID_TRANSITIONS,
    InvalidTransitionError,
    is_valid_transition,
    validate_stage_transition,
)
from runtime.event_sink import InMemoryEventSink, LoggingEventSink


# ============================================================================
# DETERMINISTIC FAKES & TEST DOUBLES
# ============================================================================

class FakeUnderstanding(RequestUnderstandingInterface):
    def __init__(self, request: Optional[Request] = None):
        self.request = request or Request(
            id="req_test_123",
            session_id="session_test_456",
            original_text="Search documentation for VFS architecture",
            normalized_text="search documentation for vfs architecture",
            timestamp=datetime.now(),
        )
        self.calls: List[Any] = []

    def understand(self, input_data: Any) -> Request:
        self.calls.append(input_data)
        if isinstance(input_data, str) and input_data.startswith("raise_error"):
            raise ValueError("Malformed input text")
        return self.request


class FakeDecisionEngine(DecisionEngineInterface):
    def __init__(self, decision: Optional[Decision] = None):
        self.decision = decision or Decision(
            request_id="req_test_123",
            primary_goal="retrieve_knowledge",
            required_capabilities=[CapabilityType.KNOWLEDGE],
            execution_mode=ExecutionMode.SINGLE_STEP,
            confidence=0.95,
            reasoning="Knowledge query identified",
        )
        self.calls: List[Request] = []

    def decide(self, request: Request) -> Decision:
        self.calls.append(request)
        return self.decision


class FakePlanner(DecisionPlannerInterface):
    def __init__(self, plan: Optional[Plan] = None):
        self.plan_instance = plan or Plan(
            goal="retrieve_knowledge",
            steps=[
                Task(id=1, type="knowledge", action="query", tool="knowledge", parameters={"query": "VFS"})
            ],
        )
        self.calls: List[Decision] = []

    def plan(self, decision: Decision) -> Plan:
        self.calls.append(decision)
        if getattr(decision, "reasoning", "") == "raise_plan_error":
            raise ValueError("Unsupported capability in plan")
        return self.plan_instance


class FakeExecutionEngine(ExecutionEngineInterface):
    def __init__(self, results: Optional[List[Result]] = None):
        self.results = results if results is not None else [
            Result(success=True, message="Document retrieved.", output="VFS provides virtual file system abstractions.")
        ]
        self.calls: List[Plan] = []

    def execute(self, plan: Plan) -> List[Result]:
        self.calls.append(plan)
        return list(self.results)


class FakeVerifier(VerificationInterface):
    def __init__(self, verified: bool = True, reason: str = "Execution succeeded"):
        self.verified = verified
        self.reason = reason
        self.calls: List[Tuple[Plan, List[Result]]] = []

    def verify(self, plan: Plan, results: List[Result]) -> VerificationResult:
        self.calls.append((plan, results))
        return VerificationResult(
            verified=self.verified,
            status="verified" if self.verified else "failed",
            confidence=1.0 if self.verified else 0.0,
            reason=self.reason,
            failed_task_id=None if self.verified else 1,
        )


class FakeComposer(ResponseComposerInterface):
    def __init__(self, response_text: str = "Composed final response"):
        self.response_text = response_text
        self.calls: List[Tuple[Any, ...]] = []

    def compose(self, request: Request, decision: Decision, plan: Plan, results: List[Result], verification: VerificationResult) -> str:
        self.calls.append((request, decision, plan, results, verification))
        return self.response_text


class FakeMemoryService(MemoryServiceInterface):
    def __init__(self):
        self.messages: List[Dict[str, Any]] = []

    def add_message(self, session_id: str, role: MessageRole, content: str) -> None:
        self.messages.append({"session_id": session_id, "role": role, "content": content})

    def get_history(self, session_id: str, limit: int = 50) -> List[Any]:
        return [
            MemoryEntry(id="m1", session_id=session_id, role=m["role"], content=m["content"])
            for m in self.messages if m["session_id"] == session_id
        ][:limit]

    def clear_session(self, session_id: str) -> None:
        self.messages = [m for m in self.messages if m["session_id"] != session_id]

    def search_memories(self, query: str, limit: int = 5) -> List[Any]:
        return []

    def get_context(self, session_id: str, max_tokens: int = 2000) -> str:
        return ""

    def save_preference(self, key: str, value: Any, user_id: str = "default_user") -> None:
        pass

    def get_preference(self, key: str, user_id: str = "default_user") -> Optional[Any]:
        return None

    def list_preferences(self, user_id: str = "default_user") -> Dict[str, Any]:
        return {}

    def delete_preference(self, key: str, user_id: str = "default_user") -> bool:
        return False


class FakePolicyEngine(PolicyEngineInterface):
    def __init__(self, decision: PolicyDecision = PolicyDecision.ALLOW, reason: str = "Approved"):
        self.decision = decision
        self.reason = reason
        self.evaluated_calls: List[Tuple[Any, Any]] = []

    def evaluate(self, tool_call: Any, context: Optional[PolicyContext] = None) -> PolicyResult:
        self.evaluated_calls.append((tool_call, context))
        return PolicyResult(
            decision=self.decision,
            reason=self.reason,
            rule_id="fake_rule_001",
        )


class FakeContextManager(ContextManagerInterface):
    def __init__(self):
        self.calls: List[CognitiveState] = []

    def build_context(self, cognitive_state: CognitiveState) -> ContextSelection:
        self.calls.append(cognitive_state)
        return ContextSelection(
            selected_items=(
                ContextItem(
                    item_id="ctx_1",
                    source=ContextSource.CURRENT_GOAL,
                    priority=ContextPriority.HIGHEST,
                    content=cognitive_state.original_goal,
                ),
            ),
            attention_focus=AttentionFocus.EXECUTION,
            budget=ContextBudget(max_tokens=2000),
        )


class FakeModelRouter(ModelRouterInterface):
    def __init__(self, provider_id: str = "fake_local_llm", success: bool = True):
        self.provider_id = provider_id
        self.success = success
        self.calls: List[ModelRequirements] = []

    def route(self, requirements: ModelRequirements) -> RoutingResult:
        self.calls.append(requirements)
        return RoutingResult(
            provider_id=self.provider_id if self.success else "",
            model_name="mock_model",
            confidence=0.99,
            estimated_cost=0.0,
            estimated_latency=10.0,
            success=self.success,
            error_reason="" if self.success else "No eligible provider",
        )

    @property
    def registry(self) -> Any:
        return None

    def get_provider(self, provider_id: str) -> Optional[Any]:
        return None

    def register_provider(self, descriptor: ModelDescriptor) -> None:
        pass


class FakeReasoningEngine:
    def __init__(self, outcome: ReasoningOutcome = ReasoningOutcome.REPORT_COMPLETION, action_proposal: Optional[ActionProposal] = None):
        self.outcome = outcome
        self.action_proposal = action_proposal
        self.calls: List[Dict[str, Any]] = []

    def run_turn(self, goal: str, task_description: Optional[str] = None, context_selection: Optional[Any] = None) -> Tuple[ReasoningResponse, Optional[ProposalValidationResult]]:
        self.calls.append({"goal": goal, "task_description": task_description, "context": context_selection})
        resp = ReasoningResponse(
            turn_id="reason_turn_01",
            outcome=self.outcome,
            proposal=self.action_proposal,
            clarification_prompt="Which directory should I check?" if self.outcome == ReasoningOutcome.ASK_CLARIFICATION else None,
            abort_reason="Unrecoverable reasoning failure" if self.outcome == ReasoningOutcome.ABORT else None,
            confidence=0.9,
        )
        val = ProposalValidationResult(is_valid=True) if self.action_proposal else None
        return resp, val


class FakeRecoveryEngine(RecoveryEngineInterface):
    def __init__(self, action: RecoveryAction = RecoveryAction.RETRY, should_succeed_on_recovery: bool = True):
        self.action = action
        self.should_succeed = should_succeed_on_recovery
        self.calls: List[Dict[str, Any]] = []

    def recover(self, original_goal: str, initial_plan: Plan, execute_fn: Any, verify_fn: Any) -> Tuple[Plan, List[Result], VerificationResult, RecoveryContext]:
        self.calls.append({"goal": original_goal, "plan": initial_plan})
        # Initial execution & failure
        results = execute_fn(initial_plan)
        verification = verify_fn(initial_plan, results)

        entry = PlanHistoryEntry(
            plan_id=getattr(initial_plan, "id", "plan_01"),
            plan=initial_plan,
            outcome=ExecutionOutcome.FAILURE,
            results=tuple(results),
            verification=verification,
            recovery_action=self.action,
            reason="Initial execution failed",
        )
        rec_ctx = RecoveryContext(
            original_goal=original_goal,
            current_plan=initial_plan,
            outcome=ExecutionOutcome.FAILURE,
            failure_reason="Initial execution failed",
            plan_history=(entry,),
        )

        if self.action in (RecoveryAction.RETRY, RecoveryAction.REPLAN) and self.should_succeed:
            # Second attempt succeeds
            results_2 = [Result(success=True, message="Recovered successfully", output="Recovered")]
            verification_2 = VerificationResult(verified=True, status="verified", confidence=1.0, reason="Recovered OK")
            return initial_plan, results_2, verification_2, rec_ctx

        return initial_plan, results, verification, rec_ctx


# ============================================================================
# A. TURN DOMAIN TESTS
# ============================================================================

def test_turn_domain_creation():
    """A1. Verify CognitiveTurn creation and initial invariant state."""
    turn = CognitiveTurn(turn_id="t_001", session_id="s_001")
    assert turn.turn_id == "t_001"
    assert turn.session_id == "s_001"
    assert turn.current_stage == CognitiveStage.RECEIVED
    assert turn.status == TurnStatus.RUNNING
    assert turn.is_active is True
    assert turn.elapsed_seconds >= 0.0


def test_turn_immutable_snapshots():
    """A2. Verify CognitiveEvent and CognitiveTrace cannot be mutated."""
    event = CognitiveEvent(
        event_id="e1",
        turn_id="t1",
        session_id="s1",
        stage=CognitiveStage.RECEIVED,
        event_type=CognitiveEventType.TURN_STARTED,
        timestamp=time.time(),
    )
    with pytest.raises(Exception):
        event.summary = "mutate"  # FrozenInstanceError

    trace = CognitiveTrace(turn_id="t1", session_id="s1", events=(event,))
    with pytest.raises(Exception):
        trace.final_status = TurnStatus.SUCCEEDED  # FrozenInstanceError


def test_correlation_ids_preserved():
    """A3. Verify turn_id, session_id, and request_id are preserved throughout trace and result."""
    event_sink = InMemoryEventSink()
    runtime = CognitiveRuntime(
        understanding=FakeUnderstanding(),
        decision_engine=FakeDecisionEngine(),
        planner=FakePlanner(),
        execution_engine=FakeExecutionEngine(),
        verifier=FakeVerifier(),
        composer=FakeComposer(),
        memory_service=FakeMemoryService(),
        event_sink=event_sink,
    )
    result = runtime.execute_turn("Test input", session_id="session_preserved_999")
    assert result.session_id == "session_preserved_999"
    assert result.trace.session_id == "session_preserved_999"
    for ev in result.trace.events:
        assert ev.turn_id == result.turn_id
        assert ev.session_id == "session_preserved_999"


# ============================================================================
# B. STAGE TRANSITIONS TESTS
# ============================================================================

def test_valid_stage_transitions_nominal():
    """B1. Verify the canonical nominal lifecycle stage transitions are recognized as valid."""
    stages = [
        CognitiveStage.RECEIVED,
        CognitiveStage.UNDERSTANDING,
        CognitiveStage.DECISION,
        CognitiveStage.PLANNING,
        CognitiveStage.CONTEXT,
        CognitiveStage.ROUTING,
        CognitiveStage.REASONING,
        CognitiveStage.PROPOSAL,
        CognitiveStage.VALIDATION,
        CognitiveStage.POLICY,
        CognitiveStage.EXECUTION,
        CognitiveStage.OBSERVATION,
        CognitiveStage.VERIFICATION,
        CognitiveStage.MEMORY,
        CognitiveStage.RESPONSE,
        CognitiveStage.COMPLETED,
    ]
    for i in range(len(stages) - 1):
        assert is_valid_transition(stages[i], stages[i + 1]) is True
        validate_stage_transition(stages[i], stages[i + 1])


def test_invalid_stage_transition_rejected():
    """B2. Verify illegal lifecycle skips raise InvalidTransitionError."""
    with pytest.raises(InvalidTransitionError):
        validate_stage_transition(CognitiveStage.RECEIVED, CognitiveStage.EXECUTION)

    with pytest.raises(InvalidTransitionError):
        validate_stage_transition(CognitiveStage.PLANNING, CognitiveStage.MEMORY)

    with pytest.raises(InvalidTransitionError):
        validate_stage_transition(CognitiveStage.ROUTING, CognitiveStage.COMPLETED)


def test_terminal_stage_transitions_rejected():
    """B3. Verify terminal stages cannot transition anywhere."""
    for term in (CognitiveStage.COMPLETED, CognitiveStage.FAILED, CognitiveStage.ABORTED):
        assert is_valid_transition(term, CognitiveStage.RECEIVED) is False
        with pytest.raises(InvalidTransitionError):
            validate_stage_transition(term, CognitiveStage.UNDERSTANDING)


def test_universal_failed_and_aborted_transitions():
    """B4. Verify active stages can always transition to FAILED or ABORTED."""
    active_stages = [s for s in CognitiveStage if s not in (CognitiveStage.COMPLETED, CognitiveStage.FAILED, CognitiveStage.ABORTED)]
    for stage in active_stages:
        assert is_valid_transition(stage, CognitiveStage.FAILED) is True
        assert is_valid_transition(stage, CognitiveStage.ABORTED) is True


# ============================================================================
# C. BASIC FULL LIFECYCLE TESTS
# ============================================================================

def test_basic_full_lifecycle_success():
    """C1. Complete nominal run through all active stages."""
    event_sink = InMemoryEventSink()
    memory = FakeMemoryService()
    runtime = CognitiveRuntime(
        understanding=FakeUnderstanding(),
        decision_engine=FakeDecisionEngine(),
        planner=FakePlanner(),
        execution_engine=FakeExecutionEngine(),
        verifier=FakeVerifier(verified=True),
        composer=FakeComposer(response_text="All steps succeeded"),
        memory_service=memory,
        event_sink=event_sink,
    )

    result = runtime.execute_turn("Execute nominal search")
    assert result.status == TurnStatus.SUCCEEDED
    assert result.stage == CognitiveStage.COMPLETED
    assert result.response == "All steps succeeded"
    assert result.verification.verified is True
    assert len(memory.messages) == 2  # user + assistant

    # Verify PipelineResult backward compatibility adapter
    p_res = result.to_pipeline_result()
    assert p_res.response == "All steps succeeded"
    assert p_res.verification.verified is True
    assert p_res.results == result.results


# ============================================================================
# D. WAITING FOR USER TESTS
# ============================================================================

def test_waiting_for_user_policy_permission():
    """D1. Turn halts at POLICY stage when permission is required without failing."""
    policy = FakePolicyEngine(decision=PolicyDecision.ASK_PERMISSION, reason="Requires admin approval")
    exec_engine = FakeExecutionEngine()
    runtime = CognitiveRuntime(
        understanding=FakeUnderstanding(),
        decision_engine=FakeDecisionEngine(),
        planner=FakePlanner(),
        policy_engine=policy,
        execution_engine=exec_engine,
        verifier=FakeVerifier(),
        composer=FakeComposer(),
        memory_service=FakeMemoryService(),
    )

    result = runtime.execute_turn("Format hard drive")
    assert result.status == TurnStatus.WAITING_FOR_USER
    assert result.stage == CognitiveStage.RESPONSE
    assert "Requires admin approval" in result.response
    assert len(exec_engine.calls) == 0  # CRITICAL: Tool was never executed


def test_waiting_for_user_policy_confirmation():
    """D2. Turn halts with WAITING_FOR_USER when confirmation is required."""
    policy = FakePolicyEngine(decision=PolicyDecision.REQUIRE_CONFIRMATION, reason="Delete file confirmation")
    exec_engine = FakeExecutionEngine()
    runtime = CognitiveRuntime(
        understanding=FakeUnderstanding(),
        decision_engine=FakeDecisionEngine(),
        planner=FakePlanner(),
        policy_engine=policy,
        execution_engine=exec_engine,
        verifier=FakeVerifier(),
        composer=FakeComposer(),
        memory_service=FakeMemoryService(),
    )

    result = runtime.execute_turn("Delete file.txt")
    assert result.status == TurnStatus.WAITING_FOR_USER
    assert len(exec_engine.calls) == 0


def test_waiting_for_user_reasoning_clarification():
    """D3. Turn halts with WAITING_FOR_USER when reasoning requests clarification."""
    reasoning = FakeReasoningEngine(outcome=ReasoningOutcome.ASK_CLARIFICATION)
    exec_engine = FakeExecutionEngine()
    runtime = CognitiveRuntime(
        understanding=FakeUnderstanding(),
        decision_engine=FakeDecisionEngine(),
        planner=FakePlanner(),
        reasoning_engine=reasoning,
        execution_engine=exec_engine,
        verifier=FakeVerifier(),
        composer=FakeComposer(),
        memory_service=FakeMemoryService(),
    )

    result = runtime.execute_turn("ambiguous query")
    assert result.status == TurnStatus.WAITING_FOR_USER
    assert "Which directory should I check?" in result.response
    assert len(exec_engine.calls) == 0


# ============================================================================
# E. RECOVERY TRANSITION TESTS
# ============================================================================

def test_recovery_transition_retry():
    """E1. Verification failure triggers RecoveryEngine retry and recovers successfully."""
    recovery = FakeRecoveryEngine(action=RecoveryAction.RETRY, should_succeed_on_recovery=True)
    runtime = CognitiveRuntime(
        understanding=FakeUnderstanding(),
        decision_engine=FakeDecisionEngine(),
        planner=FakePlanner(),
        execution_engine=FakeExecutionEngine(),
        verifier=FakeVerifier(verified=False),
        composer=FakeComposer(),
        memory_service=FakeMemoryService(),
        recovery_engine=recovery,
    )

    result = runtime.execute_turn("Execute flaky action")
    assert result.status == TurnStatus.SUCCEEDED
    assert result.recovery.plan_history[-1].recovery_action == RecoveryAction.RETRY
    assert len(recovery.calls) == 1


def test_recovery_transition_ask_user():
    """E2. RecoveryEngine decides ASK_USER, placing turn in WAITING_FOR_USER."""
    recovery = FakeRecoveryEngine(action=RecoveryAction.ASK_USER, should_succeed_on_recovery=False)
    runtime = CognitiveRuntime(
        understanding=FakeUnderstanding(),
        decision_engine=FakeDecisionEngine(),
        planner=FakePlanner(),
        execution_engine=FakeExecutionEngine(),
        verifier=FakeVerifier(verified=False),
        composer=FakeComposer(),
        memory_service=FakeMemoryService(),
        recovery_engine=recovery,
    )

    result = runtime.execute_turn("Execute failing task")
    assert result.status == TurnStatus.WAITING_FOR_USER


def test_recovery_transition_abort():
    """E3. RecoveryEngine decides ABORT, marking turn ABORTED."""
    recovery = FakeRecoveryEngine(action=RecoveryAction.ABORT, should_succeed_on_recovery=False)
    runtime = CognitiveRuntime(
        understanding=FakeUnderstanding(),
        decision_engine=FakeDecisionEngine(),
        planner=FakePlanner(),
        execution_engine=FakeExecutionEngine(),
        verifier=FakeVerifier(verified=False),
        composer=FakeComposer(),
        memory_service=FakeMemoryService(),
        recovery_engine=recovery,
    )

    result = runtime.execute_turn("Unrecoverable failure")
    assert result.status == TurnStatus.ABORTED


# ============================================================================
# F. OBSERVABILITY & EVENT SINK TESTS
# ============================================================================

def test_structured_events_emitted_in_order():
    """F1. Verify events are emitted sequentially matching the lifecycle stages."""
    sink = InMemoryEventSink()
    runtime = CognitiveRuntime(
        understanding=FakeUnderstanding(),
        decision_engine=FakeDecisionEngine(),
        planner=FakePlanner(),
        execution_engine=FakeExecutionEngine(),
        verifier=FakeVerifier(verified=True),
        composer=FakeComposer(),
        memory_service=FakeMemoryService(),
        event_sink=sink,
    )

    result = runtime.execute_turn("Observability check")
    events = sink.get_events(result.turn_id)
    event_types = [e.event_type for e in events]

    assert CognitiveEventType.TURN_STARTED in event_types
    assert CognitiveEventType.DECISION_MADE in event_types
    assert CognitiveEventType.PLAN_CREATED in event_types
    assert CognitiveEventType.TOOL_EXECUTED in event_types
    assert CognitiveEventType.VERIFICATION_COMPLETED in event_types
    assert CognitiveEventType.MEMORY_UPDATED in event_types
    assert CognitiveEventType.RESPONSE_COMPOSED in event_types
    assert CognitiveEventType.TURN_COMPLETED in event_types

    # Ensure timestamps are monotonic
    timestamps = [e.timestamp for e in events]
    assert timestamps == sorted(timestamps)


# ============================================================================
# G. EVENT IMMUTABILITY
# ============================================================================

def test_event_is_frozen_dataclass():
    """G1. Verify CognitiveEvent instances cannot be altered."""
    ev = CognitiveEvent(
        event_id="e1",
        turn_id="t1",
        session_id="s1",
        stage=CognitiveStage.RECEIVED,
        event_type=CognitiveEventType.TURN_STARTED,
        timestamp=time.time(),
        metadata={"key": "val"},
    )
    with pytest.raises(Exception):
        ev.status = "MUTATED"


# ============================================================================
# H. SENSITIVE DATA REDACTION
# ============================================================================

def test_sensitive_data_redacted_from_event_metadata():
    """H1. Passwords, secrets, credentials, and tokens are scrubbed from events."""
    raw_meta = {
        "user_password": "super_secret_password_123",
        "api_key": "sk-secret-12345",
        "auth_token": "token-xyz",
        "normal_field": "public_data",
    }
    sanitized = sanitize_event_metadata(raw_meta)
    assert sanitized["user_password"] == "[REDACTED]"
    assert sanitized["api_key"] == "[REDACTED]"
    assert sanitized["auth_token"] == "[REDACTED]"
    assert sanitized["normal_field"] == "public_data"


def test_large_image_data_summarized_not_dumped():
    """H2. Large screenshot base64 strings are summarized by length, not dumped raw."""
    large_image_bytes = "a" * 50000
    sanitized = sanitize_event_metadata({"screenshot": large_image_bytes})
    assert "IMAGE DATA: length=50000" in sanitized["screenshot"]


def test_long_typed_text_truncated():
    """H3. Very long typed text is safely truncated."""
    long_text = "x" * 500
    sanitized = sanitize_event_metadata({"typed_text": long_text})
    assert "truncated" in sanitized["typed_text"]


# ============================================================================
# I. TRACE BOUNDS TESTS
# ============================================================================

def test_trace_bounded_by_max_events():
    """I1. Verify trace snapshots never exceed max_events."""
    sink = InMemoryEventSink(max_events_per_turn=5)
    for i in range(10):
        sink.publish(
            CognitiveEvent(
                event_id=f"e_{i}",
                turn_id="t_bound",
                session_id="s1",
                stage=CognitiveStage.RECEIVED,
                event_type=CognitiveEventType.STAGE_STARTED,
                timestamp=time.time(),
            )
        )
    events = sink.get_events("t_bound")
    assert len(events) == 5  # capped at max 5


# ============================================================================
# J. ERROR BOUNDARIES TESTS
# ============================================================================

def test_expected_failure_structural_handling():
    """J1. Verification failure is captured structurally in VerificationResult, not crashed."""
    runtime = CognitiveRuntime(
        understanding=FakeUnderstanding(),
        decision_engine=FakeDecisionEngine(),
        planner=FakePlanner(),
        execution_engine=FakeExecutionEngine(),
        verifier=FakeVerifier(verified=False, reason="File not found on disk"),
        composer=FakeComposer(),
        memory_service=FakeMemoryService(),
    )
    result = runtime.execute_turn("Search file")
    assert result.status == TurnStatus.FAILED
    assert result.verification.verified is False


def test_unexpected_exception_marks_turn_failed_and_raises():
    """J2. Unexpected exceptions mark the turn FAILED, emit STAGE_FAILED, and propagate."""
    class CrashingExecutionEngine(ExecutionEngineInterface):
        def execute(self, plan: Plan) -> List[Result]:
            raise RuntimeError("Database hardware crash")

    sink = InMemoryEventSink()
    runtime = CognitiveRuntime(
        understanding=FakeUnderstanding(),
        decision_engine=FakeDecisionEngine(),
        planner=FakePlanner(),
        execution_engine=CrashingExecutionEngine(),
        verifier=FakeVerifier(),
        composer=FakeComposer(),
        memory_service=FakeMemoryService(),
        event_sink=sink,
    )

    with pytest.raises(RuntimeError) as exc_info:
        runtime.execute_turn("Execute")
    assert "Database hardware crash" in str(exc_info.value)

    events = sink.get_events()
    stage_failed_events = [e for e in events if e.event_type == CognitiveEventType.STAGE_FAILED]
    assert len(stage_failed_events) >= 1
    assert "Database hardware crash" in stage_failed_events[0].summary


# ============================================================================
# K. IDEMPOTENCY TESTS
# ============================================================================

def test_memory_written_once_per_turn():
    """K1. Memory records user message and assistant message exactly once."""
    memory = FakeMemoryService()
    runtime = CognitiveRuntime(
        understanding=FakeUnderstanding(),
        decision_engine=FakeDecisionEngine(),
        planner=FakePlanner(),
        execution_engine=FakeExecutionEngine(),
        verifier=FakeVerifier(),
        composer=FakeComposer(response_text="Answer"),
        memory_service=memory,
    )
    runtime.execute_turn("User query")
    assert len(memory.messages) == 2
    assert memory.messages[0]["role"] == MessageRole.USER
    assert memory.messages[1]["role"] == MessageRole.ASSISTANT


def test_tools_executed_once_per_attempt():
    """K2. Plan execution runs tools exactly once in a nominal turn."""
    exec_engine = FakeExecutionEngine()
    runtime = CognitiveRuntime(
        understanding=FakeUnderstanding(),
        decision_engine=FakeDecisionEngine(),
        planner=FakePlanner(),
        execution_engine=exec_engine,
        verifier=FakeVerifier(),
        composer=FakeComposer(),
        memory_service=FakeMemoryService(),
    )
    runtime.execute_turn("Run tools")
    assert len(exec_engine.calls) == 1


# ============================================================================
# L. TIME / DEADLINE / CANCELLATION TESTS
# ============================================================================

def test_turn_deadline_timeout_terminates_safely():
    """L1. Turn with elapsed time > timeout_seconds terminates with TurnStatus.ABORTED."""
    class SlowUnderstanding(RequestUnderstandingInterface):
        def understand(self, input_data: Any) -> Request:
            time.sleep(0.05)
            return Request(
                id="req_slow",
                session_id="s1",
                original_text="slow",
                normalized_text="slow",
                timestamp=datetime.now(),
            )

    runtime = CognitiveRuntime(
        understanding=SlowUnderstanding(),
        decision_engine=FakeDecisionEngine(),
        planner=FakePlanner(),
        execution_engine=FakeExecutionEngine(),
        verifier=FakeVerifier(),
        composer=FakeComposer(),
        memory_service=FakeMemoryService(),
    )
    # Give a tiny deadline of 0.01 seconds
    result = runtime.execute_turn("Execute slow", timeout_seconds=0.01)
    assert result.status == TurnStatus.ABORTED
    assert result.stage == CognitiveStage.ABORTED
    assert "exceeded" in result.trace.events[-1].summary.lower()


# ============================================================================
# M. SECURITY & SUBSYSTEM SEPARATION TESTS
# ============================================================================

def test_runtime_cannot_bypass_policy_engine():
    """M1. Policy DENY prevents tool execution unconditionally."""
    policy = FakePolicyEngine(decision=PolicyDecision.DENY, reason="Command violates safety policy")
    exec_engine = FakeExecutionEngine()
    runtime = CognitiveRuntime(
        understanding=FakeUnderstanding(),
        decision_engine=FakeDecisionEngine(),
        planner=FakePlanner(),
        policy_engine=policy,
        execution_engine=exec_engine,
        verifier=FakeVerifier(),
        composer=FakeComposer(),
        memory_service=FakeMemoryService(),
    )
    result = runtime.execute_turn("Malicious command")
    assert result.status == TurnStatus.FAILED
    assert "violates safety policy" in result.response
    assert len(exec_engine.calls) == 0


# ============================================================================
# N. END-TO-END ARCHITECTURAL TEST (Section 32)
# ============================================================================

def test_end_to_end_architectural_lifecycle():
    """
    Section 32 End-to-End Architectural Test:
    User Request -> Understand -> Decide -> Plan -> Context -> Route ->
    Reason -> Proposal -> Validate -> Policy -> Execute -> Observe ->
    Verify -> Memory -> Response -> Completed.
    """
    sink = InMemoryEventSink()
    memory = FakeMemoryService()
    context_mgr = FakeContextManager()
    router = FakeModelRouter(provider_id="qwen_local")
    proposal = ActionProposal(
        proposal_id="prop_001",
        goal_reference="search knowledge",
        action_type="tool",
        capability="knowledge",
        action="query",
        parameters={"query": "VFS"},
    )
    reasoning = FakeReasoningEngine(outcome=ReasoningOutcome.PROPOSE_ACTION, action_proposal=proposal)
    policy = FakePolicyEngine(decision=PolicyDecision.ALLOW, reason="Approved query")
    exec_engine = FakeExecutionEngine()
    verifier = FakeVerifier(verified=True)
    composer = FakeComposer(response_text="VFS architecture explained.")

    runtime = CognitiveRuntime(
        understanding=FakeUnderstanding(),
        decision_engine=FakeDecisionEngine(),
        planner=FakePlanner(),
        context_manager=context_mgr,
        model_router=router,
        reasoning_engine=reasoning,
        policy_engine=policy,
        execution_engine=exec_engine,
        verifier=verifier,
        composer=composer,
        memory_service=memory,
        event_sink=sink,
    )

    result = runtime.execute_turn("Explain VFS architecture")
    assert result.status == TurnStatus.SUCCEEDED
    assert result.stage == CognitiveStage.COMPLETED
    assert result.response == "VFS architecture explained."

    # Verify context, routing, reasoning, and policy were all invoked properly
    assert len(context_mgr.calls) == 1
    assert len(router.calls) == 1
    assert len(reasoning.calls) == 1
    assert len(policy.evaluated_calls) >= 1
    assert len(exec_engine.calls) == 1
    assert len(verifier.calls) == 1
    assert len(memory.messages) == 2


# ============================================================================
# O. FAILURE END-TO-END TEST (Section 33)
# ============================================================================

def test_end_to_end_failure_recovery_lifecycle():
    """
    Section 33 End-to-End Failure & Recovery Test:
    Request -> Plan -> Execute -> Verify FAIL -> RecoveryEngine ->
    RETRY -> Execute -> Verify SUCCESS -> Memory -> Response.
    """
    recovery = FakeRecoveryEngine(action=RecoveryAction.RETRY, should_succeed_on_recovery=True)
    runtime = CognitiveRuntime(
        understanding=FakeUnderstanding(),
        decision_engine=FakeDecisionEngine(),
        planner=FakePlanner(),
        execution_engine=FakeExecutionEngine(),
        verifier=FakeVerifier(verified=False),
        composer=FakeComposer(response_text="Recovered answer"),
        memory_service=FakeMemoryService(),
        recovery_engine=recovery,
    )

    result = runtime.execute_turn("Search with temporary network glitch")
    assert result.status == TurnStatus.SUCCEEDED
    assert result.recovery.plan_history[-1].recovery_action == RecoveryAction.RETRY
    assert result.verification.verified is True


# ============================================================================
# P. POLICY-BLOCK END-TO-END TEST (Section 34)
# ============================================================================

def test_end_to_end_policy_block_lifecycle():
    """
    Section 34 Policy-Block End-to-End Test:
    Reasoning -> ActionProposal -> Validator -> Policy DENY -> NO EXECUTION.
    """
    policy = FakePolicyEngine(decision=PolicyDecision.DENY, reason="Destructive command blocked by policy")
    exec_engine = FakeExecutionEngine()
    runtime = CognitiveRuntime(
        understanding=FakeUnderstanding(),
        decision_engine=FakeDecisionEngine(),
        planner=FakePlanner(),
        policy_engine=policy,
        execution_engine=exec_engine,
        verifier=FakeVerifier(),
        composer=FakeComposer(),
        memory_service=FakeMemoryService(),
    )

    result = runtime.execute_turn("rm -rf /")
    assert result.status == TurnStatus.FAILED
    assert "Destructive command blocked by policy" in result.response
    assert len(exec_engine.calls) == 0
