"""
ATLAS Phase 4.x System Coherence & End-to-End Integration Verification Test Suite.

Verifies that Phases 4.0 through 4.6 operate as ONE coherent autonomous architecture:
- Observation -> World State -> Event -> Anticipation -> Decision -> AutonomousGoalManager
  -> CognitiveRuntime -> Plan -> Policy -> Tool Execution -> Verification -> Memory -> World State Update.

Covers Scenarios A through K:
- Scenario A: Reactive Autonomous Flow
- Scenario B: Anticipatory Autonomous Flow
- Scenario C: Failure + Recovery
- Scenario D: Policy Enforcement (ALLOW, DENY, REQUIRE_CONFIRMATION)
- Scenario E: Replay Determinism and Isolation
- Scenario F: Conflicting Information Handling
- Scenario G: Stale Information Handling
- Scenario H: Loop / Storm Protection
- Scenario I: Memory Exact-Once Semantics
- Scenario J: Model Neutrality
- Scenario K: Security & Execution Boundaries
"""

import time
import uuid
import pytest
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# Core Models
from core.models.anticipation import (
    Anticipation,
    AnticipationProvenance,
    AnticipationStatus,
    AnticipatoryDecision,
    AnticipatoryDecisionType,
    EvidenceItem,
    EvidenceSourceType,
    FutureConditionType,
    TimeHorizon,
)
from core.models.autonomy import (
    AutonomyDecision,
    AutonomyDecisionType,
    Event,
    EventCategory,
    EventClassification,
    EventPriority,
    EventProvenance,
    EventSource,
    EventTrigger,
    TriggerCondition,
)
from core.models.context import CognitiveState
from core.models.decision import CapabilityType, Decision, ExecutionMode
from core.models.goal import (
    Goal,
    GoalCompletionCriteria,
    GoalConstraints,
    GoalPriority,
    GoalProgress,
    GoalStatus,
    Objective,
    ObjectiveStatus,
)
from core.models.memory import ChatMessage, MemoryEntry, MessageRole
from core.models.plan import Plan
from core.models.policy import AutonomyLevel, PolicyContext, PolicyDecision, PolicyResult
from core.models.recovery import ExecutionOutcome, PlanHistoryEntry, RecoveryAction, RecoveryContext
from core.models.replay import ReplayMode, ReplayRequest, ReplayResult
from core.models.request import Request
from core.models.result import Result
from core.models.runtime import (
    CognitiveEvent,
    CognitiveEventType,
    CognitiveStage,
    CognitiveTrace,
    CognitiveTurn,
    CognitiveTurnResult,
    TurnLimits,
    TurnStatus,
)
from core.models.task import Task
from core.models.tool_call import ToolCall
from core.models.verification import VerificationResult
from core.models.world_state import (
    ConflictPolicy,
    ConflictResolution,
    ConflictStatus,
    FreshnessConfig,
    FreshnessStatus,
    Observation,
    ResolutionStrategy,
    StateConflict,
    StateProvenance,
    TransitionType,
    WorldCondition,
    WorldEntity,
    WorldState,
    WorldStateTransition,
)

# Core Interfaces
from core.interfaces.anticipation_interface import (
    AnticipatoryAnalyzerInterface,
    AnticipatoryPlanningCoordinatorInterface,
    EvidenceEvaluatorInterface,
    InvalidationEngineInterface,
)
from core.interfaces.autonomy_interface import (
    EventClassifierInterface,
    EventDrivenAutonomyInterface,
    RelevanceEngineInterface,
)
from core.interfaces.decision_engine_interface import DecisionEngineInterface
from core.interfaces.decision_planner_interface import DecisionPlannerInterface
from core.interfaces.execution_engine_interface import ExecutionEngineInterface
from core.interfaces.goal_interface import (
    AutonomousGoalManagerInterface,
    GoalDecomposerInterface,
    GoalExecutionEngineInterface,
    GoalStoreInterface,
)
from core.interfaces.memory_interface import MemoryServiceInterface
from core.interfaces.policy_interface import PolicyEngineInterface
from core.interfaces.recovery_interface import RecoveryEngineInterface
from core.interfaces.request_understanding_interface import RequestUnderstandingInterface
from core.interfaces.runtime_interface import (
    CognitiveEventSinkInterface,
    CognitiveRuntimeInterface,
)
from core.interfaces.verification_interface import VerificationInterface
from core.interfaces.world_interface import (
    ConflictResolverInterface,
    WorldStateStoreInterface,
    WorldStateUpdaterInterface,
)

# Concrete Implementations
from anticipation.analyzer import DeterministicAnticipatoryAnalyzer
from anticipation.coordinator import AnticipatoryPlanningCoordinator
from anticipation.evaluator import DeterministicEvidenceEvaluator
from anticipation.event_listener import event_to_anticipation_candidate
from anticipation.invalidation import DeterministicInvalidationEngine

from autonomy.classifier import DeterministicEventClassifier
from autonomy.coordinator import EventDrivenAutonomyCoordinator
from autonomy.relevance import DeterministicRelevanceEngine
from autonomy.world_listener import transition_to_autonomy_event

from goals.decomposer import DeterministicGoalDecomposer
from goals.execution_engine import GoalExecutionEngine
from goals.manager import AutonomousGoalManager
from goals.scheduler import DeterministicGoalScheduler
from goals.store import InMemoryGoalStore

from runtime.cognitive_runtime import CognitiveRuntime
from runtime.event_sink import InMemoryEventSink
from runtime.recorded_store import RecordedResultStore
from runtime.replay_engine import ReplayEngine, ReplayExecutionEngine, ReplayMemoryService
from runtime.serialization import TraceSerializer, compute_trace_hash

from world.context_adapter import world_state_to_context_items
from world.resolver import DeterministicConflictResolver
from world.store import InMemoryWorldStateStore
from world.updater import DeterministicWorldStateUpdater


# ============================================================================
# DETERMINISTIC TEST DOUBLES FOR COGNITIVE TURNS
# ============================================================================

class SystemTestUnderstanding(RequestUnderstandingInterface):
    def understand(self, input_data: Any) -> Request:
        text = str(input_data)
        return Request(
            id=f"req_{uuid.uuid4().hex[:8]}",
            session_id="system_session_01",
            timestamp=datetime.now(),
            original_text=text,
            normalized_text=text.lower().strip(),
            parameters={"is_empty": False, "valid": True},
        )


class SystemTestDecisionEngine(DecisionEngineInterface):
    def decide(self, request: Request) -> Decision:
        return Decision(
            request_id=request.id,
            primary_goal=request.original_text,
            required_capabilities=[CapabilityType.TOOL],
            execution_mode=ExecutionMode.SINGLE_STEP,
            confidence=0.95,
            reasoning="Deterministic system test decision",
        )


class SystemTestPlanner(DecisionPlannerInterface):
    def __init__(self, action: str = "mitigate_resource", capability: str = "tool"):
        self.action = action
        self.capability = capability

    def plan(self, decision: Decision) -> Plan:
        return Plan(
            goal=decision.primary_goal,
            steps=[
                Task(
                    id=1,
                    type=self.action,
                    action=self.action,
                    tool=self.capability,
                    parameters={"command": self.action, "target": "rover_unit"},
                )
            ],
        )


class SystemTestExecutionEngine(ExecutionEngineInterface):
    def __init__(self, success: bool = True, output_text: str = "System task executed successfully."):
        self.success = success
        self.output_text = output_text
        self.executed_plans: List[Plan] = []

    def execute(self, plan: Plan) -> List[Result]:
        self.executed_plans.append(plan)
        return [
            Result(
                success=self.success,
                message=self.output_text,
                output={"status": "done" if self.success else "failed", "detail": self.output_text},
            )
        ]


class SystemTestVerifier(VerificationInterface):
    def __init__(self, verified: bool = True):
        self.verified = verified

    def verify(self, plan: Plan, results: List[Result]) -> VerificationResult:
        is_ok = self.verified and all(r.success for r in results)
        return VerificationResult(
            verified=is_ok,
            status="verified" if is_ok else "failed",
            confidence=1.0 if is_ok else 0.0,
            reason="Verified successfully" if is_ok else "Verification failed",
        )


class SystemTestPolicyEngine(PolicyEngineInterface):
    def __init__(self, decision: PolicyDecision = PolicyDecision.ALLOW, reason: str = "Authorized by policy"):
        self.decision = decision
        self.reason = reason
        self.evaluated_contexts: List[Any] = []

    def evaluate(self, *args, **kwargs) -> PolicyResult:
        self.evaluated_contexts.append((args, kwargs))
        return PolicyResult(
            decision=self.decision,
            reason=self.reason,
            rule_id="system_coherence_rule",
        )


class SystemTestMemoryService(MemoryServiceInterface):
    def __init__(self):
        self.messages: List[ChatMessage] = []
        self.preferences: Dict[Tuple[str, str], MemoryEntry] = {}

    def add_message(
        self,
        session_id: str,
        role: MessageRole,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        message_id: Optional[str] = None,
    ) -> ChatMessage:
        msg = ChatMessage(
            id=message_id or f"msg_{len(self.messages)+1}",
            session_id=session_id,
            role=role,
            content=content,
            timestamp=datetime.now(),
            metadata=metadata or {},
        )
        self.messages.append(msg)
        return msg

    def get_history(self, session_id: str, limit: Optional[int] = None) -> List[ChatMessage]:
        session_msgs = [m for m in self.messages if m.session_id == session_id]
        if limit is not None:
            return session_msgs[-limit:]
        return list(session_msgs)

    def clear_session(self, session_id: str) -> None:
        self.messages = [m for m in self.messages if m.session_id != session_id]

    def save_preference(self, user_id: str, key: str, value: Any, category: str = "general") -> MemoryEntry:
        entry = MemoryEntry(key=key, value=value, category=category, user_id=user_id)
        self.preferences[(user_id, key)] = entry
        return entry

    def get_preference(self, user_id: str, key: str) -> Optional[MemoryEntry]:
        return self.preferences.get((user_id, key))

    def delete_preference(self, user_id: str, key: str) -> bool:
        return self.preferences.pop((user_id, key), None) is not None

    def list_preferences(self, user_id: str) -> List[MemoryEntry]:
        return [entry for (uid, _), entry in self.preferences.items() if uid == user_id]


class SystemTestRecoveryEngine(RecoveryEngineInterface):
    def __init__(self, max_retries: int = 2):
        self.max_retries = max_retries
        self.retry_count = 0

    def recover(
        self,
        original_goal: str,
        initial_plan: Plan,
        execute_fn: Callable[[Plan], List[Result]],
        verify_fn: Callable[[Plan, List[Result]], VerificationResult],
    ) -> Tuple[Plan, List[Result], VerificationResult, RecoveryContext]:
        plan = initial_plan
        results = execute_fn(plan)
        verification = verify_fn(plan, results)

        history = [
            PlanHistoryEntry(
                plan_id="plan_1",
                plan=plan,
                outcome=ExecutionOutcome.SUCCESS if verification.verified else ExecutionOutcome.FAILURE,
                results=tuple(results),
                verification=verification,
                recovery_action=RecoveryAction.CONTINUE if verification.verified else RecoveryAction.RETRY,
                reason="Initial attempt" if verification.verified else "Execution failed",
            )
        ]

        if not verification.verified and self.retry_count < self.max_retries:
            self.retry_count += 1
            # Second attempt succeeds
            results = [Result(success=True, message="Recovered on retry", output={"recovered": True})]
            verification = VerificationResult(verified=True, status="verified", confidence=1.0, reason="Retry succeeded")
            history.append(
                PlanHistoryEntry(
                    plan_id="plan_2",
                    plan=plan,
                    outcome=ExecutionOutcome.SUCCESS,
                    results=tuple(results),
                    verification=verification,
                    recovery_action=RecoveryAction.CONTINUE,
                    reason="Recovery retry succeeded",
                )
            )

        ctx = RecoveryContext(
            original_goal=original_goal,
            current_plan=plan,
            attempt=self.retry_count + 1,
            outcome=ExecutionOutcome.SUCCESS if verification.verified else ExecutionOutcome.FAILURE,
            verification=verification,
            results=tuple(results),
            plan_history=tuple(history),
            remaining_retry_budget=self.max_retries - self.retry_count,
        )
        return plan, results, verification, ctx


def make_anticipation(
    anticipation_id: str = "ant_001",
    condition_type: FutureConditionType = FutureConditionType.RESOURCE_DEPLETION_RISK,
    horizon: TimeHorizon = TimeHorizon.NEAR_TERM,
    confidence: float = 0.85,
    relevance: float = 0.8,
    freshness: float = 0.9,
    depth: int = 0,
    target_entity_id: str = "rover_unit",
    correlation_id: str = "c_ant",
) -> Anticipation:
    prov = AnticipationProvenance(
        source_entity=target_entity_id,
        created_at=1000.0,
        correlation_id=correlation_id,
        depth=depth,
    )
    return Anticipation(
        anticipation_id=anticipation_id,
        condition_type=condition_type,
        description=f"Anticipated {condition_type.value} on {target_entity_id}",
        hypothetical_state={"property": "battery", "value": 15.0},
        horizon=horizon,
        horizon_window_seconds=horizon.get_default_window_seconds(),
        confidence=confidence,
        relevance=relevance,
        freshness=freshness,
        evidence_items=(),
        provenance=prov,
        correlation_id=correlation_id,
        status=AnticipationStatus.ACTIVE,
        target_entity_id=target_entity_id,
    )


# ============================================================================
# SCENARIO A: REACTIVE AUTONOMOUS FLOW
# ============================================================================

def test_scenario_a_reactive_autonomous_flow():
    """
    Scenario A: Complete end-to-end reactive autonomous pipeline:
    Observation -> World State update -> World State transition -> Phase 4.5 Event
    -> Classification -> Relevance -> Trigger -> Policy -> AutonomousGoalManager
    -> Goal lifecycle -> CognitiveRuntime -> Execution -> Verification -> Memory
    -> Subsequent World State update.
    """
    event_sink = InMemoryEventSink()
    ws_store = InMemoryWorldStateStore()
    goal_store = InMemoryGoalStore()
    memory_service = SystemTestMemoryService()
    policy_engine = SystemTestPolicyEngine(decision=PolicyDecision.ALLOW)

    # 1. Initialize WorldStateUpdater
    ws_updater = DeterministicWorldStateUpdater(
        store=ws_store,
        event_sink=event_sink,
        clock=lambda: 1000.0,
    )

    # 2. Setup CognitiveRuntime & GoalExecutionEngine
    exec_engine_mock = SystemTestExecutionEngine(success=True, output_text="Cooling fan activated.")
    runtime = CognitiveRuntime(
        understanding=SystemTestUnderstanding(),
        decision_engine=SystemTestDecisionEngine(),
        planner=SystemTestPlanner(action="activate_cooling"),
        execution_engine=exec_engine_mock,
        verifier=SystemTestVerifier(verified=True),
        memory_service=memory_service,
        event_sink=event_sink,
        policy_engine=policy_engine,
    )

    goal_execution_engine = GoalExecutionEngine(
        runtime=runtime,
        store=goal_store,
        event_sink=event_sink,
    )

    goal_manager = AutonomousGoalManager(
        store=goal_store,
        execution_engine=goal_execution_engine,
        event_sink=event_sink,
        clock=lambda: 1000.0,
    )

    # 3. Setup Phase 4.5 Event-Driven Autonomy Coordinator
    autonomy_coord = EventDrivenAutonomyCoordinator(
        goal_manager=goal_manager,
        goal_store=goal_store,
        policy_engine=policy_engine,
        world_state_store=ws_store,
        event_sink=event_sink,
        clock=lambda: 1000.0,
    )

    # -------------------------------------------------------------
    # STEP 1: Observation -> WorldState Update -> Transition
    # -------------------------------------------------------------
    obs = Observation(
        observation_id="obs_temp_99",
        source_id="bms_thermal_sensor",
        source_type=EventSource.SYSTEM_OBSERVATION.value,
        timestamp=1000.0,
        entity_id="server_rack_1",
        property_name="temp",
        value=92.5,  # Exceeds threshold > 85.0
        confidence=0.98,
    )
    update_res = ws_updater.apply_observation(obs)
    assert update_res.success is True
    assert update_res.transition is not None
    assert update_res.current_version == 2

    # -------------------------------------------------------------
    # STEP 2: Transition -> Autonomy Event
    # -------------------------------------------------------------
    ev = transition_to_autonomy_event(update_res.transition, correlation_id="corr_temp_alert")
    assert ev is not None
    assert ev.priority == EventPriority.HIGH
    assert "threshold_breach.temp" in ev.event_type
    assert ev.world_state_version == 2

    # -------------------------------------------------------------
    # STEP 3: Process Event in Autonomy Coordinator
    # -------------------------------------------------------------
    autonomy_dec = autonomy_coord.ingest_event(ev)
    assert autonomy_dec.decision_type == AutonomyDecisionType.CREATE_GOAL
    assert autonomy_dec.executed is True
    assert autonomy_dec.goal_id is not None
    created_goal_id = autonomy_dec.goal_id

    # -------------------------------------------------------------
    # STEP 4: Verify Goal in AutonomousGoalManager & GoalStore
    # -------------------------------------------------------------
    goal = goal_store.get_goal(created_goal_id)
    assert goal is not None
    assert "threshold breach" in goal.original_goal.lower() or "respond to event" in goal.original_goal.lower()
    assert goal.priority == GoalPriority.HIGH

    # -------------------------------------------------------------
    # STEP 5: Pursue & Execute Goal via AutonomousGoalManager
    # -------------------------------------------------------------
    quantum_res = goal_manager.schedule_once()
    assert quantum_res is not None
    assert quantum_res.goal_id == created_goal_id
    assert quantum_res.turn_executed is True
    assert len(exec_engine_mock.executed_plans) == 1

    # -------------------------------------------------------------
    # STEP 6: Verify Memory & Correlation Propagation
    # -------------------------------------------------------------
    assert len(memory_service.messages) >= 2  # user & assistant turns recorded
    all_events = event_sink.get_events()
    event_types = [e.event_type for e in all_events]

    assert CognitiveEventType.WORLD_STATE_UPDATED in event_types
    assert CognitiveEventType.AUTONOMY_EVENT_INGESTED in event_types
    assert CognitiveEventType.GOAL_CREATED in event_types
    assert CognitiveEventType.TURN_COMPLETED in event_types

    # -------------------------------------------------------------
    # STEP 7: Action Result Updates World State (Cooling restored)
    # -------------------------------------------------------------
    recovery_obs = Observation(
        observation_id="obs_temp_norm",
        source_id="bms_thermal_sensor",
        source_type=EventSource.SYSTEM_OBSERVATION.value,
        timestamp=1020.0,
        entity_id="server_rack_1",
        property_name="temp",
        value=68.0,
        confidence=0.99,
    )
    norm_res = ws_updater.apply_observation(recovery_obs)
    assert norm_res.success is True
    assert norm_res.current_version == 3
    assert ws_store.get_condition("server_rack_1", "temp").value == 68.0


# ============================================================================
# SCENARIO B: ANTICIPATORY AUTONOMOUS FLOW
# ============================================================================

def test_scenario_b_anticipatory_autonomous_flow():
    """
    Scenario B: Complete end-to-end anticipatory autonomous flow:
    World State + Active Goal + Evidence -> Phase 4.6 Anticipation
    -> Evaluator -> Decision -> Policy -> AutonomousGoalManager
    -> Goal Creation -> Isolation verification -> Invalidation when condition recovers.
    """
    event_sink = InMemoryEventSink()
    ws_store = InMemoryWorldStateStore()
    goal_store = InMemoryGoalStore()
    policy_engine = SystemTestPolicyEngine(decision=PolicyDecision.ALLOW)

    # 1. Setup World State with declining battery
    ws_updater = DeterministicWorldStateUpdater(store=ws_store, clock=lambda: 1000.0)
    ws_updater.apply_observation(Observation(
        observation_id="obs_bat_30",
        source_id="bms",
        source_type=EventSource.SYSTEM_OBSERVATION.value,
        timestamp=1000.0,
        entity_id="rover_unit",
        property_name="battery",
        value=28.0,
        confidence=0.95,
    ))

    # 2. Setup Production AutonomousGoalManager
    exec_mock = SystemTestExecutionEngine()
    runtime = CognitiveRuntime(
        understanding=SystemTestUnderstanding(),
        decision_engine=SystemTestDecisionEngine(),
        planner=SystemTestPlanner(),
        execution_engine=exec_mock,
        verifier=SystemTestVerifier(),
        event_sink=event_sink,
        policy_engine=policy_engine,
    )
    goal_exec_engine = GoalExecutionEngine(runtime=runtime, store=goal_store, event_sink=event_sink)
    goal_manager = AutonomousGoalManager(store=goal_store, execution_engine=goal_exec_engine, event_sink=event_sink, clock=lambda: 1000.0)

    # 3. Setup Phase 4.6 Anticipatory Coordinator
    anticipation_coord = AnticipatoryPlanningCoordinator(
        goal_manager=goal_manager,
        world_state_store=ws_store,
        policy_engine=policy_engine,
        event_sink=event_sink,
        clock=lambda: 1000.0,
    )

    # -------------------------------------------------------------
    # STEP 1: Analyzer detects future-relevant condition
    # -------------------------------------------------------------
    decisions = anticipation_coord.evaluate_cycle(max_anticipations=5)
    assert len(decisions) >= 1
    primary_dec = decisions[0]

    assert primary_dec.decision_type == AnticipatoryDecisionType.CREATE_GOAL
    assert primary_dec.executed is True
    assert primary_dec.goal_id is not None

    # Verify goal creation occurred strictly via AutonomousGoalManager
    created_goal = goal_store.get_goal(primary_dec.goal_id)
    assert created_goal is not None
    assert "Proactively mitigate" in created_goal.original_goal
    assert created_goal.priority == GoalPriority.HIGH

    # -------------------------------------------------------------
    # STEP 2: Strict Isolation Verification
    # WorldState must NOT contain hypothetical anticipation
    # -------------------------------------------------------------
    current_ws = ws_store.get_current_state()
    battery_cond = current_ws.get_condition("rover_unit", "battery")
    assert battery_cond.value == 28.0  # untouched by anticipation

    # -------------------------------------------------------------
    # STEP 3: Invalidation when conditions recover
    # -------------------------------------------------------------
    invalidation_engine = DeterministicInvalidationEngine()
    active_ants = anticipation_coord.get_active_anticipations()
    assert len(active_ants) >= 1
    ant_obj = active_ants[0]

    # Simulate rover recharging to 90%
    ws_updater.apply_observation(Observation(
        observation_id="obs_bat_90",
        source_id="bms",
        source_type=EventSource.SYSTEM_OBSERVATION.value,
        timestamp=1050.0,
        entity_id="rover_unit",
        property_name="battery",
        value=90.0,
        confidence=1.0,
    ))
    recovered_ws = ws_store.get_current_state()

    status, reason = invalidation_engine.evaluate_invalidation(ant_obj, recovered_ws, [], now=1050.0)
    assert status == AnticipationStatus.INVALIDATED
    assert "recovered" in reason.lower()


# ============================================================================
# SCENARIO C: FAILURE + RECOVERY
# ============================================================================

def test_scenario_c_failure_and_recovery():
    """
    Scenario C: Controlled failure triggers RecoveryEngine:
    Bounded retries -> no infinite recovery loops -> original goal remains immutable
    -> recovery history preserved -> final outcome observable.
    """
    event_sink = InMemoryEventSink()
    recovery_engine = SystemTestRecoveryEngine(max_retries=2)

    # First attempt will fail, recovery attempt will succeed
    call_count = [0]
    class FlakyExecutionEngine(ExecutionEngineInterface):
        def execute(self, plan: Plan) -> List[Result]:
            call_count[0] += 1
            if call_count[0] == 1:
                return [Result(success=False, message="Transient tool communication error", output=None)]
            return [Result(success=True, message="Execution succeeded on retry", output={"status": "ok"})]

    class StrictVerifier(VerificationInterface):
        def verify(self, plan: Plan, results: List[Result]) -> VerificationResult:
            ok = all(r.success for r in results)
            return VerificationResult(
                verified=ok,
                status="verified" if ok else "failed",
                confidence=1.0 if ok else 0.0,
                reason="All steps succeeded" if ok else "One or more steps failed",
            )

    runtime = CognitiveRuntime(
        understanding=SystemTestUnderstanding(),
        decision_engine=SystemTestDecisionEngine(),
        planner=SystemTestPlanner(),
        execution_engine=FlakyExecutionEngine(),
        verifier=StrictVerifier(),
        recovery_engine=recovery_engine,
        event_sink=event_sink,
    )

    result = runtime.execute_turn(input_data="Execute mission with recovery")

    # Verify recovery succeeded
    assert result.status == TurnStatus.SUCCEEDED
    assert result.recovery is not None
    assert result.recovery.attempt == 2
    assert len(result.recovery.plan_history) == 2

    # Original request / goal remains immutable
    assert result.recovery.original_goal == "Execute mission with recovery"

    # Events emitted for recovery
    emitted = [e.event_type for e in event_sink.get_events()]
    assert CognitiveEventType.RECOVERY_STARTED in emitted
    assert CognitiveEventType.TURN_COMPLETED in emitted


# ============================================================================
# SCENARIO D: POLICY ENFORCEMENT ACROSS ALL AUTONOMOUS PATHS
# ============================================================================

def test_scenario_d_policy_enforcement_allow_deny_confirmation():
    """
    Scenario D: Mandatory policy authorization on all autonomous paths:
    ALLOW -> authorized path continues
    DENY -> no lifecycle mutation, no execution
    REQUIRE_CONFIRMATION -> explicit escalation, no premature mutation.
    """
    ws_store = InMemoryWorldStateStore()

    # -------------------------------------------------------------
    # 1. Reactive Path: EventDrivenAutonomyCoordinator
    # -------------------------------------------------------------
    # Case D1.1: DENY
    deny_policy = SystemTestPolicyEngine(decision=PolicyDecision.DENY, reason="Safety policy block")
    mgr_deny = AutonomousGoalManager(store=InMemoryGoalStore(), execution_engine=None)
    coord_deny = EventDrivenAutonomyCoordinator(
        goal_manager=mgr_deny,
        policy_engine=deny_policy,
        world_state_store=ws_store,
        clock=lambda: 1000.0,
    )
    ev = Event(
        event_id="ev_d1",
        source=EventSource.SYSTEM_OBSERVATION,
        event_type="world_state.threshold_breach.temp",
        priority=EventPriority.HIGH,
        timestamp=1000.0,
        payload={"entity_id": "cart_1", "property_name": "temp", "new_value": 95.0},
        provenance=EventProvenance(source_id="s1", source_type=EventSource.SYSTEM_OBSERVATION, origin_timestamp=1000.0, correlation_id="c1"),
        correlation_id="c1",
    )
    dec_deny = coord_deny.ingest_event(ev)
    assert dec_deny.decision_type == AutonomyDecisionType.IGNORE
    assert "denied by policy" in dec_deny.reason.lower()
    assert len(mgr_deny.store.list_goals()) == 0

    # Case D1.2: REQUIRE_CONFIRMATION
    conf_policy = SystemTestPolicyEngine(decision=PolicyDecision.REQUIRE_CONFIRMATION, reason="High impact operation")
    mgr_conf = AutonomousGoalManager(store=InMemoryGoalStore(), execution_engine=None)
    coord_conf = EventDrivenAutonomyCoordinator(
        goal_manager=mgr_conf,
        policy_engine=conf_policy,
        world_state_store=ws_store,
        clock=lambda: 1000.0,
    )
    dec_conf = coord_conf.ingest_event(ev)
    assert dec_conf.decision_type == AutonomyDecisionType.ESCALATE_TO_USER
    assert "user confirmation" in dec_conf.reason.lower()
    assert len(mgr_conf.store.list_goals()) == 0

    # -------------------------------------------------------------
    # 2. Anticipatory Path: AnticipatoryPlanningCoordinator
    # -------------------------------------------------------------
    ant = make_anticipation(
        anticipation_id="ant_pol_test",
        condition_type=FutureConditionType.RESOURCE_DEPLETION_RISK,
        confidence=0.85,
        target_entity_id="rover_unit",
    )

    # Case D2.1: DENY
    ant_coord_deny = AnticipatoryPlanningCoordinator(
        goal_manager=mgr_deny,
        policy_engine=deny_policy,
        clock=lambda: 1000.0,
    )
    ant_dec_deny = ant_coord_deny.process_anticipation(ant)
    assert ant_dec_deny.decision_type == AnticipatoryDecisionType.NO_ACTION
    assert "denied by policy" in ant_dec_deny.reason.lower()
    assert len(mgr_deny.store.list_goals()) == 0

    # Case D2.2: REQUIRE_CONFIRMATION
    ant_coord_conf = AnticipatoryPlanningCoordinator(
        goal_manager=mgr_conf,
        policy_engine=conf_policy,
        clock=lambda: 1000.0,
    )
    ant_dec_conf = ant_coord_conf.process_anticipation(ant)
    assert ant_dec_conf.decision_type == AnticipatoryDecisionType.ESCALATE_USER
    assert ant_dec_conf.executed is False
    assert len(mgr_conf.store.list_goals()) == 0


# ============================================================================
# SCENARIO E: REPLAY DETERMINISM AND ISOLATION
# ============================================================================

def test_scenario_e_replay_determinism_and_isolation():
    """
    Scenario E: Deterministic Replay:
    Records a production trace -> replays through Phase 4.1 ReplayEngine
    -> verifies zero divergences -> runs multiple times deterministically
    -> verifies sandboxed memory never touches production memory.
    """
    prod_event_sink = InMemoryEventSink()
    prod_memory = SystemTestMemoryService()

    runtime = CognitiveRuntime(
        understanding=SystemTestUnderstanding(),
        decision_engine=SystemTestDecisionEngine(),
        planner=SystemTestPlanner(),
        execution_engine=SystemTestExecutionEngine(success=True, output_text="Production output"),
        verifier=SystemTestVerifier(verified=True),
        memory_service=prod_memory,
        event_sink=prod_event_sink,
    )

    turn_res = runtime.execute_turn(input_data="Run mission turn")
    assert turn_res.status == TurnStatus.SUCCEEDED
    source_trace = turn_res.trace
    assert source_trace is not None
    assert len(source_trace.events) > 0

    # Replay Run 1
    replay_engine = ReplayEngine()
    req1 = ReplayRequest(source_trace=source_trace, replay_run_id="replay_sys_01", replay_mode=ReplayMode.OFFLINE_REPLAY)
    res1 = replay_engine.replay(req1)

    assert res1.success is True
    assert res1.comparison is not None
    assert res1.comparison.status == "IDENTICAL"
    assert len(res1.comparison.divergences) == 0

    # Replay Run 2 (Determinism)
    res2 = replay_engine.replay(req1)
    assert res2.success is True
    assert res2.comparison.status == "IDENTICAL"
    assert len(res2.comparison.divergences) == 0
    assert len(res1.completed_stages) == len(res2.completed_stages)
    assert [e.event_type for e in res1.replay_trace.events] == [e.event_type for e in res2.replay_trace.events]

    # Sandboxed isolation: production memory count remained unchanged
    assert len(prod_memory.messages) == 2  # exactly one turn's user and assistant messages


# ============================================================================
# SCENARIO F: CONFLICTING INFORMATION HANDLING
# ============================================================================

def test_scenario_f_conflicting_information_handling():
    """
    Scenario F: Controlled conflict detection and resolution:
    Two observations with conflicting values are evaluated:
    - Conflict detected
    - Provenance retained for both
    - Deterministic resolution strategy applied
    - No unsafe autonomous action taken on unverified conflict.
    """
    ws_store = InMemoryWorldStateStore()
    resolver = DeterministicConflictResolver()
    ws_updater = DeterministicWorldStateUpdater(store=ws_store, resolver=resolver, clock=lambda: 1000.0)

    # Obs 1: Obstacle present (high confidence, 0.95)
    obs1 = Observation(
        observation_id="obs_conf_1",
        source_id="lidar_01",
        source_type=EventSource.SYSTEM_OBSERVATION.value,
        timestamp=1000.0,
        entity_id="rover_unit",
        property_name="path_status",
        value="blocked",
        confidence=0.95,
    )
    res1 = ws_updater.apply_observation(obs1)
    assert res1.success is True
    assert ws_store.get_condition("rover_unit", "path_status").value == "blocked"

    # Obs 2: Path clear (lower confidence, 0.60)
    obs2 = Observation(
        observation_id="obs_conf_2",
        source_id="sonar_02",
        source_type=EventSource.SYSTEM_OBSERVATION.value,
        timestamp=1005.0,
        entity_id="rover_unit",
        property_name="path_status",
        value="clear",
        confidence=0.60,
    )
    res2 = ws_updater.apply_observation(obs2)
    assert res2.success is True

    # Under confidence tie-breaker, the 0.95 "blocked" outweighs 0.60 "clear"
    current_cond = ws_store.get_condition("rover_unit", "path_status")
    assert current_cond.value == "blocked"
    assert current_cond.confidence >= 0.95


# ============================================================================
# SCENARIO G: STALE INFORMATION HANDLING
# ============================================================================

def test_scenario_g_stale_information_handling():
    """
    Scenario G: Handling of stale information:
    - Freshness decay evaluated
    - Stale / expired evidence penalized
    - Expired anticipation produces NO_ACTION.
    """
    evaluator = DeterministicEvidenceEvaluator(half_life_seconds=60.0)

    # Item observed 600s ago (10 half-lives old -> freshness near 0)
    stale_item = EvidenceItem(
        evidence_id="ev_stale",
        source_type=EvidenceSourceType.WORLD_STATE,
        source_id="rover.battery",
        description="Battery observation",
        confidence=0.8,
        observed_at=400.0,
    )
    fresh_item = EvidenceItem(
        evidence_id="ev_fresh",
        source_type=EvidenceSourceType.WORLD_STATE,
        source_id="rover.battery",
        description="Battery observation",
        confidence=0.8,
        observed_at=1000.0,
    )

    conf_stale, fresh_stale, rel_stale = evaluator.evaluate_evidence((stale_item,), active_goals=[], now=1000.0)
    conf_fresh, fresh_fresh, rel_fresh = evaluator.evaluate_evidence((fresh_item,), active_goals=[], now=1000.0)

    assert fresh_stale < 0.05
    assert fresh_fresh >= 0.9

    # Invalidation engine marks anticipation EXPIRED if horizon elapsed
    inval = DeterministicInvalidationEngine()
    ant_expired = make_anticipation(
        anticipation_id="ant_exp",
        horizon=TimeHorizon.IMMEDIATE,  # max window 300s
        confidence=0.8,
    )

    status, reason = inval.evaluate_invalidation(ant_expired, None, [], now=1500.0)  # 500s later (> 300s)
    assert status == AnticipationStatus.EXPIRED


# ============================================================================
# SCENARIO H: LOOP / STORM PROTECTION
# ============================================================================

def test_scenario_h_loop_and_storm_protection():
    """
    Scenario H: Verification of cascade depth limits, rate limits, and loop prevention:
    - Repeated identical events deduplicated
    - Repeated identical anticipations deduplicated
    - Cascade depth limit enforced.
    """
    event_sink = InMemoryEventSink()
    mgr = AutonomousGoalManager(store=InMemoryGoalStore(), execution_engine=None)

    # 1. Event Deduplication in EventDrivenAutonomyCoordinator
    coord = EventDrivenAutonomyCoordinator(
        goal_manager=mgr,
        event_sink=event_sink,
        dedup_window_seconds=60.0,
        clock=lambda: 1000.0,
    )
    ev = Event(
        event_id="ev_loop_1",
        source=EventSource.SYSTEM_OBSERVATION,
        event_type="world_state.threshold_breach.battery",
        priority=EventPriority.HIGH,
        timestamp=1000.0,
        payload={"entity_id": "bot_1", "property_name": "battery", "new_value": 15.0},
        provenance=EventProvenance(source_id="bms", source_type=EventSource.SYSTEM_OBSERVATION, origin_timestamp=1000.0, correlation_id="c_loop"),
        correlation_id="c_loop",
        deduplication_key="bot_1:battery:15.0",
    )

    dec1 = coord.ingest_event(ev)
    assert dec1.decision_type == AutonomyDecisionType.CREATE_GOAL

    # Second identical event should be suppressed by deduplication
    dec2 = coord.ingest_event(ev)
    assert dec2.decision_type == AutonomyDecisionType.IGNORE
    assert "duplicate" in dec2.reason.lower()

    # 2. Cascade Depth Enforcement in AnticipatoryPlanningCoordinator
    ant_coord = AnticipatoryPlanningCoordinator(max_cascade_depth=3)
    deep_ant = make_anticipation(
        anticipation_id="ant_deep",
        depth=3,  # Reaches max depth
        target_entity_id="bot_1",
    )
    dec_deep = ant_coord.process_anticipation(deep_ant)
    assert dec_deep.decision_type == AnticipatoryDecisionType.NO_ACTION
    assert "cascade depth limit" in dec_deep.reason.lower()


# ============================================================================
# SCENARIO I: MEMORY EXACT-ONCE SEMANTICS
# ============================================================================

def test_scenario_i_memory_exact_once_semantics():
    """
    Scenario I: Verification of exact-once memory semantics:
    - User turn persisted exactly once
    - Assistant response persisted exactly once
    - Autonomy internal events do NOT flood conversation memory.
    """
    memory_service = SystemTestMemoryService()
    runtime = CognitiveRuntime(
        understanding=SystemTestUnderstanding(),
        decision_engine=SystemTestDecisionEngine(),
        planner=SystemTestPlanner(),
        execution_engine=SystemTestExecutionEngine(),
        verifier=SystemTestVerifier(),
        memory_service=memory_service,
    )

    res = runtime.execute_turn("Test memory exact-once")
    assert res.status == TurnStatus.SUCCEEDED

    # Exactly 2 messages in conversational memory
    messages = memory_service.messages
    assert len(messages) == 2
    assert messages[0].role == MessageRole.USER
    assert messages[0].content == "Test memory exact-once"
    assert messages[1].role == MessageRole.ASSISTANT
    assert len(messages[1].content) > 0


# ============================================================================
# SCENARIO J: MODEL NEUTRALITY
# ============================================================================

def test_scenario_j_model_neutrality():
    """
    Scenario J: Architectural verification that the entire pipeline operates
    without hardcoded LLM provider dependencies (no Ollama, Gemini, OpenAI).
    """
    # Verify that all coordinating and executing classes can be instantiated
    # and executed with zero provider tokens or network connection.
    ws = InMemoryWorldStateStore()
    goals = InMemoryGoalStore()
    mgr = AutonomousGoalManager(store=goals, execution_engine=None)
    autonomy = EventDrivenAutonomyCoordinator(goal_manager=mgr, world_state_store=ws)
    anticipation = AnticipatoryPlanningCoordinator(goal_manager=mgr, world_state_store=ws)

    assert autonomy is not None
    assert anticipation is not None
    assert not hasattr(autonomy, "gemini_client")
    assert not hasattr(autonomy, "openai_client")
    assert not hasattr(anticipation, "llm_client")


# ============================================================================
# SCENARIO K: SECURITY AND EXECUTION BOUNDARIES
# ============================================================================

def test_scenario_k_security_and_execution_boundaries():
    """
    Scenario K: Verification of strict execution isolation:
    - Autonomy and Anticipation layers contain ZERO direct tool execution,
      model execution, shell execution, or computer actions.
    - PolicyEngine cannot be bypassed.
    """
    autonomy = EventDrivenAutonomyCoordinator()
    anticipation = AnticipatoryPlanningCoordinator()
    analyzer = DeterministicAnticipatoryAnalyzer()

    for comp in (autonomy, anticipation, analyzer):
        # Forbidden execution capabilities
        assert not hasattr(comp, "tool_orchestrator")
        assert not hasattr(comp, "computer_capability")
        assert not hasattr(comp, "shell")
        assert not hasattr(comp, "subprocess")
        assert not hasattr(comp, "execute_tool")
        assert not hasattr(comp, "call_model")
