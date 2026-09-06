import collections
import time
import pytest
from typing import Any, Dict, List, Optional, Tuple

from core.interfaces.goal_interface import AutonomousGoalManagerInterface, GoalStoreInterface, GoalExecutionEngineInterface
from core.interfaces.policy_interface import PolicyEngineInterface
from core.interfaces.runtime_interface import CognitiveEventSinkInterface
from core.interfaces.world_interface import WorldStateStoreInterface
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
from core.models.autonomy import Event, EventCategory, EventPriority, EventProvenance, EventSource
from core.models.goal import Goal, GoalConstraints, GoalPriority, GoalStatus
from core.models.policy import PolicyContext, PolicyDecision, PolicyResult
from core.models.runtime import CognitiveEvent, CognitiveEventType, CognitiveStage
from core.models.world_state import StateProvenance, WorldCondition, WorldEntity, WorldState
from goals.manager import AutonomousGoalManager
from goals.store import InMemoryGoalStore
from world.store import InMemoryWorldStateStore
from anticipation.analyzer import DeterministicAnticipatoryAnalyzer
from anticipation.coordinator import AnticipatoryPlanningCoordinator
from anticipation.evaluator import DeterministicEvidenceEvaluator
from anticipation.event_listener import event_to_anticipation_candidate
from anticipation.invalidation import DeterministicInvalidationEngine


# ============================================================================
# TEST DOUBLES & HARNESSES
# ============================================================================

class FakeCognitiveEventSink(CognitiveEventSinkInterface):
    def __init__(self):
        self.events: List[CognitiveEvent] = []

    def publish(self, event: CognitiveEvent) -> None:
        self.events.append(event)

    def receive_event(self, event: CognitiveEvent) -> None:
        self.events.append(event)

    def get_events(self) -> List[CognitiveEvent]:
        return list(self.events)

    def clear(self) -> None:
        self.events.clear()


class FakePolicyEngine(PolicyEngineInterface):
    def __init__(self, decision: PolicyDecision = PolicyDecision.ALLOW, reason: str = "Authorized"):
        self.decision = decision
        self.reason = reason
        self.evaluated_contexts: List[PolicyContext] = []

    def evaluate(self, context: PolicyContext) -> PolicyResult:
        self.evaluated_contexts.append(context)
        return PolicyResult(
            decision=self.decision,
            reason=self.reason,
            rule_id="anticipation_rule",
        )


class FakeGoalManager(AutonomousGoalManagerInterface):
    def __init__(self, store: Optional[GoalStoreInterface] = None):
        self.store = store or InMemoryGoalStore()
        self.created_goals: List[Goal] = []
        self.priorities: Dict[str, Any] = {}

    def create_goal(self, goal: Goal) -> Goal:
        self.created_goals.append(goal)
        return self.store.create_goal(goal)

    def schedule_once(self) -> Optional[Any]:
        return None

    def run_next_quantum(self) -> Optional[Any]:
        return None

    def claim_active_goal(self, goal_id: str) -> bool:
        return True

    def release_active_goal(self, goal_id: str) -> None:
        pass

    def get_active_goal(self) -> Optional[str]:
        return None

    def pause_goal(self, goal_id: str, reason: str = "") -> Goal:
        goal = self.store.get_goal(goal_id)
        return goal

    def resume_goal(self, goal_id: str) -> Goal:
        goal = self.store.get_goal(goal_id)
        return goal

    def cancel_goal(self, goal_id: str, reason: str = "") -> Goal:
        goal = self.store.get_goal(goal_id)
        return goal

    def set_goal_priority(self, goal_id: str, priority: Any) -> Goal:
        self.priorities[goal_id] = priority
        goal = self.store.get_goal(goal_id)
        if goal:
            goal.priority = GoalPriority.from_str(str(priority))
            self.store.update_goal(goal)
            return goal
        return None

    def set_goal_deadline(self, goal_id: str, deadline: Optional[float]) -> Goal:
        return None

    def get_management_state(self) -> Any:
        return None


def make_evidence_item(
    evidence_id: str = "ev_01",
    source_type: EvidenceSourceType = EvidenceSourceType.WORLD_STATE,
    confidence: float = 0.8,
    observed_at: float = 1000.0,
    expires_at: Optional[float] = None,
    description: str = "Battery level reported at 30%.",
) -> EvidenceItem:
    return EvidenceItem(
        evidence_id=evidence_id,
        source_type=source_type,
        source_id="rover_unit.battery",
        description=description,
        confidence=confidence,
        observed_at=observed_at,
        expires_at=expires_at,
    )


def make_anticipation(
    anticipation_id: str = "ant_001",
    condition_type: FutureConditionType = FutureConditionType.RESOURCE_DEPLETION_RISK,
    horizon: TimeHorizon = TimeHorizon.NEAR_TERM,
    confidence: float = 0.8,
    relevance: float = 0.7,
    freshness: float = 0.9,
    depth: int = 0,
    timestamp: float = 1000.0,
    target_entity_id: str = "rover_unit",
    related_goal_ids: Tuple[str, ...] = (),
) -> Anticipation:
    ev = make_evidence_item(observed_at=timestamp)
    return Anticipation(
        anticipation_id=anticipation_id,
        condition_type=condition_type,
        description=f"Anticipated {condition_type.value} on {target_entity_id}.",
        target_entity_id=target_entity_id,
        hypothetical_state={"battery": 10.0},
        horizon=horizon,
        horizon_window_seconds=horizon.get_default_window_seconds(),
        confidence=confidence,
        relevance=relevance,
        freshness=freshness,
        evidence_items=(ev,),
        provenance=AnticipationProvenance(
            source_entity="test_source",
            created_at=timestamp,
            correlation_id="corr_test_01",
            depth=depth,
        ),
        correlation_id="corr_test_01",
        related_goal_ids=related_goal_ids,
    )


# ============================================================================
# 1. ANTICIPATION DOMAIN CREATION & IMMUTABILITY
# ============================================================================

def test_anticipation_domain_creation_and_immutability():
    """1. Anticipation domain model creation, validation, and immutability."""
    ant = make_anticipation()
    assert ant.anticipation_id == "ant_001"
    assert ant.condition_type == FutureConditionType.RESOURCE_DEPLETION_RISK
    assert ant.status == AnticipationStatus.ACTIVE
    assert ant.horizon == TimeHorizon.NEAR_TERM

    # Immutability check
    with pytest.raises((AttributeError, TypeError)):
        ant.status = AnticipationStatus.EXPIRED


def test_anticipation_provenance_retention():
    """2. Anticipation retains complete provenance including depth and correlation IDs."""
    prov = AnticipationProvenance(
        source_entity="analyzer_v1",
        created_at=1000.0,
        correlation_id="corr_999",
        depth=2,
        causation_id="cause_123",
    )
    assert prov.depth == 2
    assert prov.correlation_id == "corr_999"
    assert prov.causation_id == "cause_123"


def test_time_horizon_representation_and_bounds():
    """3. TimeHorizon provides bounded min/max second ranges, never vague text."""
    for h in TimeHorizon:
        window = h.get_default_window_seconds()
        assert len(window) == 2
        assert window[0] >= 0.0
        assert window[1] >= window[0]

    assert TimeHorizon.from_seconds(60.0) == TimeHorizon.IMMEDIATE
    assert TimeHorizon.from_seconds(600.0) == TimeHorizon.NEAR_TERM
    assert TimeHorizon.from_seconds(3600.0) == TimeHorizon.MEDIUM_TERM
    assert TimeHorizon.from_seconds(10000.0) == TimeHorizon.LONGER_TERM


def test_evidence_retention_and_immutability():
    """4. EvidenceItems retain explainability and provenance metrics."""
    ev = make_evidence_item(evidence_id="ev_fixed", confidence=0.85, observed_at=500.0, expires_at=600.0)
    assert ev.evidence_id == "ev_fixed"
    assert ev.confidence == 0.85
    assert ev.is_fresh(now=550.0) is True
    assert ev.is_fresh(now=650.0) is False

    with pytest.raises((AttributeError, TypeError)):
        ev.confidence = 0.99


def test_confidence_orthogonal_to_freshness_and_relevance():
    """5. Confidence, freshness, and relevance are strictly separated metrics."""
    evaluator = DeterministicEvidenceEvaluator(half_life_seconds=100.0)
    # High confidence, but very old observation (stale)
    old_ev = make_evidence_item(confidence=0.95, observed_at=100.0)
    conf, fresh, rel = evaluator.evaluate_evidence([old_ev], active_goals=[], now=1000.0)

    # High confidence preserved
    assert conf >= 0.90
    # Freshness heavily decayed
    assert fresh < 0.01
    # Relevance baseline
    assert rel == 0.4  # 0.3 base + 0.1 world state


def test_freshness_decay_evaluation():
    """6. Exponential decay reduces freshness over elapsed time."""
    evaluator = DeterministicEvidenceEvaluator(half_life_seconds=1000.0)
    ev = make_evidence_item(observed_at=1000.0)

    _, f_now, _ = evaluator.evaluate_evidence([ev], [], now=1000.0)
    _, f_half, _ = evaluator.evaluate_evidence([ev], [], now=2000.0)

    assert pytest.approx(f_now, 0.01) == 1.0
    assert pytest.approx(f_half, 0.01) == 0.5


def test_relevance_matching_active_goals():
    """7. Active goal match elevates relevance score."""
    evaluator = DeterministicEvidenceEvaluator()
    ev = make_evidence_item(description="Chiller temperature exceeds normal operational boundaries.")

    goal_matching = Goal(original_goal="Maintain chiller temperature stability")
    goal_unrelated = Goal(original_goal="Clean workspace floor")

    _, _, rel_high = evaluator.evaluate_evidence([ev], [goal_matching], now=1000.0)
    _, _, rel_low = evaluator.evaluate_evidence([ev], [goal_unrelated], now=1000.0)

    assert rel_high > rel_low


def test_future_condition_classification_and_safety():
    """8. FutureConditionType maps known types and fails safe on UNKNOWN."""
    assert FutureConditionType.from_str("resource_depletion_risk") == FutureConditionType.RESOURCE_DEPLETION_RISK
    assert FutureConditionType.from_str("deadline_risk") == FutureConditionType.DEADLINE_RISK
    assert FutureConditionType.from_str("alien_invasion_risk") == FutureConditionType.UNKNOWN


def test_unknown_condition_type_fails_safe():
    """9. Unknown future condition types are rejected without autonomous intervention."""
    coordinator = AnticipatoryPlanningCoordinator()
    ant_unknown = make_anticipation(condition_type=FutureConditionType.UNKNOWN)

    dec = coordinator.process_anticipation(ant_unknown)
    assert dec.decision_type == AnticipatoryDecisionType.NO_ACTION
    assert "unknown condition" in dec.reason.lower()


def test_insufficient_evidence_handling():
    """10. Insufficient evidence confidence downgrades to MONITOR or NO_ACTION."""
    coordinator = AnticipatoryPlanningCoordinator(
        min_confidence_threshold=0.6,
        clock=lambda: 1000.0,
    )
    # Low confidence (0.35) -> MONITOR
    ant_moderate = make_anticipation(confidence=0.35, relevance=0.7, timestamp=1000.0)
    dec_mon = coordinator.process_anticipation(ant_moderate)
    assert dec_mon.decision_type == AnticipatoryDecisionType.MONITOR

    # Very low confidence (0.15) -> NO_ACTION
    ant_very_low = make_anticipation(confidence=0.15, relevance=0.7, timestamp=1000.0)
    dec_none = coordinator.process_anticipation(ant_very_low)
    assert dec_none.decision_type == AnticipatoryDecisionType.NO_ACTION


def test_duplicate_anticipation_suppression_via_signature():
    """11. Duplicate anticipations within deduplication window are suppressed."""
    current_time = 1000.0
    coordinator = AnticipatoryPlanningCoordinator(
        clock=lambda: current_time,
        dedup_window_seconds=60.0,
    )

    ant1 = make_anticipation(anticipation_id="a1")
    dec1 = coordinator.process_anticipation(ant1)
    assert dec1.decision_type != AnticipatoryDecisionType.NO_ACTION or "suppressed" not in dec1.reason.lower()

    # Identical signature arrives 10s later
    current_time = 1010.0
    ant2 = make_anticipation(anticipation_id="a2")
    assert ant1.get_signature() == ant2.get_signature()

    dec2 = coordinator.process_anticipation(ant2)
    assert dec2.decision_type == AnticipatoryDecisionType.NO_ACTION
    assert "duplicate" in dec2.reason.lower()


def test_duplicate_goal_prevention_on_anticipation():
    """12. Coordinator prevents creating multiple goals for the same active anticipation."""
    store = InMemoryGoalStore()
    mgr = FakeGoalManager(store=store)
    current_time = 1000.0
    coordinator = AnticipatoryPlanningCoordinator(
        goal_manager=mgr,
        clock=lambda: current_time,
        dedup_window_seconds=1.0,  # Short window
    )

    ant1 = make_anticipation(anticipation_id="a1", timestamp=1000.0)
    dec1 = coordinator.process_anticipation(ant1)
    assert dec1.decision_type == AnticipatoryDecisionType.CREATE_GOAL
    assert dec1.executed is True
    assert len(mgr.created_goals) == 1

    # Window expires, but active goal exists in store with matching dedup key
    current_time = 1010.0
    ant2 = make_anticipation(anticipation_id="a2", timestamp=1010.0)
    dec2 = coordinator.process_anticipation(ant2)
    # Proactive goal creation suppressed because goal for this condition already active!
    assert dec2.executed is False
    assert "duplicate prevented" in dec2.reason.lower()
    assert len(mgr.created_goals) == 1


def test_create_goal_routing_through_autonomous_goal_manager():
    """13. CREATE_GOAL strictly delegates to AutonomousGoalManager."""
    mgr = FakeGoalManager()
    coordinator = AnticipatoryPlanningCoordinator(goal_manager=mgr, clock=lambda: 1000.0)

    ant = make_anticipation(condition_type=FutureConditionType.RESOURCE_DEPLETION_RISK)
    dec = coordinator.process_anticipation(ant)

    assert dec.decision_type == AnticipatoryDecisionType.CREATE_GOAL
    assert dec.executed is True
    assert len(mgr.created_goals) == 1
    created = mgr.created_goals[0]
    assert "Proactively mitigate" in created.original_goal
    assert created.metadata["anticipation_id"] == ant.anticipation_id


def test_update_goal_routing_through_autonomous_goal_manager():
    """14. UPDATE_GOAL (e.g. deadline priority escalation) routes through AutonomousGoalManager."""
    store = InMemoryGoalStore()
    goal = Goal(goal_id="g_urgent", original_goal="Finish analysis", priority=GoalPriority.NORMAL)
    store.create_goal(goal)

    mgr = FakeGoalManager(store=store)
    coordinator = AnticipatoryPlanningCoordinator(goal_manager=mgr, clock=lambda: 1000.0)

    ant = make_anticipation(
        condition_type=FutureConditionType.DEADLINE_RISK,
        related_goal_ids=("g_urgent",),
    )
    dec = coordinator.process_anticipation(ant)

    assert dec.decision_type == AnticipatoryDecisionType.UPDATE_GOAL
    assert dec.executed is True
    assert mgr.priorities["g_urgent"] == "critical"
    # Verify in store
    updated = store.get_goal("g_urgent")
    assert updated.priority == GoalPriority.CRITICAL


def test_goal_store_lifecycle_cannot_be_directly_mutated_without_manager():
    """15. AnticipatoryPlanningCoordinator NEVER mutates GoalStore when goal_manager is None."""
    store = InMemoryGoalStore()
    coordinator = AnticipatoryPlanningCoordinator(goal_store=store, clock=lambda: 1000.0)

    ant = make_anticipation(condition_type=FutureConditionType.RESOURCE_DEPLETION_RISK)
    dec = coordinator.process_anticipation(ant)

    assert dec.decision_type == AnticipatoryDecisionType.CREATE_GOAL
    assert dec.executed is False
    assert dec.goal_id is None
    assert len(store.list_goals()) == 0


def test_policy_deny_enforcement():
    """16. PolicyEngine DENY blocks anticipatory goal operations completely."""
    mgr = FakeGoalManager()
    policy = FakePolicyEngine(decision=PolicyDecision.DENY, reason="Proactive autonomy disabled")
    coordinator = AnticipatoryPlanningCoordinator(
        goal_manager=mgr,
        policy_engine=policy,
        clock=lambda: 1000.0,
    )

    ant = make_anticipation(condition_type=FutureConditionType.RESOURCE_DEPLETION_RISK)
    dec = coordinator.process_anticipation(ant)

    assert dec.decision_type == AnticipatoryDecisionType.NO_ACTION
    assert dec.executed is False
    assert len(mgr.created_goals) == 0


def test_policy_confirmation_escalates_without_premature_mutation():
    """17. PolicyEngine REQUIRE_CONFIRMATION converts to user escalation without creating goals."""
    mgr = FakeGoalManager()
    policy = FakePolicyEngine(decision=PolicyDecision.REQUIRE_CONFIRMATION, reason="High impact action")
    coordinator = AnticipatoryPlanningCoordinator(
        goal_manager=mgr,
        policy_engine=policy,
        clock=lambda: 1000.0,
    )

    ant = make_anticipation(condition_type=FutureConditionType.RESOURCE_DEPLETION_RISK)
    dec = coordinator.process_anticipation(ant)

    assert dec.decision_type == AnticipatoryDecisionType.ESCALATE_USER
    assert dec.executed is False
    assert len(mgr.created_goals) == 0


def test_event_to_anticipation_integration():
    """18. Phase 4.5 prospective event transforms cleanly into candidate anticipation."""
    ev = Event(
        event_id="ev_warn_1",
        source=EventSource.SYSTEM_OBSERVATION,
        event_type="bms.battery.pre_warning",
        priority=EventPriority.HIGH,
        timestamp=1000.0,
        payload={"entity_id": "cart_4", "summary": "Battery discharging faster than expected"},
        provenance=EventProvenance(
            source_id="bms",
            source_type=EventSource.SYSTEM_OBSERVATION,
            origin_timestamp=1000.0,
            correlation_id="corr_ev_1",
            depth=1,
        ),
        correlation_id="corr_ev_1",
    )

    candidate = event_to_anticipation_candidate(ev)
    assert candidate is not None
    assert candidate.condition_type == FutureConditionType.RESOURCE_DEPLETION_RISK
    assert candidate.target_entity_id == "cart_4"
    assert candidate.provenance.depth == 2
    assert candidate.provenance.causation_id == "ev_warn_1"


def test_hypothetical_vs_actual_world_state_separation():
    """19. Anticipations NEVER write hypothetical future states into WorldState."""
    entity = WorldEntity(entity_id="rover_unit", entity_type="robot")
    cond = WorldCondition(
        entity_id="rover_unit",
        property_name="battery",
        value=35.0,
        confidence=1.0,
        observed_at=1000.0,
    )
    ws = WorldState(
        state_id="ws_sep_01",
        version=1,
        timestamp=1000.0,
        entities=(entity,),
        conditions=(cond,),
    )
    store = InMemoryWorldStateStore(initial_state=ws)

    analyzer = DeterministicAnticipatoryAnalyzer(clock=lambda: 1000.0)
    candidates = analyzer.analyze(ws, active_goals=[], recent_events=[], evidence=[])
    assert len(candidates) >= 1

    # World state must remain untouched at 35.0%
    curr_state = store.get_current_state()
    assert curr_state.get_condition("rover_unit", "battery").value == 35.0


def test_invalidation_when_conditions_recover():
    """20. InvalidationEngine marks hypothesis INVALIDATED when condition recovers."""
    invalidation_engine = DeterministicInvalidationEngine()
    ant = make_anticipation(target_entity_id="rover_unit")

    # Later: battery is recharged to 80%
    entity_recovered = WorldEntity(entity_id="rover_unit", entity_type="robot")
    cond_recovered = WorldCondition(
        entity_id="rover_unit",
        property_name="battery",
        value=80.0,
        confidence=1.0,
        observed_at=1050.0,
    )
    ws_recovered = WorldState(
        state_id="ws_rec_02",
        version=2,
        timestamp=1050.0,
        entities=(entity_recovered,),
        conditions=(cond_recovered,),
    )

    status, reason = invalidation_engine.evaluate_invalidation(ant, ws_recovered, active_goals=[], now=1050.0)
    assert status == AnticipationStatus.INVALIDATED
    assert "recovered" in reason.lower()


def test_temporal_expiration_evaluation():
    """21. Anticipation marks EXPIRED when horizon deadline elapsed."""
    invalidation_engine = DeterministicInvalidationEngine()
    # Created at 1000.0 with horizon NEAR_TERM (max window 1800.0s -> expires at 2800.0)
    ant = make_anticipation(timestamp=1000.0, horizon=TimeHorizon.NEAR_TERM)

    status_valid, _ = invalidation_engine.evaluate_invalidation(ant, None, [], now=2000.0)
    assert status_valid == AnticipationStatus.ACTIVE

    status_exp, reason = invalidation_engine.evaluate_invalidation(ant, None, [], now=3000.0)
    assert status_exp == AnticipationStatus.EXPIRED
    assert "exceeded" in reason.lower()


def test_cascade_depth_limit_and_loop_prevention():
    """22. Anticipations exceeding max_cascade_depth are suppressed to avoid loops."""
    coordinator = AnticipatoryPlanningCoordinator(max_cascade_depth=3)
    ant_deep = make_anticipation(depth=3)

    dec = coordinator.process_anticipation(ant_deep)
    assert dec.decision_type == AnticipatoryDecisionType.NO_ACTION
    assert "cascade depth limit" in dec.reason.lower()


def test_bounded_evaluation_cycle():
    """23. evaluate_cycle processes at most max_anticipations per invocation."""
    coordinator = AnticipatoryPlanningCoordinator()
    decisions = coordinator.evaluate_cycle(max_anticipations=5)
    assert len(decisions) <= 5


def test_cognitive_event_observability_emission():
    """24. Coordinator emits CognitiveEvents for evaluation, decision, and dispatch."""
    sink = FakeCognitiveEventSink()
    mgr = FakeGoalManager()
    coordinator = AnticipatoryPlanningCoordinator(
        goal_manager=mgr,
        event_sink=sink,
        clock=lambda: 1000.0,
    )

    ant = make_anticipation(condition_type=FutureConditionType.RESOURCE_DEPLETION_RISK)
    coordinator.process_anticipation(ant)

    emitted_types = [e.event_type for e in sink.get_events()]
    assert CognitiveEventType.ANTICIPATION_ACCEPTED in emitted_types
    assert CognitiveEventType.ANTICIPATION_DECISION_PRODUCED in emitted_types
    assert CognitiveEventType.ANTICIPATION_ACTION_DISPATCHED in emitted_types


def test_replay_determinism_without_llm():
    """25. Identical input conditions produce bit-for-bit identical anticipations."""
    analyzer1 = DeterministicAnticipatoryAnalyzer(clock=lambda: 1000.0)
    analyzer2 = DeterministicAnticipatoryAnalyzer(clock=lambda: 1000.0)

    entity = WorldEntity(entity_id="drone_1", entity_type="robot")
    cond = WorldCondition(
        entity_id="drone_1",
        property_name="battery",
        value=25.0,
        confidence=0.9,
        observed_at=1000.0,
    )
    ws = WorldState(
        state_id="ws_rep_01",
        version=1,
        timestamp=1000.0,
        entities=(entity,),
        conditions=(cond,),
    )

    res1 = analyzer1.analyze(ws, [], [], [])
    res2 = analyzer2.analyze(ws, [], [], [])

    assert len(res1) == len(res2)
    assert res1[0].get_signature() == res2[0].get_signature()
    assert res1[0].confidence == res2[0].confidence
    assert res1[0].relevance == res2[0].relevance
    assert res1[0].freshness == res2[0].freshness


def test_security_no_direct_tools_models_or_subprocesses():
    """26. Anticipatory planning components contain zero direct tool/LLM/computer references."""
    coordinator = AnticipatoryPlanningCoordinator()
    analyzer = DeterministicAnticipatoryAnalyzer()

    for comp in (coordinator, analyzer):
        assert not hasattr(comp, "tool_orchestrator")
        assert not hasattr(comp, "llm")
        assert not hasattr(comp, "model_router")
        assert not hasattr(comp, "computer_capability")
        assert not hasattr(comp, "shell")
        assert not hasattr(comp, "subprocess")


def test_end_to_end_anticipation_with_production_goal_manager():
    """27. End-to-end integration: WorldState -> Analyzer -> Coordinator -> Production AutonomousGoalManager."""
    class FakeExecutionEngine(GoalExecutionEngineInterface):
        def execute_goal(self, goal_id: str, max_steps: Optional[int] = None) -> Goal: pass
        def step_goal(self, goal_id: str) -> Goal: pass
        def pause_goal(self, goal_id: str, reason: str = "") -> Goal: pass
        def resume_goal(self, goal_id: str) -> Goal: pass
        def abort_goal(self, goal_id: str, reason: str = "") -> Goal: pass

    store = InMemoryGoalStore()
    event_sink = FakeCognitiveEventSink()
    real_manager = AutonomousGoalManager(
        store=store,
        execution_engine=FakeExecutionEngine(),
        event_sink=event_sink,
        clock=lambda: 1000.0,
    )

    entity = WorldEntity(entity_id="rover_unit", entity_type="robot")
    cond = WorldCondition(
        entity_id="rover_unit",
        property_name="battery",
        value=20.0,
        confidence=1.0,
        observed_at=1000.0,
    )
    ws = WorldState(
        state_id="ws_e2e_01",
        version=1,
        timestamp=1000.0,
        entities=(entity,),
        conditions=(cond,),
    )
    ws_store = InMemoryWorldStateStore(initial_state=ws)

    coordinator = AnticipatoryPlanningCoordinator(
        goal_manager=real_manager,
        world_state_store=ws_store,
        event_sink=event_sink,
        clock=lambda: 1000.0,
    )

    decisions = coordinator.evaluate_cycle(max_anticipations=5)
    assert len(decisions) >= 1
    primary_decision = decisions[0]

    assert primary_decision.decision_type == AnticipatoryDecisionType.CREATE_GOAL
    assert primary_decision.executed is True
    assert primary_decision.goal_id is not None

    # Verify goal in store
    created_goal = store.get_goal(primary_decision.goal_id)
    assert created_goal is not None
    assert "Proactively mitigate" in created_goal.original_goal
    assert created_goal.priority == GoalPriority.HIGH

    # Verify CognitiveEvents
    emitted = [e.event_type for e in event_sink.get_events()]
    assert CognitiveEventType.ANTICIPATION_EVALUATED in emitted
    assert CognitiveEventType.GOAL_CREATED in emitted
    assert CognitiveEventType.ANTICIPATION_ACTION_DISPATCHED in emitted
