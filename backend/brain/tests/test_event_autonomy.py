import collections
import time
import pytest
from typing import Any, Dict, List, Optional

from core.interfaces.goal_interface import AutonomousGoalManagerInterface, GoalStoreInterface
from core.interfaces.policy_interface import PolicyEngineInterface
from core.interfaces.runtime_interface import CognitiveEventSinkInterface
from core.models.autonomy import (
    AutonomyDecision,
    AutonomyDecisionType,
    Event,
    EventCategory,
    EventClassification,
    EventPriority,
    EventProvenance,
    EventRelevance,
    EventSource,
    EventTrigger,
    TriggerCondition,
)
from core.models.goal import Goal, GoalConstraints, GoalPriority, GoalStatus
from core.models.policy import AutonomyLevel, PolicyContext, PolicyDecision, PolicyResult, RiskLevel
from core.models.runtime import CognitiveEvent, CognitiveEventType, CognitiveStage
from core.models.world_state import StateProvenance, TransitionType, WorldStateTransition
from goals.store import InMemoryGoalStore
from autonomy.classifier import DeterministicEventClassifier
from autonomy.coordinator import EventDrivenAutonomyCoordinator
from autonomy.relevance import DeterministicRelevanceEngine
from autonomy.world_listener import transition_to_autonomy_event


# ============================================================================
# TEST FIXTURES & HARNESSES
# ============================================================================

class FakeCognitiveEventSink(CognitiveEventSinkInterface):
    """Test double capturing CognitiveEvents."""

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
    """Mock PolicyEngine to test policy gating and denial."""

    def __init__(self, decision: PolicyDecision = PolicyDecision.ALLOW, reason: str = "Authorized"):
        self.decision = decision
        self.reason = reason
        self.evaluated_contexts: List[PolicyContext] = []

    def evaluate(self, context: PolicyContext) -> PolicyResult:
        self.evaluated_contexts.append(context)
        return PolicyResult(
            decision=self.decision,
            reason=self.reason,
            rule_id="test_rule",
        )


class FakeGoalManager(AutonomousGoalManagerInterface):
    """Mock AutonomousGoalManager to test goal lifecycle delegation."""

    def __init__(self, store: Optional[GoalStoreInterface] = None):
        self.store = store or InMemoryGoalStore()
        self.created_goals: List[Goal] = []
        self.paused_goals: List[Tuple[str, str]] = []
        self.resumed_goals: List[str] = []
        self.cancelled_goals: List[Tuple[str, str]] = []
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

    def pause_goal(self, goal_id: str, reason: str = "") -> Any:
        self.paused_goals.append((goal_id, reason))
        goal = self.store.get_goal(goal_id)
        if goal:
            goal.status = GoalStatus.PAUSED
            self.store.update_goal(goal)
        return goal

    def resume_goal(self, goal_id: str) -> Any:
        self.resumed_goals.append(goal_id)
        goal = self.store.get_goal(goal_id)
        if goal:
            goal.status = GoalStatus.RUNNING
            self.store.update_goal(goal)
        return goal

    def cancel_goal(self, goal_id: str, reason: str = "") -> Any:
        self.cancelled_goals.append((goal_id, reason))
        goal = self.store.get_goal(goal_id)
        if goal:
            goal.status = GoalStatus.CANCELLED
            self.store.update_goal(goal)
        return goal

    def set_goal_priority(self, goal_id: str, priority: Any) -> Any:
        self.priorities[goal_id] = priority
        return None

    def set_goal_deadline(self, goal_id: str, deadline: Optional[float]) -> Any:
        return None

    def get_management_state(self) -> Any:
        return None


def make_event(
    event_id: str = "ev_001",
    event_type: str = "world_state.threshold_breach.battery",
    source: EventSource = EventSource.WORLD_STATE,
    priority: EventPriority = EventPriority.HIGH,
    timestamp: float = 1000.0,
    payload: Optional[Dict[str, Any]] = None,
    depth: int = 0,
    correlation_id: str = "corr_001",
    dedup_key: Optional[str] = None,
) -> Event:
    prov = EventProvenance(
        source_id="src_sys",
        source_type=source,
        origin_timestamp=timestamp,
        correlation_id=correlation_id,
        depth=depth,
    )
    return Event(
        event_id=event_id,
        source=source,
        event_type=event_type,
        priority=priority,
        timestamp=timestamp,
        payload=payload or {"entity_id": "robot_1", "property_name": "battery", "value": 15},
        provenance=prov,
        correlation_id=correlation_id,
        deduplication_key=dedup_key,
    )


# ============================================================================
# 1. EVENT CREATION & IMMUTABILITY
# ============================================================================

def test_event_domain_creation_and_immutability():
    """1. Event is frozen dataclass with complete identity and rejects direct mutation."""
    ev = make_event()
    assert ev.event_id == "ev_001"
    assert ev.source == EventSource.WORLD_STATE
    assert ev.priority == EventPriority.HIGH

    with pytest.raises((AttributeError, TypeError)):
        ev.priority = EventPriority.CRITICAL  # type: ignore


# ============================================================================
# 2. PROVENANCE RETENTION
# ============================================================================

def test_event_provenance_retention():
    """2. Event preserves source ID, modality, timestamp, correlation, and depth."""
    ev = make_event(depth=1, correlation_id="corr_test_99")
    prov = ev.provenance
    assert prov.source_id == "src_sys"
    assert prov.source_type == EventSource.WORLD_STATE
    assert prov.correlation_id == "corr_test_99"
    assert prov.depth == 1
    assert prov.to_dict()["depth"] == 1


# ============================================================================
# 3. EVENT IDENTITY & DEDUPLICATION KEY
# ============================================================================

def test_event_identity_and_deduplication_key():
    """3. Event computes deterministic default dedup key when not explicitly set."""
    ev1 = make_event(payload={"entity_id": "drone", "property_name": "battery"})
    assert ev1.get_dedup_key() == "WORLD_STATE:world_state.threshold_breach.battery:drone:battery"

    ev2 = make_event(dedup_key="custom_unique_key")
    assert ev2.get_dedup_key() == "custom_unique_key"


# ============================================================================
# 4. DETERMINISTIC CLASSIFICATION
# ============================================================================

def test_deterministic_classification_categories():
    """4. DeterministicEventClassifier correctly identifies standard categories without LLM."""
    classifier = DeterministicEventClassifier()

    ev_safety = make_event(event_type="hazard_detected", priority=EventPriority.CRITICAL)
    assert classifier.classify(ev_safety).category == EventCategory.SAFETY_ALERT

    ev_threshold = make_event(event_type="cpu_threshold_exceeded", priority=EventPriority.HIGH)
    assert classifier.classify(ev_threshold).category == EventCategory.THRESHOLD_BREACH

    ev_failure = make_event(event_type="component_failed", priority=EventPriority.HIGH)
    assert classifier.classify(ev_failure).category == EventCategory.FAILURE

    ev_timeout = make_event(event_type="sensor_timeout", priority=EventPriority.NORMAL)
    assert classifier.classify(ev_timeout).category == EventCategory.TIMEOUT

    ev_progress = make_event(event_type="step_progress", priority=EventPriority.LOW)
    assert classifier.classify(ev_progress).category == EventCategory.GOAL_PROGRESS


# ============================================================================
# 5. CLASSIFICATION SEVERITY & ACTION MAPPING
# ============================================================================

def test_classification_severity_and_priority_mapping():
    """5. Critical and safety events require action; low progress events do not."""
    classifier = DeterministicEventClassifier()

    c_crit = classifier.classify(make_event(event_type="fire_alarm", priority=EventPriority.CRITICAL))
    assert c_crit.severity == "critical"
    assert c_crit.requires_action is True

    c_low = classifier.classify(make_event(event_type="step_progress", priority=EventPriority.LOW))
    assert c_low.severity == "low"
    assert c_low.requires_action is False


# ============================================================================
# 6. SAFE HANDLING OF UNKNOWN EVENTS
# ============================================================================

def test_safe_handling_of_unknown_events():
    """6. Unmapped event types classify as UNKNOWN and fail safe with requires_action=False."""
    classifier = DeterministicEventClassifier()
    ev_unknown = make_event(event_type="unregistered_bizarre_event_type_xyz", priority=EventPriority.LOW)

    c = classifier.classify(ev_unknown)
    assert c.category == EventCategory.UNKNOWN
    assert c.requires_action is False
    assert "unknown" in c.rationale.lower()


# ============================================================================
# 7. RELEVANCE EVALUATION FACTORS
# ============================================================================

def test_relevance_evaluation_factors():
    """7. DeterministicRelevanceEngine scores critical safety higher than routine changes."""
    relevance_engine = DeterministicRelevanceEngine(clock=lambda: 1000.0)
    classifier = DeterministicEventClassifier()

    ev_crit = make_event(event_type="safety_alert_breach", priority=EventPriority.CRITICAL, timestamp=1000.0)
    c_crit = classifier.classify(ev_crit)
    rel_crit = relevance_engine.evaluate_relevance(ev_crit, c_crit)
    assert rel_crit.is_relevant is True
    assert rel_crit.score >= 0.7

    ev_low = make_event(event_type="state_change_ambient", priority=EventPriority.LOW, timestamp=1000.0)
    c_low = classifier.classify(ev_low)
    rel_low = relevance_engine.evaluate_relevance(ev_low, c_low)
    assert rel_low.score < rel_crit.score


# ============================================================================
# 8. RELEVANCE FRESHNESS DECAY
# ============================================================================

def test_relevance_freshness_decay():
    """8. Relevance decays gracefully as event age increases."""
    relevance_engine = DeterministicRelevanceEngine(clock=lambda: 1500.0)
    classifier = DeterministicEventClassifier()

    # Fresh event (age 5s)
    ev_fresh = make_event(timestamp=1495.0)
    c_fresh = classifier.classify(ev_fresh)
    rel_fresh = relevance_engine.evaluate_relevance(ev_fresh, c_fresh)

    # Stale event (age 400s)
    ev_stale = make_event(timestamp=1100.0)
    c_stale = classifier.classify(ev_stale)
    rel_stale = relevance_engine.evaluate_relevance(ev_stale, c_stale)

    assert rel_fresh.score > rel_stale.score
    assert rel_fresh.freshness_factor == 1.0
    assert rel_stale.freshness_factor == 0.2


# ============================================================================
# 9. RELEVANCE ACTIVE GOAL MATCHING
# ============================================================================

def test_relevance_active_goal_matching():
    """9. Events correlated with currently running goals receive a relevance boost."""
    relevance_engine = DeterministicRelevanceEngine(clock=lambda: 1000.0)
    classifier = DeterministicEventClassifier()

    active_goal = Goal(
        goal_id="g_robot_dock",
        original_goal="Dock robot_1 safely",
        metadata={"entity_id": "robot_1"},
    )

    ev_match = make_event(payload={"entity_id": "robot_1", "property_name": "battery"}, timestamp=1000.0)
    c_match = classifier.classify(ev_match)
    rel_match = relevance_engine.evaluate_relevance(ev_match, c_match, active_goals=[active_goal])

    assert "g_robot_dock" in rel_match.matched_goals
    assert rel_match.score > 0.6


# ============================================================================
# 10. DUPLICATE SUPPRESSION WINDOW
# ============================================================================

def test_duplicate_suppression_window():
    """10. Repeated identical events within dedup window are suppressed to prevent storms."""
    current_time = 1000.0
    coordinator = EventDrivenAutonomyCoordinator(
        clock=lambda: current_time,
        dedup_window_seconds=30.0,
    )

    ev1 = make_event(event_id="e1", timestamp=1000.0)
    dec1 = coordinator.ingest_event(ev1)
    assert dec1.decision_type != AutonomyDecisionType.IGNORE

    # Repeat same event 5 seconds later
    current_time = 1005.0
    ev2 = make_event(event_id="e2", timestamp=1005.0)
    dec2 = coordinator.ingest_event(ev2)
    assert dec2.decision_type == AutonomyDecisionType.IGNORE
    assert "duplicate suppression" in dec2.reason.lower()

    # Repeat after window expires (35 seconds later) -> allowed
    current_time = 1036.0
    ev3 = make_event(event_id="e3", timestamp=1036.0)
    dec3 = coordinator.ingest_event(ev3)
    assert dec3.decision_type != AutonomyDecisionType.IGNORE


# ============================================================================
# 11. TRIGGER CONDITION MATCHING
# ============================================================================

def test_trigger_condition_matching_predicate():
    """11. EventTrigger correctly evaluates threshold and priority predicates."""
    cond = TriggerCondition(
        condition_id="cond_battery_low",
        event_types=("battery_warning",),
        min_priority=EventPriority.HIGH,
        min_relevance=0.5,
        predicate_type="threshold",
        predicate_params={"key": "battery_level", "operator": "<", "threshold": 20},
    )
    trig = EventTrigger(
        trigger_id="trig_recharge",
        condition=cond,
        action_type=AutonomyDecisionType.CREATE_GOAL,
        target_goal_template={"original_goal": "Recharge immediately"},
    )

    ev_match = make_event(
        event_type="battery_warning",
        priority=EventPriority.HIGH,
        payload={"battery_level": 15},
    )
    clf = EventClassification(EventCategory.THRESHOLD_BREACH, EventPriority.HIGH, "high", False, True, "")
    rel = EventRelevance(0.8, True)
    assert cond.matches(ev_match, clf, rel) is True

    # Battery above threshold -> no match
    ev_nomatch = make_event(
        event_type="battery_warning",
        priority=EventPriority.HIGH,
        payload={"battery_level": 50},
    )
    assert cond.matches(ev_nomatch, clf, rel) is False


# ============================================================================
# 12. TRIGGER COOLDOWN & RATE LIMITING
# ============================================================================

def test_trigger_cooldown_and_rate_limiting():
    """12. Trigger adheres to cooldown seconds before re-triggering."""
    current_time = 1000.0
    coordinator = EventDrivenAutonomyCoordinator(clock=lambda: current_time)

    cond = TriggerCondition(condition_id="c_ping", predicate_type="always", min_priority=EventPriority.NORMAL, min_relevance=0.0)
    trig = EventTrigger(
        trigger_id="trig_cooldown",
        condition=cond,
        action_type=AutonomyDecisionType.RECORD_ONLY,
        cooldown_seconds=15.0,
        max_triggers_per_window=1,
        window_seconds=60.0,
    )
    coordinator.register_trigger(trig)

    ev1 = make_event(event_id="e1", dedup_key="k1", timestamp=1000.0)
    dec1 = coordinator.ingest_event(ev1)
    assert dec1.matched_trigger_id == "trig_cooldown"

    # Second event 5s later with different dedup key -> trigger rate limited
    current_time = 1005.0
    ev2 = make_event(event_id="e2", dedup_key="k2", timestamp=1005.0)
    dec2 = coordinator.ingest_event(ev2)
    assert dec2.matched_trigger_id != "trig_cooldown"


# ============================================================================
# 13. AUTONOMOUS GOAL CREATION REQUEST
# ============================================================================

def test_autonomy_decision_create_goal():
    """13. High-relevance threshold breach creates goal through AutonomousGoalManager."""
    store = InMemoryGoalStore()
    mgr = FakeGoalManager(store=store)
    coordinator = EventDrivenAutonomyCoordinator(goal_manager=mgr, clock=lambda: 1000.0)

    ev = make_event(
        event_id="e_temp",
        event_type="world_state.threshold_breach.temperature",
        priority=EventPriority.HIGH,
        payload={"entity_id": "chiller", "temp": 95},
    )

    dec = coordinator.ingest_event(ev)
    assert dec.decision_type == AutonomyDecisionType.CREATE_GOAL
    assert dec.executed is True
    assert dec.goal_id is not None
    assert len(mgr.created_goals) == 1

    # Goal in store via GoalManager
    saved_goal = store.get_goal(dec.goal_id)
    assert saved_goal is not None
    assert "Resolve threshold breach" in saved_goal.original_goal
    assert saved_goal.priority == GoalPriority.HIGH


# ============================================================================
# 14. AUTONOMOUS PAUSE & RESUME GOAL
# ============================================================================

def test_autonomy_decision_pause_and_resume_goal():
    """14. Safety alert with matched goal pauses active goal, and clear alert resumes it via AutonomousGoalManager."""
    mgr = FakeGoalManager()
    coordinator = EventDrivenAutonomyCoordinator(goal_manager=mgr, clock=lambda: 1000.0)

    cond_pause = TriggerCondition(condition_id="c_safety", event_types=("hazard_stop",), predicate_type="always")
    trig_pause = EventTrigger(trigger_id="t_pause", condition=cond_pause, action_type=AutonomyDecisionType.PAUSE_GOAL)
    coordinator.register_trigger(trig_pause)

    cond_resume = TriggerCondition(condition_id="c_clear", event_types=("hazard_clear",), predicate_type="always")
    trig_resume = EventTrigger(trigger_id="t_resume", condition=cond_resume, action_type=AutonomyDecisionType.RESUME_GOAL)
    coordinator.register_trigger(trig_resume)

    # 1. Pause
    ev_pause = make_event(
        event_id="ev_p",
        event_type="hazard_stop",
        priority=EventPriority.CRITICAL,
        payload={"goal_id": "goal_active_9"},
    )
    dec_pause = coordinator.ingest_event(ev_pause)
    assert dec_pause.decision_type == AutonomyDecisionType.PAUSE_GOAL
    assert dec_pause.executed is True
    assert len(mgr.paused_goals) == 1
    assert mgr.paused_goals[0][0] == "goal_active_9"

    # 2. Resume
    ev_resume = make_event(
        event_id="ev_r",
        event_type="hazard_clear",
        priority=EventPriority.HIGH,
        payload={"goal_id": "goal_active_9"},
    )
    dec_resume = coordinator.ingest_event(ev_resume)
    assert dec_resume.decision_type == AutonomyDecisionType.RESUME_GOAL
    assert dec_resume.executed is True
    assert len(mgr.resumed_goals) == 1
    assert mgr.resumed_goals[0] == "goal_active_9"


# ============================================================================
# 15. DUPLICATE GOAL PREVENTION
# ============================================================================

def test_duplicate_goal_creation_prevention():
    """15. Coordinator prevents creating multiple goals for the same active condition."""
    store = InMemoryGoalStore()
    mgr = FakeGoalManager(store=store)
    current_time = 1000.0
    coordinator = EventDrivenAutonomyCoordinator(
        goal_manager=mgr,
        clock=lambda: current_time,
        dedup_window_seconds=1.0,  # short event window
    )

    ev1 = make_event(event_id="e1", timestamp=1000.0)
    dec1 = coordinator.ingest_event(ev1)
    assert dec1.decision_type == AutonomyDecisionType.CREATE_GOAL
    assert dec1.executed is True
    assert len(mgr.created_goals) == 1

    # After dedup window expires, new event for same condition arrives
    current_time = 1010.0
    ev2 = make_event(event_id="e2", timestamp=1010.0)
    dec2 = coordinator.ingest_event(ev2)
    # Goal creation suppressed because active goal with dedup_key already exists!
    assert dec2.executed is False
    assert "duplicate prevented" in dec2.reason.lower()
    assert len(mgr.created_goals) == 1


# ============================================================================
# 16. EVENT LOOP & CASCADE DEPTH LIMIT
# ============================================================================

def test_event_loop_and_cascade_depth_limit():
    """16. Events exceeding max_cascade_depth are suppressed to prevent infinite loops."""
    coordinator = EventDrivenAutonomyCoordinator(max_cascade_depth=3)

    ev_deep = make_event(event_id="e_loop", depth=3)
    dec = coordinator.ingest_event(ev_deep)

    assert dec.decision_type == AutonomyDecisionType.IGNORE
    assert "Cascade depth limit" in dec.reason


# ============================================================================
# 17. BOUNDED BATCH PROCESSING
# ============================================================================

def test_bounded_batch_processing():
    """17. process_pending_events drains at most max_events per call."""
    coordinator = EventDrivenAutonomyCoordinator()

    for i in range(15):
        coordinator.queue_event(make_event(event_id=f"e_{i}", dedup_key=f"k_{i}"))

    batch = coordinator.process_pending_events(max_events=10)
    assert len(batch) == 10

    # Remaining 5
    batch2 = coordinator.process_pending_events(max_events=10)
    assert len(batch2) == 5


# ============================================================================
# 18. POLICY ENGINE ENFORCEMENT & DENIAL
# ============================================================================

def test_policy_engine_enforcement_and_denial():
    """18. PolicyEngine DENY blocks autonomous goal action and marks decision IGNORE."""
    policy = FakePolicyEngine(decision=PolicyDecision.DENY, reason="Autonomy restricted during maintenance")
    coordinator = EventDrivenAutonomyCoordinator(policy_engine=policy)

    cond = TriggerCondition(condition_id="c_act", predicate_type="always")
    trig = EventTrigger(trigger_id="t_act", condition=cond, action_type=AutonomyDecisionType.CREATE_GOAL)
    coordinator.register_trigger(trig)

    ev = make_event()
    dec = coordinator.ingest_event(ev)

    assert dec.decision_type == AutonomyDecisionType.IGNORE
    assert dec.policy_result is not None
    assert dec.policy_result.decision == PolicyDecision.DENY
    assert "denied by policy" in dec.reason.lower()


def test_policy_escalates_to_user_on_confirmation():
    """19. PolicyEngine REQUIRE_CONFIRMATION converts action to ESCALATE_TO_USER."""
    policy = FakePolicyEngine(decision=PolicyDecision.REQUIRE_CONFIRMATION, reason="High risk action")
    coordinator = EventDrivenAutonomyCoordinator(policy_engine=policy)

    cond = TriggerCondition(condition_id="c_act", predicate_type="always")
    trig = EventTrigger(trigger_id="t_act", condition=cond, action_type=AutonomyDecisionType.CREATE_GOAL)
    coordinator.register_trigger(trig)

    ev = make_event()
    dec = coordinator.ingest_event(ev)

    assert dec.decision_type == AutonomyDecisionType.ESCALATE_TO_USER
    assert "user confirmation" in dec.reason.lower()


# ============================================================================
# 20. WORLD STATE TRANSITION BRIDGE
# ============================================================================

def test_world_state_transition_bridge():
    """20. transition_to_autonomy_event converts Phase 4.4 transitions to autonomy events."""
    trans_battery = WorldStateTransition(
        transition_id="tr_01",
        from_version=1,
        to_version=2,
        transition_type=TransitionType.UPDATE,
        observation_id="obs_01",
        entity_id="drone_alpha",
        property_name="battery_pct",
        old_value=72,
        new_value=14,
        timestamp=1000.0,
        provenance=StateProvenance(source_id="s1", source_type="sensor", observation_id="obs_01"),
    )

    ev = transition_to_autonomy_event(trans_battery)
    assert ev is not None
    assert ev.source == EventSource.WORLD_STATE
    assert ev.priority == EventPriority.HIGH
    assert "threshold_breach" in ev.event_type
    assert ev.payload["new_value"] == 14

    # Unchanged transition is benign and ignored
    trans_unchanged = WorldStateTransition(
        transition_id="tr_02",
        from_version=2,
        to_version=3,
        transition_type=TransitionType.UNCHANGED,
        observation_id="obs_02",
        entity_id="drone_alpha",
        property_name="battery_pct",
        old_value=14,
        new_value=14,
        timestamp=1005.0,
        provenance=StateProvenance(source_id="s1", source_type="sensor", observation_id="obs_02"),
    )
    assert transition_to_autonomy_event(trans_unchanged) is None


# ============================================================================
# 21. COGNITIVE EVENT OBSERVABILITY EMISSION
# ============================================================================

def test_cognitive_event_observability_emission():
    """21. Ingestion, classification, relevance, and decisions emit CognitiveEvents."""
    sink = FakeCognitiveEventSink()
    coordinator = EventDrivenAutonomyCoordinator(event_sink=sink)

    ev = make_event(correlation_id="corr_obs_123")
    coordinator.ingest_event(ev)

    events = sink.get_events()
    assert len(events) >= 3
    event_types = [e.event_type for e in events]
    assert CognitiveEventType.AUTONOMY_EVENT_INGESTED in event_types
    assert CognitiveEventType.AUTONOMY_EVENT_CLASSIFIED in event_types
    assert CognitiveEventType.AUTONOMY_RELEVANCE_EVALUATED in event_types
    assert CognitiveEventType.AUTONOMY_DECISION_PRODUCED in event_types

    # Correlation ID propagated
    assert all("corr_obs_123" in e.metadata.get("correlation_id", "") for e in events)


# ============================================================================
# 22. REPLAY DETERMINISM WITHOUT LLM
# ============================================================================

def test_replay_determinism_without_llm():
    """22. Re-evaluating identical events produces strictly identical decisions."""
    coord1 = EventDrivenAutonomyCoordinator(clock=lambda: 1000.0)
    coord2 = EventDrivenAutonomyCoordinator(clock=lambda: 1000.0)

    ev = make_event(event_id="e_rep", timestamp=1000.0)

    dec1 = coord1.ingest_event(ev)
    dec2 = coord2.ingest_event(ev)

    assert dec1.decision_type == dec2.decision_type
    assert dec1.classification.category == dec2.classification.category
    assert dec1.relevance.score == dec2.relevance.score
    assert dec1.reason == dec2.reason


# ============================================================================
# 23. SECURITY INVARIANTS: NO DIRECT TOOL OR MODEL EXECUTION
# ============================================================================

def test_security_no_direct_tool_or_model_execution():
    """23. EventDrivenAutonomyCoordinator contains no tool orchestrator or LLM runners."""
    coordinator = EventDrivenAutonomyCoordinator()

    assert not hasattr(coordinator, "tool_orchestrator")
    assert not hasattr(coordinator, "llm")
    assert not hasattr(coordinator, "model_router")
    assert not hasattr(coordinator, "computer_capability")
    assert not hasattr(coordinator, "shell")


# ============================================================================
# 24. END-TO-END WORLD EVENT TO GOAL CREATION
# ============================================================================

def test_end_to_end_world_event_to_goal_creation():
    """24. Full chain: WorldStateTransition -> Event -> Classification -> Relevance -> GoalManager -> Store."""
    goal_store = InMemoryGoalStore()
    mgr = FakeGoalManager(store=goal_store)
    coordinator = EventDrivenAutonomyCoordinator(goal_manager=mgr, clock=lambda: 1000.0)

    # 1. State transition from sensor breach
    trans = WorldStateTransition(
        transition_id="tr_battery_crit",
        from_version=4,
        to_version=5,
        transition_type=TransitionType.UPDATE,
        observation_id="obs_crit",
        entity_id="rover_unit",
        property_name="battery",
        old_value=50,
        new_value=12,
        timestamp=1000.0,
        provenance=StateProvenance(source_id="bms", source_type="sensor", observation_id="obs_crit"),
    )

    # 2. Bridge to autonomy event
    ev = transition_to_autonomy_event(trans)
    assert ev is not None

    # 3. Coordinator processes event
    dec = coordinator.ingest_event(ev)

    # 4. Routed through GoalManager into GoalStore
    assert dec.decision_type == AutonomyDecisionType.CREATE_GOAL
    assert dec.executed is True
    assert len(mgr.created_goals) == 1

    goals = goal_store.list_goals()
    assert len(goals) == 1
    assert "Resolve threshold breach" in goals[0].original_goal
    assert goals[0].metadata["triggered_by_event"] == ev.event_id
    assert goals[0].metadata["correlation_id"] == ev.correlation_id


# ============================================================================
# 25. GOAL LIFECYCLE AUTHORITY INVARIANT TESTS
# ============================================================================

def test_autonomy_does_not_directly_mutate_goal_store_without_manager():
    """25. Proves EventDrivenAutonomyCoordinator NEVER directly mutates GoalStore when goal_manager is None."""
    store = InMemoryGoalStore()
    # No goal_manager provided; only goal_store
    coordinator = EventDrivenAutonomyCoordinator(goal_store=store, clock=lambda: 1000.0)

    ev = make_event(
        event_id="e_threshold",
        event_type="world_state.threshold_breach.temperature",
        priority=EventPriority.HIGH,
        payload={"entity_id": "chiller", "temp": 95},
    )

    dec = coordinator.ingest_event(ev)
    assert dec.decision_type == AutonomyDecisionType.CREATE_GOAL
    # Must NOT execute or mutate store directly without manager authority
    assert dec.executed is False
    assert dec.goal_id is None
    assert len(store.list_goals()) == 0


def test_cancel_goal_goes_through_goal_manager_authority():
    """26. CANCEL_GOAL trigger routes strictly through AutonomousGoalManager."""
    mgr = FakeGoalManager()
    coordinator = EventDrivenAutonomyCoordinator(goal_manager=mgr, clock=lambda: 1000.0)

    cond = TriggerCondition(condition_id="c_abort", event_types=("fatal_fault",), predicate_type="always")
    trig = EventTrigger(trigger_id="t_cancel", condition=cond, action_type=AutonomyDecisionType.CANCEL_GOAL)
    coordinator.register_trigger(trig)

    ev = make_event(
        event_type="fatal_fault",
        priority=EventPriority.CRITICAL,
        payload={"goal_id": "goal_target_77"},
    )
    dec = coordinator.ingest_event(ev)

    assert dec.decision_type == AutonomyDecisionType.CANCEL_GOAL
    assert dec.executed is True
    assert len(mgr.cancelled_goals) == 1
    assert mgr.cancelled_goals[0][0] == "goal_target_77"


def test_policy_deny_prevents_manager_lifecycle_changes():
    """27. PolicyEngine DENY ensures AutonomousGoalManager is never invoked."""
    store = InMemoryGoalStore()
    mgr = FakeGoalManager(store=store)
    policy = FakePolicyEngine(decision=PolicyDecision.DENY, reason="Safety interlock engaged")
    coordinator = EventDrivenAutonomyCoordinator(
        goal_manager=mgr,
        policy_engine=policy,
        clock=lambda: 1000.0,
    )

    ev = make_event(
        event_id="e_temp",
        event_type="world_state.threshold_breach.temperature",
        priority=EventPriority.HIGH,
        payload={"entity_id": "chiller", "temp": 95},
    )
    dec = coordinator.ingest_event(ev)

    assert dec.decision_type == AutonomyDecisionType.IGNORE
    assert dec.executed is False
    assert len(mgr.created_goals) == 0
    assert len(store.list_goals()) == 0


def test_policy_confirmation_prevents_premature_manager_mutation():
    """28. PolicyEngine REQUIRE_CONFIRMATION converts to user escalation without mutating manager."""
    store = InMemoryGoalStore()
    mgr = FakeGoalManager(store=store)
    policy = FakePolicyEngine(decision=PolicyDecision.REQUIRE_CONFIRMATION, reason="Operator confirmation required")
    coordinator = EventDrivenAutonomyCoordinator(
        goal_manager=mgr,
        policy_engine=policy,
        clock=lambda: 1000.0,
    )

    ev = make_event(
        event_id="e_temp",
        event_type="world_state.threshold_breach.temperature",
        priority=EventPriority.HIGH,
        payload={"entity_id": "chiller", "temp": 95},
    )
    dec = coordinator.ingest_event(ev)

    assert dec.decision_type == AutonomyDecisionType.ESCALATE_TO_USER
    assert dec.executed is False
    assert len(mgr.created_goals) == 0
    assert len(store.list_goals()) == 0


def test_real_autonomous_goal_manager_integration():
    """29. End-to-end integration with production Phase 4.3 AutonomousGoalManager."""
    from goals.manager import AutonomousGoalManager
    from core.interfaces.goal_interface import GoalExecutionEngineInterface

    class FakeExecutionEngine(GoalExecutionEngineInterface):
        def execute_goal(self, goal_id: str, max_steps: Optional[int] = None) -> Goal:
            raise NotImplementedError()
        def step_goal(self, goal_id: str) -> Goal:
            raise NotImplementedError()
        def pause_goal(self, goal_id: str, reason: str = "") -> Goal:
            raise NotImplementedError()
        def resume_goal(self, goal_id: str) -> Goal:
            raise NotImplementedError()
        def abort_goal(self, goal_id: str, reason: str = "") -> Goal:
            raise NotImplementedError()

    store = InMemoryGoalStore()
    event_sink = FakeCognitiveEventSink()
    real_manager = AutonomousGoalManager(
        store=store,
        execution_engine=FakeExecutionEngine(),
        event_sink=event_sink,
        clock=lambda: 1000.0,
    )

    coordinator = EventDrivenAutonomyCoordinator(
        goal_manager=real_manager,
        event_sink=event_sink,
        clock=lambda: 1000.0,
    )

    ev = make_event(
        event_id="ev_sensor_heat",
        event_type="world_state.threshold_breach.temperature",
        priority=EventPriority.HIGH,
        payload={"entity_id": "server_rack_1", "temperature": 85},
    )

    dec = coordinator.ingest_event(ev)

    assert dec.decision_type == AutonomyDecisionType.CREATE_GOAL
    assert dec.executed is True
    assert dec.goal_id is not None

    # Verify goal is persisted in store through AutonomousGoalManager
    saved = store.get_goal(dec.goal_id)
    assert saved is not None
    assert "Resolve threshold breach" in saved.original_goal
    assert saved.priority == GoalPriority.HIGH
    assert saved.metadata["triggered_by_event"] == "ev_sensor_heat"
    assert saved.metadata["correlation_id"] == "corr_001"

    # Verify GOAL_CREATED and AUTONOMY_ACTION_DISPATCHED events were published
    event_types = [e.event_type for e in event_sink.get_events()]
    assert CognitiveEventType.GOAL_CREATED in event_types
    assert CognitiveEventType.AUTONOMY_ACTION_DISPATCHED in event_types
