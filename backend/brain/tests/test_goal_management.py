import pytest
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from core.interfaces.runtime_interface import CognitiveRuntimeInterface
from core.models.goal import (
    Goal,
    GoalCompletionCriteria,
    GoalConstraints,
    GoalProgress,
    GoalStatus,
    Objective,
    ObjectiveStatus,
)
from core.models.goal_management import (
    DeadlineStatus,
    GoalFreshnessStatus,
    GoalManagementState,
    GoalPriority,
    QuantumResult,
    SchedulerConfig,
    SchedulingDecision,
)
from core.models.runtime import (
    CognitiveEvent,
    CognitiveEventType,
    CognitiveStage,
    CognitiveTrace,
    CognitiveTurnResult,
    TurnStatus,
)
from goals.execution_engine import GoalExecutionEngine
from goals.manager import AutonomousGoalManager
from goals.scheduler import DeterministicGoalScheduler
from goals.store import InMemoryGoalStore, SQLiteGoalStore
from runtime.event_sink import InMemoryEventSink


# ============================================================================
# TEST HELPERS & FAKES
# ============================================================================

def _make_turn_result(
    turn_id: str = "t1",
    session_id: str = "s1",
    status: TurnStatus = TurnStatus.SUCCEEDED,
    response: str = "Objective task completed successfully.",
    waiting_reason: Optional[str] = None,
    error: Optional[str] = None,
) -> CognitiveTurnResult:
    trace = CognitiveTrace(turn_id=turn_id, session_id=session_id, events=(), final_status=status)
    meta = {"error": error} if error else {}
    return CognitiveTurnResult(
        turn_id=turn_id,
        session_id=session_id,
        status=status,
        stage=CognitiveStage.COMPLETED if status == TurnStatus.SUCCEEDED else CognitiveStage.FAILED,
        response=response,
        trace=trace,
        waiting_reason=waiting_reason,
        metadata=meta,
    )


class FakeCognitiveRuntime(CognitiveRuntimeInterface):
    """Controlled CognitiveRuntime fake for testing GoalManager and ExecutionEngine."""

    def __init__(self, responses: Optional[List[CognitiveTurnResult]] = None):
        self.responses = list(responses or [])
        self.call_history: List[Tuple[Any, Optional[str]]] = []

    def execute_turn(
        self,
        input_data: Any,
        session_id: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
    ) -> CognitiveTurnResult:
        self.call_history.append((input_data, session_id))
        if self.responses:
            return self.responses.pop(0)
        return _make_turn_result(
            turn_id=f"t_{len(self.call_history)}",
            session_id=session_id or "default",
            status=TurnStatus.SUCCEEDED,
            response="Default fake turn completed.",
        )

    def run(
        self,
        input_data: Any,
        session_id: Optional[str] = None,
    ) -> str:
        return self.execute_turn(input_data, session_id).response


class ControlledClock:
    """Deterministic injectable clock for testing deadlines, freshness, and leases."""

    def __init__(self, initial_time: float = 1000.0):
        self._current_time = initial_time

    def __call__(self) -> float:
        return self._current_time

    def advance(self, seconds: float) -> None:
        self._current_time += seconds

    def set(self, timestamp: float) -> None:
        self._current_time = timestamp


# ============================================================================
# 18 DETERMINISTIC FIXTURES
# ============================================================================

def fixture_one_goal() -> Goal:
    """Fixture 1: Single simple runnable goal."""
    return Goal(
        goal_id="g_one",
        original_goal="Process single invoice",
        status=GoalStatus.RUNNING,
        priority=GoalPriority.NORMAL,
        objectives=(
            Objective(objective_id="g_one_o1", description="Extract invoice fields", order=1, status=ObjectiveStatus.READY),
        ),
    )


def fixture_multiple_goals() -> Tuple[Goal, Goal, Goal]:
    """Fixture 2: Three distinct goals with default priorities."""
    g1 = Goal(goal_id="g_multi_1", original_goal="Task 1", status=GoalStatus.RUNNING, priority=GoalPriority.NORMAL)
    g2 = Goal(goal_id="g_multi_2", original_goal="Task 2", status=GoalStatus.RUNNING, priority=GoalPriority.NORMAL)
    g3 = Goal(goal_id="g_multi_3", original_goal="Task 3", status=GoalStatus.RUNNING, priority=GoalPriority.NORMAL)
    return g1, g2, g3


def fixture_different_priorities() -> Tuple[Goal, Goal, Goal, Goal]:
    """Fixture 3: Four goals spanning all priority levels."""
    g_crit = Goal(goal_id="g_crit", original_goal="Critical incident", status=GoalStatus.RUNNING, priority=GoalPriority.CRITICAL)
    g_high = Goal(goal_id="g_high", original_goal="High priority fix", status=GoalStatus.RUNNING, priority=GoalPriority.HIGH)
    g_norm = Goal(goal_id="g_norm", original_goal="Normal report", status=GoalStatus.RUNNING, priority=GoalPriority.NORMAL)
    g_low = Goal(goal_id="g_low", original_goal="Low cleanup", status=GoalStatus.RUNNING, priority=GoalPriority.LOW)
    return g_crit, g_high, g_norm, g_low


def fixture_equal_priorities() -> Tuple[Goal, Goal]:
    """Fixture 4: Two goals with identical priority for tie-breaker verification."""
    g1 = Goal(goal_id="g_tie_b", original_goal="Task B", status=GoalStatus.RUNNING, priority=GoalPriority.HIGH)
    g2 = Goal(goal_id="g_tie_a", original_goal="Task A", status=GoalStatus.RUNNING, priority=GoalPriority.HIGH)
    return g1, g2


def fixture_deadlines() -> Tuple[Goal, Goal]:
    """Fixture 5: Goals with explicit deadline timestamps."""
    g_on_track = Goal(
        goal_id="g_deadline_far",
        original_goal="End of week audit",
        status=GoalStatus.RUNNING,
        priority=GoalPriority.NORMAL,
        constraints=GoalConstraints(deadline=5000.0),  # Well in future
    )
    g_no_deadline = Goal(
        goal_id="g_no_deadline",
        original_goal="Open-ended research",
        status=GoalStatus.RUNNING,
        priority=GoalPriority.NORMAL,
        constraints=GoalConstraints(deadline=None),
    )
    return g_on_track, g_no_deadline


def fixture_due_soon_goals() -> Goal:
    """Fixture 6: Goal nearing its deadline window (within 300s)."""
    return Goal(
        goal_id="g_due_soon",
        original_goal="Urgent billing cutoff",
        status=GoalStatus.RUNNING,
        priority=GoalPriority.NORMAL,
        constraints=GoalConstraints(deadline=1200.0),  # 200s from base time 1000s
    )


def fixture_overdue_goals() -> Goal:
    """Fixture 7: Goal past its deadline."""
    return Goal(
        goal_id="g_overdue",
        original_goal="Expired SLA ticket",
        status=GoalStatus.RUNNING,
        priority=GoalPriority.NORMAL,
        constraints=GoalConstraints(deadline=900.0),  # Past base time 1000s
    )


def fixture_paused_goals() -> Goal:
    """Fixture 8: Explicitly paused goal."""
    return Goal(
        goal_id="g_paused",
        original_goal="Paused export job",
        status=GoalStatus.PAUSED,
        priority=GoalPriority.HIGH,
    )


def fixture_blocked_goals() -> Goal:
    """Fixture 9: Blocked goal."""
    return Goal(
        goal_id="g_blocked",
        original_goal="Blocked database migration",
        status=GoalStatus.BLOCKED,
        priority=GoalPriority.CRITICAL,
    )


def fixture_waiting_for_user_goals() -> Goal:
    """Fixture 10: Waiting for user action."""
    return Goal(
        goal_id="g_waiting",
        original_goal="Pending user password confirmation",
        status=GoalStatus.WAITING_FOR_USER,
        priority=GoalPriority.HIGH,
    )


def fixture_cancelled_goals() -> Goal:
    """Fixture 11: Cancelled goal."""
    return Goal(
        goal_id="g_cancelled",
        original_goal="Cancelled cluster spin-up",
        status=GoalStatus.CANCELLED,
        priority=GoalPriority.HIGH,
    )


def fixture_stale_goals() -> Goal:
    """Fixture 12: Stale goal inactive for > 3600 seconds."""
    return Goal(
        goal_id="g_stale",
        original_goal="Abandoned draft goal",
        status=GoalStatus.CREATED,
        priority=GoalPriority.NORMAL,
        updated_at=100.0,  # 900s older than base time 1000s (with threshold = 500s)
    )


def fixture_fairness_aging() -> Tuple[Goal, Goal]:
    """Fixture 13: High priority goal vs starved low priority goal."""
    g_high = Goal(goal_id="g_starve_high", original_goal="High priority continuous job", status=GoalStatus.RUNNING, priority=GoalPriority.HIGH)
    g_low = Goal(goal_id="g_starve_low", original_goal="Low priority worker", status=GoalStatus.RUNNING, priority=GoalPriority.LOW)
    return g_high, g_low


def fixture_priority_change() -> Goal:
    """Fixture 14: Goal subject to priority elevation."""
    return Goal(goal_id="g_prio_change", original_goal="Escalating customer ticket", status=GoalStatus.RUNNING, priority=GoalPriority.LOW)


def fixture_deadline_change() -> Goal:
    """Fixture 15: Goal subject to deadline modification."""
    return Goal(goal_id="g_dl_change", original_goal="Dynamic deadline task", status=GoalStatus.RUNNING, priority=GoalPriority.NORMAL, constraints=GoalConstraints(deadline=5000.0))


def fixture_single_active_goal() -> Tuple[Goal, Goal]:
    """Fixture 16: Two ready goals contending for single active lease."""
    g1 = Goal(goal_id="g_active_1", original_goal="Goal 1", status=GoalStatus.RUNNING, priority=GoalPriority.NORMAL)
    g2 = Goal(goal_id="g_active_2", original_goal="Goal 2", status=GoalStatus.RUNNING, priority=GoalPriority.NORMAL)
    return g1, g2


def fixture_atomic_goal_claim() -> Goal:
    """Fixture 17: Goal used to test concurrent lease claim collisions."""
    return Goal(goal_id="g_claim_test", original_goal="Claim contested goal", status=GoalStatus.RUNNING, priority=GoalPriority.NORMAL)


def fixture_scheduling_replay() -> Tuple[Goal, Goal]:
    """Fixture 18: Goals with recorded factors for replay auditability."""
    g1 = Goal(goal_id="g_rep_1", original_goal="Audit job 1", status=GoalStatus.RUNNING, priority=GoalPriority.HIGH)
    g2 = Goal(goal_id="g_rep_2", original_goal="Audit job 2", status=GoalStatus.RUNNING, priority=GoalPriority.NORMAL)
    return g1, g2


# ============================================================================
# A. DOMAIN TESTS
# ============================================================================

def test_domain_goal_priority_values_and_weights():
    """A1. Verify GoalPriority enum values, string parsing, and score weights."""
    assert GoalPriority.LOW.score_weight < GoalPriority.NORMAL.score_weight
    assert GoalPriority.NORMAL.score_weight < GoalPriority.HIGH.score_weight
    assert GoalPriority.HIGH.score_weight < GoalPriority.CRITICAL.score_weight

    assert GoalPriority.from_str("critical") == GoalPriority.CRITICAL
    assert GoalPriority.from_str("HIGH") == GoalPriority.HIGH
    assert GoalPriority.from_str("unknown") == GoalPriority.NORMAL


def test_domain_original_goal_immutability_preserved():
    """A2. Modifying original_goal raises AttributeError."""
    g = Goal(original_goal="Root intention", priority=GoalPriority.HIGH)
    with pytest.raises(AttributeError, match="immutable"):
        g.original_goal = "Altered root intention"


def test_domain_scheduling_decision_structure():
    """A3. SchedulingDecision holds candidates, skipped reasons, and factors."""
    dec = SchedulingDecision(
        selected_goal_id="g1",
        reason="Selected by score",
        candidates_considered=("g1", "g2"),
        skipped_candidates={"g2": "PAUSED"},
        scheduling_factors={"g1": {"score": 3000}},
    )
    assert dec.is_selection_made() is True
    assert dec.selected_goal_id == "g1"
    assert "g2" in dec.skipped_candidates


# ============================================================================
# B. ELIGIBILITY TESTS
# ============================================================================

def test_eligibility_runnable_selected():
    """B1. Runnable CREATED or RUNNING goal is eligible."""
    g = fixture_one_goal()
    sched = DeterministicGoalScheduler()
    dec = sched.select_next_goal([g])
    assert dec.selected_goal_id == "g_one"


def test_eligibility_paused_excluded():
    """B2. Paused goal is excluded from scheduling."""
    g = fixture_paused_goals()
    sched = DeterministicGoalScheduler()
    dec = sched.select_next_goal([g])
    assert dec.selected_goal_id is None
    assert "PAUSED" in dec.skipped_candidates["g_paused"]


def test_eligibility_blocked_excluded():
    """B3. Blocked goal is excluded from scheduling."""
    g = fixture_blocked_goals()
    sched = DeterministicGoalScheduler()
    dec = sched.select_next_goal([g])
    assert dec.selected_goal_id is None
    assert "BLOCKED" in dec.skipped_candidates["g_blocked"]


def test_eligibility_waiting_for_user_excluded():
    """B4. WAITING_FOR_USER goal is excluded from scheduling."""
    g = fixture_waiting_for_user_goals()
    sched = DeterministicGoalScheduler()
    dec = sched.select_next_goal([g])
    assert dec.selected_goal_id is None
    assert "WAITING_FOR_USER" in dec.skipped_candidates["g_waiting"]


def test_eligibility_cancelled_and_completed_excluded():
    """B5. Terminal cancelled/completed goals are excluded."""
    g_cancel = fixture_cancelled_goals()
    g_comp = Goal(goal_id="g_comp", original_goal="Done", status=GoalStatus.COMPLETED)
    sched = DeterministicGoalScheduler()
    dec = sched.select_next_goal([g_cancel, g_comp])
    assert dec.selected_goal_id is None
    assert "Terminal" in dec.skipped_candidates["g_cancelled"]
    assert "Terminal" in dec.skipped_candidates["g_comp"]


# ============================================================================
# C. PRIORITY TESTS
# ============================================================================

def test_priority_high_beats_normal_beats_low():
    """C1. Higher priority goals win over lower priority goals."""
    g_crit, g_high, g_norm, g_low = fixture_different_priorities()
    sched = DeterministicGoalScheduler()

    # CRITICAL vs HIGH
    dec1 = sched.select_next_goal([g_high, g_crit])
    assert dec1.selected_goal_id == "g_crit"

    # HIGH vs NORMAL
    dec2 = sched.select_next_goal([g_norm, g_high])
    assert dec2.selected_goal_id == "g_high"

    # NORMAL vs LOW
    dec3 = sched.select_next_goal([g_low, g_norm])
    assert dec3.selected_goal_id == "g_norm"


def test_priority_changes_reflected_next_selection():
    """C2. Updating priority dynamically alters the next scheduler selection."""
    g1 = Goal(goal_id="g1", original_goal="Task 1", status=GoalStatus.RUNNING, priority=GoalPriority.LOW)
    g2 = Goal(goal_id="g2", original_goal="Task 2", status=GoalStatus.RUNNING, priority=GoalPriority.NORMAL)

    sched = DeterministicGoalScheduler()
    dec1 = sched.select_next_goal([g1, g2])
    assert dec1.selected_goal_id == "g2"

    # Elevate g1 to CRITICAL
    g1.priority = GoalPriority.CRITICAL
    dec2 = sched.select_next_goal([g1, g2])
    assert dec2.selected_goal_id == "g1"


# ============================================================================
# D. DEADLINE TESTS
# ============================================================================

def test_deadlines_due_soon_and_overdue_urgency():
    """D1. Due-soon and overdue goals receive priority urgency boosts."""
    clock = ControlledClock(1000.0)
    cfg = SchedulerConfig(due_soon_window_seconds=300.0)
    sched = DeterministicGoalScheduler()

    g_norm_urgent = fixture_due_soon_goals()  # deadline 1200 (due soon at 1000)
    g_high_far = Goal(
        goal_id="g_high_far",
        original_goal="High priority but deadline next week",
        status=GoalStatus.RUNNING,
        priority=GoalPriority.HIGH,
        constraints=GoalConstraints(deadline=10000.0),
    )

    # g_norm_urgent (score: 2000 + 1500 = 3500) beats g_high_far (score: 3000 + 0 = 3000)
    dec = sched.select_next_goal([g_high_far, g_norm_urgent], config=cfg, clock=clock)
    assert dec.selected_goal_id == "g_due_soon"


def test_deadline_status_transitions():
    """D2. Verify classification transitions from ON_TRACK -> DUE_SOON -> OVERDUE."""
    clock = ControlledClock(1000.0)
    cfg = SchedulerConfig(due_soon_window_seconds=300.0)
    sched = DeterministicGoalScheduler()

    g = Goal(goal_id="g_dl", original_goal="Timed", status=GoalStatus.RUNNING, constraints=GoalConstraints(deadline=1500.0))

    # At 1000, remaining = 500 (> 300) -> ON_TRACK
    assert sched._evaluate_deadline(g, clock(), cfg) == DeadlineStatus.ON_TRACK

    # Advance to 1250, remaining = 250 (<= 300) -> DUE_SOON
    clock.set(1250.0)
    assert sched._evaluate_deadline(g, clock(), cfg) == DeadlineStatus.DUE_SOON

    # Advance to 1501, remaining < 0 -> OVERDUE
    clock.set(1501.0)
    assert sched._evaluate_deadline(g, clock(), cfg) == DeadlineStatus.OVERDUE


# ============================================================================
# E. DETERMINISM TESTS
# ============================================================================

def test_determinism_stable_selection_and_tie_breaker():
    """E1. Given identical candidates and state, selection is 100% deterministic."""
    g1, g2 = fixture_equal_priorities()
    sched = DeterministicGoalScheduler()

    # Pass in both orders, verify winning goal is consistently g_tie_a (lexicographical)
    dec1 = sched.select_next_goal([g1, g2])
    dec2 = sched.select_next_goal([g2, g1])

    assert dec1.selected_goal_id == dec2.selected_goal_id == "g_tie_a"


# ============================================================================
# F. FAIRNESS & ANTI-STARVATION TESTS
# ============================================================================

def test_fairness_low_priority_eventually_overtakes_high():
    """F1. Starved low-priority goal accumulates aging score and overtakes high-priority goal."""
    g_high, g_low = fixture_fairness_aging()
    sched = DeterministicGoalScheduler()
    cfg = SchedulerConfig(fairness_starvation_limit=5, fairness_boost_per_skip=250)

    # Initial state: g_high (3000) beats g_low (1000)
    state = GoalManagementState(fairness_counters={"g_starve_low": 0})
    dec1 = sched.select_next_goal([g_high, g_low], state=state, config=cfg)
    assert dec1.selected_goal_id == "g_starve_high"

    # After 5 skips, g_low aging bonus = 5 * 250 + 1200 starvation bonus = 2450.
    # Total score = 1000 + 2450 = 3450 > g_high (3000)!
    state.fairness_counters["g_starve_low"] = 5
    dec2 = sched.select_next_goal([g_high, g_low], state=state, config=cfg)
    assert dec2.selected_goal_id == "g_starve_low"


# ============================================================================
# G. SINGLE ACTIVE GOAL TESTS
# ============================================================================

def test_single_active_goal_enforcement():
    """G1. Only one goal can hold the active execution lease at a time."""
    store = InMemoryGoalStore()
    g1, g2 = fixture_single_active_goal()
    store.create_goal(g1)
    store.create_goal(g2)

    runtime = FakeCognitiveRuntime()
    engine = GoalExecutionEngine(runtime=runtime, store=store)
    mgr = AutonomousGoalManager(store=store, execution_engine=engine)

    # Claim active lease for g1
    assert mgr.claim_active_goal("g_active_1") is True
    assert mgr.get_active_goal() == "g_active_1"

    # Attempt to claim g2 while g1 active -> rejected
    assert mgr.claim_active_goal("g_active_2") is False
    assert mgr.get_active_goal() == "g_active_1"

    # Release g1 -> now g2 can be claimed
    mgr.release_active_goal("g_active_1")
    assert mgr.get_active_goal() is None
    assert mgr.claim_active_goal("g_active_2") is True
    assert mgr.get_active_goal() == "g_active_2"


# ============================================================================
# H. GOAL QUANTA TESTS
# ============================================================================

def test_goal_quantum_executes_single_step_and_yields():
    """H1. Each call to schedule_once() executes exactly one cognitive turn and yields."""
    store = InMemoryGoalStore()
    g = fixture_one_goal()
    store.create_goal(g)

    runtime = FakeCognitiveRuntime()
    engine = GoalExecutionEngine(runtime=runtime, store=store)
    mgr = AutonomousGoalManager(store=store, execution_engine=engine)

    res = mgr.schedule_once()
    assert res is not None
    assert res.turn_executed is True
    assert res.goal_id == "g_one"
    assert len(runtime.call_history) == 1


# ============================================================================
# I. PAUSE AND RESUME TESTS
# ============================================================================

def test_manager_pause_and_resume_lifecycle():
    """I1. Pausing a goal releases active lease and revokes scheduling eligibility; resume restores it."""
    store = InMemoryGoalStore()
    g = fixture_one_goal()
    store.create_goal(g)

    runtime = FakeCognitiveRuntime()
    engine = GoalExecutionEngine(runtime=runtime, store=store)
    mgr = AutonomousGoalManager(store=store, execution_engine=engine)

    # Claim and pause
    mgr.claim_active_goal(g.goal_id)
    paused = mgr.pause_goal(g.goal_id, reason="User hold")
    assert paused.status == GoalStatus.PAUSED
    assert mgr.get_active_goal() is None

    # Next scheduling attempt finds no eligible goals
    res = mgr.schedule_once()
    assert res is None

    # Resume goal
    resumed = mgr.resume_goal(g.goal_id)
    assert resumed.status == GoalStatus.RUNNING

    # Now eligible again
    res2 = mgr.schedule_once()
    assert res2 is not None
    assert res2.goal_id == g.goal_id


# ============================================================================
# J. CANCEL TESTS
# ============================================================================

def test_manager_cancel_goal_permanently_excludes_from_scheduling():
    """J1. Cancelled goal is marked CANCELLED and never scheduled again."""
    store = InMemoryGoalStore()
    g = fixture_one_goal()
    store.create_goal(g)

    runtime = FakeCognitiveRuntime()
    engine = GoalExecutionEngine(runtime=runtime, store=store)
    mgr = AutonomousGoalManager(store=store, execution_engine=engine)

    cancelled = mgr.cancel_goal(g.goal_id, reason="Scope abandoned")
    assert cancelled.status == GoalStatus.CANCELLED

    # Check store preserves record
    persisted = store.get_goal(g.goal_id)
    assert persisted is not None
    assert persisted.status == GoalStatus.CANCELLED

    # Scheduler never selects it
    assert mgr.schedule_once() is None


# ============================================================================
# K. WAITING FOR USER TESTS
# ============================================================================

def test_waiting_for_user_releases_lease_and_suspends():
    """K1. When turn returns WAITING_FOR_USER, lease is released and future quanta skip it."""
    store = InMemoryGoalStore()
    g = fixture_one_goal()
    store.create_goal(g)

    wait_res = _make_turn_result(status=TurnStatus.WAITING_FOR_USER, response="Confirmation needed")
    runtime = FakeCognitiveRuntime([wait_res])
    engine = GoalExecutionEngine(runtime=runtime, store=store)
    mgr = AutonomousGoalManager(store=store, execution_engine=engine)

    res = mgr.schedule_once()
    assert res is not None
    assert res.goal_status == GoalStatus.WAITING_FOR_USER.value
    assert mgr.get_active_goal() is None

    # Next quantum skips it
    assert mgr.schedule_once() is None


# ============================================================================
# L. BLOCKED GOAL TESTS
# ============================================================================

def test_blocked_goal_excluded_until_resumed():
    """L1. Blocked goal is excluded; resuming restores scheduling eligibility."""
    store = InMemoryGoalStore()
    g = fixture_blocked_goals()
    store.create_goal(g)

    runtime = FakeCognitiveRuntime()
    engine = GoalExecutionEngine(runtime=runtime, store=store)
    mgr = AutonomousGoalManager(store=store, execution_engine=engine)

    assert mgr.schedule_once() is None

    # User resumes after resolving external blocker
    mgr.resume_goal(g.goal_id)
    res = mgr.schedule_once()
    assert res is not None
    assert res.goal_id == g.goal_id


# ============================================================================
# M. STALE GOAL TESTS
# ============================================================================

def test_stale_goal_detection():
    """M1. Goals older than stale threshold are classified STALE without failing."""
    clock = ControlledClock(1000.0)
    cfg = SchedulerConfig(stale_threshold_seconds=500.0)

    store = InMemoryGoalStore()
    g = fixture_stale_goals()  # updated_at = 100.0, age = 900s > 500s
    store.create_goal(g)

    event_sink = InMemoryEventSink()
    runtime = FakeCognitiveRuntime()
    engine = GoalExecutionEngine(runtime=runtime, store=store)
    mgr = AutonomousGoalManager(
        store=store,
        execution_engine=engine,
        event_sink=event_sink,
        config=cfg,
        clock=clock,
    )

    # Calling schedule_once executes partition check
    mgr.schedule_once()
    state = mgr.get_management_state()
    assert g.goal_id in state.stale_goal_ids

    # Stale event emitted
    stale_events = [e for e in event_sink.get_events() if e.event_type == CognitiveEventType.GOAL_STALE]
    assert len(stale_events) >= 1
    assert stale_events[0].metadata["goal_id"] == g.goal_id


# ============================================================================
# N. ACTIVE GOAL CLAIM TESTS (SQLITE)
# ============================================================================

def test_sqlite_store_atomic_claim(tmp_path):
    """N1. SQLiteGoalStore enforces atomic mutual exclusion for active claims."""
    db_file = str(tmp_path / "claim_test.db")
    store = SQLiteGoalStore(db_path=db_file)

    g1, g2 = fixture_single_active_goal()
    store.create_goal(g1)
    store.create_goal(g2)

    # Owner 1 claims g1
    assert store.claim_goal(g1.goal_id, owner_id="owner_1", lease_duration=60.0) is True

    # Owner 2 attempts to claim g2 -> rejected
    assert store.claim_goal(g2.goal_id, owner_id="owner_2", lease_duration=60.0) is False

    # Owner 1 releases g1
    assert store.release_goal(g1.goal_id, owner_id="owner_1") is True

    # Now owner 2 can claim g2
    assert store.claim_goal(g2.goal_id, owner_id="owner_2", lease_duration=60.0) is True


# ============================================================================
# O. GOAL EXECUTION INTEGRATION TESTS
# ============================================================================

def test_manager_delegates_to_execution_engine_only():
    """O1. Manager drives turns strictly via GoalExecutionEngine."""
    store = InMemoryGoalStore()
    g = fixture_one_goal()
    store.create_goal(g)

    runtime = FakeCognitiveRuntime()
    engine = GoalExecutionEngine(runtime=runtime, store=store)
    mgr = AutonomousGoalManager(store=store, execution_engine=engine)

    res = mgr.schedule_once()
    assert res is not None
    # Engine executed step
    assert len(runtime.call_history) == 1
    assert "g_one" in runtime.call_history[0][1]


# ============================================================================
# P. OBSERVABILITY & EVENT CORRELATION TESTS
# ============================================================================

def test_manager_observability_event_correlation():
    """P1. Manager emits GOAL_SELECTED and priority events with correct IDs."""
    store = InMemoryGoalStore()
    g = fixture_one_goal()
    store.create_goal(g)

    event_sink = InMemoryEventSink()
    runtime = FakeCognitiveRuntime()
    engine = GoalExecutionEngine(runtime=runtime, store=store, event_sink=event_sink)
    mgr = AutonomousGoalManager(store=store, execution_engine=engine, event_sink=event_sink)

    # Priority update event
    mgr.set_goal_priority(g.goal_id, GoalPriority.CRITICAL)
    prio_events = [e for e in event_sink.get_events() if e.event_type == CognitiveEventType.GOAL_PRIORITY_CHANGED]
    assert len(prio_events) == 1
    assert prio_events[0].metadata["priority"] == GoalPriority.CRITICAL.value

    # Selection event
    mgr.schedule_once()
    sel_events = [e for e in event_sink.get_events() if e.event_type == CognitiveEventType.GOAL_SELECTED]
    assert len(sel_events) == 1
    assert sel_events[0].metadata["goal_id"] == g.goal_id


# ============================================================================
# Q. REPLAY COMPATIBILITY TESTS
# ============================================================================

def test_scheduling_replay_explainability():
    """Q1. SchedulingDecision records all factor weights and skip reasons for deterministic replay."""
    g1, g2 = fixture_scheduling_replay()
    sched = DeterministicGoalScheduler()

    dec = sched.select_next_goal([g1, g2])
    assert dec.selected_goal_id == "g_rep_1"

    # Verify decision metadata contains complete scoring explanation
    factors = dec.scheduling_factors["g_rep_1"]
    assert factors["base_priority_score"] == 3000
    assert "score" in factors


# ============================================================================
# R. SECURITY & ARCHITECTURAL INVARIANT TESTS
# ============================================================================

def test_manager_security_invariants():
    """R1. AutonomousGoalManager has no direct access to tools, models, computer, or memory."""
    store = InMemoryGoalStore()
    runtime = FakeCognitiveRuntime()
    engine = GoalExecutionEngine(runtime=runtime, store=store)
    mgr = AutonomousGoalManager(store=store, execution_engine=engine)

    for attr in ("tool_orchestrator", "computer", "model_router", "memory_service", "policy_engine"):
        assert not hasattr(mgr, attr)


# ============================================================================
# S. MULTI-GOAL END-TO-END QUANTUM TEST
# ============================================================================

def test_multi_goal_end_to_end_quanta_progression():
    """S1. Three goals (HIGH, NORMAL, LOW) execute sequentially in bounded quanta."""
    store = InMemoryGoalStore()

    g_high = Goal(
        goal_id="g_e2e_high",
        original_goal="Process critical payroll",
        status=GoalStatus.RUNNING,
        priority=GoalPriority.HIGH,
        objectives=(Objective(objective_id="o_h1", description="Calculate pay", order=1, status=ObjectiveStatus.READY),),
    )
    g_norm = Goal(
        goal_id="g_e2e_norm",
        original_goal="Generate weekly report",
        status=GoalStatus.RUNNING,
        priority=GoalPriority.NORMAL,
        objectives=(Objective(objective_id="o_n1", description="Compile metrics", order=1, status=ObjectiveStatus.READY),),
    )
    g_low = Goal(
        goal_id="g_e2e_low",
        original_goal="Archive old logs",
        status=GoalStatus.RUNNING,
        priority=GoalPriority.LOW,
        objectives=(Objective(objective_id="o_l1", description="Compress logs", order=1, status=ObjectiveStatus.READY),),
    )

    store.create_goal(g_high)
    store.create_goal(g_norm)
    store.create_goal(g_low)

    runtime = FakeCognitiveRuntime()
    engine = GoalExecutionEngine(runtime=runtime, store=store)
    mgr = AutonomousGoalManager(store=store, execution_engine=engine)

    # Quantum 1: HIGH priority runs and completes
    q1 = mgr.schedule_once()
    assert q1 is not None
    assert q1.goal_id == "g_e2e_high"
    assert store.get_goal("g_e2e_high").status == GoalStatus.COMPLETED

    # Quantum 2: NORMAL priority runs and completes
    q2 = mgr.schedule_once()
    assert q2 is not None
    assert q2.goal_id == "g_e2e_norm"
    assert store.get_goal("g_e2e_norm").status == GoalStatus.COMPLETED

    # Quantum 3: LOW priority runs and completes
    q3 = mgr.schedule_once()
    assert q3 is not None
    assert q3.goal_id == "g_e2e_low"
    assert store.get_goal("g_e2e_low").status == GoalStatus.COMPLETED

    # Quantum 4: All completed -> nothing to schedule
    q4 = mgr.schedule_once()
    assert q4 is None
