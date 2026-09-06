import pytest
import time
from typing import Any, Dict, List, Optional, Tuple

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
from core.models.runtime import (
    CognitiveEvent,
    CognitiveEventType,
    CognitiveStage,
    CognitiveTrace,
    CognitiveTurnResult,
    TurnStatus,
)
from goals.decomposer import DecomposerError, DeterministicGoalDecomposer
from goals.execution_engine import GoalExecutionEngine
from goals.store import InMemoryGoalStore, SQLiteGoalStore, serialize_goal, deserialize_goal
from runtime.event_sink import InMemoryEventSink


# ============================================================================
# FAKE COGNITIVE RUNTIME FOR GOAL TESTING
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
    """Controlled CognitiveRuntime fake for testing GoalExecutionEngine."""

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
            turn_id=f"turn_{len(self.call_history)}",
            session_id=session_id or "default_session",
            status=TurnStatus.SUCCEEDED,
            response="Objective task completed successfully.",
        )

    def run(
        self,
        input_data: Any,
        session_id: Optional[str] = None,
    ) -> str:
        return self.execute_turn(input_data, session_id).response


# ============================================================================
# 12 DETERMINISTIC TEST FIXTURES
# ============================================================================

def fixture_one_objective_goal() -> Goal:
    """Fixture 1: Atomic single-objective goal."""
    return Goal(
        goal_id="g_one_01",
        original_goal="Summarize research paper",
        objectives=(
            Objective(objective_id="g_one_01_obj_1", description="Summarize paper", order=1, status=ObjectiveStatus.READY),
        ),
    )


def fixture_multi_objective_goal() -> Goal:
    """Fixture 2: Independent multi-objective goal."""
    return Goal(
        goal_id="g_multi_01",
        original_goal="Gather system diagnostics",
        objectives=(
            Objective(objective_id="g_multi_01_obj_1", description="Inspect CPU", order=1, status=ObjectiveStatus.READY),
            Objective(objective_id="g_multi_01_obj_2", description="Inspect RAM", order=2, status=ObjectiveStatus.READY),
            Objective(objective_id="g_multi_01_obj_3", description="Inspect Disk", order=3, status=ObjectiveStatus.READY),
        ),
    )


def fixture_dependent_objectives_goal() -> Goal:
    """Fixture 3: Sequentially dependent objectives."""
    return Goal(
        goal_id="g_dep_01",
        original_goal="Download, compile, and run benchmarks",
        objectives=(
            Objective(objective_id="g_dep_01_obj_1", description="Download source", order=1, status=ObjectiveStatus.READY),
            Objective(objective_id="g_dep_01_obj_2", description="Compile source", order=2, dependencies=("g_dep_01_obj_1",), status=ObjectiveStatus.PENDING),
            Objective(objective_id="g_dep_01_obj_3", description="Run benchmarks", order=3, dependencies=("g_dep_01_obj_2",), status=ObjectiveStatus.PENDING),
        ),
    )


def fixture_partial_progress_goal() -> Goal:
    """Fixture 4: Goal with an objective currently in partial progress."""
    return Goal(
        goal_id="g_partial_01",
        original_goal="Scan 1000 files",
        active_objective_id="g_partial_01_obj_1",
        status=GoalStatus.RUNNING,
        objectives=(
            Objective(objective_id="g_partial_01_obj_1", description="Scan files", order=1, status=ObjectiveStatus.PARTIAL, progress=0.4, attempts=1),
        ),
    )


def fixture_blocked_objective_goal() -> Goal:
    """Fixture 5: Goal blocked by missing prerequisite or environmental blocker."""
    return Goal(
        goal_id="g_blocked_01",
        original_goal="Deploy to restricted server",
        status=GoalStatus.BLOCKED,
        objectives=(
            Objective(objective_id="g_blocked_01_obj_1", description="Deploy artifact", order=1, status=ObjectiveStatus.BLOCKED, blocker_reason="SSH credentials unavailable"),
        ),
        progress=GoalProgress(blockers=("SSH credentials unavailable",), progress_reason="Blocked"),
    )


def fixture_waiting_for_user_goal() -> Goal:
    """Fixture 6: Goal waiting for explicit user confirmation."""
    return Goal(
        goal_id="g_wait_01",
        original_goal="Format production drive",
        status=GoalStatus.WAITING_FOR_USER,
        objectives=(
            Objective(objective_id="g_wait_01_obj_1", description="Format disk", order=1, status=ObjectiveStatus.BLOCKED, blocker_reason="Confirmation required"),
        ),
    )


def fixture_paused_goal() -> Goal:
    """Fixture 7: Manually paused goal."""
    return Goal(
        goal_id="g_pause_01",
        original_goal="Index filesystem",
        status=GoalStatus.PAUSED,
        objectives=(
            Objective(objective_id="g_pause_01_obj_1", description="Index directories", order=1, status=ObjectiveStatus.READY),
        ),
    )


def fixture_failed_goal() -> Goal:
    """Fixture 8: Terminal failed goal."""
    return Goal(
        goal_id="g_fail_01",
        original_goal="Connect to offline server",
        status=GoalStatus.FAILED,
        objectives=(
            Objective(objective_id="g_fail_01_obj_1", description="Connect socket", order=1, status=ObjectiveStatus.FAILED, blocker_reason="Network unreachable", attempts=3),
        ),
    )


def fixture_successful_goal() -> Goal:
    """Fixture 9: Terminal completed goal."""
    return Goal(
        goal_id="g_succ_01",
        original_goal="Backup configurations",
        status=GoalStatus.COMPLETED,
        objectives=(
            Objective(objective_id="g_succ_01_obj_1", description="Archive configs", order=1, status=ObjectiveStatus.COMPLETED, progress=1.0, result_summary="Archive created"),
        ),
        progress=GoalProgress(completed_objectives=1, total_objectives=1, percentage=100.0, progress_reason="Completed"),
    )


def fixture_no_progress_goal() -> Goal:
    """Fixture 10: Goal subject to repeated no-progress detection."""
    return Goal(
        goal_id="g_loop_01",
        original_goal="Flaky service poll",
        status=GoalStatus.RUNNING,
        objectives=(
            Objective(objective_id="g_loop_01_obj_1", description="Poll service", order=1, status=ObjectiveStatus.READY, max_attempts=5),
        ),
        constraints=GoalConstraints(max_consecutive_no_progress=3),
    )


def fixture_objective_retry_goal() -> Goal:
    """Fixture 11: Goal requiring retry across multiple turns."""
    return Goal(
        goal_id="g_retry_01",
        original_goal="Fetch unstable endpoint",
        status=GoalStatus.RUNNING,
        objectives=(
            Objective(objective_id="g_retry_01_obj_1", description="HTTP Fetch", order=1, status=ObjectiveStatus.READY, max_attempts=3),
        ),
    )


def fixture_replay_linked_goal() -> Goal:
    """Fixture 12: Goal with explicit correlation IDs for Phase 4.1 replay."""
    return Goal(
        goal_id="g_replay_01",
        original_goal="Replay compatible audit task",
        status=GoalStatus.RUNNING,
        objectives=(
            Objective(objective_id="g_replay_01_obj_1", description="Audit logs", order=1, status=ObjectiveStatus.READY),
        ),
    )


# ============================================================================
# A. GOAL DOMAIN TESTS
# ============================================================================

def test_goal_domain_valid_instantiation():
    """A1. Goal instantiates with default constraints and progress."""
    goal = Goal(original_goal="Build robot")
    assert goal.original_goal == "Build robot"
    assert goal.status == GoalStatus.CREATED
    assert goal.constraints.max_turns_total == 20
    assert goal.completion_criteria.require_all_objectives is True
    assert goal.progress.percentage == 0.0


def test_original_goal_immutability():
    """A2. Attempting to mutate original_goal raises AttributeError."""
    goal = Goal(original_goal="Permanent high-level goal")
    with pytest.raises(AttributeError, match="immutable"):
        goal.original_goal = "Altered narrow goal"


def test_legacy_goal_compatibility():
    """A3. Legacy Goal(goal='...') initialization works seamlessly."""
    legacy = Goal(goal="Launch Calculator")
    assert legacy.original_goal == "Launch Calculator"
    assert legacy.goal == "Launch Calculator"
    assert legacy.priority == "normal"


# ============================================================================
# B. OBJECTIVE DOMAIN TESTS
# ============================================================================

def test_objective_domain_dependency_check():
    """B1. Objective readiness respects dependency satisfaction."""
    obj1 = Objective(objective_id="o1", description="Step 1", order=1, status=ObjectiveStatus.COMPLETED)
    obj2 = Objective(objective_id="o2", description="Step 2", order=2, dependencies=("o1",), status=ObjectiveStatus.PENDING)

    assert obj2.is_ready(completed_objective_ids=("o1",)) is True
    assert obj2.is_ready(completed_objective_ids=()) is False


def test_objective_terminal_detection():
    """B2. Objective correctly reports terminal states."""
    obj = Objective(objective_id="o1", description="Task", status=ObjectiveStatus.COMPLETED)
    assert obj.is_terminal() is True

    obj.status = ObjectiveStatus.RUNNING
    assert obj.is_terminal() is False


# ============================================================================
# C. DECOMPOSITION TESTS
# ============================================================================

def test_deterministic_decomposer_numbered_steps():
    """C1. Decomposer correctly parses numbered steps."""
    decomposer = DeterministicGoalDecomposer()
    goal = Goal(original_goal="1. Clone repo\n2. Run tests\n3. Deploy app")
    objectives = decomposer.decompose(goal)

    assert len(objectives) == 3
    assert objectives[0].description == "Clone repo"
    assert objectives[0].status == ObjectiveStatus.READY
    assert objectives[1].description == "Run tests"
    assert objectives[1].dependencies == (objectives[0].objective_id,)
    assert objectives[2].description == "Deploy app"
    assert objectives[2].dependencies == (objectives[1].objective_id,)


def test_deterministic_decomposer_rejects_duplicates():
    """C2. Decomposer flags duplicate objectives."""
    decomposer = DeterministicGoalDecomposer()
    goal = Goal(original_goal="1. Clone repo\n2. Clone repo")
    with pytest.raises(DecomposerError, match="Duplicate objective"):
        decomposer.decompose(goal)


def test_deterministic_decomposer_atomic_goal():
    """C3. Atomic goal creates single bounded objective."""
    decomposer = DeterministicGoalDecomposer()
    goal = Goal(original_goal="Calculate prime numbers")
    objectives = decomposer.decompose(goal)

    assert len(objectives) == 1
    assert objectives[0].description == "Calculate prime numbers"
    assert objectives[0].status == ObjectiveStatus.READY


# ============================================================================
# D. PROGRESS TESTS
# ============================================================================

def test_progress_calculation():
    """D1. GoalExecutionEngine accurately calculates progress percentage."""
    goal = fixture_multi_objective_goal()
    store = InMemoryGoalStore()
    store.create_goal(goal)

    runtime = FakeCognitiveRuntime()
    engine = GoalExecutionEngine(runtime=runtime, store=store)

    # Step 1
    updated = engine.step_goal(goal.goal_id)
    assert updated.progress.completed_objectives == 1
    assert updated.progress.total_objectives == 3
    assert updated.progress.percentage == 33.3


# ============================================================================
# E. OBJECTIVE SELECTION TESTS
# ============================================================================

def test_deterministic_objective_selection_ordering():
    """E1. Objective selection respects sequential dependencies."""
    goal = fixture_dependent_objectives_goal()
    store = InMemoryGoalStore()
    store.create_goal(goal)

    runtime = FakeCognitiveRuntime()
    engine = GoalExecutionEngine(runtime=runtime, store=store)

    # First step runs obj 1
    g1 = engine.step_goal(goal.goal_id)
    assert g1.objectives[0].status == ObjectiveStatus.COMPLETED

    # Second step runs obj 2 (which depended on obj 1)
    g2 = engine.step_goal(goal.goal_id)
    assert g2.objectives[1].status == ObjectiveStatus.COMPLETED


# ============================================================================
# F. GOAL EXECUTION TESTS
# ============================================================================

def test_goal_execution_invokes_runtime_once_per_step():
    """F1. step_goal invokes CognitiveRuntime exactly once."""
    goal = fixture_one_objective_goal()
    store = InMemoryGoalStore()
    store.create_goal(goal)

    runtime = FakeCognitiveRuntime()
    engine = GoalExecutionEngine(runtime=runtime, store=store)

    engine.step_goal(goal.goal_id)
    assert len(runtime.call_history) == 1
    assert "Summarize paper" in runtime.call_history[0][0]


# ============================================================================
# G. MULTI-TURN OBJECTIVE CONTINUATION TESTS
# ============================================================================

def test_objective_multi_turn_continuation():
    """G1. Partial objective progress continues across turns without restart."""
    goal = fixture_one_objective_goal()
    store = InMemoryGoalStore()
    store.create_goal(goal)

    # Turn 1: Partial progress
    res1 = _make_turn_result(
        turn_id="t1", session_id="s1", status=TurnStatus.SUCCEEDED,
        response="Partially scanned index. Continue in progress.",
    )
    # Turn 2: Completion
    res2 = _make_turn_result(
        turn_id="t2", session_id="s1", status=TurnStatus.SUCCEEDED,
        response="Scan finished completely.",
    )

    runtime = FakeCognitiveRuntime([res1, res2])
    engine = GoalExecutionEngine(runtime=runtime, store=store)

    # Step 1: Partial
    g1 = engine.step_goal(goal.goal_id)
    assert g1.objectives[0].status == ObjectiveStatus.PARTIAL
    assert g1.objectives[0].progress > 0.0
    assert g1.status == GoalStatus.RUNNING

    # Step 2: Complete
    g2 = engine.step_goal(goal.goal_id)
    assert g2.objectives[0].status == ObjectiveStatus.COMPLETED
    assert g2.objectives[0].progress == 1.0
    assert g2.status == GoalStatus.COMPLETED


# ============================================================================
# H. PAUSE AND RESUME TESTS
# ============================================================================

def test_pause_stops_execution_and_resume_continues():
    """H1. Paused goal halts turn creation; resume continues from saved state."""
    goal = fixture_multi_objective_goal()
    store = InMemoryGoalStore()
    store.create_goal(goal)

    runtime = FakeCognitiveRuntime()
    engine = GoalExecutionEngine(runtime=runtime, store=store)

    # Step 1: Objective 1 completes
    engine.step_goal(goal.goal_id)
    assert store.get_goal(goal.goal_id).progress.completed_objectives == 1

    # Pause goal
    engine.pause_goal(goal.goal_id, reason="User interrupted")
    assert store.get_goal(goal.goal_id).status == GoalStatus.PAUSED

    # Attempt step while paused: should not execute runtime turns
    engine.step_goal(goal.goal_id)
    assert len(runtime.call_history) == 1

    # Resume goal
    engine.resume_goal(goal.goal_id)
    assert store.get_goal(goal.goal_id).status == GoalStatus.RUNNING

    # Step again: Objective 2 executes
    engine.step_goal(goal.goal_id)
    assert len(runtime.call_history) == 2
    assert store.get_goal(goal.goal_id).progress.completed_objectives == 2


# ============================================================================
# I. WAITING FOR USER TESTS
# ============================================================================

def test_waiting_for_user_halts_automatic_execution():
    """I1. TurnStatus.WAITING_FOR_USER transitions goal to WAITING_FOR_USER."""
    goal = fixture_one_objective_goal()
    store = InMemoryGoalStore()
    store.create_goal(goal)

    turn_res = _make_turn_result(
        turn_id="t_wait", session_id="s1", status=TurnStatus.WAITING_FOR_USER,
        response="Please grant permission to write to /etc.",
    )
    runtime = FakeCognitiveRuntime([turn_res])
    engine = GoalExecutionEngine(runtime=runtime, store=store)

    g = engine.step_goal(goal.goal_id)
    assert g.status == GoalStatus.WAITING_FOR_USER
    assert "permission" in g.objectives[0].blocker_reason.lower()


# ============================================================================
# J. BLOCKED GOAL TESTS
# ============================================================================

def test_blocked_prerequisite_blocks_goal():
    """J1. Failed prerequisite marks dependent objective and goal as BLOCKED."""
    goal = fixture_dependent_objectives_goal()
    store = InMemoryGoalStore()
    store.create_goal(goal)

    # Objective 1 fails 3 times
    err_res = _make_turn_result(
        turn_id="err", session_id="s1", status=TurnStatus.FAILED,
        response="", error="Failed to download",
    )
    runtime = FakeCognitiveRuntime([err_res, err_res, err_res])
    engine = GoalExecutionEngine(runtime=runtime, store=store)

    # 3 attempts on objective 1
    engine.step_goal(goal.goal_id)
    engine.step_goal(goal.goal_id)
    engine.step_goal(goal.goal_id)

    g = store.get_goal(goal.goal_id)
    assert g.objectives[0].status == ObjectiveStatus.FAILED

    # Next step attempts obj 2, detects prerequisite failed -> goal blocked
    g_blocked = engine.step_goal(goal.goal_id)
    assert g_blocked.status in (GoalStatus.BLOCKED, GoalStatus.FAILED)


# ============================================================================
# K. FAILURE HANDLING TESTS
# ============================================================================

def test_repeated_failure_exceeds_threshold_and_fails_goal():
    """K1. Exceeding max failed objectives transitions goal to FAILED."""
    goal = fixture_multi_objective_goal()
    # Allow at most 1 failed objective
    goal = Goal(
        goal_id="g_fail_thresh",
        original_goal="Multi task",
        objectives=goal.objectives,
        constraints=GoalConstraints(max_failed_objectives=1, max_turns_per_objective=1),
    )
    store = InMemoryGoalStore()
    store.create_goal(goal)

    err = _make_turn_result(
        turn_id="err", session_id="s1", status=TurnStatus.FAILED,
        response="", error="Critical crash",
    )
    runtime = FakeCognitiveRuntime([err])
    engine = GoalExecutionEngine(runtime=runtime, store=store)

    g = engine.step_goal(goal.goal_id)
    assert g.status == GoalStatus.FAILED


# ============================================================================
# L. COMPLETION CRITERIA TESTS
# ============================================================================

def test_completion_criteria_required_objectives_only():
    """L1. Completion criteria can require specific objective subset."""
    obj1 = Objective(objective_id="obj_req", description="Required task", order=1, status=ObjectiveStatus.READY)
    obj2 = Objective(objective_id="obj_opt", description="Optional task", order=2, status=ObjectiveStatus.PENDING)

    goal = Goal(
        goal_id="g_partial_req",
        original_goal="Partial required goal",
        objectives=(obj1, obj2),
        completion_criteria=GoalCompletionCriteria(
            require_all_objectives=False,
            required_objective_ids=("obj_req",),
            min_completion_percentage=50.0,
        ),
    )
    store = InMemoryGoalStore()
    store.create_goal(goal)

    runtime = FakeCognitiveRuntime()
    engine = GoalExecutionEngine(runtime=runtime, store=store)

    # Step 1 completes required objective
    g = engine.step_goal(goal.goal_id)
    assert g.status == GoalStatus.COMPLETED


# ============================================================================
# M. NO-PROGRESS LOOP PROTECTION TESTS
# ============================================================================

def test_no_progress_loop_protection():
    """M1. Consecutive identical turns trigger no-progress BLOCKED state."""
    goal = fixture_no_progress_goal()
    store = InMemoryGoalStore()
    store.create_goal(goal)

    # 3 identical turns
    res = _make_turn_result(
        turn_id="t_same", session_id="s1", status=TurnStatus.SUCCEEDED,
        response="continue in progress with no change",
    )
    runtime = FakeCognitiveRuntime([res, res, res, res])
    engine = GoalExecutionEngine(runtime=runtime, store=store)

    engine.step_goal(goal.goal_id)
    engine.step_goal(goal.goal_id)
    g3 = engine.step_goal(goal.goal_id)

    assert g3.status == GoalStatus.BLOCKED
    assert "no-progress" in g3.progress.progress_reason.lower()


# ============================================================================
# N. BUDGET BOUNDS TESTS
# ============================================================================

def test_goal_budget_max_total_turns():
    """N1. Goal aborts when max total turns budget is exceeded."""
    goal = Goal(
        goal_id="g_budget",
        original_goal="Endless loop task",
        objectives=(Objective(objective_id="o1", description="Repeat task", status=ObjectiveStatus.PARTIAL),),
        constraints=GoalConstraints(max_turns_total=2),
    )
    store = InMemoryGoalStore()
    store.create_goal(goal)

    part_res = _make_turn_result(
        turn_id="t", session_id="s1", status=TurnStatus.SUCCEEDED,
        response="continue in progress",
    )
    runtime = FakeCognitiveRuntime([part_res, part_res, part_res])
    engine = GoalExecutionEngine(runtime=runtime, store=store)

    engine.step_goal(goal.goal_id)
    engine.step_goal(goal.goal_id)
    g3 = engine.step_goal(goal.goal_id)

    assert g3.status == GoalStatus.ABORTED


# ============================================================================
# O. PERSISTENCE TESTS (IN-MEMORY & SQLITE)
# ============================================================================

def test_sqlite_goal_store_persistence(tmp_path):
    """O1. SQLiteGoalStore persists and restores goal across connections."""
    db_file = str(tmp_path / "goals_test.db")
    store1 = SQLiteGoalStore(db_path=db_file)

    goal = fixture_multi_objective_goal()
    store1.create_goal(goal)

    # Open fresh connection to same DB
    store2 = SQLiteGoalStore(db_path=db_file)
    loaded = store2.get_goal(goal.goal_id)

    assert loaded is not None
    assert loaded.original_goal == goal.original_goal
    assert len(loaded.objectives) == 3
    assert loaded.objectives[0].description == "Inspect CPU"


# ============================================================================
# P. OBSERVABILITY & CORRELATION TESTS
# ============================================================================

def test_goal_observability_event_correlation():
    """P1. Goal events emit with correlated goal_id, objective_id, and turn_id."""
    goal = fixture_one_objective_goal()
    store = InMemoryGoalStore()
    store.create_goal(goal)

    sink = InMemoryEventSink()
    runtime = FakeCognitiveRuntime()
    engine = GoalExecutionEngine(runtime=runtime, store=store, event_sink=sink)

    engine.step_goal(goal.goal_id)

    events = sink.get_events()
    assert len(events) >= 2  # GOAL_STARTED, OBJECTIVE_STARTED, OBJECTIVE_COMPLETED, GOAL_COMPLETED
    goal_events = [e for e in events if "goal_id" in e.metadata]
    assert len(goal_events) > 0
    assert goal_events[0].metadata["goal_id"] == goal.goal_id


# ============================================================================
# Q. REPLAY INTEGRATION TESTS
# ============================================================================

def test_replay_compatibility_turn_linkage():
    """Q1. Objective turns preserve goal_id correlation for Phase 4.1 replay."""
    goal = fixture_replay_linked_goal()
    store = InMemoryGoalStore()
    store.create_goal(goal)

    runtime = FakeCognitiveRuntime()
    engine = GoalExecutionEngine(runtime=runtime, store=store)

    engine.step_goal(goal.goal_id)
    assert len(runtime.call_history) == 1
    # Verify session_id links to goal_id
    assert runtime.call_history[0][1] == f"goal_{goal.goal_id}"


# ============================================================================
# R. SECURITY BOUNDARY CHECKS
# ============================================================================

def test_security_engine_has_no_direct_tool_or_model_access():
    """R1. GoalExecutionEngine does not import or hold tool orchestrators or model providers."""
    import goals.execution_engine as ee_mod
    source = open(ee_mod.__file__, "r").read()

    assert "ToolOrchestrator" not in source
    assert "ComputerBackend" not in source
    assert "OpenAI" not in source
    assert "Gemini" not in source
    assert "Anthropic" not in source
    assert "subprocess" not in source


# ============================================================================
# S. ARCHITECTURAL & END-TO-END LONG-HORIZON TEST
# ============================================================================

def test_long_horizon_end_to_end_3_objective_task():
    """S1. 3-objective task completes with partial progress and verified state."""
    goal = Goal(
        goal_id="g_e2e_3step",
        original_goal="1. Collect data\n2. Analyze metrics\n3. Write report",
    )
    store = InMemoryGoalStore()
    store.create_goal(goal)

    # Objective 1: turn 1 -> success
    # Objective 2: turn 2 -> partial, turn 3 -> success
    # Objective 3: turn 4 -> success
    res_obj1 = _make_turn_result(
        turn_id="t1", session_id="s", status=TurnStatus.SUCCEEDED,
        response="Data collected successfully.",
    )
    res_obj2_part = _make_turn_result(
        turn_id="t2", session_id="s", status=TurnStatus.SUCCEEDED,
        response="Metrics partially computed; continue processing.",
    )
    res_obj2_done = _make_turn_result(
        turn_id="t3", session_id="s", status=TurnStatus.SUCCEEDED,
        response="Metrics analysis finished.",
    )
    res_obj3_done = _make_turn_result(
        turn_id="t4", session_id="s", status=TurnStatus.SUCCEEDED,
        response="Report generated.",
    )

    runtime = FakeCognitiveRuntime([res_obj1, res_obj2_part, res_obj2_done, res_obj3_done])
    engine = GoalExecutionEngine(runtime=runtime, store=store)

    final_goal = engine.execute_goal(goal.goal_id)

    assert final_goal.status == GoalStatus.COMPLETED
    assert final_goal.original_goal == "1. Collect data\n2. Analyze metrics\n3. Write report"
    assert len(final_goal.objectives) == 3
    assert all(o.status == ObjectiveStatus.COMPLETED for o in final_goal.objectives)
    assert final_goal.progress.percentage == 100.0
    assert len(runtime.call_history) == 4
