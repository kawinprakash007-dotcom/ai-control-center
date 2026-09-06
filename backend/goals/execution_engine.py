import hashlib
import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from core.interfaces.goal_interface import (
    GoalDecomposerInterface,
    GoalExecutionEngineInterface,
    GoalStoreInterface,
)
from core.interfaces.runtime_interface import (
    CognitiveEventSinkInterface,
    CognitiveRuntimeInterface,
)
from core.models.goal import (
    Goal,
    GoalCompletionCriteria,
    GoalProgress,
    GoalStatus,
    Objective,
    ObjectiveStatus,
)
from core.models.runtime import (
    CognitiveEvent,
    CognitiveEventType,
    CognitiveStage,
    CognitiveTurnResult,
    TurnStatus,
)
from goals.decomposer import DeterministicGoalDecomposer
from goals.store import InMemoryGoalStore

logger = logging.getLogger(__name__)


class GoalExecutionEngine(GoalExecutionEngineInterface):
    """
    Thin, bounded coordination engine for pursuing multi-turn Goals.
    Dispatches each objective turn strictly to CognitiveRuntime.
    Guarantees:
    - No secondary agent loops
    - No direct tool execution
    - No direct model access
    - No direct memory mutations
    - Strict bounds and no-progress protection
    """

    def __init__(
        self,
        runtime: CognitiveRuntimeInterface,
        store: Optional[GoalStoreInterface] = None,
        decomposer: Optional[GoalDecomposerInterface] = None,
        event_sink: Optional[CognitiveEventSinkInterface] = None,
    ):
        self.runtime = runtime
        self.store = store or InMemoryGoalStore()
        self.decomposer = decomposer or DeterministicGoalDecomposer()
        self.event_sink = event_sink

        # Operational tracking per goal
        self._goal_turns: Dict[str, int] = {}
        self._no_progress_state: Dict[str, Tuple[str, int]] = {}  # goal_id -> (state_sig, count)

    def execute_goal(self, goal_id: str, max_steps: Optional[int] = None) -> Goal:
        """
        Execute bounded cognitive turns for a goal until terminal, paused, blocked,
        waiting for user, or step limit is reached.
        """
        step_count = 0
        limit = max_steps if max_steps is not None else 100

        while step_count < limit:
            goal = self.store.get_goal(goal_id)
            if goal is None:
                raise KeyError(f"Goal '{goal_id}' not found in goal store.")

            if goal.is_terminal() or goal.status in (
                GoalStatus.PAUSED,
                GoalStatus.BLOCKED,
                GoalStatus.WAITING_FOR_USER,
            ):
                return goal

            self.step_goal(goal_id)
            step_count += 1

        return self.store.get_goal(goal_id)

    def step_goal(self, goal_id: str) -> Goal:
        """
        Execute exactly one bounded cognitive turn for the next ready objective.
        """
        goal = self.store.get_goal(goal_id)
        if goal is None:
            raise KeyError(f"Goal '{goal_id}' not found in goal store.")

        # Guard: Non-runnable states
        if goal.is_terminal():
            return goal

        if goal.status in (GoalStatus.PAUSED, GoalStatus.BLOCKED, GoalStatus.WAITING_FOR_USER):
            return goal

        # 1. Initialization / Decomposition
        if goal.status == GoalStatus.CREATED:
            if not goal.objectives:
                objectives = self.decomposer.decompose(goal)
                goal.objectives = objectives
            goal.status = GoalStatus.RUNNING
            self._publish_goal_event(
                goal=goal,
                event_type=CognitiveEventType.GOAL_STARTED,
                summary=f"Goal started: {goal.original_goal}",
            )
            self.store.update_goal(goal)

        # 2. Check goal-level total turns budget
        turns_executed = self._goal_turns.get(goal_id, 0)
        max_total_turns = goal.constraints.max_turns_total
        if turns_executed >= max_total_turns:
            goal.status = GoalStatus.ABORTED
            self._update_progress(goal, f"Goal aborted: reached maximum total turns ({max_total_turns}).")
            self._publish_goal_event(
                goal=goal,
                event_type=CognitiveEventType.GOAL_ABORTED,
                summary="Goal aborted due to budget exhaustion.",
            )
            return self.store.update_goal(goal)

        # 3. Select next active or ready objective
        objective = self._select_next_objective(goal)

        if objective is None:
            # No objective available to run
            if self._evaluate_completion(goal):
                goal.status = GoalStatus.COMPLETED
                goal.active_objective_id = None
                self._update_progress(goal, "All completion criteria satisfied.")
                self._publish_goal_event(
                    goal=goal,
                    event_type=CognitiveEventType.GOAL_COMPLETED,
                    summary="Goal completed successfully.",
                )
            else:
                goal.status = GoalStatus.BLOCKED
                self._update_progress(goal, "No ready objectives available; goal blocked.")
                self._publish_goal_event(
                    goal=goal,
                    event_type=CognitiveEventType.GOAL_FAILED,
                    summary="Goal blocked: unresolved objective dependencies.",
                )
            return self.store.update_goal(goal)

        # 4. Construct bounded objective context for CognitiveRuntime
        context_payload = self._build_turn_context(goal, objective)

        # 5. Execute bounded cognitive turn via CognitiveRuntime
        goal.active_objective_id = objective.objective_id
        objective.status = ObjectiveStatus.RUNNING
        objective.attempts += 1
        self._goal_turns[goal_id] = turns_executed + 1

        self._publish_goal_event(
            goal=goal,
            event_type=CognitiveEventType.OBJECTIVE_STARTED,
            objective=objective,
            summary=f"Objective turn started: {objective.description}",
        )

        turn_result: CognitiveTurnResult
        try:
            turn_result = self.runtime.execute_turn(
                input_data=context_payload,
                session_id=f"goal_{goal.goal_id}",
            )
        except Exception as ex:
            logger.error("CognitiveRuntime turn execution error: %s", ex, exc_info=True)
            err_turn_id = f"err_{uuid.uuid4().hex[:8]}"
            err_trace = CognitiveTrace(
                turn_id=err_turn_id,
                session_id=f"goal_{goal.goal_id}",
                events=(),
                final_status=TurnStatus.FAILED,
            )
            turn_result = CognitiveTurnResult(
                turn_id=err_turn_id,
                session_id=f"goal_{goal.goal_id}",
                status=TurnStatus.FAILED,
                stage=CognitiveStage.FAILED,
                response="Internal runtime error during objective turn.",
                trace=err_trace,
                metadata={"error": str(ex)},
            )

        # 6. Interpret Turn Result and update objective state
        self._process_turn_result(goal, objective, turn_result)

        # 7. No-Progress Detection
        self._check_no_progress(goal, objective, turn_result)

        # 8. Update Goal progress metrics and completion status
        self._recalculate_progress(goal)

        if self._evaluate_completion(goal):
            goal.status = GoalStatus.COMPLETED
            goal.active_objective_id = None
            self._update_progress(goal, "All completion criteria satisfied.")
            self._publish_goal_event(
                goal=goal,
                event_type=CognitiveEventType.GOAL_COMPLETED,
                summary="Goal completed successfully.",
            )

        return self.store.update_goal(goal)

    def pause_goal(self, goal_id: str, reason: str = "") -> Goal:
        goal = self.store.get_goal(goal_id)
        if goal is None:
            raise KeyError(f"Goal '{goal_id}' not found in goal store.")

        if not goal.is_terminal():
            goal.status = GoalStatus.PAUSED
            self._update_progress(goal, f"Goal paused: {reason or 'User/system request'}")
            self._publish_goal_event(
                goal=goal,
                event_type=CognitiveEventType.GOAL_PAUSED,
                summary=f"Goal paused: {reason}",
            )
            return self.store.update_goal(goal)
        return goal

    def resume_goal(self, goal_id: str) -> Goal:
        goal = self.store.get_goal(goal_id)
        if goal is None:
            raise KeyError(f"Goal '{goal_id}' not found in goal store.")

        if goal.status in (GoalStatus.PAUSED, GoalStatus.BLOCKED, GoalStatus.WAITING_FOR_USER):
            goal.status = GoalStatus.RUNNING
            # Reset no-progress tracking upon explicit resume
            self._no_progress_state.pop(goal_id, None)
            self._update_progress(goal, "Goal resumed.")
            self._publish_goal_event(
                goal=goal,
                event_type=CognitiveEventType.GOAL_RESUMED,
                summary="Goal resumed from preserved state.",
            )
            return self.store.update_goal(goal)
        return goal

    def abort_goal(self, goal_id: str, reason: str = "") -> Goal:
        goal = self.store.get_goal(goal_id)
        if goal is None:
            raise KeyError(f"Goal '{goal_id}' not found in goal store.")

        if not goal.is_terminal():
            goal.status = GoalStatus.ABORTED
            self._update_progress(goal, f"Goal aborted: {reason or 'Explicit abort'}")
            self._publish_goal_event(
                goal=goal,
                event_type=CognitiveEventType.GOAL_ABORTED,
                summary=f"Goal aborted: {reason}",
            )
            return self.store.update_goal(goal)
        return goal

    # ------------------------------------------------------------------------
    # INTERNAL HELPERS
    # ------------------------------------------------------------------------

    def _select_next_objective(self, goal: Goal) -> Optional[Objective]:
        """Deterministically select the next runnable objective."""
        completed_ids = goal.get_completed_objective_ids()

        # If an objective is currently active and partially completed, continue it
        if goal.active_objective_id:
            active = goal.get_objective(goal.active_objective_id)
            if active and active.status == ObjectiveStatus.PARTIAL:
                return active

        # Find first ready or pending objective whose dependencies are satisfied
        for obj in goal.objectives:
            if obj.is_terminal():
                continue

            # Check if any dependency failed
            has_failed_dep = any(
                dep_obj and dep_obj.status in (ObjectiveStatus.FAILED, ObjectiveStatus.CANCELLED)
                for dep_id in obj.dependencies
                for dep_obj in [goal.get_objective(dep_id)]
            )
            if has_failed_dep:
                obj.status = ObjectiveStatus.BLOCKED
                obj.blocker_reason = "Prerequisite dependency failed."
                continue

            if obj.is_ready(completed_ids):
                if obj.status != ObjectiveStatus.PARTIAL:
                    obj.status = ObjectiveStatus.READY
                return obj

        return None

    def _build_turn_context(self, goal: Goal, objective: Objective) -> str:
        """Construct bounded context prompt for CognitiveRuntime."""
        completed = [
            f"- {o.description}: {o.result_summary or 'Completed'}"
            for o in goal.objectives
            if o.status == ObjectiveStatus.COMPLETED
        ]
        history_text = "\n".join(completed) if completed else "None"

        return (
            f"GOAL: {goal.original_goal}\n"
            f"CURRENT OBJECTIVE: {objective.description}\n"
            f"OBJECTIVE PROGRESS: {int(objective.progress * 100)}%\n"
            f"PREVIOUS COMPLETED WORK:\n{history_text}\n"
            f"INSTRUCTION: Accomplish the current objective towards the overall goal."
        )

    def _process_turn_result(
        self,
        goal: Goal,
        objective: Objective,
        turn_result: CognitiveTurnResult,
    ) -> None:
        """Interpret CognitiveTurnResult and update objective / goal states."""
        if turn_result.status == TurnStatus.WAITING_FOR_USER:
            goal.status = GoalStatus.WAITING_FOR_USER
            objective.status = ObjectiveStatus.BLOCKED
            objective.blocker_reason = turn_result.response or "Waiting for user action / permission."
            self._publish_goal_event(
                goal=goal,
                event_type=CognitiveEventType.GOAL_WAITING_FOR_USER,
                objective=objective,
                turn_id=turn_result.turn_id,
                summary=f"Turn waiting for user: {objective.blocker_reason}",
            )
            return

        if turn_result.status == TurnStatus.SUCCEEDED:
            # Check if turn indicated partial or complete completion
            resp_lower = (turn_result.response or "").lower()
            is_partial = "partial" in resp_lower or "in progress" in resp_lower or "continue" in resp_lower
            effective_max_attempts = min(objective.max_attempts, goal.constraints.max_turns_per_objective)

            if is_partial and objective.attempts < effective_max_attempts:
                objective.status = ObjectiveStatus.PARTIAL
                if "progress" in turn_result.metadata:
                    objective.progress = float(turn_result.metadata["progress"])
                elif "no change" not in resp_lower and "no progress" not in resp_lower:
                    objective.progress = min(0.8, round(objective.progress + 0.4, 2))
                objective.result_summary = turn_result.response
                self._publish_goal_event(
                    goal=goal,
                    event_type=CognitiveEventType.OBJECTIVE_PROGRESS,
                    objective=objective,
                    turn_id=turn_result.turn_id,
                    summary=f"Objective partial progress: {int(objective.progress * 100)}%",
                )
            else:
                objective.status = ObjectiveStatus.COMPLETED
                objective.progress = 1.0
                objective.result_summary = turn_result.response
                self._publish_goal_event(
                    goal=goal,
                    event_type=CognitiveEventType.OBJECTIVE_COMPLETED,
                    objective=objective,
                    turn_id=turn_result.turn_id,
                    summary=f"Objective completed: {objective.description}",
                )
            return

        # Turn failed or aborted
        error_msg = turn_result.metadata.get("error") or getattr(turn_result, "error", None) or "Turn failed."
        effective_max_attempts = min(objective.max_attempts, goal.constraints.max_turns_per_objective)
        if objective.attempts >= effective_max_attempts:
            objective.status = ObjectiveStatus.FAILED
            objective.blocker_reason = error_msg or f"Exceeded max attempts ({effective_max_attempts})."
            self._publish_goal_event(
                goal=goal,
                event_type=CognitiveEventType.OBJECTIVE_BLOCKED,
                objective=objective,
                turn_id=turn_result.turn_id,
                summary=f"Objective failed: {objective.blocker_reason}",
            )

            # Check max failed objectives threshold
            failed_count = sum(1 for o in goal.objectives if o.status == ObjectiveStatus.FAILED)
            if failed_count >= goal.constraints.max_failed_objectives:
                goal.status = GoalStatus.FAILED
                self._publish_goal_event(
                    goal=goal,
                    event_type=CognitiveEventType.GOAL_FAILED,
                    objective=objective,
                    turn_id=turn_result.turn_id,
                    summary=f"Goal failed: exceeded failed objectives limit ({failed_count}/{goal.constraints.max_failed_objectives}).",
                )
        else:
            # Allow next turn retry
            objective.status = ObjectiveStatus.PARTIAL
            objective.blocker_reason = error_msg or "Turn failed; retrying in next turn."

    def _check_no_progress(
        self,
        goal: Goal,
        objective: Objective,
        turn_result: CognitiveTurnResult,
    ) -> None:
        """Detect repeated identical turn states to prevent infinite execution loops."""
        if objective.is_terminal():
            return
        state_payload = f"{objective.objective_id}|{objective.progress}|{turn_result.status.value}|{turn_result.response[:100]}"
        state_sig = hashlib.sha256(state_payload.encode("utf-8")).hexdigest()

        last_sig, count = self._no_progress_state.get(goal.goal_id, ("", 0))
        if last_sig == state_sig:
            count += 1
        else:
            count = 1

        self._no_progress_state[goal.goal_id] = (state_sig, count)

        max_limit = goal.constraints.max_consecutive_no_progress
        if count >= max_limit:
            goal.status = GoalStatus.BLOCKED
            objective.status = ObjectiveStatus.BLOCKED
            objective.blocker_reason = f"No-progress detected across {count} consecutive identical turns."
            self._update_progress(goal, objective.blocker_reason)
            self._publish_goal_event(
                goal=goal,
                event_type=CognitiveEventType.OBJECTIVE_BLOCKED,
                objective=objective,
                summary=f"Loop protection triggered: {objective.blocker_reason}",
            )

    def _recalculate_progress(self, goal: Goal) -> None:
        """Update Goal progress percentage and blocker list."""
        total = len(goal.objectives)
        completed = sum(1 for o in goal.objectives if o.status == ObjectiveStatus.COMPLETED)
        percentage = round((completed / total) * 100.0, 1) if total > 0 else 0.0

        blockers = tuple(
            o.blocker_reason for o in goal.objectives
            if o.status == ObjectiveStatus.BLOCKED and o.blocker_reason
        )

        reason = (
            goal.progress.progress_reason
            if (goal.status in (GoalStatus.BLOCKED, GoalStatus.FAILED, GoalStatus.ABORTED, GoalStatus.PAUSED) and goal.progress.progress_reason)
            else f"{completed}/{total} objectives completed ({percentage}%)"
        )

        goal.progress = GoalProgress(
            completed_objectives=completed,
            total_objectives=total,
            percentage=percentage,
            active_objective_id=goal.active_objective_id,
            blockers=blockers,
            last_update=time.time(),
            progress_reason=reason,
        )

    def _evaluate_completion(self, goal: Goal) -> bool:
        """Evaluate explicit goal completion criteria."""
        criteria = goal.completion_criteria

        if criteria.required_objective_ids:
            # Explicit required objectives check
            for req_id in criteria.required_objective_ids:
                obj = goal.get_objective(req_id)
                if not obj or obj.status != ObjectiveStatus.COMPLETED:
                    return False

        if criteria.require_all_objectives:
            # All objectives check
            if any(o.status != ObjectiveStatus.COMPLETED for o in goal.objectives):
                return False

        if goal.progress.percentage < criteria.min_completion_percentage:
            return False

        return True

    def _update_progress(self, goal: Goal, reason: str) -> None:
        """Update progress reason text."""
        goal.progress = GoalProgress(
            completed_objectives=goal.progress.completed_objectives,
            total_objectives=goal.progress.total_objectives,
            percentage=goal.progress.percentage,
            active_objective_id=goal.active_objective_id,
            blockers=goal.progress.blockers,
            last_update=time.time(),
            progress_reason=reason,
        )

    def _publish_goal_event(
        self,
        goal: Goal,
        event_type: CognitiveEventType,
        summary: str = "",
        objective: Optional[Objective] = None,
        turn_id: Optional[str] = None,
    ) -> None:
        """Publish structured goal lifecycle event with correlation IDs."""
        if self.event_sink is None:
            return

        active_obj_id = objective.objective_id if objective else goal.active_objective_id
        meta = {
            "goal_id": goal.goal_id,
            "objective_id": active_obj_id,
            "turn_id": turn_id,
            "goal_status": goal.status.value,
            "progress_pct": goal.progress.percentage,
        }

        event = CognitiveEvent(
            event_id=f"evt_g_{uuid.uuid4().hex[:10]}",
            turn_id=turn_id or f"goal_{goal.goal_id}",
            session_id=f"goal_{goal.goal_id}",
            stage=CognitiveStage.RECEIVED,
            event_type=event_type,
            timestamp=time.time(),
            duration=0.0,
            status="OK" if goal.status != GoalStatus.FAILED else "FAILED",
            component="goal_engine",
            summary=summary,
            metadata=meta,
        )
        self.event_sink.publish(event)
