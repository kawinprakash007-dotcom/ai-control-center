import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from core.interfaces.goal_interface import GoalSchedulerInterface
from core.models.goal import Goal, GoalStatus, ObjectiveStatus
from core.models.goal_management import (
    DeadlineStatus,
    GoalManagementState,
    GoalPriority,
    SchedulerConfig,
    SchedulingDecision,
)


class DeterministicGoalScheduler(GoalSchedulerInterface):
    """
    Deterministic, model-neutral goal scheduler for Phase 4.3.
    Evaluates candidate goals and deterministically selects the single goal
    eligible to receive the next execution quantum.

    Hierarchy of evaluation:
    1. Eligibility (runnable state, not paused, not blocked, not waiting, not terminal)
    2. Deadline Urgency (OVERDUE > DUE_SOON > ON_TRACK / NO_DEADLINE)
    3. User Priority (CRITICAL > HIGH > NORMAL > LOW)
    4. Fairness / Anti-Starvation Aging
    5. Active Goal Continuation
    6. Deterministic goal_id Tie-breaker
    """

    def select_next_goal(
        self,
        goals: Sequence[Goal],
        state: Optional[GoalManagementState] = None,
        config: Optional[SchedulerConfig] = None,
        clock: Optional[Callable[[], float]] = None,
    ) -> SchedulingDecision:
        current_time = clock() if clock is not None else time.time()
        cfg = config or SchedulerConfig()
        fairness_state = state.fairness_counters if state else {}

        considered_ids: List[str] = []
        skipped: Dict[str, str] = {}
        eligible_candidates: List[Tuple[int, str, Goal, DeadlineStatus, Dict[str, Any]]] = []

        for goal in goals:
            g_id = goal.goal_id
            considered_ids.append(g_id)

            # 1. Eligibility evaluation
            is_eligible, skip_reason = self._evaluate_eligibility(goal)
            if not is_eligible:
                skipped[g_id] = skip_reason
                continue

            # 2. Deadline status evaluation
            deadline_status = self._evaluate_deadline(goal, current_time, cfg)

            # 3. Calculate deterministic score
            score, factors = self._calculate_score(
                goal=goal,
                deadline_status=deadline_status,
                fairness_counter=fairness_state.get(g_id, 0),
                is_currently_active=(state.active_goal_id == g_id) if state else False,
                config=cfg,
            )

            eligible_candidates.append((score, g_id, goal, deadline_status, factors))

        if not eligible_candidates:
            return SchedulingDecision(
                selected_goal_id=None,
                reason="No eligible candidate goals available.",
                candidates_considered=tuple(considered_ids),
                skipped_candidates=skipped,
                scheduling_factors={},
                timestamp=current_time,
            )

        # 4. Sort deterministically:
        # - Primary: Score descending (-score)
        # - Tie-breaker: goal_id ascending (for stable, predictable ordering)
        eligible_candidates.sort(key=lambda item: (-item[0], item[1]))

        winning_score, winning_id, winning_goal, winning_deadline, winning_factors = eligible_candidates[0]

        # Record why the winning goal was selected
        decision_reason = (
            f"Selected goal '{winning_id}' with score {winning_score} "
            f"(priority={winning_goal.priority}, deadline={winning_deadline.value}, "
            f"fairness_skips={winning_factors.get('fairness_skips', 0)})."
        )

        all_factors = {
            item[1]: {
                "score": item[0],
                "priority": str(item[2].priority),
                "deadline_status": item[3].value,
                **item[4],
            }
            for item in eligible_candidates
        }

        return SchedulingDecision(
            selected_goal_id=winning_id,
            reason=decision_reason,
            candidates_considered=tuple(considered_ids),
            skipped_candidates=skipped,
            scheduling_factors=all_factors,
            timestamp=current_time,
        )

    def _evaluate_eligibility(self, goal: Goal) -> Tuple[bool, str]:
        """Check if a goal is eligible for scheduling in the current cycle."""
        if goal.is_terminal():
            return False, f"Terminal status: {goal.status.value}"

        if goal.status == GoalStatus.PAUSED:
            return False, "Goal is PAUSED"

        if goal.status == GoalStatus.WAITING_FOR_USER:
            return False, "Goal is WAITING_FOR_USER"

        if goal.status == GoalStatus.BLOCKED:
            return False, "Goal is BLOCKED"

        if goal.status not in (GoalStatus.CREATED, GoalStatus.RUNNING):
            return False, f"Non-runnable status: {goal.status.value}"

        # If goal already decomposed, ensure it has runnable non-terminal objectives
        if goal.objectives:
            completed_ids = goal.get_completed_objective_ids()
            has_runnable = any(obj.is_ready(completed_ids) for obj in goal.objectives if not obj.is_terminal())
            if not has_runnable and not goal.active_objective_id:
                return False, "No ready objectives available"

        return True, ""

    def _evaluate_deadline(
        self,
        goal: Goal,
        current_time: float,
        config: SchedulerConfig,
    ) -> DeadlineStatus:
        """Classify goal deadline urgency relative to the current clock."""
        deadline = goal.constraints.deadline
        if deadline is None:
            return DeadlineStatus.NO_DEADLINE

        if current_time >= deadline:
            return DeadlineStatus.OVERDUE

        time_remaining = deadline - current_time
        if time_remaining <= config.due_soon_window_seconds:
            return DeadlineStatus.DUE_SOON

        return DeadlineStatus.ON_TRACK

    def _calculate_score(
        self,
        goal: Goal,
        deadline_status: DeadlineStatus,
        fairness_counter: int,
        is_currently_active: bool,
        config: SchedulerConfig,
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Compute deterministic composite scheduling score.
        Higher score receives priority.
        """
        # Base priority score
        p = goal.priority if isinstance(goal.priority, GoalPriority) else GoalPriority.from_str(str(goal.priority))
        base_priority_score = p.score_weight

        # Deadline urgency bonus
        deadline_bonus = 0
        if deadline_status == DeadlineStatus.OVERDUE:
            deadline_bonus = 2500
        elif deadline_status == DeadlineStatus.DUE_SOON:
            deadline_bonus = 1500

        # Fairness / Anti-starvation aging bonus
        fairness_aging_bonus = fairness_counter * config.fairness_boost_per_skip
        if fairness_counter >= config.fairness_starvation_limit:
            # Extra starvation boost to overtake higher-priority non-urgent goals
            fairness_aging_bonus += 1200

        # Active continuation bonus (prefer finishing active work over context-switching)
        continuation_bonus = config.active_continuation_boost if is_currently_active else 0

        total_score = base_priority_score + deadline_bonus + fairness_aging_bonus + continuation_bonus

        factors = {
            "base_priority_score": base_priority_score,
            "deadline_bonus": deadline_bonus,
            "fairness_aging_bonus": fairness_aging_bonus,
            "fairness_skips": fairness_counter,
            "continuation_bonus": continuation_bonus,
        }

        return total_score, factors
