from typing import List, Tuple, Callable, Optional, Dict, Any

from core.interfaces.recovery_interface import (
    RecoveryEngineInterface,
    RecoveryPlannerInterface,
)
from core.models.plan import Plan
from core.models.result import Result
from core.models.verification import VerificationResult
from core.models.recovery import (
    RecoveryContext,
    RecoveryDecision,
    RecoveryAction,
    RecoveryLimits,
    PlanHistoryEntry,
    ExecutionOutcome,
    FailureClassification,
)
from brain.recovery.recovery_planner import StandardRecoveryPlanner


class StandardRecoveryEngine(RecoveryEngineInterface):
    """
    Standard implementation of RecoveryEngineInterface.
    Orchestrates deterministic, bounded recovery around Plan execution and verification.
    """

    def __init__(
        self,
        planner: Optional[RecoveryPlannerInterface] = None,
        limits: Optional[RecoveryLimits] = None,
    ):
        self.planner = planner or StandardRecoveryPlanner()
        self.limits = limits or RecoveryLimits()

    def recover(
        self,
        original_goal: str,
        initial_plan: Plan,
        execute_fn: Callable[[Plan], List[Result]],
        verify_fn: Callable[[Plan, List[Result]], VerificationResult],
    ) -> Tuple[Plan, List[Result], VerificationResult, RecoveryContext]:
        """
        Bounded recovery loop.
        Ensures:
        1. original_goal is preserved and never mutated.
        2. All execution passes through execute_fn (guaranteeing ToolOrchestrator & PolicyEngine enforcement).
        3. Attempts and replans are strictly bounded by limits.
        4. Anti-loop protection prevents repeating identical failed strategies.
        """
        current_plan = initial_plan
        attempt = 1
        replan_count = 0
        plan_history: List[PlanHistoryEntry] = []
        action_retry_counts: Dict[str, int] = {}
        cumulative_results: List[Result] = []

        while True:
            # 1. Execute current plan via execute_fn
            results = execute_fn(current_plan)

            # Preserve successful results across partial success attempts
            effective_results = list(cumulative_results) + list(results)

            # 2. Verify execution outcome
            verification = verify_fn(current_plan, effective_results)

            # 3. Assess ExecutionOutcome
            outcome = self._assess_outcome(current_plan, effective_results, verification)

            # 4. Success termination
            if outcome == ExecutionOutcome.SUCCESS:
                final_context = RecoveryContext(
                    original_goal=original_goal,
                    current_plan=current_plan,
                    attempt=attempt,
                    outcome=outcome,
                    failure_classification=None,
                    failure_reason=None,
                    verification=verification,
                    results=tuple(effective_results),
                    plan_history=tuple(plan_history),
                    action_retry_counts=dict(action_retry_counts),
                    replan_count=replan_count,
                    remaining_retry_budget=max(0, self.limits.max_retries_per_action),
                    remaining_replan_budget=max(0, self.limits.max_replans - replan_count),
                    limits=self.limits,
                )
                return current_plan, effective_results, verification, final_context

            # 5. Classify Failure
            classification = self.planner.classify_failure(current_plan, results, verification)

            # Check if policy blocked
            if classification == FailureClassification.POLICY_DENIED:
                outcome = ExecutionOutcome.POLICY_BLOCKED

            # 6. Build RecoveryContext
            context = RecoveryContext(
                original_goal=original_goal,
                current_plan=current_plan,
                attempt=attempt,
                outcome=outcome,
                failure_classification=classification,
                failure_reason=verification.reason,
                verification=verification,
                results=tuple(effective_results),
                plan_history=tuple(plan_history),
                action_retry_counts=dict(action_retry_counts),
                replan_count=replan_count,
                remaining_retry_budget=max(
                    0,
                    self.limits.max_retries_per_action
                    - max(action_retry_counts.values(), default=0),
                ),
                remaining_replan_budget=max(0, self.limits.max_replans - replan_count),
                limits=self.limits,
            )

            # 7. Decide Recovery Action
            decision = self.planner.decide_recovery(context)

            # 8. Record Snapshot in Plan History
            entry = PlanHistoryEntry(
                plan_id=f"plan-attempt-{attempt}",
                plan=current_plan,
                outcome=outcome,
                results=tuple(results),
                verification=verification,
                recovery_action=decision.action,
                reason=decision.reason,
            )
            plan_history.append(entry)

            # 9. Handle Non-Executing Actions (CONTINUE, ABORT, ASK_USER)
            if decision.action in (
                RecoveryAction.CONTINUE,
                RecoveryAction.ABORT,
                RecoveryAction.ASK_USER,
            ):
                updated_context = RecoveryContext(
                    original_goal=original_goal,
                    current_plan=current_plan,
                    attempt=attempt,
                    outcome=outcome,
                    failure_classification=classification,
                    failure_reason=decision.reason,
                    verification=verification,
                    results=tuple(effective_results),
                    plan_history=tuple(plan_history),
                    action_retry_counts=dict(action_retry_counts),
                    replan_count=replan_count,
                    remaining_retry_budget=max(
                        0,
                        self.limits.max_retries_per_action
                        - max(action_retry_counts.values(), default=0),
                    ),
                    remaining_replan_budget=max(0, self.limits.max_replans - replan_count),
                    limits=self.limits,
                    metadata=dict(decision.metadata),
                )
                return current_plan, effective_results, verification, updated_context

            # 10. Handle RETRY (same plan, reset pending steps)
            if decision.action == RecoveryAction.RETRY:
                retry_target = decision.metadata.get("retry_target", "action")
                action_retry_counts[retry_target] = (
                    action_retry_counts.get(retry_target, 0) + 1
                )
                attempt += 1

                # Reset task statuses to pending for retry
                for task in current_plan.steps:
                    task.status = "pending"
                    task.result = None
                current_plan.status = "pending"
                continue

            # 11. Handle REPLAN (different execution strategy for original goal)
            if decision.action == RecoveryAction.REPLAN:
                replan_count += 1
                attempt += 1

                revised_plan = decision.revised_plan
                if revised_plan is None:
                    revised_plan = self.planner.create_replan(context)

                if revised_plan is None:
                    # Cannot generate compliant non-repeating plan -> abort
                    abort_entry = PlanHistoryEntry(
                        plan_id=f"plan-attempt-{attempt}",
                        plan=current_plan,
                        outcome=outcome,
                        results=tuple(results),
                        verification=verification,
                        recovery_action=RecoveryAction.ABORT,
                        reason="No valid alternative plan available.",
                    )
                    plan_history.append(abort_entry)
                    final_context = RecoveryContext(
                        original_goal=original_goal,
                        current_plan=current_plan,
                        attempt=attempt,
                        outcome=outcome,
                        failure_classification=classification,
                        failure_reason="No alternative plan available without repeating failed strategies.",
                        verification=verification,
                        results=tuple(effective_results),
                        plan_history=tuple(plan_history),
                        action_retry_counts=dict(action_retry_counts),
                        replan_count=replan_count,
                        remaining_retry_budget=0,
                        remaining_replan_budget=0,
                        limits=self.limits,
                    )
                    return current_plan, effective_results, verification, final_context

                # Accumulate successful results from earlier tasks if partial success
                for r in results:
                    if getattr(r, "success", False):
                        cumulative_results.append(r)

                current_plan = revised_plan
                continue

    def _assess_outcome(
        self,
        plan: Plan,
        results: List[Result],
        verification: VerificationResult,
    ) -> ExecutionOutcome:
        if verification.verified and verification.status == "verified":
            return ExecutionOutcome.SUCCESS

        # Check for partial success (some tasks completed successfully, but not all)
        has_successful_result = any(getattr(r, "success", False) is True for r in results)
        has_failed_result = any(getattr(r, "success", True) is False for r in results)

        if has_successful_result and has_failed_result:
            return ExecutionOutcome.PARTIAL_SUCCESS

        return ExecutionOutcome.FAILURE
