from typing import List

from core.interfaces.verification_interface import VerificationInterface
from core.models.plan import Plan
from core.models.result import Result
from core.models.verification import VerificationResult


class StandardVerifier(VerificationInterface):
    """
    Deterministic implementation of VerificationInterface.
    Performs structural verification over Plan and execution Results without
    calling external tools, models, or services.
    """

    def verify(
        self,
        plan: Plan,
        results: List[Result]
    ) -> VerificationResult:
        if not isinstance(plan, Plan):
            raise TypeError(
                f"StandardVerifier.verify expects a Plan instance, got {type(plan).__name__}"
            )

        if not isinstance(results, list):
            raise TypeError(
                f"StandardVerifier.verify expects a list for results, got {type(results).__name__}"
            )

        # Rule 4: Empty plan
        if not plan.steps:
            return VerificationResult(
                verified=True,
                status="verified",
                confidence=1.0,
                reason="Plan completed with no tasks.",
                failed_task_id=None,
            )

        # Find first explicitly failed task, if any
        failed_task = next(
            (t for t in plan.steps if getattr(t, "status", None) == "failed"),
            None,
        )

        # Find first explicitly failed result, if any
        failed_result_idx = next(
            (i for i, r in enumerate(results) if not getattr(r, "success", True)),
            None,
        )

        # Rule 5 & 6: Execution failure via plan status, failed task, or failed result
        if plan.status == "failed" or failed_task is not None or failed_result_idx is not None:
            failed_task_id = None
            failure_reason = "Execution failed."

            if failed_task is not None:
                failed_task_id = getattr(failed_task, "id", None)
                msg = getattr(failed_task, "result", None) or getattr(failed_task, "action", "Task failed")
                failure_reason = f"Execution failed on task {failed_task_id}: {msg}"
            elif failed_result_idx is not None:
                if failed_result_idx < len(plan.steps):
                    failed_task_id = getattr(plan.steps[failed_result_idx], "id", None)
                r_msg = getattr(results[failed_result_idx], "message", "Result reported failure")
                failure_reason = f"Execution failed: {r_msg}"

            return VerificationResult(
                verified=False,
                status="failed",
                confidence=1.0,
                reason=failure_reason,
                failed_task_id=failed_task_id,
            )

        # Rule 7: Skipped tasks
        skipped_task = next(
            (t for t in plan.steps if getattr(t, "status", None) == "skipped"),
            None,
        )
        if skipped_task is not None:
            return VerificationResult(
                verified=False,
                status="failed",
                confidence=1.0,
                reason=f"Execution did not complete all planned tasks: task {skipped_task.id} was skipped.",
                failed_task_id=skipped_task.id,
            )

        # Rule 8: Successful completion
        all_tasks_completed = all(getattr(t, "status", "completed") == "completed" for t in plan.steps)
        all_results_successful = all(getattr(r, "success", False) is True for r in results)
        consistent_result_count = (len(results) == len(plan.steps))

        if plan.status == "completed" and all_tasks_completed and all_results_successful and consistent_result_count:
            return VerificationResult(
                verified=True,
                status="verified",
                confidence=1.0,
                reason="All planned tasks completed successfully.",
                failed_task_id=None,
            )

        # Rule 9: Incomplete / inconsistent state
        return VerificationResult(
            verified=False,
            status="unverified",
            confidence=0.5,
            reason="Execution state is incomplete or inconsistent.",
            failed_task_id=None,
        )
