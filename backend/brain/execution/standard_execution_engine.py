from typing import List, Optional, Any

from core.interfaces.execution_engine_interface import ExecutionEngineInterface
from core.models.plan import Plan
from core.models.result import Result
from brain.router import Router


class StandardExecutionEngine(ExecutionEngineInterface):
    """
    Plan-level execution engine implementing ExecutionEngineInterface.
    Orchestrates sequential Task execution by delegating each Task to Router.route(),
    manages task/plan statuses, enforces fail-fast error handling, and collects Results.
    """

    def __init__(
        self,
        router: Optional[Any] = None,
        policy_engine: Optional[Any] = None,
        demo_mode: Optional[bool] = None,
    ):
        """
        Initialize StandardExecutionEngine.

        Args:
            router: Optional Router instance or stub for dependency injection.
                    Defaults to standard Router() if omitted.
            policy_engine: Optional PolicyEngineInterface for execution authorization.
            demo_mode: Optional boolean flag for demo mode execution.
        """
        self.router = router if router is not None else Router(
            policy_engine=policy_engine,
            demo_mode=demo_mode,
        )

    def execute(self, plan: Plan) -> List[Result]:
        if not isinstance(plan, Plan):
            raise TypeError(
                f"StandardExecutionEngine.execute expects a Plan instance, got {type(plan).__name__}"
            )

        # Empty Plan handling
        if not plan.steps:
            plan.status = "completed"
            return []

        results: List[Result] = []

        for idx, task in enumerate(plan.steps):
            task.status = "running"

            try:
                result = self.router.route(task)
            except Exception as e:
                result = Result(
                    success=False,
                    message=f"Router execution exception: {e}",
                    output=None,
                )

            results.append(result)

            if result.success:
                task.status = "completed"
                task.result = result.output or result.message
            else:
                task.status = "failed"
                task.result = result.output or result.message

                # Fail-fast: mark remaining unexecuted tasks as 'skipped'
                for remaining_task in plan.steps[idx + 1 :]:
                    remaining_task.status = "skipped"

                plan.status = "failed"
                return results

        plan.status = "completed"
        return results
