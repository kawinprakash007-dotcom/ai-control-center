from typing import Optional, Any
from tools.executor import Executor
from tools.capability_registry import CapabilityRegistry
from tools.tool_orchestrator import ToolOrchestrator
from core.models.result import Result
from core.models.tool_call import ToolCall


class Router:
    """
    Plan-level router and compatibility bridge.
    Delegates capability execution to ToolOrchestrator for validated execution,
    while maintaining full backward compatibility for legacy tasks and desktop tools.
    """

    def __init__(
        self,
        registry: Optional[CapabilityRegistry] = None,
        executor: Optional[Executor] = None,
        orchestrator: Optional[ToolOrchestrator] = None,
    ):
        self.registry = registry if registry is not None else (orchestrator.registry if orchestrator else CapabilityRegistry())
        self.executor = executor if executor is not None else (orchestrator.executor if orchestrator else Executor())
        self.orchestrator = orchestrator if orchestrator is not None else ToolOrchestrator(
            registry=self.registry,
            executor=self.executor,
        )

    def route(self, task: Any) -> Result:
        """
        Route a Task or ToolCall to the appropriate capability or tool executor.
        """
        # 1. Direct ToolCall execution
        if isinstance(task, ToolCall):
            return self.orchestrator.execute(task)

        tool_name = getattr(task, "tool", None)

        # 2. If task targets a whitelisted capability, route through orchestrator
        if tool_name and str(tool_name).strip().lower() in self.orchestrator.get_allowed_capabilities():
            from core.models.policy import PolicyContext
            context = PolicyContext.from_tool_call(
                ToolCall.from_task(task),
                source="pipeline",
            )
            return self.orchestrator.execute_task(task, context=context)

        # 3. Legacy fallback for desktop tools or unconstrained task routing
        tool_function = self.registry.get_executor(tool_name)

        if tool_function is None:
            return Result(
                success=False,
                message=f"No capability found for '{tool_name}'"
            )

        return self.executor.execute(tool_function, task)