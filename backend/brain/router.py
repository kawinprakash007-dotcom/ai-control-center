from tools.executor import Executor
from tools.capability_registry import CapabilityRegistry
from core.models.result import Result


class Router:

    def __init__(self):

        self.registry = CapabilityRegistry()
        self.executor = Executor()

    def route(self, task):

        tool_function = self.registry.get_executor(task.tool)

        if tool_function is None:

            return Result(
                success=False,
                message=f"No capability found for '{task.tool}'"
            )

        return self.executor.execute(tool_function)