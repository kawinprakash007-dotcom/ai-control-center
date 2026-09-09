import inspect
from typing import Optional, Any

from core.models.result import Result


class Executor:
    """
    Executes tool and capability functions safely.
    Supports task-aware capabilities while maintaining backward compatibility
    with legacy parameterless tools.
    """

    def execute(
        self,
        tool_function,
        task: Optional[Any] = None
    ) -> Result:
        try:
            if task is not None:
                accepts_args = False
                try:
                    sig = inspect.signature(tool_function)
                    for param in sig.parameters.values():
                        if param.kind in (
                            inspect.Parameter.POSITIONAL_ONLY,
                            inspect.Parameter.POSITIONAL_OR_KEYWORD,
                            inspect.Parameter.VAR_POSITIONAL,
                        ):
                            accepts_args = True
                            break
                except (ValueError, TypeError):
                    accepts_args = False

                if accepts_args:
                    output = tool_function(task)
                else:
                    output = tool_function()
            else:
                output = tool_function()

            if isinstance(output, Result):
                return output

            data = None
            if hasattr(tool_function, "last_research_result") and getattr(tool_function, "last_research_result", None) is not None:
                data = getattr(tool_function, "last_research_result", None)
            elif hasattr(tool_function, "last_evidence"):
                data = getattr(tool_function, "last_evidence", None)
            elif task is not None and hasattr(task, "parameters") and isinstance(task.parameters, dict):
                data = task.parameters.get("research_result") or task.parameters.get("evidence")

            return Result(
                success=True,
                message="Task completed.",
                output=str(output),
                data=data,
            )

        except Exception as e:
            return Result(
                success=False,
                message=str(e)
            )