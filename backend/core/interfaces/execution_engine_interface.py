from abc import ABC, abstractmethod
from typing import List

from core.models.plan import Plan
from core.models.result import Result


class ExecutionEngineInterface(ABC):
    """
    Abstract interface for plan-level execution orchestration.
    """

    @abstractmethod
    def execute(self, plan: Plan) -> List[Result]:
        """
        Execute all tasks in the given Plan sequentially.

        Args:
            plan: The Plan containing ordered Task steps.

        Returns:
            Ordered list of Result objects for executed tasks.
        """
        pass
