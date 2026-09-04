from abc import ABC, abstractmethod

from core.models.decision import Decision
from core.models.plan import Plan


class DecisionPlannerInterface(ABC):
    """
    Abstract interface for translating a Decision into an actionable Plan.
    """

    @abstractmethod
    def plan(self, decision: Decision) -> Plan:
        """
        Translate a Decision into an executable sequence of Task steps within a Plan.

        Args:
            decision: Validated Decision model.

        Returns:
            Plan containing ordered Task objects.
        """
        pass
