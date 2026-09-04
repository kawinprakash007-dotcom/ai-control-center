from abc import ABC, abstractmethod

from core.models.request import Request
from core.models.decision import Decision


class DecisionEngineInterface(ABC):
    """
    Abstract interface for evaluating requests and determining execution decisions.
    """

    @abstractmethod
    def decide(self, request: Request) -> Decision:
        """
        Evaluate a Request and determine goals, required capabilities, and execution mode.

        Args:
            request: The validated Request to analyze.

        Returns:
            Structured Decision specifying required capabilities and execution strategy.
        """
        pass
