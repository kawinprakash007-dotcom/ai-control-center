from abc import ABC, abstractmethod
from typing import List

from core.models.plan import Plan
from core.models.result import Result
from core.models.verification import VerificationResult


class VerificationInterface(ABC):
    """
    Abstract interface for deterministic post-execution plan verification.
    """

    @abstractmethod
    def verify(
        self,
        plan: Plan,
        results: List[Result]
    ) -> VerificationResult:
        """
        Verify whether the executed plan satisfies the planned operation.

        Args:
            plan: The Plan containing ordered Task steps and final statuses.
            results: List of Result objects returned by execution.

        Returns:
            VerificationResult summarizing the verification outcome.
        """
        pass
