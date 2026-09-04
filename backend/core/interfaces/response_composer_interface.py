from abc import ABC, abstractmethod
from typing import List

from core.models.request import Request
from core.models.decision import Decision
from core.models.plan import Plan
from core.models.result import Result
from core.models.verification import VerificationResult


class ResponseComposerInterface(ABC):
    """
    Abstract interface for deterministic response composition.
    """

    @abstractmethod
    def compose(
        self,
        request: Request,
        decision: Decision,
        plan: Plan,
        results: List[Result],
        verification: VerificationResult,
    ) -> str:
        """
        Compose the final user-facing response string deterministically.

        Args:
            request: The validated user Request.
            decision: The architectural Decision.
            plan: The planned/executed Plan.
            results: List of execution Results.
            verification: The verification outcome.

        Returns:
            Concise, deterministic user-facing response string.
        """
        pass
