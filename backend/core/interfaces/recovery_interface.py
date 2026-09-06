from abc import ABC, abstractmethod
from typing import Optional, Any, Callable, List, Tuple

from core.models.recovery import (
    RecoveryContext,
    RecoveryDecision,
    FailureClassification,
)
from core.models.plan import Plan
from core.models.result import Result
from core.models.verification import VerificationResult


class RecoveryPlannerInterface(ABC):
    """
    Model-neutral interface for failure classification and recovery planning.
    """

    @abstractmethod
    def classify_failure(
        self,
        plan: Plan,
        results: List[Result],
        verification: VerificationResult,
    ) -> FailureClassification:
        """
        Deterministically classify the root cause of an execution or verification failure.
        """
        pass

    @abstractmethod
    def decide_recovery(
        self,
        context: RecoveryContext,
    ) -> RecoveryDecision:
        """
        Evaluate recovery context and decide the next recovery action.
        """
        pass

    @abstractmethod
    def create_replan(
        self,
        context: RecoveryContext,
    ) -> Optional[Plan]:
        """
        Generate a revised Plan targeting the original goal.
        """
        pass


class RecoveryEngineInterface(ABC):
    """
    Interface for orchestrating recovery around plan execution and verification.
    """

    @abstractmethod
    def recover(
        self,
        original_goal: str,
        initial_plan: Plan,
        execute_fn: Callable[[Plan], List[Result]],
        verify_fn: Callable[[Plan, List[Result]], VerificationResult],
    ) -> Tuple[Plan, List[Result], VerificationResult, RecoveryContext]:
        """
        Execute bounded recovery loop around execution and verification.

        Args:
            original_goal: The user's original immutable goal.
            initial_plan: The original plan attempt.
            execute_fn: Callable executing a Plan and returning List[Result].
            verify_fn: Callable verifying Plan and List[Result].

        Returns:
            Tuple of (final_plan, final_results, final_verification, final_recovery_context).
        """
        pass
