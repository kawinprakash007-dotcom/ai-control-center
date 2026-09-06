from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from core.models.plan import Plan
from core.models.result import Result
from core.models.verification import VerificationResult


class ExecutionOutcome(str, Enum):
    """
    Standard evaluation outcome of an execution and verification cycle.
    """
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILURE = "failure"
    BLOCKED = "blocked"
    POLICY_BLOCKED = "policy_blocked"


class RecoveryAction(str, Enum):
    """
    Action to take following an execution/verification outcome.
    """
    CONTINUE = "continue"
    RETRY = "retry"
    REPLAN = "replan"
    ASK_USER = "ask_user"
    ABORT = "abort"


class FailureClassification(str, Enum):
    """
    Deterministic classification of the failure root cause.
    """
    TRANSIENT = "transient"
    EXECUTION_ERROR = "execution_error"
    POLICY_DENIED = "policy_denied"
    PERMISSION_REQUIRED = "permission_required"
    CONFIRMATION_REQUIRED = "confirmation_required"
    INSUFFICIENT_RESULT = "insufficient_result"
    UNSUPPORTED = "unsupported"
    MALFORMED = "malformed"
    REPEATED_FAILURE = "repeated_failure"


@dataclass(frozen=True)
class RecoveryLimits:
    """
    Hard deterministic bounds and budgets to prevent infinite loops and resource exhaustion.
    """
    max_attempts: int = 3
    max_retries_per_action: int = 2
    max_replans: int = 2
    max_total_recovery_steps: int = 4
    max_plan_history: int = 10


@dataclass(frozen=True)
class PlanHistoryEntry:
    """
    Immutable snapshot of an executed plan attempt and its outcome.
    """
    plan_id: str
    plan: Plan
    outcome: ExecutionOutcome
    results: Tuple[Result, ...]
    verification: VerificationResult
    recovery_action: RecoveryAction
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "goal": getattr(self.plan, "goal", ""),
            "outcome": self.outcome.value,
            "verification_status": getattr(self.verification, "status", ""),
            "recovery_action": self.recovery_action.value,
            "reason": self.reason,
            "results_count": len(self.results),
        }


@dataclass(frozen=True)
class RecoveryDecision:
    """
    Decision produced by the recovery planner.
    """
    action: RecoveryAction
    classification: FailureClassification
    reason: str
    revised_plan: Optional[Plan] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RecoveryContext:
    """
    Bounded, immutable recovery state.
    Preserves the critical invariant: ORIGINAL GOAL != CURRENT PLAN.
    """
    original_goal: str
    current_plan: Plan
    attempt: int = 1
    outcome: ExecutionOutcome = ExecutionOutcome.FAILURE
    failure_classification: Optional[FailureClassification] = None
    failure_reason: Optional[str] = None
    verification: Optional[VerificationResult] = None
    results: Tuple[Result, ...] = field(default_factory=tuple)
    plan_history: Tuple[PlanHistoryEntry, ...] = field(default_factory=tuple)
    action_retry_counts: Dict[str, int] = field(default_factory=dict)
    replan_count: int = 0
    remaining_retry_budget: int = 2
    remaining_replan_budget: int = 2
    limits: RecoveryLimits = field(default_factory=RecoveryLimits)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_budget(self) -> bool:
        """Check if any recovery budget remains."""
        return (
            self.attempt < self.limits.max_attempts
            and self.replan_count < self.limits.max_replans
            and len(self.plan_history) < self.limits.max_total_recovery_steps
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_goal": self.original_goal,
            "attempt": self.attempt,
            "outcome": self.outcome.value if self.outcome else None,
            "failure_classification": (
                self.failure_classification.value
                if self.failure_classification
                else None
            ),
            "failure_reason": self.failure_reason,
            "replan_count": self.replan_count,
            "remaining_retry_budget": self.remaining_retry_budget,
            "remaining_replan_budget": self.remaining_replan_budget,
            "plan_history_count": len(self.plan_history),
            "metadata": self.metadata,
        }
