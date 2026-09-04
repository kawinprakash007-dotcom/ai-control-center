from dataclasses import dataclass
from typing import Optional


ALLOWED_STATUSES = {"verified", "failed", "unverified"}


@dataclass(frozen=True)
class VerificationResult:
    """
    Immutable representation of the verification outcome for an executed Plan.

    Attributes:
        verified: Boolean flag indicating if the plan execution was verified.
        status: Canonical verification status ('verified', 'failed', 'unverified').
        confidence: Confidence score of the verification outcome [0.0 - 1.0].
        reason: Explanation of the verification evaluation.
        failed_task_id: Optional ID of the task that failed or was skipped.
    """

    verified: bool
    status: str
    confidence: float
    reason: str
    failed_task_id: Optional[int] = None

    def __post_init__(self):
        if not isinstance(self.status, str) or self.status not in ALLOWED_STATUSES:
            raise ValueError(
                f"Invalid status '{self.status}'. Must be one of {sorted(list(ALLOWED_STATUSES))}"
            )

        if not isinstance(self.confidence, (int, float)) or not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"Confidence {self.confidence} must be a float between 0.0 and 1.0"
            )

        if self.status == "verified" and not self.verified:
            raise ValueError("When status is 'verified', verified must be True")

        if self.status in {"failed", "unverified"} and self.verified:
            raise ValueError(f"When status is '{self.status}', verified must be False")
