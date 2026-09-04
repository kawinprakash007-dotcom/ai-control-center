from dataclasses import dataclass
from typing import List

from core.models.request import Request
from core.models.decision import Decision
from core.models.plan import Plan
from core.models.result import Result
from core.models.verification import VerificationResult


@dataclass(frozen=True)
class PipelineResult:
    """
    Immutable representation of the end-to-end outcome of a Phase 2 pipeline execution.

    Attributes:
        response: Final user-facing response string.
        request: The structured user Request.
        decision: The architectural Decision determined for the request.
        plan: The execution Plan with final task statuses and results.
        results: Ordered list of execution Results.
        verification: Structural VerificationResult evaluating execution outcome.
    """

    response: str
    request: Request
    decision: Decision
    plan: Plan
    results: List[Result]
    verification: VerificationResult
