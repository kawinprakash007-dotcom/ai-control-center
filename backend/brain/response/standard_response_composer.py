from typing import List

from core.interfaces.response_composer_interface import ResponseComposerInterface
from core.models.request import Request
from core.models.decision import Decision
from core.models.plan import Plan
from core.models.result import Result
from core.models.verification import VerificationResult


class StandardResponseComposer(ResponseComposerInterface):
    """
    Deterministic implementation of ResponseComposerInterface.
    Composes concise, deterministic user-facing responses based on
    Request, Decision, Plan, Results, and VerificationResult.
    """

    def compose(
        self,
        request: Request,
        decision: Decision,
        plan: Plan,
        results: List[Result],
        verification: VerificationResult,
    ) -> str:
        # Type safety validation
        if not isinstance(request, Request):
            raise TypeError(
                f"StandardResponseComposer.compose expects a Request instance, got {type(request).__name__}"
            )
        if not isinstance(decision, Decision):
            raise TypeError(
                f"StandardResponseComposer.compose expects a Decision instance, got {type(decision).__name__}"
            )
        if not isinstance(plan, Plan):
            raise TypeError(
                f"StandardResponseComposer.compose expects a Plan instance, got {type(plan).__name__}"
            )
        if not isinstance(results, list):
            raise TypeError(
                f"StandardResponseComposer.compose expects a list for results, got {type(results).__name__}"
            )
        if not isinstance(verification, VerificationResult):
            raise TypeError(
                f"StandardResponseComposer.compose expects a VerificationResult instance, got {type(verification).__name__}"
            )

        # Priority A: Empty input / prompt_user_input
        if decision.primary_goal == "prompt_user_input":
            return "Please provide an instruction or question."

        # Priority B: Clarification request
        if decision.primary_goal == "clarify_request":
            ambiguity_reason = (
                decision.routing_hints.get("ambiguity_reason")
                or request.parameters.get("ambiguity_reason")
            )
            if ambiguity_reason:
                return f"Please clarify your request: {ambiguity_reason}."
            if verification.reason and verification.reason not in (
                "All planned tasks completed successfully.",
                "Plan completed with no tasks.",
            ):
                return f"Please clarify your request: {verification.reason}"
            return "Please clarify your request."

        # Priority C: Execution failure
        if not verification.verified and verification.status == "failed":
            failed_result = next(
                (r for r in results if not getattr(r, "success", True)), None
            )
            detail = None
            if failed_result is not None:
                detail = getattr(failed_result, "output", None) or getattr(
                    failed_result, "message", None
                )
            if not detail or not str(detail).strip():
                detail = verification.reason

            if verification.failed_task_id is not None:
                return f"Execution failed on task {verification.failed_task_id}: {detail}"
            return f"Execution failed: {detail}"

        # Priority D: Successful execution
        if verification.verified:
            useful_pieces: List[str] = []
            for r in results:
                out = getattr(r, "output", None)
                msg = getattr(r, "message", None)
                if out is not None and str(out).strip():
                    useful_pieces.append(str(out))
                elif msg is not None and str(msg).strip():
                    useful_pieces.append(str(msg))

            if useful_pieces:
                if len(useful_pieces) == 1:
                    return useful_pieces[0]
                return "\n".join(useful_pieces)

            return "Completed successfully."

        # Priority E: Unverified state
        if verification.status == "unverified":
            return f"Execution completed but could not be fully verified: {verification.reason}"

        return "Execution completed."
