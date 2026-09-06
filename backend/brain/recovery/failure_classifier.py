from typing import List, Optional
import re

from core.models.plan import Plan
from core.models.result import Result
from core.models.verification import VerificationResult
from core.models.recovery import FailureClassification

TRANSIENT_PATTERNS = [
    r"timeout",
    r"timed out",
    r"connection reset",
    r"connection refused",
    r"network error",
    r"temporary",
    r"\b503\b",
    r"\b429\b",
    r"rate limit",
    r"transient",
]

MALFORMED_PATTERNS = [
    r"malformed",
    r"validation failed",
    r"invalid parameter",
    r"missing required",
    r"parameter schema",
]

UNSUPPORTED_PATTERNS = [
    r"no capability found",
    r"unknown capability",
    r"unauthorized capability",
    r"not permitted for capability",
    r"unsupported action",
]

INSUFFICIENT_PATTERNS = [
    r"insufficient",
    r"no evidence found",
    r"empty evidence",
    r"missing evidence",
    r"partial results",
    r"incomplete evidence",
]


class FailureClassifier:
    """
    Deterministic failure classifier analyzing Plan, Results, and VerificationResult.
    """

    @staticmethod
    def classify(
        plan: Plan,
        results: List[Result],
        verification: VerificationResult,
    ) -> FailureClassification:
        # 1. Inspect results first
        failed_result = next((r for r in results if not getattr(r, "success", True)), None)

        if failed_result is not None:
            msg = (failed_result.message or "").lower()
            out = (failed_result.output or "").lower()
            combined_text = f"{msg} {out}"
            data = getattr(failed_result, "data", {}) or {}

            # Check structured data flags
            if isinstance(data, dict):
                if data.get("requires_permission") or "permission required" in combined_text:
                    return FailureClassification.PERMISSION_REQUIRED
                if data.get("requires_confirmation") or "confirmation required" in combined_text:
                    return FailureClassification.CONFIRMATION_REQUIRED
                policy_res = data.get("policy_result")
                if isinstance(policy_res, dict) and policy_res.get("decision") == "deny":
                    return FailureClassification.POLICY_DENIED
                if data.get("policy_error") or "denied by policy" in combined_text:
                    return FailureClassification.POLICY_DENIED

            # Check transient network / rate-limit failures
            if any(re.search(pat, combined_text) for pat in TRANSIENT_PATTERNS):
                return FailureClassification.TRANSIENT

            # Check malformed parameters
            if any(re.search(pat, combined_text) for pat in MALFORMED_PATTERNS):
                return FailureClassification.MALFORMED

            # Check unsupported capabilities / actions
            if any(re.search(pat, combined_text) for pat in UNSUPPORTED_PATTERNS):
                return FailureClassification.UNSUPPORTED

            # Check insufficient content
            if any(re.search(pat, combined_text) for pat in INSUFFICIENT_PATTERNS):
                return FailureClassification.INSUFFICIENT_RESULT

            return FailureClassification.EXECUTION_ERROR

        # 2. Inspect verification reason if all results reported success or no failed result found
        ver_reason = (getattr(verification, "reason", "") or "").lower()
        if "permission required" in ver_reason:
            return FailureClassification.PERMISSION_REQUIRED
        if "confirmation required" in ver_reason:
            return FailureClassification.CONFIRMATION_REQUIRED
        if "denied by policy" in ver_reason:
            return FailureClassification.POLICY_DENIED
        if any(re.search(pat, ver_reason) for pat in INSUFFICIENT_PATTERNS):
            return FailureClassification.INSUFFICIENT_RESULT
        if any(re.search(pat, ver_reason) for pat in TRANSIENT_PATTERNS):
            return FailureClassification.TRANSIENT
        if any(re.search(pat, ver_reason) for pat in UNSUPPORTED_PATTERNS):
            return FailureClassification.UNSUPPORTED

        return FailureClassification.EXECUTION_ERROR
