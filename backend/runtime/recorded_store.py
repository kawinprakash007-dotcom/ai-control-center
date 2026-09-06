from typing import Any, Dict, List, Optional

from core.models.runtime import (
    CognitiveEvent,
    CognitiveEventType,
    CognitiveTrace,
)
from core.models.result import Result
from core.models.policy import PolicyDecision, PolicyResult
from core.models.verification import VerificationResult
from core.models.reasoning import ReasoningResponse, ReasoningOutcome, ActionProposal


class RecordedResultStore:
    """
    Bounded in-memory repository of sanitized execution, policy, reasoning,
    and verification results extracted from historical traces or explicitly recorded.
    Guarantees replay never invokes real tools when recorded results exist,
    and safely triggers structured divergence when results are missing.
    """

    def __init__(self, max_items: int = 200):
        self.max_items = max_items
        self._store: Dict[str, Any] = {}

    def _evict_if_needed(self) -> None:
        """Evict oldest recorded entries if capacity is exceeded."""
        if len(self._store) >= self.max_items:
            # Remove earliest recorded key
            oldest_key = next(iter(self._store))
            del self._store[oldest_key]

    def record(self, key: str, value: Any) -> None:
        """Store a result under an arbitrary key."""
        self._evict_if_needed()
        self._store[key] = value

    def get(self, key: str) -> Optional[Any]:
        """Retrieve a stored result or None if absent."""
        return self._store.get(key)

    def has(self, key: str) -> bool:
        """Check if a result is present."""
        return key in self._store

    def record_tool_result(self, turn_id: str, call_id: str, result: Any) -> None:
        """Store a tool execution result."""
        key = f"tool:{turn_id}:{call_id}"
        self.record(key, result)

    def get_tool_result(self, turn_id: str, call_id: str) -> Optional[Any]:
        """Retrieve a tool execution result by turn_id and call_id."""
        key = f"tool:{turn_id}:{call_id}"
        return self.get(key)

    def record_policy_result(self, turn_id: str, step_id: str, result: Any) -> None:
        """Store a policy evaluation result."""
        key = f"policy:{turn_id}:{step_id}"
        self.record(key, result)

    def get_policy_result(self, turn_id: str, step_id: str) -> Optional[Any]:
        """Retrieve a policy result."""
        key = f"policy:{turn_id}:{step_id}"
        return self.get(key)

    def record_reasoning_result(self, turn_id: str, result: Any) -> None:
        """Store a reasoning result."""
        key = f"reasoning:{turn_id}"
        self.record(key, result)

    def get_reasoning_result(self, turn_id: str) -> Optional[Any]:
        """Retrieve a reasoning result."""
        key = f"reasoning:{turn_id}"
        return self.get(key)

    def extract_from_trace(self, trace: CognitiveTrace) -> int:
        """
        Extract and record synthetic results from CognitiveTrace events.
        Enables instant OFFLINE_REPLAY from any exported trace.
        Returns the count of extracted records.
        """
        extracted = 0
        turn_id = trace.turn_id
        tool_call_counter = 0

        for event in trace.events:
            meta = event.metadata or {}

            # 1. TOOL_EXECUTED events
            if event.event_type == CognitiveEventType.TOOL_EXECUTED:
                tool_call_counter += 1
                call_id = meta.get("call_id", f"call_{tool_call_counter}")
                success = bool(meta.get("success", event.status == "OK"))
                message = event.summary or ("Tool executed" if success else "Tool execution failed")
                output_preview = meta.get("output_preview")

                res = Result(
                    success=success,
                    message=message,
                    output=output_preview,
                    call_id=call_id,
                )
                self.record_tool_result(turn_id, call_id, res)
                # Also record generic step index
                self.record_tool_result(turn_id, f"step_{tool_call_counter}", res)
                extracted += 1

            # 2. POLICY_DECIDED events
            elif event.event_type == CognitiveEventType.POLICY_DECIDED:
                decision_str = str(meta.get("decision", "allow")).lower()
                try:
                    dec = PolicyDecision(decision_str)
                except ValueError:
                    dec = PolicyDecision.ALLOW
                reason = meta.get("reason", event.summary)
                rule_id = meta.get("rule_id", "recorded_rule")

                p_res = PolicyResult(
                    decision=dec,
                    rule_id=rule_id,
                    reason=reason,
                )
                self.record_policy_result(turn_id, f"step_{extracted + 1}", p_res)
                self.record(f"policy:{turn_id}:latest", p_res)
                extracted += 1

            # 3. REASONING_COMPLETED events
            elif event.event_type == CognitiveEventType.REASONING_COMPLETED:
                outcome_str = meta.get("outcome", "PROPOSE_ACTION")
                try:
                    outcome = ReasoningOutcome(outcome_str)
                except ValueError:
                    outcome = ReasoningOutcome.PROPOSE_ACTION

                reason_res = ReasoningResponse(
                    turn_id=f"reason_{turn_id}",
                    outcome=outcome,
                    proposal=None,
                    confidence=float(meta.get("confidence", 1.0)),
                )
                self.record_reasoning_result(turn_id, reason_res)
                extracted += 1

            # 4. VERIFICATION_COMPLETED events
            elif event.event_type == CognitiveEventType.VERIFICATION_COMPLETED:
                verified = bool(meta.get("verified", event.status == "OK"))
                v_res = VerificationResult(
                    verified=verified,
                    status="verified" if verified else "failed",
                    confidence=1.0,
                    reason=event.summary or ("Verification succeeded" if verified else "Verification failed"),
                )
                self.record(f"verification:{turn_id}", v_res)
                extracted += 1

            # 5. RECOVERY_STARTED events
            elif event.event_type == CognitiveEventType.RECOVERY_STARTED:
                action_str = meta.get("action", "retry")
                self.record(f"recovery:{turn_id}", action_str)
                extracted += 1

            # 6. RESPONSE_COMPOSED events
            elif event.event_type == CognitiveEventType.RESPONSE_COMPOSED:
                resp_text = meta.get("response", event.summary)
                if resp_text:
                    self.record(f"response:{turn_id}", resp_text)
                extracted += 1

        return extracted

    def clear(self) -> None:
        """Clear all stored results."""
        self._store.clear()
