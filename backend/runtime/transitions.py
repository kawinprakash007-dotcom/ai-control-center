from typing import Dict, Set

from core.models.runtime import CognitiveStage


class InvalidTransitionError(RuntimeError):
    """Raised when an illegal lifecycle stage transition is attempted."""
    def __init__(self, current_stage: CognitiveStage, target_stage: CognitiveStage, reason: str = ""):
        msg = f"Invalid stage transition from '{current_stage.value}' to '{target_stage.value}'."
        if reason:
            msg += f" Reason: {reason}"
        super().__init__(msg)
        self.current_stage = current_stage
        self.target_stage = target_stage
        self.reason = reason


# Deterministic transition graph defining all allowable lifecycle state advancements
VALID_TRANSITIONS: Dict[CognitiveStage, Set[CognitiveStage]] = {
    CognitiveStage.RECEIVED: {
        CognitiveStage.UNDERSTANDING,
        CognitiveStage.FAILED,
        CognitiveStage.ABORTED,
    },
    CognitiveStage.UNDERSTANDING: {
        CognitiveStage.DECISION,
        CognitiveStage.RESPONSE,  # E.g. empty or non-executable prompt
        CognitiveStage.FAILED,
        CognitiveStage.ABORTED,
    },
    CognitiveStage.DECISION: {
        CognitiveStage.PLANNING,
        CognitiveStage.RESPONSE,  # E.g. clarification needed or direct response
        CognitiveStage.FAILED,
        CognitiveStage.ABORTED,
    },
    CognitiveStage.PLANNING: {
        CognitiveStage.CONTEXT,
        CognitiveStage.ROUTING,
        CognitiveStage.REASONING,
        CognitiveStage.POLICY,     # Direct plan policy evaluation
        CognitiveStage.EXECUTION,  # Deterministic plan without reasoning
        CognitiveStage.RESPONSE,   # Direct answer or planning rejection
        CognitiveStage.FAILED,
        CognitiveStage.ABORTED,
    },
    CognitiveStage.CONTEXT: {
        CognitiveStage.ROUTING,
        CognitiveStage.REASONING,
        CognitiveStage.POLICY,
        CognitiveStage.FAILED,
        CognitiveStage.ABORTED,
    },
    CognitiveStage.ROUTING: {
        CognitiveStage.REASONING,
        CognitiveStage.POLICY,
        CognitiveStage.FAILED,
        CognitiveStage.ABORTED,
    },
    CognitiveStage.REASONING: {
        CognitiveStage.PROPOSAL,
        CognitiveStage.RESPONSE,  # Direct text response without proposal
        CognitiveStage.FAILED,
        CognitiveStage.ABORTED,
    },
    CognitiveStage.PROPOSAL: {
        CognitiveStage.VALIDATION,
        CognitiveStage.FAILED,
        CognitiveStage.ABORTED,
    },
    CognitiveStage.VALIDATION: {
        CognitiveStage.POLICY,
        CognitiveStage.REASONING,  # Validation failed, re-reason
        CognitiveStage.FAILED,
        CognitiveStage.ABORTED,
    },
    CognitiveStage.POLICY: {
        CognitiveStage.EXECUTION,
        CognitiveStage.RESPONSE,  # Policy denied or requires user confirmation (WAITING_FOR_USER)
        CognitiveStage.FAILED,
        CognitiveStage.ABORTED,
    },
    CognitiveStage.EXECUTION: {
        CognitiveStage.OBSERVATION,
        CognitiveStage.VERIFICATION,  # Direct verification of execution result
        CognitiveStage.FAILED,
        CognitiveStage.ABORTED,
    },
    CognitiveStage.OBSERVATION: {
        CognitiveStage.VERIFICATION,
        CognitiveStage.REASONING,  # Multi-step cycle
        CognitiveStage.FAILED,
        CognitiveStage.ABORTED,
    },
    CognitiveStage.VERIFICATION: {
        CognitiveStage.MEMORY,
        CognitiveStage.RECOVERY,   # Verification failure triggers RecoveryEngine
        CognitiveStage.RESPONSE,
        CognitiveStage.FAILED,
        CognitiveStage.ABORTED,
    },
    CognitiveStage.RECOVERY: {
        CognitiveStage.EXECUTION,  # Retry action
        CognitiveStage.PLANNING,   # Re-plan
        CognitiveStage.CONTEXT,    # Update context for replan
        CognitiveStage.ROUTING,
        CognitiveStage.REASONING,  # Re-reason
        CognitiveStage.MEMORY,     # Succeeded recovery advances to memory
        CognitiveStage.RESPONSE,   # Ask user or aborted
        CognitiveStage.FAILED,
        CognitiveStage.ABORTED,
    },
    CognitiveStage.MEMORY: {
        CognitiveStage.RESPONSE,
        CognitiveStage.COMPLETED,
        CognitiveStage.FAILED,
        CognitiveStage.ABORTED,
    },
    CognitiveStage.RESPONSE: {
        CognitiveStage.MEMORY,     # Optional memory write after response
        CognitiveStage.COMPLETED,
        CognitiveStage.FAILED,
        CognitiveStage.ABORTED,
    },
    # Terminal stages have no outbound transitions
    CognitiveStage.COMPLETED: set(),
    CognitiveStage.FAILED: set(),
    CognitiveStage.ABORTED: set(),
}


def is_valid_transition(current_stage: CognitiveStage, target_stage: CognitiveStage) -> bool:
    """Check if transitioning from current_stage to target_stage is allowed."""
    valid_targets = VALID_TRANSITIONS.get(current_stage, set())
    return target_stage in valid_targets


def validate_stage_transition(current_stage: CognitiveStage, target_stage: CognitiveStage) -> None:
    """
    Validate that transitioning from current_stage to target_stage is allowed.
    Raises InvalidTransitionError if the transition violates the lifecycle transition graph.
    """
    if not is_valid_transition(current_stage, target_stage):
        raise InvalidTransitionError(
            current_stage=current_stage,
            target_stage=target_stage,
            reason=f"Allowed transitions from '{current_stage.value}': {[s.value for s in VALID_TRANSITIONS.get(current_stage, set())]}"
        )
