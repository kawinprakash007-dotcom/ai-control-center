from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Set

from core.models.tool_call import ToolCall
from core.models.verification import VerificationResult
from core.models.recovery import RecoveryContext
from core.models.perception import VisualScene, GroundedTarget


class ReasoningOutcome(str, Enum):
    """
    Bounded set of architectural outcomes from a reasoning turn.
    Ensures model outputs conform to deterministic control paths.
    """
    PROPOSE_ACTION = "propose_action"
    REQUEST_OBSERVATION = "request_observation"
    ASK_CLARIFICATION = "ask_clarification"
    REPORT_COMPLETION = "report_completion"
    ABORT = "abort"


class ModelCapabilityType(str, Enum):
    """
    Minimal capability vocabulary enabling future Model Router selection.
    """
    TEXT = "text"
    VISION = "vision"
    STRUCTURED_OUTPUT = "structured_output"
    TOOL_REASONING = "tool_reasoning"


@dataclass(frozen=True)
class ProviderMetadata:
    """
    Immutable specification of a reasoning provider's profile and capabilities.
    Used by future Model Router to choose the optimal provider for a task.
    """
    provider_id: str
    version: str = "1.0.0"
    supported_capabilities: Tuple[ModelCapabilityType, ...] = (
        ModelCapabilityType.TEXT,
        ModelCapabilityType.STRUCTURED_OUTPUT,
    )
    is_local: bool = True
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def has_capability(self, cap: ModelCapabilityType) -> bool:
        return cap in self.supported_capabilities


@dataclass(frozen=True)
class ActionProposal:
    """
    Explicit, immutable proposal of an action emitted by a reasoning provider.
    CRITICAL ARCHITECTURAL BOUNDARY:
    An ActionProposal represents PROPOSED INTENT, NOT an authorized or executable action.
    It MUST be deterministically validated by ActionProposalValidator before any
    ToolCall can be instantiated.
    """
    proposal_id: str
    goal_reference: str
    action_type: str
    capability: str
    action: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    target_reference: Optional[str] = None
    rationale: str = ""
    confidence: float = 1.0
    observation_reference: Optional[str] = None
    requires_reobservation: bool = False
    constraints: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"ActionProposal confidence must be in range [0.0, 1.0], got {self.confidence}")
        object.__setattr__(self, "capability", str(self.capability or "").strip().lower())
        object.__setattr__(self, "action", str(self.action or "").strip().lower())
        object.__setattr__(self, "action_type", str(self.action_type or "").strip().lower())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "goal_reference": self.goal_reference,
            "action_type": self.action_type,
            "capability": self.capability,
            "action": self.action,
            "parameters": dict(self.parameters),
            "target_reference": self.target_reference,
            "rationale": self.rationale,
            "confidence": self.confidence,
            "observation_reference": self.observation_reference,
            "requires_reobservation": self.requires_reobservation,
            "constraints": dict(self.constraints),
        }


@dataclass(frozen=True)
class ReasoningRequest:
    """
    Structured, bounded input context passed to a reasoning provider.
    Prevents passing uncontrolled, raw application state or unbounded memory.
    Separates structured domain state from raw multimodal artifacts.
    """
    goal: str
    task_description: Optional[str] = None
    history: Tuple[Dict[str, str], ...] = field(default_factory=tuple)
    memory_context: Tuple[str, ...] = field(default_factory=tuple)
    visual_scene: Optional[VisualScene] = None
    grounded_candidates: Tuple[GroundedTarget, ...] = field(default_factory=tuple)
    screenshot_ref: Optional[str] = None  # Opaque identifier or file path reference; never raw base64 bloat
    execution_history: Tuple[Dict[str, Any], ...] = field(default_factory=tuple)
    verification_result: Optional[VerificationResult] = None
    recovery_context: Optional[RecoveryContext] = None
    available_capabilities: Tuple[str, ...] = field(default_factory=tuple)
    turn_index: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ReasoningResponse:
    """
    Structured domain output from a reasoning provider.
    Fails closed on any ambiguity; prevents free-form text execution.
    """
    turn_id: str
    outcome: ReasoningOutcome
    proposal: Optional[ActionProposal] = None
    clarification_prompt: Optional[str] = None
    completion_summary: Optional[str] = None
    observation_request_reason: Optional[str] = None
    abort_reason: Optional[str] = None
    confidence: float = 1.0
    raw_output: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"ReasoningResponse confidence must be in range [0.0, 1.0], got {self.confidence}")
        if self.outcome == ReasoningOutcome.PROPOSE_ACTION and self.proposal is None:
            raise ValueError("ReasoningResponse with outcome PROPOSE_ACTION must provide an ActionProposal")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "outcome": self.outcome.value,
            "proposal": self.proposal.to_dict() if self.proposal else None,
            "clarification_prompt": self.clarification_prompt,
            "completion_summary": self.completion_summary,
            "observation_request_reason": self.observation_request_reason,
            "abort_reason": self.abort_reason,
            "confidence": self.confidence,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ProposalValidationResult:
    """
    Deterministic result of validating an ActionProposal against ATLAS domain boundaries.
    """
    is_valid: bool
    tool_call: Optional[ToolCall] = None
    errors: Tuple[str, ...] = field(default_factory=tuple)
    stale_target: bool = False
    unknown_capability: bool = False
    unknown_action: bool = False
    policy_blocked: bool = False
    prohibited_action: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def primary_error(self) -> Optional[str]:
        return self.errors[0] if self.errors else None


@dataclass(frozen=True)
class ReasoningLimits:
    """
    Hard deterministic bounds and budgets to prevent infinite reasoning loops.
    """
    max_reasoning_turns: int = 10
    max_invalid_proposals: int = 3
    max_observation_requests: int = 3
    max_consecutive_no_progress: int = 2
    min_action_confidence: float = 0.2
