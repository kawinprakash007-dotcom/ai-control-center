from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any


class CapabilityType(Enum):
    """
    Supported capability categories in AI Control Center.
    """
    CHAT = "chat"
    KNOWLEDGE = "knowledge"
    WEB = "web"
    MEMORY = "memory"
    TOOL = "tool"
    VISION = "vision"
    DEVICE = "device"


class ExecutionMode(Enum):
    """
    Execution strategy modes for fulfilling a decision.
    """
    DIRECT = "direct"
    SINGLE_STEP = "single_step"
    MULTI_STEP = "multi_step"


@dataclass(frozen=True)
class CapabilityRequirement:
    """
    Detailed requirement specification for an individual capability.

    Attributes:
        capability: Target capability type.
        priority: Priority level (1 is standard/highest).
        parameters: Specific arguments or configuration for the capability.
        mandatory: Whether failure of this capability fails the overall goal.
    """
    capability: CapabilityType
    priority: int = 1
    parameters: Dict[str, Any] = field(default_factory=dict)
    mandatory: bool = True


@dataclass(frozen=True)
class Decision:
    """
    Structured outcome of the Decision Engine evaluation.

    Attributes:
        request_id: Correlated Request ID.
        primary_goal: High-level intended objective.
        required_capabilities: List of required capabilities to fulfill the goal.
        execution_mode: Chosen execution strategy (direct, single_step, multi_step).
        confidence: Confidence score of the decision [0.0 - 1.0].
        reasoning: Semantic justification for the chosen strategy.
        routing_hints: Optional domain or capability-specific routing guidance.
    """
    request_id: str
    primary_goal: str
    required_capabilities: List[CapabilityType]
    execution_mode: ExecutionMode
    confidence: float
    reasoning: str
    routing_hints: Dict[str, Any] = field(default_factory=dict)
