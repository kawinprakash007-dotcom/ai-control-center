from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional
from core.models.tool_call import ToolCall


class PolicyDecision(str, Enum):
    """
    Deterministic authorization decision for tool execution.
    """
    ALLOW = "allow"
    DENY = "deny"
    ASK_PERMISSION = "ask_permission"
    REQUIRE_CONFIRMATION = "require_confirmation"


class AutonomyLevel(str, Enum):
    """
    Operational autonomy level governing runtime evaluation sensitivity.
    """
    MANUAL = "manual"
    ASSISTED = "assisted"
    AUTONOMOUS = "autonomous"


class RiskLevel(str, Enum):
    """
    Assessed risk tier of an operation.
    """
    SAFE = "safe"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass(frozen=True)
class PolicyContext:
    """
    Contextual metadata supplied to the Policy Engine for deterministic evaluation.

    Attributes:
        capability: Target capability name (e.g. 'web', 'memory', 'knowledge').
        action: Target action on the capability.
        parameters: Sanitized parameter dictionary.
        reason: Justification from the caller / model.
        call_id: Optional correlation identifier.
        source: Origin of the request ('brain', 'orchestrator', 'agent').
        session_id: Optional user/session identifier.
        autonomy_level: Current operating autonomy level.
        risk_level: Assessed risk level if predetermined.
        metadata: Optional auxiliary contextual metadata.
    """
    capability: str
    action: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    call_id: Optional[str] = None
    source: str = "orchestrator"
    session_id: Optional[str] = None
    autonomy_level: AutonomyLevel = AutonomyLevel.ASSISTED
    risk_level: Optional[RiskLevel] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "capability", str(self.capability or "").strip().lower())
        object.__setattr__(self, "action", str(self.action or "").strip().lower())

    @classmethod
    def from_tool_call(
        cls,
        tool_call: ToolCall,
        session_id: Optional[str] = None,
        autonomy_level: AutonomyLevel = AutonomyLevel.ASSISTED,
        source: str = "orchestrator",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "PolicyContext":
        """
        Build a PolicyContext from a ToolCall.
        """
        return cls(
            capability=tool_call.capability,
            action=tool_call.action,
            parameters=dict(tool_call.parameters),
            reason=tool_call.reason,
            call_id=tool_call.call_id,
            source=source,
            session_id=session_id,
            autonomy_level=autonomy_level,
            metadata=dict(metadata or {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "capability": self.capability,
            "action": self.action,
            "parameters": dict(self.parameters),
            "reason": self.reason,
            "call_id": self.call_id,
            "source": self.source,
            "session_id": self.session_id,
            "autonomy_level": self.autonomy_level.value,
            "risk_level": self.risk_level.value if self.risk_level else None,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class PolicyResult:
    """
    Structured, deterministic outcome of policy evaluation.

    Attributes:
        decision: The PolicyDecision (ALLOW, DENY, ASK_PERMISSION, REQUIRE_CONFIRMATION).
        reason: Technical or semantic justification for the decision.
        rule_id: Unique identifier of the policy rule that produced this result.
        explanation: Optional human-readable explanation suitable for user display.
        metadata: Optional structured metadata (e.g. risk level, required scopes).
    """
    decision: PolicyDecision
    reason: str
    rule_id: str
    explanation: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_allowed(self) -> bool:
        return self.decision == PolicyDecision.ALLOW

    @property
    def is_denied(self) -> bool:
        return self.decision == PolicyDecision.DENY

    @property
    def requires_permission(self) -> bool:
        return self.decision in (PolicyDecision.ASK_PERMISSION, PolicyDecision.REQUIRE_CONFIRMATION)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision.value,
            "reason": self.reason,
            "rule_id": self.rule_id,
            "explanation": self.explanation,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def allow(
        cls,
        rule_id: str,
        reason: str = "Operation permitted by policy.",
        explanation: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "PolicyResult":
        return cls(
            decision=PolicyDecision.ALLOW,
            reason=reason,
            rule_id=rule_id,
            explanation=explanation,
            metadata=dict(metadata or {}),
        )

    @classmethod
    def deny(
        cls,
        rule_id: str,
        reason: str = "Operation denied by policy.",
        explanation: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "PolicyResult":
        return cls(
            decision=PolicyDecision.DENY,
            reason=reason,
            rule_id=rule_id,
            explanation=explanation,
            metadata=dict(metadata or {}),
        )

    @classmethod
    def ask_permission(
        cls,
        rule_id: str,
        reason: str = "Operation requires user permission.",
        explanation: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "PolicyResult":
        return cls(
            decision=PolicyDecision.ASK_PERMISSION,
            reason=reason,
            rule_id=rule_id,
            explanation=explanation,
            metadata=dict(metadata or {}),
        )

    @classmethod
    def require_confirmation(
        cls,
        rule_id: str,
        reason: str = "Operation requires user confirmation.",
        explanation: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "PolicyResult":
        return cls(
            decision=PolicyDecision.REQUIRE_CONFIRMATION,
            reason=reason,
            rule_id=rule_id,
            explanation=explanation,
            metadata=dict(metadata or {}),
        )
