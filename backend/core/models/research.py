from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Union, Tuple


class AgentActionType(str, Enum):
    """
    Allowed actions in Phase 3.0 Autonomous Research Agent runtime.
    Strictly bounded; no arbitrary tool execution or system access allowed.
    """
    SEARCH = "search"
    FETCH = "fetch"
    FINISH = "finish"


@dataclass(frozen=True)
class AgentAction:
    """
    Structured action proposed by a reasoning model or provider.
    Subject to strict runtime validation and bounds before execution.

    Attributes:
        action_type: Action to perform ('search', 'fetch', 'finish').
        parameters: Action-specific arguments (e.g. query, url).
        reason: Justification or reasoning behind selecting this action.
        confidence: Optional confidence score [0.0 - 1.0].
    """
    action_type: Union[AgentActionType, str]
    parameters: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        atype = self.action_type.value if isinstance(self.action_type, AgentActionType) else str(self.action_type)
        return {
            "action_type": atype,
            "parameters": dict(self.parameters),
            "reason": self.reason,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentAction":
        raw_type = data.get("action_type", "")
        try:
            action_type = AgentActionType(str(raw_type).strip().lower())
        except (ValueError, KeyError):
            action_type = raw_type
        return cls(
            action_type=action_type,
            parameters=dict(data.get("parameters", {})),
            reason=str(data.get("reason", "")),
            confidence=float(data.get("confidence", 1.0)),
        )


@dataclass(frozen=True)
class ResearchObservation:
    """
    Bounded observation returned to the reasoning provider after an action executes.
    Strictly bounded to prevent prompt explosion and context poisoning.

    Attributes:
        action_type: The action that produced this observation.
        success: Whether the action succeeded at runtime.
        summary: Short human-readable summary of the outcome.
        data: Bounded structured data (e.g. snippets, titles, status codes).
        error: Optional error description if the action failed.
    """
    action_type: str
    success: bool
    summary: str
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_type": self.action_type,
            "success": self.success,
            "summary": self.summary,
            "data": dict(self.data),
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResearchObservation":
        return cls(
            action_type=str(data.get("action_type", "")),
            success=bool(data.get("success", False)),
            summary=str(data.get("summary", "")),
            data=dict(data.get("data", {})),
            error=data.get("error"),
        )


@dataclass(frozen=True)
class ResearchLimits:
    """
    Runtime-owned resource and execution limits for the autonomous research loop.
    Guarantees strict termination and prevents infinite loops or runaway execution.

    The model can NEVER alter these limits.
    """
    max_iterations: int = 10
    max_searches: int = 10
    max_fetches: int = 5
    max_evidence: int = 50
    min_evidence: int = 1
    max_invalid_actions: int = 3
    timeout_seconds: float = 10.0

    def enforce_caps(self) -> "ResearchLimits":
        """Defensively cap all limits to hard runtime maxima."""
        return ResearchLimits(
            max_iterations=max(1, min(self.max_iterations, 10)),
            max_searches=max(1, min(self.max_searches, 10)),
            max_fetches=max(0, min(self.max_fetches, 5)),
            max_evidence=max(1, min(self.max_evidence, 50)),
            min_evidence=max(1, self.min_evidence),
            max_invalid_actions=max(1, min(self.max_invalid_actions, 5)),
            timeout_seconds=max(0.1, min(self.timeout_seconds, 60.0)),
        )
