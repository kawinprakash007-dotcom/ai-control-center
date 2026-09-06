import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from core.models.policy import PolicyResult


class TimeHorizon(str, Enum):
    """
    Bounded time horizons representing the future temporal scope of an anticipation.
    Always backed by bounded duration windows in seconds, never vague text.
    """
    IMMEDIATE = "IMMEDIATE"        # 0 to 5 minutes (0 - 300s)
    NEAR_TERM = "NEAR_TERM"        # 5 to 30 minutes (300 - 1800s)
    MEDIUM_TERM = "MEDIUM_TERM"    # 30 minutes to 2 hours (1800 - 7200s)
    LONGER_TERM = "LONGER_TERM"    # 2 hours to 24 hours (7200 - 86400s)

    def get_default_window_seconds(self) -> Tuple[float, float]:
        """Return the bounded (min_seconds, max_seconds) window for this horizon."""
        if self == TimeHorizon.IMMEDIATE:
            return (0.0, 300.0)
        elif self == TimeHorizon.NEAR_TERM:
            return (300.0, 1800.0)
        elif self == TimeHorizon.MEDIUM_TERM:
            return (1800.0, 7200.0)
        elif self == TimeHorizon.LONGER_TERM:
            return (7200.0, 86400.0)
        return (0.0, 300.0)

    @classmethod
    def from_seconds(cls, seconds: float) -> "TimeHorizon":
        """Deterministically map a duration in seconds to a bounded TimeHorizon."""
        if seconds <= 300.0:
            return cls.IMMEDIATE
        elif seconds <= 1800.0:
            return cls.NEAR_TERM
        elif seconds <= 7200.0:
            return cls.MEDIUM_TERM
        else:
            return cls.LONGER_TERM


class AnticipationStatus(str, Enum):
    """
    Lifecycle states of an anticipation.
    Anticipations are dynamic hypotheses that can be confirmed, invalidated, or expired.
    """
    ACTIVE = "ACTIVE"
    CONFIRMED = "CONFIRMED"
    INVALIDATED = "INVALIDATED"
    EXPIRED = "EXPIRED"
    DISMISSED = "DISMISSED"


class FutureConditionType(str, Enum):
    """
    Categorization of candidate future conditions.
    Unknown condition types must fail safely without autonomous execution.
    """
    RESOURCE_DEPLETION_RISK = "RESOURCE_DEPLETION_RISK"
    GOAL_COMPLETION_RISK = "GOAL_COMPLETION_RISK"
    DEADLINE_RISK = "DEADLINE_RISK"
    DEPENDENCY_RISK = "DEPENDENCY_RISK"
    ENVIRONMENT_CHANGE_RISK = "ENVIRONMENT_CHANGE_RISK"
    CAPABILITY_RISK = "CAPABILITY_RISK"
    SAFETY_RISK = "SAFETY_RISK"
    UNKNOWN = "UNKNOWN"

    @classmethod
    def from_str(cls, val: str) -> "FutureConditionType":
        s = str(val).strip().upper()
        for member in cls:
            if member.value == s or member.name == s:
                return member
        return cls.UNKNOWN


class EvidenceSourceType(str, Enum):
    """
    Identifies the architectural source of supporting evidence for an anticipation.
    Unsupported model output is never treated as evidence.
    """
    WORLD_STATE = "WORLD_STATE"
    EVENT = "EVENT"
    ACTIVE_GOAL = "ACTIVE_GOAL"
    GOAL_PROGRESS = "GOAL_PROGRESS"
    HISTORICAL_OBSERVATION = "HISTORICAL_OBSERVATION"
    KNOWLEDGE = "KNOWLEDGE"


class AnticipatoryDecisionType(str, Enum):
    """
    Action routing decision resulting from an anticipatory planning cycle.
    Anticipations NEVER directly execute tools, models, or computer actions.
    """
    NO_ACTION = "NO_ACTION"
    MONITOR = "MONITOR"
    PREPARE = "PREPARE"
    CREATE_GOAL = "CREATE_GOAL"
    UPDATE_GOAL = "UPDATE_GOAL"
    ESCALATE_USER = "ESCALATE_USER"


@dataclass(frozen=True)
class EvidenceItem:
    """
    An immutable unit of explainable evidence supporting an anticipation hypothesis.
    Retains full provenance for deterministic replay and auditability.
    Freshness is evaluated separately from confidence.
    """
    evidence_id: str
    source_type: EvidenceSourceType
    source_id: str
    description: str
    confidence: float
    observed_at: float
    expires_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.evidence_id or not isinstance(self.evidence_id, str):
            raise ValueError("EvidenceItem evidence_id must be a non-empty string.")
        if not isinstance(self.source_type, EvidenceSourceType):
            raise ValueError(f"EvidenceItem source_type must be an EvidenceSourceType enum, got {type(self.source_type)}")
        if not self.source_id or not isinstance(self.source_id, str):
            raise ValueError("EvidenceItem source_id must be a non-empty string.")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"EvidenceItem confidence must be in [0.0, 1.0], got {self.confidence}")
        if self.observed_at <= 0.0:
            raise ValueError(f"EvidenceItem observed_at must be positive, got {self.observed_at}")
        if self.expires_at is not None and self.expires_at < self.observed_at:
            raise ValueError(f"EvidenceItem expires_at ({self.expires_at}) cannot precede observed_at ({self.observed_at})")

    def is_fresh(self, now: float) -> bool:
        """Return False if expires_at is in the past."""
        if self.expires_at is not None and now >= self.expires_at:
            return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "source_type": self.source_type.value,
            "source_id": self.source_id,
            "description": self.description,
            "confidence": self.confidence,
            "observed_at": self.observed_at,
            "expires_at": self.expires_at,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class AnticipationProvenance:
    """
    Mandatory provenance record tracking the origin and cascade depth of an anticipation.
    Prevents infinite prediction feedback loops.
    """
    source_entity: str
    created_at: float
    correlation_id: str
    depth: int = 0
    causation_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.source_entity or not isinstance(self.source_entity, str):
            raise ValueError("AnticipationProvenance source_entity must be a non-empty string.")
        if self.created_at <= 0.0:
            raise ValueError(f"AnticipationProvenance created_at must be positive, got {self.created_at}")
        if not self.correlation_id or not isinstance(self.correlation_id, str):
            raise ValueError("AnticipationProvenance correlation_id must be a non-empty string.")
        if self.depth < 0:
            raise ValueError(f"AnticipationProvenance depth must be non-negative, got {self.depth}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_entity": self.source_entity,
            "created_at": self.created_at,
            "correlation_id": self.correlation_id,
            "depth": self.depth,
            "causation_id": self.causation_id,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class Anticipation:
    """
    Model-neutral, immutable anticipation domain model.
    Represents a bounded hypothesis about a future-relevant condition.
    NEVER treated as an observed fact in World State.
    """
    anticipation_id: str
    condition_type: FutureConditionType
    description: str
    hypothetical_state: Dict[str, Any]
    horizon: TimeHorizon
    horizon_window_seconds: Tuple[float, float]
    confidence: float
    relevance: float
    freshness: float
    evidence_items: Tuple[EvidenceItem, ...]
    provenance: AnticipationProvenance
    correlation_id: str
    status: AnticipationStatus = AnticipationStatus.ACTIVE
    target_entity_id: Optional[str] = None
    related_event_ids: Tuple[str, ...] = field(default_factory=tuple)
    related_goal_ids: Tuple[str, ...] = field(default_factory=tuple)
    signature: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.anticipation_id or not isinstance(self.anticipation_id, str):
            raise ValueError("Anticipation anticipation_id must be a non-empty string.")
        if not isinstance(self.condition_type, FutureConditionType):
            raise ValueError(f"Anticipation condition_type must be a FutureConditionType enum, got {type(self.condition_type)}")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"Anticipation confidence must be in [0.0, 1.0], got {self.confidence}")
        if not (0.0 <= self.relevance <= 1.0):
            raise ValueError(f"Anticipation relevance must be in [0.0, 1.0], got {self.relevance}")
        if not (0.0 <= self.freshness <= 1.0):
            raise ValueError(f"Anticipation freshness must be in [0.0, 1.0], got {self.freshness}")
        if not isinstance(self.horizon, TimeHorizon):
            raise ValueError(f"Anticipation horizon must be a TimeHorizon enum, got {type(self.horizon)}")
        if len(self.horizon_window_seconds) != 2 or self.horizon_window_seconds[0] < 0.0 or self.horizon_window_seconds[1] < self.horizon_window_seconds[0]:
            raise ValueError(f"Anticipation horizon_window_seconds must be a non-negative (min, max) range, got {self.horizon_window_seconds}")

        # Compute deterministic signature if not provided
        if self.signature is None:
            sig = self._compute_signature()
            object.__setattr__(self, "signature", sig)

    def _compute_signature(self) -> str:
        """Deterministic signature computed from condition type, target entity, and hypothetical state."""
        state_repr = json.dumps(self.hypothetical_state, sort_keys=True)
        raw = f"{self.condition_type.value}:{self.target_entity_id or ''}:{state_repr}:{self.horizon.value}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def get_signature(self) -> str:
        return self.signature or self._compute_signature()

    def is_actionable(self, min_confidence: float = 0.5, min_relevance: float = 0.4) -> bool:
        """
        Return True if active, not expired or invalidated, and meets minimum thresholds.
        """
        if self.status != AnticipationStatus.ACTIVE:
            return False
        if self.condition_type == FutureConditionType.UNKNOWN:
            return False
        return self.confidence >= min_confidence and self.relevance >= min_relevance and self.freshness > 0.0

    def is_stale(self, now: float) -> bool:
        """
        Return True if all supporting evidence is expired, or if created_at + max horizon is exceeded.
        """
        if self.status in (AnticipationStatus.EXPIRED, AnticipationStatus.INVALIDATED):
            return True
        max_horizon_deadline = self.provenance.created_at + self.horizon_window_seconds[1]
        if now > max_horizon_deadline:
            return True
        # If all evidence items with expires_at are expired
        expirable_evidence = [e for e in self.evidence_items if e.expires_at is not None]
        if expirable_evidence and all(now >= e.expires_at for e in expirable_evidence):
            return True
        return False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "anticipation_id": self.anticipation_id,
            "condition_type": self.condition_type.value,
            "description": self.description,
            "target_entity_id": self.target_entity_id,
            "hypothetical_state": dict(self.hypothetical_state),
            "horizon": self.horizon.value,
            "horizon_window_seconds": list(self.horizon_window_seconds),
            "confidence": self.confidence,
            "relevance": self.relevance,
            "freshness": self.freshness,
            "status": self.status.value,
            "evidence_items": [e.to_dict() for e in self.evidence_items],
            "related_event_ids": list(self.related_event_ids),
            "related_goal_ids": list(self.related_goal_ids),
            "provenance": self.provenance.to_dict(),
            "signature": self.get_signature(),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class AnticipatoryDecision:
    """
    Result of evaluating an anticipation through policy and goal management authority.
    Records whether an autonomous goal operation was dispatched, monitored, or rejected.
    """
    decision_id: str
    anticipation_id: str
    correlation_id: str
    decision_type: AnticipatoryDecisionType
    timestamp: float
    goal_id: Optional[str] = None
    goal_payload: Optional[Dict[str, Any]] = None
    policy_result: Optional[PolicyResult] = None
    reason: str = ""
    executed: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "anticipation_id": self.anticipation_id,
            "correlation_id": self.correlation_id,
            "decision_type": self.decision_type.value,
            "timestamp": self.timestamp,
            "goal_id": self.goal_id,
            "goal_payload": self.goal_payload,
            "policy_result": self.policy_result.to_dict() if self.policy_result else None,
            "reason": self.reason,
            "executed": self.executed,
            "metadata": dict(self.metadata),
        }
