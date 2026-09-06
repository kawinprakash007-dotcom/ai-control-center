import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any, Dict, List, Optional, Tuple, Union

from core.models.policy import PolicyResult


class EventSource(str, Enum):
    """
    Origins from which an autonomy event may arise.
    """
    WORLD_STATE = "WORLD_STATE"
    GOAL_LIFECYCLE = "GOAL_LIFECYCLE"
    SYSTEM_OBSERVATION = "SYSTEM_OBSERVATION"
    COGNITIVE_EVENT = "COGNITIVE_EVENT"
    EXTERNAL = "EXTERNAL"
    USER = "USER"


class EventPriority(IntEnum):
    """
    Standard priority hierarchy for autonomy events.
    """
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

    @classmethod
    def from_str(cls, val: str) -> "EventPriority":
        s = str(val).strip().upper()
        for p in cls:
            if p.name == s:
                return p
        return cls.NORMAL


class EventCategory(str, Enum):
    """
    Deterministic categorization of detected events.
    """
    THRESHOLD_BREACH = "THRESHOLD_BREACH"
    STATE_CHANGE = "STATE_CHANGE"
    FAILURE = "FAILURE"
    SAFETY_ALERT = "SAFETY_ALERT"
    GOAL_PROGRESS = "GOAL_PROGRESS"
    TIMEOUT = "TIMEOUT"
    EXTERNAL_SIGNAL = "EXTERNAL_SIGNAL"
    UNKNOWN = "UNKNOWN"


class AutonomyDecisionType(str, Enum):
    """
    Deterministic action routing decision for an event.
    Events NEVER directly execute tools or capabilities.
    """
    IGNORE = "IGNORE"
    RECORD_ONLY = "RECORD_ONLY"
    EVALUATE = "EVALUATE"
    CREATE_GOAL = "CREATE_GOAL"
    UPDATE_GOAL = "UPDATE_GOAL"
    RESUME_GOAL = "RESUME_GOAL"
    PAUSE_GOAL = "PAUSE_GOAL"
    CANCEL_GOAL = "CANCEL_GOAL"
    ESCALATE_TO_USER = "ESCALATE_TO_USER"


@dataclass(frozen=True)
class EventProvenance:
    """
    Mandatory provenance record for an autonomy event.
    Tracks origins, correlation identifiers, and cascade depth to prevent loops.
    """
    source_id: str
    source_type: EventSource
    origin_timestamp: float
    correlation_id: str
    depth: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.source_id or not isinstance(self.source_id, str):
            raise ValueError("EventProvenance source_id must be a non-empty string.")
        if not isinstance(self.source_type, EventSource):
            raise ValueError(f"EventProvenance source_type must be an EventSource enum, got {type(self.source_type)}")
        if self.origin_timestamp <= 0.0:
            raise ValueError(f"EventProvenance origin_timestamp must be positive, got {self.origin_timestamp}")
        if not self.correlation_id or not isinstance(self.correlation_id, str):
            raise ValueError("EventProvenance correlation_id must be a non-empty string.")
        if self.depth < 0:
            raise ValueError(f"EventProvenance depth must be non-negative, got {self.depth}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "source_type": self.source_type.value,
            "origin_timestamp": self.origin_timestamp,
            "correlation_id": self.correlation_id,
            "depth": self.depth,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EventProvenance":
        return cls(
            source_id=data["source_id"],
            source_type=EventSource(data["source_type"]),
            origin_timestamp=float(data["origin_timestamp"]),
            correlation_id=data["correlation_id"],
            depth=int(data.get("depth", 0)),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class Event:
    """
    Model-neutral autonomy event model.
    Immutable, provenance-preserving representation of a detected system or world change.
    """
    event_id: str
    source: EventSource
    event_type: str
    priority: EventPriority
    timestamp: float
    payload: Dict[str, Any]
    provenance: EventProvenance
    correlation_id: str
    world_state_version: Optional[int] = None
    deduplication_key: Optional[str] = None

    def __post_init__(self):
        if not self.event_id or not isinstance(self.event_id, str):
            raise ValueError("Event event_id must be a non-empty string.")
        if not isinstance(self.source, EventSource):
            raise ValueError(f"Event source must be an EventSource enum, got {type(self.source)}")
        if not self.event_type or not isinstance(self.event_type, str):
            raise ValueError("Event event_type must be a non-empty string.")
        if not isinstance(self.priority, EventPriority):
            raise ValueError(f"Event priority must be an EventPriority enum, got {type(self.priority)}")
        if self.timestamp <= 0.0:
            raise ValueError(f"Event timestamp must be positive, got {self.timestamp}")
        if not self.correlation_id or not isinstance(self.correlation_id, str):
            raise ValueError("Event correlation_id must be a non-empty string.")

    def get_dedup_key(self) -> str:
        """Return the effective deduplication key for storm/loop protection."""
        if self.deduplication_key:
            return self.deduplication_key
        # Default deterministic key: source:event_type:entity_id/key
        entity = self.payload.get("entity_id", "")
        prop = self.payload.get("property_name", "")
        return f"{self.source.value}:{self.event_type}:{entity}:{prop}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "source": self.source.value,
            "event_type": self.event_type,
            "priority": self.priority.value,
            "timestamp": self.timestamp,
            "payload": dict(self.payload),
            "provenance": self.provenance.to_dict(),
            "correlation_id": self.correlation_id,
            "world_state_version": self.world_state_version,
            "deduplication_key": self.deduplication_key,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Event":
        prov = EventProvenance.from_dict(data["provenance"]) if isinstance(data["provenance"], dict) else data["provenance"]
        return cls(
            event_id=data["event_id"],
            source=EventSource(data["source"]),
            event_type=data["event_type"],
            priority=EventPriority(data["priority"]),
            timestamp=float(data["timestamp"]),
            payload=dict(data.get("payload", {})),
            provenance=prov,
            correlation_id=data["correlation_id"],
            world_state_version=data.get("world_state_version"),
            deduplication_key=data.get("deduplication_key"),
        )


@dataclass(frozen=True)
class EventClassification:
    """
    Deterministic classification outcome for an autonomy event.
    Evaluated without LLMs.
    """
    category: EventCategory
    priority: EventPriority
    severity: str
    is_novel: bool
    requires_action: bool
    rationale: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category.value,
            "priority": self.priority.value,
            "severity": self.severity,
            "is_novel": self.is_novel,
            "requires_action": self.requires_action,
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class EventRelevance:
    """
    Deterministic relevance evaluation of an event relative to system goals and state.
    """
    score: float
    is_relevant: bool
    matched_goals: Tuple[str, ...] = field(default_factory=tuple)
    freshness_factor: float = 1.0
    priority_factor: float = 1.0
    rationale: str = ""

    def __post_init__(self):
        if not (0.0 <= self.score <= 1.0):
            raise ValueError(f"EventRelevance score must be in [0.0, 1.0], got {self.score}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": self.score,
            "is_relevant": self.is_relevant,
            "matched_goals": list(self.matched_goals),
            "freshness_factor": self.freshness_factor,
            "priority_factor": self.priority_factor,
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class TriggerCondition:
    """
    Matching criteria for an autonomous event trigger.
    """
    condition_id: str
    event_types: Tuple[str, ...] = field(default_factory=tuple)
    min_priority: EventPriority = EventPriority.NORMAL
    min_relevance: float = 0.5
    predicate_type: str = "always"
    predicate_params: Dict[str, Any] = field(default_factory=dict)

    def matches(self, event: Event, classification: EventClassification, relevance: EventRelevance) -> bool:
        """Deterministically evaluate whether an event satisfies the condition."""
        if self.event_types and event.event_type not in self.event_types:
            return False
        if classification.priority < self.min_priority:
            return False
        if relevance.score < self.min_relevance:
            return False

        if self.predicate_type == "always":
            return True
        elif self.predicate_type == "payload_match":
            key = self.predicate_params.get("key")
            val = self.predicate_params.get("value")
            return event.payload.get(key) == val
        elif self.predicate_type == "threshold":
            key = self.predicate_params.get("key")
            threshold = self.predicate_params.get("threshold")
            op = self.predicate_params.get("operator", "lt")
            actual = event.payload.get(key)
            if actual is None or threshold is None:
                return False
            try:
                num_actual = float(actual)
                num_thresh = float(threshold)
                if op in ("lt", "<"):
                    return num_actual < num_thresh
                elif op in ("lte", "<="):
                    return num_actual <= num_thresh
                elif op in ("gt", ">"):
                    return num_actual > num_thresh
                elif op in ("gte", ">="):
                    return num_actual >= num_thresh
                elif op in ("eq", "=="):
                    return num_actual == num_thresh
            except (ValueError, TypeError):
                return False
        return True


@dataclass(frozen=True)
class EventTrigger:
    """
    Autonomous rule linking satisfied trigger conditions to an autonomy decision.
    """
    trigger_id: str
    condition: TriggerCondition
    action_type: AutonomyDecisionType
    target_goal_template: Optional[Dict[str, Any]] = None
    cooldown_seconds: float = 0.0
    max_triggers_per_window: int = 1
    window_seconds: float = 60.0
    is_active: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trigger_id": self.trigger_id,
            "condition_id": self.condition.condition_id,
            "action_type": self.action_type.value,
            "target_goal_template": self.target_goal_template,
            "cooldown_seconds": self.cooldown_seconds,
            "max_triggers_per_window": self.max_triggers_per_window,
            "window_seconds": self.window_seconds,
            "is_active": self.is_active,
        }


@dataclass(frozen=True)
class AutonomyDecision:
    """
    Immutable, auditable outcome of event processing by the autonomy layer.
    """
    decision_id: str
    event_id: str
    correlation_id: str
    decision_type: AutonomyDecisionType
    classification: EventClassification
    relevance: EventRelevance
    matched_trigger_id: Optional[str] = None
    goal_id: Optional[str] = None
    goal_payload: Optional[Dict[str, Any]] = None
    policy_result: Optional[PolicyResult] = None
    reason: str = ""
    timestamp: float = field(default_factory=time.time)
    executed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "event_id": self.event_id,
            "correlation_id": self.correlation_id,
            "decision_type": self.decision_type.value,
            "classification": self.classification.to_dict(),
            "relevance": self.relevance.to_dict(),
            "matched_trigger_id": self.matched_trigger_id,
            "goal_id": self.goal_id,
            "goal_payload": self.goal_payload,
            "policy_decision": self.policy_result.decision.value if self.policy_result else None,
            "reason": self.reason,
            "timestamp": self.timestamp,
            "executed": self.executed,
        }


@dataclass(frozen=True)
class EventSubscription:
    """
    Subscription filter for subscribing components to autonomy events.
    """
    subscription_id: str
    source_filter: Optional[Tuple[EventSource, ...]] = None
    type_filter: Optional[Tuple[str, ...]] = None
    min_priority: EventPriority = EventPriority.LOW

    def matches(self, event: Event) -> bool:
        if self.source_filter and event.source not in self.source_filter:
            return False
        if self.type_filter and event.event_type not in self.type_filter:
            return False
        if event.priority < self.min_priority:
            return False
        return True
