import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union


class FreshnessStatus(str, Enum):
    """
    Deterministic freshness status of a world condition.
    Freshness is orthogonal to confidence: high-confidence facts can be stale or expired.
    """
    FRESH = "FRESH"
    AGING = "AGING"
    STALE = "STALE"
    EXPIRED = "EXPIRED"


class TransitionType(str, Enum):
    """
    Deterministic transition category resulting from evaluating an observation.
    """
    ADD = "ADD"
    UPDATE = "UPDATE"
    UNCHANGED = "UNCHANGED"
    REJECTED = "REJECTED"
    CONFLICT = "CONFLICT"


class ConflictStatus(str, Enum):
    """
    Status of an identified state conflict.
    """
    DETECTED = "DETECTED"
    RESOLVED = "RESOLVED"
    UNRESOLVED = "UNRESOLVED"


class ResolutionStrategy(str, Enum):
    """
    Deterministic strategy applied to resolve competing observations.
    """
    SOURCE_AUTHORITY = "SOURCE_AUTHORITY"
    HIGHER_CONFIDENCE = "HIGHER_CONFIDENCE"
    RECENCY = "RECENCY"
    MANUAL = "MANUAL"


@dataclass(frozen=True)
class StateProvenance:
    """
    Identifies the origin, modality, and recorded time of a world-state belief.
    """
    source_id: str
    source_type: str
    observation_id: str
    recorded_at: float = field(default_factory=time.time)

    def __post_init__(self):
        if not self.source_id or not isinstance(self.source_id, str):
            raise ValueError("StateProvenance source_id must be a non-empty string.")
        if not self.source_type or not isinstance(self.source_type, str):
            raise ValueError("StateProvenance source_type must be a non-empty string.")
        if not self.observation_id or not isinstance(self.observation_id, str):
            raise ValueError("StateProvenance observation_id must be a non-empty string.")
        if self.recorded_at <= 0.0:
            raise ValueError(f"StateProvenance recorded_at must be positive, got {self.recorded_at}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "source_type": self.source_type,
            "observation_id": self.observation_id,
            "recorded_at": self.recorded_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StateProvenance":
        return cls(
            source_id=data["source_id"],
            source_type=data["source_type"],
            observation_id=data["observation_id"],
            recorded_at=data.get("recorded_at", time.time()),
        )


@dataclass(frozen=True)
class WorldCondition:
    """
    A single property assertion about an entity in the world state.
    """
    entity_id: str
    property_name: str
    value: Any
    confidence: float
    observed_at: float
    expires_at: Optional[float] = None
    provenance: StateProvenance = field(
        default_factory=lambda: StateProvenance("system", "internal", "init_obs")
    )
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.entity_id or not isinstance(self.entity_id, str):
            raise ValueError("WorldCondition entity_id must be a non-empty string.")
        if not self.property_name or not isinstance(self.property_name, str):
            raise ValueError("WorldCondition property_name must be a non-empty string.")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"WorldCondition confidence must be in [0.0, 1.0], got {self.confidence}")
        if self.observed_at <= 0.0:
            raise ValueError(f"WorldCondition observed_at must be positive, got {self.observed_at}")
        if self.expires_at is not None and self.expires_at < self.observed_at:
            raise ValueError(f"WorldCondition expires_at ({self.expires_at}) cannot be before observed_at ({self.observed_at})")

    @property
    def condition_key(self) -> Tuple[str, str]:
        return (self.entity_id, self.property_name)

    def is_expired(self, now: Optional[float] = None) -> bool:
        current = now if now is not None else time.time()
        return self.expires_at is not None and current >= self.expires_at

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "property_name": self.property_name,
            "value": self.value,
            "confidence": self.confidence,
            "observed_at": self.observed_at,
            "expires_at": self.expires_at,
            "provenance": self.provenance.to_dict(),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorldCondition":
        prov = StateProvenance.from_dict(data["provenance"]) if isinstance(data.get("provenance"), dict) else data["provenance"]
        return cls(
            entity_id=data["entity_id"],
            property_name=data["property_name"],
            value=data["value"],
            confidence=float(data["confidence"]),
            observed_at=float(data["observed_at"]),
            expires_at=float(data["expires_at"]) if data.get("expires_at") is not None else None,
            provenance=prov,
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class WorldRelationship:
    """
    Directed relationship connecting two world entities.
    """
    source_entity_id: str
    relationship_type: str
    target_entity_id: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    provenance: StateProvenance = field(
        default_factory=lambda: StateProvenance("system", "internal", "init_rel")
    )

    def __post_init__(self):
        if not self.source_entity_id or not isinstance(self.source_entity_id, str):
            raise ValueError("WorldRelationship source_entity_id must be a non-empty string.")
        if not self.relationship_type or not isinstance(self.relationship_type, str):
            raise ValueError("WorldRelationship relationship_type must be a non-empty string.")
        if not self.target_entity_id or not isinstance(self.target_entity_id, str):
            raise ValueError("WorldRelationship target_entity_id must be a non-empty string.")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"WorldRelationship confidence must be in [0.0, 1.0], got {self.confidence}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_entity_id": self.source_entity_id,
            "relationship_type": self.relationship_type,
            "target_entity_id": self.target_entity_id,
            "attributes": dict(self.attributes),
            "confidence": self.confidence,
            "provenance": self.provenance.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorldRelationship":
        prov = StateProvenance.from_dict(data["provenance"]) if isinstance(data.get("provenance"), dict) else data["provenance"]
        return cls(
            source_entity_id=data["source_entity_id"],
            relationship_type=data["relationship_type"],
            target_entity_id=data["target_entity_id"],
            attributes=dict(data.get("attributes", {})),
            confidence=float(data.get("confidence", 1.0)),
            provenance=prov,
        )


@dataclass(frozen=True)
class WorldEntity:
    """
    Identified entity within the world state.
    """
    entity_id: str
    entity_type: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    provenance: StateProvenance = field(
        default_factory=lambda: StateProvenance("system", "internal", "init_entity")
    )

    def __post_init__(self):
        if not self.entity_id or not isinstance(self.entity_id, str):
            raise ValueError("WorldEntity entity_id must be a non-empty string.")
        if not self.entity_type or not isinstance(self.entity_type, str):
            raise ValueError("WorldEntity entity_type must be a non-empty string.")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"WorldEntity confidence must be in [0.0, 1.0], got {self.confidence}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "entity_type": self.entity_type,
            "attributes": dict(self.attributes),
            "confidence": self.confidence,
            "provenance": self.provenance.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorldEntity":
        prov = StateProvenance.from_dict(data["provenance"]) if isinstance(data.get("provenance"), dict) else data["provenance"]
        return cls(
            entity_id=data["entity_id"],
            entity_type=data["entity_type"],
            attributes=dict(data.get("attributes", {})),
            confidence=float(data.get("confidence", 1.0)),
            provenance=prov,
        )


@dataclass(frozen=True)
class WorldState:
    """
    Immutable snapshot of ATLAS's structured representation of the world.
    Contains monotonically increasing versioning.
    """
    state_id: str
    version: int
    timestamp: float
    entities: Tuple[WorldEntity, ...] = field(default_factory=tuple)
    conditions: Tuple[WorldCondition, ...] = field(default_factory=tuple)
    relationships: Tuple[WorldRelationship, ...] = field(default_factory=tuple)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.state_id or not isinstance(self.state_id, str):
            raise ValueError("WorldState state_id must be a non-empty string.")
        if self.version < 1:
            raise ValueError(f"WorldState version must be >= 1, got {self.version}")
        if self.timestamp <= 0.0:
            raise ValueError(f"WorldState timestamp must be positive, got {self.timestamp}")

    def get_condition(self, entity_id: str, property_name: str) -> Optional[WorldCondition]:
        for cond in self.conditions:
            if cond.entity_id == entity_id and cond.property_name == property_name:
                return cond
        return None

    def get_entity(self, entity_id: str) -> Optional[WorldEntity]:
        for ent in self.entities:
            if ent.entity_id == entity_id:
                return ent
        return None

    def get_relationships_for_entity(self, entity_id: str) -> Tuple[WorldRelationship, ...]:
        return tuple(
            rel for rel in self.relationships
            if rel.source_entity_id == entity_id or rel.target_entity_id == entity_id
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state_id": self.state_id,
            "version": self.version,
            "timestamp": self.timestamp,
            "entities": [ent.to_dict() for ent in self.entities],
            "conditions": [cond.to_dict() for cond in self.conditions],
            "relationships": [rel.to_dict() for rel in self.relationships],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorldState":
        entities = tuple(WorldEntity.from_dict(item) for item in data.get("entities", []))
        conditions = tuple(WorldCondition.from_dict(item) for item in data.get("conditions", []))
        relationships = tuple(WorldRelationship.from_dict(item) for item in data.get("relationships", []))
        return cls(
            state_id=data["state_id"],
            version=int(data["version"]),
            timestamp=float(data["timestamp"]),
            entities=entities,
            conditions=conditions,
            relationships=relationships,
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class Observation:
    """
    Model-neutral observation submitted from perception, tool outputs, sensors, or user inputs.
    Observations never mutate world state directly; they pass through an explicit update boundary.
    """
    observation_id: str
    source_id: str
    source_type: str
    timestamp: float
    entity_id: str
    property_name: str
    value: Any
    confidence: float = 1.0
    expires_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.observation_id or not isinstance(self.observation_id, str):
            raise ValueError("Observation observation_id must be a non-empty string.")
        if not self.source_id or not isinstance(self.source_id, str):
            raise ValueError("Observation source_id must be a non-empty string.")
        if not self.source_type or not isinstance(self.source_type, str):
            raise ValueError("Observation source_type must be a non-empty string.")
        if not self.entity_id or not isinstance(self.entity_id, str):
            raise ValueError("Observation entity_id must be a non-empty string.")
        if not self.property_name or not isinstance(self.property_name, str):
            raise ValueError("Observation property_name must be a non-empty string.")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"Observation confidence must be in [0.0, 1.0], got {self.confidence}")
        if self.timestamp <= 0.0:
            raise ValueError(f"Observation timestamp must be positive, got {self.timestamp}")
        if self.expires_at is not None and self.expires_at < self.timestamp:
            raise ValueError(f"Observation expires_at ({self.expires_at}) cannot be before timestamp ({self.timestamp})")

    def to_provenance(self) -> StateProvenance:
        return StateProvenance(
            source_id=self.source_id,
            source_type=self.source_type,
            observation_id=self.observation_id,
            recorded_at=self.timestamp,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "source_id": self.source_id,
            "source_type": self.source_type,
            "timestamp": self.timestamp,
            "entity_id": self.entity_id,
            "property_name": self.property_name,
            "value": self.value,
            "confidence": self.confidence,
            "expires_at": self.expires_at,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Observation":
        return cls(
            observation_id=data["observation_id"],
            source_id=data["source_id"],
            source_type=data["source_type"],
            timestamp=float(data["timestamp"]),
            entity_id=data["entity_id"],
            property_name=data["property_name"],
            value=data["value"],
            confidence=float(data.get("confidence", 1.0)),
            expires_at=float(data["expires_at"]) if data.get("expires_at") is not None else None,
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class ConflictResolution:
    """
    Deterministic resolution record for a state conflict.
    """
    resolution_id: str
    conflict_id: str
    resolved_value: Any
    winning_source: str
    strategy: ResolutionStrategy
    rationale: str
    resolved_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "resolution_id": self.resolution_id,
            "conflict_id": self.conflict_id,
            "resolved_value": self.resolved_value,
            "winning_source": self.winning_source,
            "strategy": self.strategy.value,
            "rationale": self.rationale,
            "resolved_at": self.resolved_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConflictResolution":
        return cls(
            resolution_id=data["resolution_id"],
            conflict_id=data["conflict_id"],
            resolved_value=data["resolved_value"],
            winning_source=data["winning_source"],
            strategy=ResolutionStrategy(data["strategy"]),
            rationale=data["rationale"],
            resolved_at=float(data.get("resolved_at", time.time())),
        )


@dataclass(frozen=True)
class StateConflict:
    """
    Explicit representation of competing or contradictory assertions.
    Conflicts are never silently overwritten.
    """
    conflict_id: str
    entity_id: str
    property_name: str
    existing_condition: WorldCondition
    competing_observation: Observation
    detected_at: float
    status: ConflictStatus = ConflictStatus.DETECTED
    resolution: Optional[ConflictResolution] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "conflict_id": self.conflict_id,
            "entity_id": self.entity_id,
            "property_name": self.property_name,
            "existing_condition": self.existing_condition.to_dict(),
            "competing_observation": self.competing_observation.to_dict(),
            "detected_at": self.detected_at,
            "status": self.status.value,
            "resolution": self.resolution.to_dict() if self.resolution else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StateConflict":
        existing = WorldCondition.from_dict(data["existing_condition"])
        competing = Observation.from_dict(data["competing_observation"])
        resolution = ConflictResolution.from_dict(data["resolution"]) if data.get("resolution") else None
        return cls(
            conflict_id=data["conflict_id"],
            entity_id=data["entity_id"],
            property_name=data["property_name"],
            existing_condition=existing,
            competing_observation=competing,
            detected_at=float(data["detected_at"]),
            status=ConflictStatus(data["status"]),
            resolution=resolution,
        )


@dataclass(frozen=True)
class WorldStateTransition:
    """
    Auditable record of a single atomic state transition.
    Supports deterministic replay and complete temporal provenance.
    """
    transition_id: str
    from_version: int
    to_version: int
    transition_type: TransitionType
    observation_id: str
    entity_id: str
    property_name: str
    old_value: Any
    new_value: Any
    timestamp: float
    provenance: StateProvenance
    conflict_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "transition_id": self.transition_id,
            "from_version": self.from_version,
            "to_version": self.to_version,
            "transition_type": self.transition_type.value,
            "observation_id": self.observation_id,
            "entity_id": self.entity_id,
            "property_name": self.property_name,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "timestamp": self.timestamp,
            "provenance": self.provenance.to_dict(),
            "conflict_id": self.conflict_id,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorldStateTransition":
        prov = StateProvenance.from_dict(data["provenance"]) if isinstance(data.get("provenance"), dict) else data["provenance"]
        return cls(
            transition_id=data["transition_id"],
            from_version=int(data["from_version"]),
            to_version=int(data["to_version"]),
            transition_type=TransitionType(data["transition_type"]),
            observation_id=data["observation_id"],
            entity_id=data["entity_id"],
            property_name=data["property_name"],
            old_value=data.get("old_value"),
            new_value=data.get("new_value"),
            timestamp=float(data["timestamp"]),
            provenance=prov,
            conflict_id=data.get("conflict_id"),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class WorldStateUpdateResult:
    """
    Structured outcome of an observation update attempt.
    """
    success: bool
    transition_type: TransitionType
    previous_version: int
    current_version: int
    condition: Optional[WorldCondition] = None
    transition: Optional[WorldStateTransition] = None
    conflict: Optional[StateConflict] = None
    error: Optional[str] = None


@dataclass(frozen=True)
class FreshnessConfig:
    """
    Explicit, configurable thresholds for evaluating condition freshness.
    No hard-coded global constants.
    """
    fresh_duration: float = 30.0
    aging_duration: float = 120.0
    stale_duration: float = 300.0
    allow_expired_query: bool = False

    def __post_init__(self):
        if self.fresh_duration < 0 or self.aging_duration < 0 or self.stale_duration < 0:
            raise ValueError("Freshness durations must be non-negative.")
        if not (self.fresh_duration <= self.aging_duration <= self.stale_duration):
            raise ValueError(
                f"Freshness durations must satisfy fresh_duration ({self.fresh_duration}) <= "
                f"aging_duration ({self.aging_duration}) <= stale_duration ({self.stale_duration})"
            )

    def evaluate(self, observed_at: float, expires_at: Optional[float], now: float) -> FreshnessStatus:
        if expires_at is not None and now >= expires_at:
            return FreshnessStatus.EXPIRED
        age = max(0.0, now - observed_at)
        if age <= self.fresh_duration:
            return FreshnessStatus.FRESH
        elif age <= self.aging_duration:
            return FreshnessStatus.AGING
        elif age <= self.stale_duration:
            return FreshnessStatus.STALE
        return FreshnessStatus.EXPIRED


@dataclass(frozen=True)
class ConflictPolicy:
    """
    Deterministic conflict resolution configuration.
    Defines explicit source authorities and tie-breaking behavior.
    """
    source_authorities: Dict[str, int] = field(default_factory=lambda: {
        "user": 100,
        "direct_sensor": 85,
        "perception": 70,
        "tool_result": 60,
        "web": 40,
        "inferred": 20,
        "default": 10,
    })
    confidence_delta_threshold: float = 0.15
    confidence_tie_breaker: bool = True
    recency_tie_breaker: bool = True

    def get_authority(self, source_type: str) -> int:
        return self.source_authorities.get(source_type.lower(), self.source_authorities.get("default", 10))
