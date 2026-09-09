"""
ATLAS Phase 6.4 — Multi-Product Situation & Mission Intelligence Models.

Defines canonical domain models for:
1. MultiProductSituation & ProductEvidence
2. Cross-Product EntityCorrelation & SituationContradiction
3. Multi-Product Mission, MissionObjective, MissionStatus, and TimelineEntry
4. MissionLimits for bounded resource safety

CRITICAL ARCHITECTURAL RULES:
1. NO SECOND BRAIN: Models represent semantic situational views and mission coordination contracts.
2. NO DIRECT EXECUTION: Models carry ZERO executable authority or direct tool dispatch code.
3. IMMUTABILITY & BOUNDS: Dataclasses are frozen where appropriate; collections are bounded.
4. SANITIZATION: All sensitive tokens/credentials in metadata are scrubbed at serialization.
"""

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from core.models.device_contract import (
    ProductRole,
    ProductType,
    sanitize_contract_metadata,
)
from core.models.goal import GoalPriority
from core.models.orchestration import (
    GeoLocation,
    ModalityType,
    Situation,
    SituationCategory,
    SituationEvidence,
    SituationSeverity,
    SituationStatus,
)


# ============================================================================
# Enums
# ============================================================================

class MissionStatus(str, Enum):
    """Lifecycle status of a multi-product coordinated mission."""
    PLANNED = "PLANNED"
    ACTIVE = "ACTIVE"
    INVESTIGATING = "INVESTIGATING"
    RESPONDING = "RESPONDING"
    MONITORING = "MONITORING"
    RESOLVING = "RESOLVING"
    COMPLETED = "COMPLETED"
    PAUSED = "PAUSED"
    BLOCKED = "BLOCKED"
    FAILED = "FAILED"
    ABORTED = "ABORTED"

    @classmethod
    def from_str(cls, val: Union[str, "MissionStatus"]) -> "MissionStatus":
        if isinstance(val, cls):
            return val
        norm = str(val).strip().upper()
        for member in cls:
            if member.value == norm or member.name == norm:
                return member
        return cls.PLANNED


class MissionObjectiveType(str, Enum):
    """Standardized tactical objective types for multi-product edge work."""
    VERIFY_INCIDENT = "VERIFY_INCIDENT"
    LOCATE_TARGET = "LOCATE_TARGET"
    ASSESS_THREAT = "ASSESS_THREAT"
    MAINTAIN_OBSERVATION = "MAINTAIN_OBSERVATION"
    NOTIFY_WEARER = "NOTIFY_WEARER"
    INSPECT_ROUTE = "INSPECT_ROUTE"
    DISPATCH_RESPONSE = "DISPATCH_RESPONSE"
    MONITOR_AREA = "MONITOR_AREA"
    CONFIRM_RESOLUTION = "CONFIRM_RESOLUTION"

    @classmethod
    def from_str(cls, val: Union[str, "MissionObjectiveType"]) -> "MissionObjectiveType":
        if isinstance(val, cls):
            return val
        norm = str(val).strip().upper()
        for member in cls:
            if member.value == norm or member.name == norm:
                return member
        return cls.VERIFY_INCIDENT


class ObjectiveStatus(str, Enum):
    """Lifecycle status of an individual mission objective."""
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
    BLOCKED = "BLOCKED"

    @classmethod
    def from_str(cls, val: Union[str, "ObjectiveStatus"]) -> "ObjectiveStatus":
        if isinstance(val, cls):
            return val
        norm = str(val).strip().upper()
        for member in cls:
            if member.value == norm or member.name == norm:
                return member
        return cls.PENDING


class EntityCorrelationStatus(str, Enum):
    """Evidence status of a cross-product entity correlation."""
    CANDIDATE = "CANDIDATE"
    CORRELATED = "CORRELATED"
    DISPROVED = "DISPROVED"
    UNKNOWN = "UNKNOWN"

    @classmethod
    def from_str(cls, val: Union[str, "EntityCorrelationStatus"]) -> "EntityCorrelationStatus":
        if isinstance(val, cls):
            return val
        norm = str(val).strip().upper()
        for member in cls:
            if member.value == norm or member.name == norm:
                return member
        return cls.CANDIDATE


# ============================================================================
# Limits & Safety Bounds
# ============================================================================

@dataclass(frozen=True)
class MissionLimits:
    """Bounded resource safety limits for multi-product situation & mission models."""
    max_active_missions: int = 20
    max_objectives_per_mission: int = 20
    max_evidence_per_mission: int = 100
    max_timeline_entries: int = 200
    max_replanning_attempts: int = 5
    max_situations_correlated: int = 50
    max_entity_correlations: int = 100
    max_contradictions: int = 50
    temporal_correlation_window_s: float = 300.0


# ============================================================================
# Product Evidence & Contradiction Models
# ============================================================================

@dataclass(frozen=True)
class ProductEvidence:
    """
    Immutable representation of evidence originating from an edge product.
    References the observation and underlying situation without copying large media buffers.
    """
    evidence_id: str
    source_id: str
    product_type: ProductType
    product_role: ProductRole
    observation_id: str
    situation_id: str
    timestamp: float
    confidence: float
    modality: ModalityType
    summary: str
    data: Dict[str, Any] = field(default_factory=dict)
    correlation_id: str = ""
    causation_id: Optional[str] = None
    provenance: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.evidence_id or not isinstance(self.evidence_id, str):
            raise ValueError("ProductEvidence evidence_id must be a non-empty string.")
        if not self.source_id or not isinstance(self.source_id, str):
            raise ValueError("ProductEvidence source_id must be a non-empty string.")
        if not self.observation_id or not isinstance(self.observation_id, str):
            raise ValueError("ProductEvidence observation_id must be a non-empty string.")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"ProductEvidence confidence must be in [0.0, 1.0], got {self.confidence}")

        # Ensure credentials and sensitive keys in data & provenance are scrubbed
        object.__setattr__(self, "data", sanitize_contract_metadata(self.data))
        object.__setattr__(self, "provenance", sanitize_contract_metadata(self.provenance))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "source_id": self.source_id,
            "product_type": self.product_type.value,
            "product_role": self.product_role.value,
            "observation_id": self.observation_id,
            "situation_id": self.situation_id,
            "timestamp": self.timestamp,
            "confidence": self.confidence,
            "modality": self.modality.value if hasattr(self.modality, "value") else str(self.modality),
            "summary": self.summary,
            "data": sanitize_contract_metadata(self.data),
            "correlation_id": self.correlation_id,
            "causation_id": self.causation_id,
            "provenance": sanitize_contract_metadata(self.provenance),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProductEvidence":
        return cls(
            evidence_id=str(data["evidence_id"]),
            source_id=str(data["source_id"]),
            product_type=ProductType.from_str(data.get("product_type", "UNKNOWN")),
            product_role=ProductRole.from_str(data.get("product_role", "OBSERVATION_SOURCE")),
            observation_id=str(data["observation_id"]),
            situation_id=str(data.get("situation_id", "")),
            timestamp=float(data.get("timestamp", time.time())),
            confidence=float(data.get("confidence", 0.5)),
            modality=ModalityType.from_str(data.get("modality", "EVENT")),
            summary=str(data.get("summary", "")),
            data=dict(data.get("data", {})),
            correlation_id=str(data.get("correlation_id", "")),
            causation_id=str(data["causation_id"]) if data.get("causation_id") else None,
            provenance=dict(data.get("provenance", {})),
        )


@dataclass(frozen=True)
class SituationContradiction:
    """
    Explicit representation of conflicting evidence between two or more edge observations.
    Preserves disagreement without overwriting or 'last observation wins' truncation.
    """
    contradiction_id: str
    situation_id: str
    source_a: str
    source_b: str
    claim_a: str
    claim_b: str
    confidence_a: float
    confidence_b: float
    timestamp_a: float
    timestamp_b: float
    resolution_status: str = "UNRESOLVED"  # "UNRESOLVED", "RESOLVED_BY_CONFIDENCE", "SPLIT_SITUATION", "DISMISSED"
    resolution_notes: Optional[str] = None
    detected_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "contradiction_id": self.contradiction_id,
            "situation_id": self.situation_id,
            "source_a": self.source_a,
            "source_b": self.source_b,
            "claim_a": self.claim_a,
            "claim_b": self.claim_b,
            "confidence_a": self.confidence_a,
            "confidence_b": self.confidence_b,
            "timestamp_a": self.timestamp_a,
            "timestamp_b": self.timestamp_b,
            "resolution_status": self.resolution_status,
            "resolution_notes": self.resolution_notes,
            "detected_at": self.detected_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SituationContradiction":
        return cls(
            contradiction_id=str(data["contradiction_id"]),
            situation_id=str(data["situation_id"]),
            source_a=str(data["source_a"]),
            source_b=str(data["source_b"]),
            claim_a=str(data["claim_a"]),
            claim_b=str(data["claim_b"]),
            confidence_a=float(data.get("confidence_a", 0.5)),
            confidence_b=float(data.get("confidence_b", 0.5)),
            timestamp_a=float(data.get("timestamp_a", 0.0)),
            timestamp_b=float(data.get("timestamp_b", 0.0)),
            resolution_status=str(data.get("resolution_status", "UNRESOLVED")),
            resolution_notes=str(data["resolution_notes"]) if data.get("resolution_notes") else None,
            detected_at=float(data.get("detected_at", time.time())),
        )


@dataclass(frozen=True)
class EntityCorrelation:
    """
    Deterministic cross-product association between entities detected by different products.
    Example: 'person_17' (Vision) linked to 'person_A' (Drone).
    """
    correlation_id: str
    primary_entity_id: str
    correlated_entity_id: str
    entity_type: str  # "PERSON", "VEHICLE", "OBSTACLE", "HAZARD", "DEVICE"
    status: EntityCorrelationStatus = EntityCorrelationStatus.CANDIDATE
    confidence: float = 0.5
    supporting_evidence_ids: Tuple[str, ...] = ()
    spatial_distance_meters: Optional[float] = None
    time_delta_seconds: Optional[float] = None
    provenance: Dict[str, Any] = field(default_factory=dict)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "correlation_id": self.correlation_id,
            "primary_entity_id": self.primary_entity_id,
            "correlated_entity_id": self.correlated_entity_id,
            "entity_type": self.entity_type,
            "status": self.status.value,
            "confidence": self.confidence,
            "supporting_evidence_ids": list(self.supporting_evidence_ids),
            "spatial_distance_meters": self.spatial_distance_meters,
            "time_delta_seconds": self.time_delta_seconds,
            "provenance": sanitize_contract_metadata(self.provenance),
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EntityCorrelation":
        return cls(
            correlation_id=str(data["correlation_id"]),
            primary_entity_id=str(data["primary_entity_id"]),
            correlated_entity_id=str(data["correlated_entity_id"]),
            entity_type=str(data.get("entity_type", "PERSON")),
            status=EntityCorrelationStatus.from_str(data.get("status", "CANDIDATE")),
            confidence=float(data.get("confidence", 0.5)),
            supporting_evidence_ids=tuple(data.get("supporting_evidence_ids", [])),
            spatial_distance_meters=float(data["spatial_distance_meters"]) if data.get("spatial_distance_meters") is not None else None,
            time_delta_seconds=float(data["time_delta_seconds"]) if data.get("time_delta_seconds") is not None else None,
            provenance=dict(data.get("provenance", {})),
            updated_at=float(data.get("updated_at", time.time())),
        )


# ============================================================================
# Multi-Product Situation Model
# ============================================================================

@dataclass(frozen=True)
class MultiProductSituation:
    """
    Unified situational intelligence synthesized across multiple edge products.
    References underlying Situation objects rather than duplicating their storage.
    """
    situation_id: str
    category: SituationCategory
    title: str
    description: str
    severity: SituationSeverity
    confidence: float
    status: SituationStatus
    involved_products: Tuple[ProductType, ...]
    involved_product_ids: Tuple[str, ...]
    involved_entities: Tuple[str, ...]
    supporting_situation_ids: Tuple[str, ...]
    evidence_references: Tuple[ProductEvidence, ...]
    contradictions: Tuple[SituationContradiction, ...] = ()
    entity_correlations: Tuple[EntityCorrelation, ...] = ()
    location: Optional[GeoLocation] = None
    spatial_bounds: Optional[Dict[str, float]] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    valid_until: Optional[float] = None
    correlation_id: str = ""
    causation_id: Optional[str] = None
    recommended_missions: Tuple[str, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.situation_id or not isinstance(self.situation_id, str):
            raise ValueError("MultiProductSituation situation_id must be a non-empty string.")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"MultiProductSituation confidence must be in [0.0, 1.0], got {self.confidence}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "situation_id": self.situation_id,
            "category": self.category.value,
            "title": self.title,
            "description": self.description,
            "severity": self.severity.value,
            "confidence": self.confidence,
            "status": self.status.value,
            "involved_products": [p.value for p in self.involved_products],
            "involved_product_ids": list(self.involved_product_ids),
            "involved_entities": list(self.involved_entities),
            "supporting_situation_ids": list(self.supporting_situation_ids),
            "evidence_references": [e.to_dict() for e in self.evidence_references],
            "contradictions": [c.to_dict() for c in self.contradictions],
            "entity_correlations": [ec.to_dict() for ec in self.entity_correlations],
            "location": self.location.to_dict() if self.location else None,
            "spatial_bounds": dict(self.spatial_bounds) if self.spatial_bounds else None,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "valid_until": self.valid_until,
            "correlation_id": self.correlation_id,
            "causation_id": self.causation_id,
            "recommended_missions": list(self.recommended_missions),
            "metadata": sanitize_contract_metadata(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MultiProductSituation":
        loc_data = data.get("location")
        loc = GeoLocation.from_dict(loc_data) if loc_data else None

        prods = tuple(ProductType.from_str(p) for p in data.get("involved_products", []))
        ev_refs = tuple(ProductEvidence.from_dict(e) for e in data.get("evidence_references", []))
        contras = tuple(SituationContradiction.from_dict(c) for c in data.get("contradictions", []))
        e_corrs = tuple(EntityCorrelation.from_dict(ec) for ec in data.get("entity_correlations", []))

        return cls(
            situation_id=str(data["situation_id"]),
            category=SituationCategory.from_str(data.get("category", "UNKNOWN")),
            title=str(data.get("title", "")),
            description=str(data.get("description", "")),
            severity=SituationSeverity.from_str(data.get("severity", "INFO")),
            confidence=float(data.get("confidence", 0.5)),
            status=SituationStatus.from_str(data.get("status", "DETECTED")),
            involved_products=prods,
            involved_product_ids=tuple(data.get("involved_product_ids", [])),
            involved_entities=tuple(data.get("involved_entities", [])),
            supporting_situation_ids=tuple(data.get("supporting_situation_ids", [])),
            evidence_references=ev_refs,
            contradictions=contras,
            entity_correlations=e_corrs,
            location=loc,
            spatial_bounds=dict(data.get("spatial_bounds", {})) if data.get("spatial_bounds") else None,
            created_at=float(data.get("created_at", time.time())),
            updated_at=float(data.get("updated_at", time.time())),
            valid_until=float(data["valid_until"]) if data.get("valid_until") is not None else None,
            correlation_id=str(data.get("correlation_id", "")),
            causation_id=str(data["causation_id"]) if data.get("causation_id") else None,
            recommended_missions=tuple(data.get("recommended_missions", [])),
            metadata=dict(data.get("metadata", {})),
        )


# ============================================================================
# Mission & Objective Models
# ============================================================================

@dataclass(frozen=True)
class MissionObjective:
    """
    Discrete semantic objective of a multi-product mission.
    Contains NO direct executable code; translates to Goal requests for AutonomousGoalManager.
    """
    objective_id: str
    mission_id: str
    type: MissionObjectiveType
    description: str
    dependencies: Tuple[str, ...] = ()
    required_capabilities: Tuple[str, ...] = ()
    candidate_products: Tuple[str, ...] = ()
    assigned_product_id: Optional[str] = None
    status: ObjectiveStatus = ObjectiveStatus.PENDING
    completion_criteria: Dict[str, Any] = field(default_factory=dict)
    evidence_requirements: Tuple[str, ...] = ()
    progress: float = 0.0
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    result_summary: Optional[str] = None

    def __post_init__(self):
        if not self.objective_id or not isinstance(self.objective_id, str):
            raise ValueError("MissionObjective objective_id must be a non-empty string.")
        if not self.mission_id or not isinstance(self.mission_id, str):
            raise ValueError("MissionObjective mission_id must be a non-empty string.")
        if not (0.0 <= self.progress <= 1.0):
            raise ValueError(f"MissionObjective progress must be in [0.0, 1.0], got {self.progress}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "objective_id": self.objective_id,
            "mission_id": self.mission_id,
            "type": self.type.value,
            "description": self.description,
            "dependencies": list(self.dependencies),
            "required_capabilities": list(self.required_capabilities),
            "candidate_products": list(self.candidate_products),
            "assigned_product_id": self.assigned_product_id,
            "status": self.status.value,
            "completion_criteria": sanitize_contract_metadata(self.completion_criteria),
            "evidence_requirements": list(self.evidence_requirements),
            "progress": self.progress,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "result_summary": self.result_summary,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MissionObjective":
        return cls(
            objective_id=str(data["objective_id"]),
            mission_id=str(data["mission_id"]),
            type=MissionObjectiveType.from_str(data.get("type", "VERIFY_INCIDENT")),
            description=str(data.get("description", "")),
            dependencies=tuple(data.get("dependencies", [])),
            required_capabilities=tuple(data.get("required_capabilities", [])),
            candidate_products=tuple(data.get("candidate_products", [])),
            assigned_product_id=str(data["assigned_product_id"]) if data.get("assigned_product_id") else None,
            status=ObjectiveStatus.from_str(data.get("status", "PENDING")),
            completion_criteria=dict(data.get("completion_criteria", {})),
            evidence_requirements=tuple(data.get("evidence_requirements", [])),
            progress=float(data.get("progress", 0.0)),
            created_at=float(data.get("created_at", time.time())),
            completed_at=float(data["completed_at"]) if data.get("completed_at") is not None else None,
            result_summary=str(data["result_summary"]) if data.get("result_summary") else None,
        )


@dataclass(frozen=True)
class MissionTimelineEntry:
    """
    Chronological milestone entry recording key state transitions in a mission.
    """
    entry_id: str
    timestamp: float
    source_id: str
    event_type: str
    description: str
    evidence_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "timestamp": self.timestamp,
            "source_id": self.source_id,
            "event_type": self.event_type,
            "description": self.description,
            "evidence_id": self.evidence_id,
            "metadata": sanitize_contract_metadata(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MissionTimelineEntry":
        return cls(
            entry_id=str(data["entry_id"]),
            timestamp=float(data.get("timestamp", time.time())),
            source_id=str(data.get("source_id", "")),
            event_type=str(data.get("event_type", "INFO")),
            description=str(data.get("description", "")),
            evidence_id=str(data["evidence_id"]) if data.get("evidence_id") else None,
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class Mission:
    """
    Immutable representation of a bounded, multi-product tactical mission.
    DOES NOT replace Goal or GoalManager. Coordinates objectives and maps to Goals.
    """
    mission_id: str
    mission_type: str
    description: str
    trigger_situation_ids: Tuple[str, ...] = ()
    involved_product_ids: Tuple[str, ...] = ()
    objectives: Tuple[MissionObjective, ...] = ()
    current_status: MissionStatus = MissionStatus.PLANNED
    priority: Union[GoalPriority, str] = GoalPriority.NORMAL
    constraints: Dict[str, Any] = field(default_factory=dict)
    evidence_references: Tuple[ProductEvidence, ...] = ()
    timeline: Tuple[MissionTimelineEntry, ...] = ()
    progress: float = 0.0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    correlation_id: str = ""
    causation_id: Optional[str] = None
    replanning_count: int = 0
    failure_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.mission_id or not isinstance(self.mission_id, str):
            raise ValueError("Mission mission_id must be a non-empty string.")
        if not (0.0 <= self.progress <= 1.0):
            raise ValueError(f"Mission progress must be in [0.0, 1.0], got {self.progress}")

    def to_dict(self) -> Dict[str, Any]:
        prio_val = self.priority.value if hasattr(self.priority, "value") else str(self.priority)
        return {
            "mission_id": self.mission_id,
            "mission_type": self.mission_type,
            "description": self.description,
            "trigger_situation_ids": list(self.trigger_situation_ids),
            "involved_product_ids": list(self.involved_product_ids),
            "objectives": [o.to_dict() for o in self.objectives],
            "current_status": self.current_status.value,
            "priority": prio_val,
            "constraints": sanitize_contract_metadata(self.constraints),
            "evidence_references": [e.to_dict() for e in self.evidence_references],
            "timeline": [t.to_dict() for t in self.timeline],
            "progress": self.progress,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "completed_at": self.completed_at,
            "correlation_id": self.correlation_id,
            "causation_id": self.causation_id,
            "replanning_count": self.replanning_count,
            "failure_reason": self.failure_reason,
            "metadata": sanitize_contract_metadata(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Mission":
        objs = tuple(MissionObjective.from_dict(o) for o in data.get("objectives", []))
        evs = tuple(ProductEvidence.from_dict(e) for e in data.get("evidence_references", []))
        tl = tuple(MissionTimelineEntry.from_dict(t) for t in data.get("timeline", []))

        return cls(
            mission_id=str(data["mission_id"]),
            mission_type=str(data.get("mission_type", "TACTICAL")),
            description=str(data.get("description", "")),
            trigger_situation_ids=tuple(data.get("trigger_situation_ids", [])),
            involved_product_ids=tuple(data.get("involved_product_ids", [])),
            objectives=objs,
            current_status=MissionStatus.from_str(data.get("current_status", "PLANNED")),
            priority=GoalPriority.from_str(data.get("priority", "NORMAL")),
            constraints=dict(data.get("constraints", {})),
            evidence_references=evs,
            timeline=tl,
            progress=float(data.get("progress", 0.0)),
            created_at=float(data.get("created_at", time.time())),
            updated_at=float(data.get("updated_at", time.time())),
            completed_at=float(data["completed_at"]) if data.get("completed_at") is not None else None,
            correlation_id=str(data.get("correlation_id", "")),
            causation_id=str(data["causation_id"]) if data.get("causation_id") else None,
            replanning_count=int(data.get("replanning_count", 0)),
            failure_reason=str(data["failure_reason"]) if data.get("failure_reason") else None,
            metadata=dict(data.get("metadata", {})),
        )
