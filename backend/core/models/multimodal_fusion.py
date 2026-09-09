"""ATLAS Phase 6.5d — Temporal & Cross-Modal Fusion Domain Models.

Establishes transport-neutral, immutable domain models for:
- Temporal relationship categorization (BEFORE, AFTER, OVERLAPS, COINCIDENT, WITHIN_WINDOW, SEQUENCE)
- Evidence relationships (SUPPORTS, CORROBORATES, CONTRADICTS, PRECEDES, FOLLOWS, OVERLAPS, COLOCATED_WITH, SAME_ENTITY_CANDIDATE, DUPLICATES)
- Modality compatibility and cross-modal correlation
- Bounded temporal window abstractions (without implicit wall-clock time)
- Pre-situation fusion clusters with deterministic semantic identifiers
- Aggregate fusion results and resource limits

CRITICAL INVARIANTS:
1. Pre-situation evidence organization ONLY: answers "When are observations related?" and
   "Which observations probably describe the same underlying event/entity?".
2. ZERO operational decision-making: does NOT produce actions, missions, goals, or dispatch commands.
3. ZERO WorldState mutation: does NOT resolve contradictions into final WorldState facts.
4. Models are frozen/immutable dataclasses with deterministic serialization and validation.
5. All collections are strictly bounded to prevent memory growth or denial-of-service.
6. IDs are deterministic (derived from semantic content / SHA-256 hashes), never random UUIDs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import math
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

from core.models.device_contract import sanitize_contract_metadata
from core.models.orchestration import GeoLocation, ModalityType, MultimodalObservation
from core.models.perception import REQUIRED_PROVENANCE_KEYS, PerceptionEvidence
from core.models.spatial_telemetry import SpatialRelationType


# ============================================================================
# 1. Enums
# ============================================================================

class TemporalRelation(str, Enum):
    """
    Categorical temporal relationship between two observation timestamps or intervals.
    """
    BEFORE = "BEFORE"
    AFTER = "AFTER"
    OVERLAPS = "OVERLAPS"
    COINCIDENT = "COINCIDENT"
    WITHIN_WINDOW = "WITHIN_WINDOW"
    SEQUENCE = "SEQUENCE"

    @classmethod
    def from_str(cls, val: Any) -> "TemporalRelation":
        if isinstance(val, cls):
            return val
        norm = str(val or "").strip().upper()
        for member in cls:
            if member.value == norm or member.name == norm:
                return member
        raise ValueError(f"Unknown TemporalRelation: '{val}'")


class EvidenceRelationType(str, Enum):
    """
    Semantic relationship between two distinct pieces of evidence or observations.
    """
    SUPPORTS = "SUPPORTS"
    CORROBORATES = "CORROBORATES"
    CONTRADICTS = "CONTRADICTS"
    PRECEDES = "PRECEDES"
    FOLLOWS = "FOLLOWS"
    OVERLAPS = "OVERLAPS"
    COLOCATED_WITH = "COLOCATED_WITH"
    SAME_ENTITY_CANDIDATE = "SAME_ENTITY_CANDIDATE"
    DUPLICATES = "DUPLICATES"

    @classmethod
    def from_str(cls, val: Any) -> "EvidenceRelationType":
        if isinstance(val, cls):
            return val
        norm = str(val or "").strip().upper()
        for member in cls:
            if member.value == norm or member.name == norm:
                return member
        raise ValueError(f"Unknown EvidenceRelationType: '{val}'")


class ModalityCompatibilityLevel(str, Enum):
    """
    Compatibility classification between two distinct observation modalities.
    """
    COMPLEMENTARY = "COMPLEMENTARY"
    COMPATIBLE = "COMPATIBLE"
    INDEPENDENT = "INDEPENDENT"
    CONFLICTING = "CONFLICTING"

    @classmethod
    def from_str(cls, val: Any) -> "ModalityCompatibilityLevel":
        if isinstance(val, cls):
            return val
        norm = str(val or "").strip().upper()
        for member in cls:
            if member.value == norm or member.name == norm:
                return member
        return cls.INDEPENDENT


class FusionFreshnessStatus(str, Enum):
    """
    Freshness classification of evidence during cross-modal fusion.
    """
    VALID = "VALID"
    STALE = "STALE"
    EXPIRED = "EXPIRED"

    @classmethod
    def from_str(cls, val: Any) -> "FusionFreshnessStatus":
        if isinstance(val, cls):
            return val
        norm = str(val or "").strip().upper()
        for member in cls:
            if member.value == norm or member.name == norm:
                return member
        return cls.VALID


# ============================================================================
# 2. Limits & Safety Bounds
# ============================================================================

@dataclass(frozen=True)
class FusionLimits:
    """
    Operational resource bounds for temporal and cross-modal fusion.
    Guarantees predictable execution time and prevents unbounded buffer allocations.
    """
    max_input_observations: int = 500
    max_evidence_relationships: int = 200
    max_clusters: int = 50
    max_evidence_per_cluster: int = 25
    max_sources_per_cluster: int = 10
    max_modalities_per_cluster: int = 8
    max_duplicate_cache: int = 500
    max_temporal_window_seconds: float = 300.0
    max_spatial_distance_meters: float = 500.0
    max_retained_history: int = 50
    max_string_length: int = 128
    freshness_threshold_seconds: float = 30.0
    expiration_threshold_seconds: float = 300.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_input_observations": self.max_input_observations,
            "max_evidence_relationships": self.max_evidence_relationships,
            "max_clusters": self.max_clusters,
            "max_evidence_per_cluster": self.max_evidence_per_cluster,
            "max_sources_per_cluster": self.max_sources_per_cluster,
            "max_modalities_per_cluster": self.max_modalities_per_cluster,
            "max_duplicate_cache": self.max_duplicate_cache,
            "max_temporal_window_seconds": self.max_temporal_window_seconds,
            "max_spatial_distance_meters": self.max_spatial_distance_meters,
            "max_retained_history": self.max_retained_history,
            "max_string_length": self.max_string_length,
            "freshness_threshold_seconds": self.freshness_threshold_seconds,
            "expiration_threshold_seconds": self.expiration_threshold_seconds,
        }

    def model_dump(self) -> Dict[str, Any]:
        return self.to_dict()

    def model_dump_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FusionLimits":
        return cls(
            max_input_observations=int(data.get("max_input_observations", 500)),
            max_evidence_relationships=int(data.get("max_evidence_relationships", 200)),
            max_clusters=int(data.get("max_clusters", 50)),
            max_evidence_per_cluster=int(data.get("max_evidence_per_cluster", 25)),
            max_sources_per_cluster=int(data.get("max_sources_per_cluster", 10)),
            max_modalities_per_cluster=int(data.get("max_modalities_per_cluster", 8)),
            max_duplicate_cache=int(data.get("max_duplicate_cache", 500)),
            max_temporal_window_seconds=float(data.get("max_temporal_window_seconds", 300.0)),
            max_spatial_distance_meters=float(data.get("max_spatial_distance_meters", 500.0)),
            max_retained_history=int(data.get("max_retained_history", 50)),
            max_string_length=int(data.get("max_string_length", 128)),
            freshness_threshold_seconds=float(data.get("freshness_threshold_seconds", 30.0)),
            expiration_threshold_seconds=float(data.get("expiration_threshold_seconds", 300.0)),
        )

    @classmethod
    def model_validate(cls, obj: Any) -> "FusionLimits":
        if isinstance(obj, cls):
            return obj
        if isinstance(obj, dict):
            return cls.from_dict(obj)
        raise TypeError(f"Cannot validate {type(obj)} as FusionLimits")


# ============================================================================
# 3. Temporal Window Abstraction
# ============================================================================

@dataclass(frozen=True)
class TemporalWindow:
    """
    Bounded temporal window abstraction for deterministic correlation.
    Strictly validates start <= end, duration >= 0, duration <= max_duration.
    Explicitly decoupled from wall-clock time: caller or simulation supplies all timestamps.
    """
    start_time: float
    end_time: float
    max_duration: float = 300.0
    source_timestamps: Tuple[float, ...] = field(default_factory=tuple)
    reference_timestamp: Optional[float] = None

    def __post_init__(self):
        st = float(self.start_time)
        et = float(self.end_time)
        md = float(self.max_duration)

        if math.isnan(st) or math.isinf(st) or st < 0.0:
            raise ValueError(f"TemporalWindow start_time must be finite and non-negative, got {st}")
        if math.isnan(et) or math.isinf(et) or et < 0.0:
            raise ValueError(f"TemporalWindow end_time must be finite and non-negative, got {et}")
        if math.isnan(md) or math.isinf(md) or md <= 0.0:
            raise ValueError(f"TemporalWindow max_duration must be finite and positive, got {md}")

        if st > et:
            raise ValueError(f"TemporalWindow start_time ({st}) cannot exceed end_time ({et})")

        duration = et - st
        if duration > md:
            raise ValueError(f"TemporalWindow duration ({duration}s) exceeds max_duration ({md}s)")

        object.__setattr__(self, "start_time", round(st, 4))
        object.__setattr__(self, "end_time", round(et, 4))
        object.__setattr__(self, "max_duration", round(md, 4))

        if self.reference_timestamp is not None:
            rt = float(self.reference_timestamp)
            if math.isnan(rt) or math.isinf(rt) or rt < 0.0:
                raise ValueError(f"TemporalWindow reference_timestamp must be finite and non-negative, got {rt}")
            object.__setattr__(self, "reference_timestamp", round(rt, 4))

        sorted_ts = tuple(sorted(float(t) for t in self.source_timestamps))
        object.__setattr__(self, "source_timestamps", sorted_ts)

    @property
    def duration(self) -> float:
        return round(self.end_time - self.start_time, 4)

    def contains(self, timestamp: float, tolerance: float = 0.0) -> bool:
        """Check if a timestamp falls within this temporal window."""
        t = float(timestamp)
        return (self.start_time - tolerance) <= t <= (self.end_time + tolerance)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "max_duration": self.max_duration,
            "source_timestamps": list(self.source_timestamps),
            "reference_timestamp": self.reference_timestamp,
        }

    def model_dump(self) -> Dict[str, Any]:
        return self.to_dict()

    def model_dump_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TemporalWindow":
        return cls(
            start_time=float(data["start_time"]),
            end_time=float(data["end_time"]),
            max_duration=float(data.get("max_duration", 300.0)),
            source_timestamps=tuple(float(t) for t in data.get("source_timestamps", ())),
            reference_timestamp=float(data["reference_timestamp"]) if data.get("reference_timestamp") is not None else None,
        )

    @classmethod
    def model_validate(cls, obj: Any) -> "TemporalWindow":
        if isinstance(obj, cls):
            return obj
        if isinstance(obj, dict):
            return cls.from_dict(obj)
        raise TypeError(f"Cannot validate {type(obj)} as TemporalWindow")

    @classmethod
    def model_validate_json(cls, json_str: str) -> "TemporalWindow":
        return cls.from_dict(json.loads(json_str))


# ============================================================================
# 4. Spatial Link & Evidence Relationship Models
# ============================================================================

@dataclass(frozen=True)
class SpatialEvidenceLink:
    """
    Immutable spatial proximity link connecting two observations or evidence items.
    """
    source_id_1: str
    source_id_2: str
    distance_meters: float
    spatial_relation: SpatialRelationType
    spatial_confidence: float = 1.0
    bearing_degrees: Optional[float] = None
    location_1: Optional[GeoLocation] = None
    location_2: Optional[GeoLocation] = None

    def __post_init__(self):
        if not self.source_id_1 or not self.source_id_2:
            raise ValueError("SpatialEvidenceLink source IDs must be non-empty strings.")

        d = float(self.distance_meters)
        if math.isnan(d) or math.isinf(d) or d < 0.0:
            raise ValueError(f"SpatialEvidenceLink distance_meters must be finite and non-negative, got {d}")
        object.__setattr__(self, "distance_meters", round(d, 2))

        conf = float(self.spatial_confidence)
        if math.isnan(conf) or math.isinf(conf) or not (0.0 <= conf <= 1.0):
            raise ValueError(f"SpatialEvidenceLink spatial_confidence must be in [0.0, 1.0], got {conf}")
        object.__setattr__(self, "spatial_confidence", round(conf, 4))

        if self.bearing_degrees is not None:
            b = float(self.bearing_degrees)
            if not math.isnan(b) and not math.isinf(b):
                object.__setattr__(self, "bearing_degrees", round(b, 2))

        rel = self.spatial_relation
        if isinstance(rel, str):
            rel = SpatialRelationType.from_str(rel)
        object.__setattr__(self, "spatial_relation", rel)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id_1": self.source_id_1,
            "source_id_2": self.source_id_2,
            "distance_meters": self.distance_meters,
            "spatial_relation": self.spatial_relation.value,
            "spatial_confidence": self.spatial_confidence,
            "bearing_degrees": self.bearing_degrees,
            "location_1": self.location_1.to_dict() if self.location_1 else None,
            "location_2": self.location_2.to_dict() if self.location_2 else None,
        }

    def model_dump(self) -> Dict[str, Any]:
        return self.to_dict()

    def model_dump_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SpatialEvidenceLink":
        l1 = GeoLocation.from_dict(data["location_1"]) if data.get("location_1") else None
        l2 = GeoLocation.from_dict(data["location_2"]) if data.get("location_2") else None
        return cls(
            source_id_1=str(data["source_id_1"]),
            source_id_2=str(data["source_id_2"]),
            distance_meters=float(data["distance_meters"]),
            spatial_relation=SpatialRelationType.from_str(data.get("spatial_relation", "NEAR")),
            spatial_confidence=float(data.get("spatial_confidence", 1.0)),
            bearing_degrees=float(data["bearing_degrees"]) if data.get("bearing_degrees") is not None else None,
            location_1=l1,
            location_2=l2,
        )

    @classmethod
    def model_validate(cls, obj: Any) -> "SpatialEvidenceLink":
        if isinstance(obj, cls):
            return obj
        if isinstance(obj, dict):
            return cls.from_dict(obj)
        raise TypeError(f"Cannot validate {type(obj)} as SpatialEvidenceLink")

    @classmethod
    def model_validate_json(cls, json_str: str) -> "SpatialEvidenceLink":
        return cls.from_dict(json.loads(json_str))


@dataclass(frozen=True)
class EvidenceRelationship:
    """
    Immutable semantic relationship between two specific pieces of evidence.
    """
    relation_type: EvidenceRelationType
    source_evidence_id: str
    target_evidence_id: str
    confidence: float = 1.0
    reason: str = ""
    temporal_relation: Optional[TemporalRelation] = None
    spatial_link: Optional[SpatialEvidenceLink] = None
    created_at: float = field(default_factory=time.time)

    def __post_init__(self):
        if not self.source_evidence_id or not self.target_evidence_id:
            raise ValueError("EvidenceRelationship evidence IDs must be non-empty strings.")

        rt = self.relation_type
        if isinstance(rt, str):
            rt = EvidenceRelationType.from_str(rt)
        object.__setattr__(self, "relation_type", rt)

        conf = float(self.confidence)
        if math.isnan(conf) or math.isinf(conf) or not (0.0 <= conf <= 1.0):
            raise ValueError(f"EvidenceRelationship confidence must be in [0.0, 1.0], got {conf}")
        object.__setattr__(self, "confidence", round(conf, 4))

        if self.temporal_relation is not None:
            tr = self.temporal_relation
            if isinstance(tr, str):
                tr = TemporalRelation.from_str(tr)
            object.__setattr__(self, "temporal_relation", tr)

        ts = float(self.created_at)
        if ts <= 0.0 or math.isnan(ts) or math.isinf(ts):
            raise ValueError(f"EvidenceRelationship created_at must be positive and finite, got {ts}")
        object.__setattr__(self, "created_at", round(ts, 4))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "relation_type": self.relation_type.value,
            "source_evidence_id": self.source_evidence_id,
            "target_evidence_id": self.target_evidence_id,
            "confidence": self.confidence,
            "reason": self.reason,
            "temporal_relation": self.temporal_relation.value if self.temporal_relation else None,
            "spatial_link": self.spatial_link.to_dict() if self.spatial_link else None,
            "created_at": self.created_at,
        }

    def model_dump(self) -> Dict[str, Any]:
        return self.to_dict()

    def model_dump_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvidenceRelationship":
        slink = SpatialEvidenceLink.from_dict(data["spatial_link"]) if data.get("spatial_link") else None
        return cls(
            relation_type=EvidenceRelationType.from_str(data["relation_type"]),
            source_evidence_id=str(data["source_evidence_id"]),
            target_evidence_id=str(data["target_evidence_id"]),
            confidence=float(data.get("confidence", 1.0)),
            reason=str(data.get("reason", "")),
            temporal_relation=TemporalRelation.from_str(data["temporal_relation"]) if data.get("temporal_relation") else None,
            spatial_link=slink,
            created_at=float(data.get("created_at", time.time())),
        )

    @classmethod
    def model_validate(cls, obj: Any) -> "EvidenceRelationship":
        if isinstance(obj, cls):
            return obj
        if isinstance(obj, dict):
            return cls.from_dict(obj)
        raise TypeError(f"Cannot validate {type(obj)} as EvidenceRelationship")

    @classmethod
    def model_validate_json(cls, json_str: str) -> "EvidenceRelationship":
        return cls.from_dict(json.loads(json_str))


# ============================================================================
# 5. Cross-Modal Correlation & Fusion Cluster Models
# ============================================================================

@dataclass(frozen=True)
class CrossModalCorrelation:
    """
    Immutable record of a validated cross-modal correlation across multiple evidence items.
    """
    correlation_id: str
    evidence_ids: Tuple[str, ...]
    observation_ids: Tuple[str, ...] = field(default_factory=tuple)
    participating_modalities: Tuple[ModalityType, ...] = field(default_factory=tuple)
    participating_sources: Tuple[str, ...] = field(default_factory=tuple)
    temporal_relationship: TemporalRelation = TemporalRelation.WITHIN_WINDOW
    spatial_relationship: Optional[SpatialRelationType] = None
    relationship_type: EvidenceRelationType = EvidenceRelationType.CORROBORATES
    confidence: float = 1.0
    source_diversity: float = 1.0  # Proportion of unique sources [0.0, 1.0]
    freshness: FusionFreshnessStatus = FusionFreshnessStatus.VALID
    provenance: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def __post_init__(self):
        if not self.correlation_id:
            raise ValueError("CrossModalCorrelation correlation_id must be a non-empty string.")
        if not self.evidence_ids and not self.observation_ids:
            raise ValueError("CrossModalCorrelation must reference at least one evidence or observation ID.")

        conf = float(self.confidence)
        if math.isnan(conf) or math.isinf(conf) or not (0.0 <= conf <= 1.0):
            raise ValueError(f"CrossModalCorrelation confidence must be in [0.0, 1.0], got {conf}")
        object.__setattr__(self, "confidence", round(conf, 4))

        div = float(self.source_diversity)
        if math.isnan(div) or math.isinf(div) or not (0.0 <= div <= 1.0):
            raise ValueError(f"CrossModalCorrelation source_diversity must be in [0.0, 1.0], got {div}")
        object.__setattr__(self, "source_diversity", round(div, 4))

        # Sort and freeze tuples for deterministic ordering
        object.__setattr__(self, "evidence_ids", tuple(sorted(self.evidence_ids)))
        object.__setattr__(self, "observation_ids", tuple(sorted(self.observation_ids)))
        object.__setattr__(self, "participating_sources", tuple(sorted(set(self.participating_sources))))

        norm_mods = tuple(
            m if isinstance(m, ModalityType) else ModalityType.from_str(m)
            for m in self.participating_modalities
        )
        object.__setattr__(self, "participating_modalities", tuple(sorted(set(norm_mods), key=lambda x: x.value)))

        object.__setattr__(self, "provenance", sanitize_contract_metadata(self.provenance))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "correlation_id": self.correlation_id,
            "evidence_ids": list(self.evidence_ids),
            "observation_ids": list(self.observation_ids),
            "participating_modalities": [m.value for m in self.participating_modalities],
            "participating_sources": list(self.participating_sources),
            "temporal_relationship": self.temporal_relationship.value,
            "spatial_relationship": self.spatial_relationship.value if self.spatial_relationship else None,
            "relationship_type": self.relationship_type.value,
            "confidence": self.confidence,
            "source_diversity": self.source_diversity,
            "freshness": self.freshness.value,
            "provenance": dict(self.provenance),
            "created_at": self.created_at,
        }

    def model_dump(self) -> Dict[str, Any]:
        return self.to_dict()

    def model_dump_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CrossModalCorrelation":
        mods = tuple(ModalityType.from_str(m) for m in data.get("participating_modalities", ()))
        srel = SpatialRelationType.from_str(data["spatial_relationship"]) if data.get("spatial_relationship") else None
        return cls(
            correlation_id=str(data["correlation_id"]),
            evidence_ids=tuple(data.get("evidence_ids", ())),
            observation_ids=tuple(data.get("observation_ids", ())),
            participating_modalities=mods,
            participating_sources=tuple(data.get("participating_sources", ())),
            temporal_relationship=TemporalRelation.from_str(data.get("temporal_relationship", "WITHIN_WINDOW")),
            spatial_relationship=srel,
            relationship_type=EvidenceRelationType.from_str(data.get("relationship_type", "CORROBORATES")),
            confidence=float(data.get("confidence", 1.0)),
            source_diversity=float(data.get("source_diversity", 1.0)),
            freshness=FusionFreshnessStatus.from_str(data.get("freshness", "VALID")),
            provenance=dict(data.get("provenance", {})),
            created_at=float(data.get("created_at", time.time())),
        )

    @classmethod
    def model_validate(cls, obj: Any) -> "CrossModalCorrelation":
        if isinstance(obj, cls):
            return obj
        if isinstance(obj, dict):
            return cls.from_dict(obj)
        raise TypeError(f"Cannot validate {type(obj)} as CrossModalCorrelation")

    @classmethod
    def model_validate_json(cls, json_str: str) -> "CrossModalCorrelation":
        return cls.from_dict(json.loads(json_str))


@dataclass(frozen=True)
class FusionCluster:
    """
    Pre-situation evidence organization grouping temporally and spatially related observations.
    CRITICAL: This is an evidence cluster, NOT a Situation.
    Cluster IDs are deterministically generated via content hash, never random UUIDs.
    """
    cluster_id: str
    evidence_ids: Tuple[str, ...]
    observation_ids: Tuple[str, ...] = field(default_factory=tuple)
    source_ids: Tuple[str, ...] = field(default_factory=tuple)
    modalities: Tuple[ModalityType, ...] = field(default_factory=tuple)
    temporal_window: Optional[TemporalWindow] = None
    relationships: Tuple[EvidenceRelationship, ...] = field(default_factory=tuple)
    confidence: float = 1.0
    source_diversity: float = 1.0
    contradiction_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.cluster_id:
            raise ValueError("FusionCluster cluster_id must be a non-empty string.")

        conf = float(self.confidence)
        if math.isnan(conf) or math.isinf(conf) or not (0.0 <= conf <= 1.0):
            raise ValueError(f"FusionCluster confidence must be in [0.0, 1.0], got {conf}")
        object.__setattr__(self, "confidence", round(conf, 4))

        object.__setattr__(self, "evidence_ids", tuple(sorted(self.evidence_ids)))
        object.__setattr__(self, "observation_ids", tuple(sorted(self.observation_ids)))
        object.__setattr__(self, "source_ids", tuple(sorted(set(self.source_ids))))

        norm_mods = tuple(
            m if isinstance(m, ModalityType) else ModalityType.from_str(m)
            for m in self.modalities
        )
        object.__setattr__(self, "modalities", tuple(sorted(set(norm_mods), key=lambda x: x.value)))
        object.__setattr__(self, "metadata", sanitize_contract_metadata(self.metadata))

    @staticmethod
    def generate_deterministic_id(
        evidence_ids: Sequence[str],
        source_ids: Sequence[str],
        start_time: float,
        end_time: float,
    ) -> str:
        """Deterministically compute cluster ID from sorted member IDs and temporal window."""
        ev_part = ",".join(sorted(evidence_ids))
        src_part = ",".join(sorted(source_ids))
        payload = f"{ev_part}|{src_part}|{start_time:.2f}|{end_time:.2f}".encode("utf-8")
        h = hashlib.sha256(payload).hexdigest()[:16]
        return f"fcluster_{h}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "evidence_ids": list(self.evidence_ids),
            "observation_ids": list(self.observation_ids),
            "source_ids": list(self.source_ids),
            "modalities": [m.value for m in self.modalities],
            "temporal_window": self.temporal_window.to_dict() if self.temporal_window else None,
            "relationships": [r.to_dict() for r in self.relationships],
            "confidence": self.confidence,
            "source_diversity": self.source_diversity,
            "contradiction_count": self.contradiction_count,
            "metadata": dict(self.metadata),
        }

    def model_dump(self) -> Dict[str, Any]:
        return self.to_dict()

    def model_dump_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FusionCluster":
        tw = TemporalWindow.from_dict(data["temporal_window"]) if data.get("temporal_window") else None
        rels = tuple(EvidenceRelationship.from_dict(r) for r in data.get("relationships", ()))
        mods = tuple(ModalityType.from_str(m) for m in data.get("modalities", ()))
        return cls(
            cluster_id=str(data["cluster_id"]),
            evidence_ids=tuple(data.get("evidence_ids", ())),
            observation_ids=tuple(data.get("observation_ids", ())),
            source_ids=tuple(data.get("source_ids", ())),
            modalities=mods,
            temporal_window=tw,
            relationships=rels,
            confidence=float(data.get("confidence", 1.0)),
            source_diversity=float(data.get("source_diversity", 1.0)),
            contradiction_count=int(data.get("contradiction_count", 0)),
            metadata=dict(data.get("metadata", {})),
        )

    @classmethod
    def model_validate(cls, obj: Any) -> "FusionCluster":
        if isinstance(obj, cls):
            return obj
        if isinstance(obj, dict):
            return cls.from_dict(obj)
        raise TypeError(f"Cannot validate {type(obj)} as FusionCluster")

    @classmethod
    def model_validate_json(cls, json_str: str) -> "FusionCluster":
        return cls.from_dict(json.loads(json_str))


# ============================================================================
# 6. Fusion Result Aggregate Model
# ============================================================================

@dataclass(frozen=True)
class FusionResult:
    """
    Immutable comprehensive result emitted by TemporalCrossModalFusionEngine.
    Organizes correlated clusters, relationships, contradictions, and normalized observations.
    """
    result_id: str
    temporal_window: TemporalWindow
    clusters: Tuple[FusionCluster, ...] = field(default_factory=tuple)
    relationships: Tuple[EvidenceRelationship, ...] = field(default_factory=tuple)
    correlations: Tuple[CrossModalCorrelation, ...] = field(default_factory=tuple)
    contradictions: Tuple[EvidenceRelationship, ...] = field(default_factory=tuple)
    duplicate_evidence_ids: Tuple[str, ...] = field(default_factory=tuple)
    normalized_observations: Tuple[MultimodalObservation, ...] = field(default_factory=tuple)
    provenance: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    semantic_hash: str = ""

    def __post_init__(self):
        if not self.result_id:
            raise ValueError("FusionResult result_id must be a non-empty string.")

        if not self.semantic_hash:
            # Deterministically compute semantic hash
            cluster_ids = ",".join(sorted(c.cluster_id for c in self.clusters))
            rel_keys = ",".join(sorted(f"{r.source_evidence_id}->{r.target_evidence_id}:{r.relation_type.value}" for r in self.relationships))
            hash_in = f"{self.result_id}|{self.temporal_window.start_time}|{self.temporal_window.end_time}|{cluster_ids}|{rel_keys}".encode("utf-8")
            shash = hashlib.sha256(hash_in).hexdigest()
            object.__setattr__(self, "semantic_hash", shash)

        object.__setattr__(self, "provenance", sanitize_contract_metadata(self.provenance))

    @staticmethod
    def generate_deterministic_result_id(window: TemporalWindow, input_count: int) -> str:
        h = hashlib.sha256(f"{window.start_time:.2f}|{window.end_time:.2f}|{input_count}".encode("utf-8")).hexdigest()[:12]
        return f"fres_{h}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "result_id": self.result_id,
            "temporal_window": self.temporal_window.to_dict(),
            "clusters": [c.to_dict() for c in self.clusters],
            "relationships": [r.to_dict() for r in self.relationships],
            "correlations": [c.to_dict() for c in self.correlations],
            "contradictions": [r.to_dict() for r in self.contradictions],
            "duplicate_evidence_ids": list(self.duplicate_evidence_ids),
            "normalized_observation_count": len(self.normalized_observations),
            "provenance": dict(self.provenance),
            "created_at": self.created_at,
            "semantic_hash": self.semantic_hash,
        }

    def model_dump(self) -> Dict[str, Any]:
        return self.to_dict()

    def model_dump_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FusionResult":
        tw = TemporalWindow.from_dict(data["temporal_window"])
        cls_list = tuple(FusionCluster.from_dict(c) for c in data.get("clusters", ()))
        rels = tuple(EvidenceRelationship.from_dict(r) for r in data.get("relationships", ()))
        corrs = tuple(CrossModalCorrelation.from_dict(c) for c in data.get("correlations", ()))
        contras = tuple(EvidenceRelationship.from_dict(r) for r in data.get("contradictions", ()))
        return cls(
            result_id=str(data["result_id"]),
            temporal_window=tw,
            clusters=cls_list,
            relationships=rels,
            correlations=corrs,
            contradictions=contras,
            duplicate_evidence_ids=tuple(data.get("duplicate_evidence_ids", ())),
            provenance=dict(data.get("provenance", {})),
            created_at=float(data.get("created_at", time.time())),
            semantic_hash=str(data.get("semantic_hash", "")),
        )

    @classmethod
    def model_validate(cls, obj: Any) -> "FusionResult":
        if isinstance(obj, cls):
            return obj
        if isinstance(obj, dict):
            return cls.from_dict(obj)
        raise TypeError(f"Cannot validate {type(obj)} as FusionResult")

    @classmethod
    def model_validate_json(cls, json_str: str) -> "FusionResult":
        return cls.from_dict(json.loads(json_str))
