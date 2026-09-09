import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from core.models.autonomy import (
    Event,
    EventCategory,
    EventPriority,
    EventProvenance,
    EventSource,
)
from core.models.world_state import Observation, StateProvenance


class ModalityType(str, Enum):
    """
    Approved modalities supported by the Phase 5.0 Central Orchestration Layer.
    Covers sensory, spatial, stateful, and intentional data streams.
    """
    TEXT = "TEXT"
    VOICE_TRANSCRIPT = "VOICE_TRANSCRIPT"
    AUDIO_EVENT = "AUDIO_EVENT"
    IMAGE = "IMAGE"
    VIDEO_FRAME = "VIDEO_FRAME"
    GPS = "GPS"
    TELEMETRY = "TELEMETRY"
    DEVICE_STATE = "DEVICE_STATE"
    WORLD_STATE = "WORLD_STATE"
    EVENT = "EVENT"
    USER_ACTION = "USER_ACTION"
    UNKNOWN = "UNKNOWN"

    @classmethod
    def from_str(cls, val: Any) -> "ModalityType":
        """Deterministic mapping with safe fallback to UNKNOWN for unrecognized modalities."""
        if isinstance(val, cls):
            return val
        s = str(val).strip().upper()
        for member in cls:
            if member.value == s or member.name == s:
                return member
        return cls.UNKNOWN


@dataclass(frozen=True)
class GeoLocation:
    """
    Immutable WGS-84 geographic coordinate representation.
    Encodes outdoor spatial coordinates: latitude, longitude, altitude, and horizontal accuracy.

    NOTE: Does not encode indoor SLAM coordinates or local cartesian frames.
    Future local coordinate reference frames (e.g. SLAM pose, local XYZ, frame_id)
    will require an architectural extension.
    """
    latitude: float
    longitude: float
    altitude: Optional[float] = None
    accuracy: Optional[float] = None

    def __post_init__(self):
        lat = float(self.latitude)
        lon = float(self.longitude)
        if not (-90.0 <= lat <= 90.0):
            raise ValueError(f"GeoLocation latitude must be in [-90.0, 90.0], got {lat}")
        if not (-180.0 <= lon <= 180.0):
            raise ValueError(f"GeoLocation longitude must be in [-180.0, 180.0], got {lon}")
        if self.accuracy is not None:
            acc = float(self.accuracy)
            if acc < 0.0:
                raise ValueError(f"GeoLocation accuracy must be non-negative, got {acc}")
        object.__setattr__(self, "latitude", lat)
        object.__setattr__(self, "longitude", lon)
        if self.altitude is not None:
            object.__setattr__(self, "altitude", float(self.altitude))
        if self.accuracy is not None:
            object.__setattr__(self, "accuracy", float(self.accuracy))

    def distance_to(self, other: "GeoLocation") -> float:
        """
        Calculate the great-circle distance in meters between two coordinates using the Haversine formula.
        """
        if not isinstance(other, GeoLocation):
            raise TypeError(f"Expected GeoLocation, got {type(other)}")
        r = 6371000.0  # Mean radius of Earth in meters
        phi1 = math.radians(self.latitude)
        phi2 = math.radians(other.latitude)
        delta_phi = math.radians(other.latitude - self.latitude)
        delta_lambda = math.radians(other.longitude - self.longitude)
        a = (
            math.sin(delta_phi / 2.0) ** 2
            + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
        )
        c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
        return r * c

    def to_dict(self) -> Dict[str, Any]:
        """Deterministic, JSON-safe dictionary representation."""
        data: Dict[str, Any] = {
            "latitude": round(self.latitude, 7),
            "longitude": round(self.longitude, 7),
        }
        if self.altitude is not None:
            data["altitude"] = round(self.altitude, 2)
        if self.accuracy is not None:
            data["accuracy"] = round(self.accuracy, 2)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GeoLocation":
        return cls(
            latitude=float(data["latitude"]),
            longitude=float(data["longitude"]),
            altitude=float(data["altitude"]) if data.get("altitude") is not None else None,
            accuracy=float(data["accuracy"]) if data.get("accuracy") is not None else None,
        )


@dataclass(frozen=True)
class MultimodalObservation:
    """
    Model-neutral, immutable multimodal observation submitted from perception, sensors,
    edge devices, audio/video pipelines, or user interactions.

    Observations represent objective raw facts or perception outputs; they NEVER directly
    mutate WorldState or execute actions.
    """
    observation_id: str
    source_id: str
    source_type: str
    modality: ModalityType
    timestamp: float
    payload: Any
    confidence: float = 1.0
    location: Optional[GeoLocation] = None
    device_id: Optional[str] = None
    correlation_id: str = ""
    causation_id: Optional[str] = None
    artifact_reference: Optional[str] = None
    expires_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.observation_id or not isinstance(self.observation_id, str):
            raise ValueError("MultimodalObservation observation_id must be a non-empty string.")
        if not self.source_id or not isinstance(self.source_id, str):
            raise ValueError("MultimodalObservation source_id must be a non-empty string.")
        if not self.source_type or not isinstance(self.source_type, str):
            raise ValueError("MultimodalObservation source_type must be a non-empty string.")

        mod = self.modality
        if isinstance(mod, str):
            mod = ModalityType.from_str(mod)
        elif not isinstance(mod, ModalityType):
            raise ValueError(f"MultimodalObservation modality must be ModalityType, got {type(self.modality)}")
        object.__setattr__(self, "modality", mod)

        ts = float(self.timestamp)
        if ts <= 0.0:
            raise ValueError(f"MultimodalObservation timestamp must be positive, got {ts}")
        object.__setattr__(self, "timestamp", ts)

        conf = float(self.confidence)
        if not (0.0 <= conf <= 1.0):
            raise ValueError(f"MultimodalObservation confidence must be in [0.0, 1.0], got {conf}")
        object.__setattr__(self, "confidence", conf)

        cid = self.correlation_id if self.correlation_id else self.observation_id
        object.__setattr__(self, "correlation_id", cid)

        if self.expires_at is not None:
            exp = float(self.expires_at)
            if exp < ts:
                raise ValueError(
                    f"MultimodalObservation expires_at ({exp}) cannot be earlier than timestamp ({ts})"
                )
            object.__setattr__(self, "expires_at", exp)

        if self.location is not None and isinstance(self.location, dict):
            object.__setattr__(self, "location", GeoLocation.from_dict(self.location))

        if self.metadata is not None:
            object.__setattr__(self, "metadata", dict(self.metadata))

    def is_expired(self, now: Optional[float] = None) -> bool:
        """
        Evaluate observation expiration status.
        Freshness is distinct from confidence.
        """
        current = now if now is not None else time.time()
        return self.expires_at is not None and current >= self.expires_at

    def get_age(self, now: Optional[float] = None) -> float:
        """Calculate elapsed seconds since observation creation."""
        current = now if now is not None else time.time()
        return max(0.0, current - self.timestamp)

    def freshness_score(self, now: Optional[float] = None, half_life_seconds: float = 60.0) -> float:
        """
        Deterministic freshness score [0.0, 1.0] calculated via exponential decay.
        Freshness remains strictly orthogonal to confidence (reliability vs recency).
        """
        if half_life_seconds <= 0.0:
            return 1.0 if self.get_age(now) == 0.0 else 0.0
        age = self.get_age(now)
        decay_constant = 0.6931471805599453 / half_life_seconds  # ln(2) / half_life
        return math.exp(-decay_constant * age)

    def to_world_state_observation(
        self,
        entity_id: str,
        property_name: str,
        value: Optional[Any] = None,
    ) -> Observation:
        """
        Bridge to Phase 4.4 WorldState Observation model.
        Allows downstream WorldState updater to ingest multimodal facts cleanly.
        """
        val = self.payload if value is None else value
        return Observation(
            observation_id=self.observation_id,
            source_id=self.source_id,
            source_type=self.source_type,
            timestamp=self.timestamp,
            entity_id=entity_id,
            property_name=property_name,
            value=val,
            confidence=self.confidence,
            expires_at=self.expires_at,
            metadata={
                **self.metadata,
                "modality": self.modality.value,
                "correlation_id": self.correlation_id,
                "causation_id": self.causation_id,
                "device_id": self.device_id,
                "artifact_reference": self.artifact_reference,
            },
        )

    def to_dict(self) -> Dict[str, Any]:
        """Deterministic, JSON-safe dictionary representation."""
        data: Dict[str, Any] = {
            "observation_id": self.observation_id,
            "source_id": self.source_id,
            "source_type": self.source_type,
            "modality": self.modality.value,
            "timestamp": self.timestamp,
            "payload": self.payload,
            "confidence": self.confidence,
            "correlation_id": self.correlation_id,
        }
        if self.location is not None:
            data["location"] = self.location.to_dict()
        if self.device_id is not None:
            data["device_id"] = self.device_id
        if self.causation_id is not None:
            data["causation_id"] = self.causation_id
        if self.artifact_reference is not None:
            data["artifact_reference"] = self.artifact_reference
        if self.expires_at is not None:
            data["expires_at"] = self.expires_at
        if self.metadata:
            data["metadata"] = dict(self.metadata)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MultimodalObservation":
        loc = GeoLocation.from_dict(data["location"]) if data.get("location") is not None else None
        return cls(
            observation_id=data["observation_id"],
            source_id=data["source_id"],
            source_type=data["source_type"],
            modality=ModalityType.from_str(data["modality"]),
            timestamp=float(data["timestamp"]),
            payload=data["payload"],
            confidence=float(data.get("confidence", 1.0)),
            location=loc,
            device_id=data.get("device_id"),
            correlation_id=data.get("correlation_id", ""),
            causation_id=data.get("causation_id"),
            artifact_reference=data.get("artifact_reference"),
            expires_at=float(data["expires_at"]) if data.get("expires_at") is not None else None,
            metadata=dict(data.get("metadata", {})),
        )


class SituationCategory(str, Enum):
    """
    Deterministic categorization of detected Situations.
    """
    ENVIRONMENTAL = "ENVIRONMENTAL"
    NAVIGATIONAL = "NAVIGATIONAL"
    OPERATIONAL = "OPERATIONAL"
    SYSTEM_HEALTH = "SYSTEM_HEALTH"
    SECURITY = "SECURITY"
    USER_INTERACTION = "USER_INTERACTION"
    ANOMALY = "ANOMALY"
    UNKNOWN = "UNKNOWN"

    @classmethod
    def from_str(cls, val: Any) -> "SituationCategory":
        if isinstance(val, cls):
            return val
        s = str(val).strip().upper()
        for member in cls:
            if member.value == s or member.name == s:
                return member
        return cls.UNKNOWN


class SituationSeverity(str, Enum):
    """
    Standard severity ranking for Situations.
    """
    INFO = "INFO"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

    @classmethod
    def from_str(cls, val: Any) -> "SituationSeverity":
        if isinstance(val, cls):
            return val
        s = str(val).strip().upper()
        for member in cls:
            if member.value == s or member.name == s:
                return member
        return cls.INFO


class SituationStatus(str, Enum):
    """
    Lifecycle status of an active or resolved Situation.
    """
    DETECTED = "DETECTED"
    ACTIVE = "ACTIVE"
    UPDATING = "UPDATING"
    RESOLVED = "RESOLVED"
    DISMISSED = "DISMISSED"
    EXPIRED = "EXPIRED"

    @classmethod
    def from_str(cls, val: Any) -> "SituationStatus":
        if isinstance(val, cls):
            return val
        s = str(val).strip().upper()
        for member in cls:
            if member.value == s or member.name == s:
                return member
        return cls.DETECTED


@dataclass(frozen=True)
class SituationEvidence:
    """
    Immutable evidence record connecting a Situation to its underlying observation(s).
    CRITICAL: Does NOT copy large multimedia buffers into the situation; references observations/artifacts instead.
    """
    evidence_id: str
    observation_id: str
    source_id: str
    modality: ModalityType
    evidence_weight: float
    timestamp: float
    concise_summary: str
    provenance: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if not self.evidence_id or not isinstance(self.evidence_id, str):
            raise ValueError("SituationEvidence evidence_id must be a non-empty string.")
        if not self.observation_id or not isinstance(self.observation_id, str):
            raise ValueError("SituationEvidence observation_id must be a non-empty string.")
        if not self.source_id or not isinstance(self.source_id, str):
            raise ValueError("SituationEvidence source_id must be a non-empty string.")
        if not self.concise_summary or not isinstance(self.concise_summary, str):
            raise ValueError("SituationEvidence concise_summary must be a non-empty string.")

        mod = self.modality
        if isinstance(mod, str):
            mod = ModalityType.from_str(mod)
        elif not isinstance(mod, ModalityType):
            raise ValueError(f"SituationEvidence modality must be ModalityType, got {type(self.modality)}")
        object.__setattr__(self, "modality", mod)

        ew = float(self.evidence_weight)
        if not (0.0 <= ew <= 1.0):
            raise ValueError(f"SituationEvidence evidence_weight must be in [0.0, 1.0], got {ew}")
        object.__setattr__(self, "evidence_weight", ew)

        ts = float(self.timestamp)
        if ts <= 0.0:
            raise ValueError(f"SituationEvidence timestamp must be positive, got {ts}")
        object.__setattr__(self, "timestamp", ts)

        if self.provenance is not None:
            object.__setattr__(self, "provenance", dict(self.provenance))

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "evidence_id": self.evidence_id,
            "observation_id": self.observation_id,
            "source_id": self.source_id,
            "modality": self.modality.value,
            "evidence_weight": self.evidence_weight,
            "timestamp": self.timestamp,
            "concise_summary": self.concise_summary,
        }
        if self.provenance:
            data["provenance"] = dict(self.provenance)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SituationEvidence":
        return cls(
            evidence_id=data["evidence_id"],
            observation_id=data["observation_id"],
            source_id=data["source_id"],
            modality=ModalityType.from_str(data["modality"]),
            evidence_weight=float(data["evidence_weight"]),
            timestamp=float(data["timestamp"]),
            concise_summary=data["concise_summary"],
            provenance=dict(data.get("provenance", {})) if data.get("provenance") else None,
        )


@dataclass(frozen=True)
class Situation:
    """
    An intermediate semantic interpretation of related observations across modalities.

    A Situation is NOT:
    - WorldState (does not store authoritative entity facts)
    - Event (does not directly trigger autonomy workflows without classification)
    - Anticipation (does not predict future risks/probabilities)
    - Goal / Action / ToolCall (contains ZERO executable instructions)
    """
    situation_id: str
    category: SituationCategory
    title: str
    description: str
    severity: SituationSeverity
    confidence: float
    status: SituationStatus
    involved_entities: Tuple[str, ...]
    supporting_evidence: Tuple[SituationEvidence, ...]
    location: Optional[GeoLocation] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    valid_until: Optional[float] = None
    related_event_ids: Tuple[str, ...] = ()
    related_goal_ids: Tuple[str, ...] = ()
    correlation_id: str = ""
    causation_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.situation_id or not isinstance(self.situation_id, str):
            raise ValueError("Situation situation_id must be a non-empty string.")
        if not self.title or not isinstance(self.title, str):
            raise ValueError("Situation title must be a non-empty string.")

        cat = self.category
        if isinstance(cat, str):
            cat = SituationCategory.from_str(cat)
        elif not isinstance(cat, SituationCategory):
            raise ValueError(f"Situation category must be SituationCategory, got {type(self.category)}")
        object.__setattr__(self, "category", cat)

        sev = self.severity
        if isinstance(sev, str):
            sev = SituationSeverity.from_str(sev)
        elif not isinstance(sev, SituationSeverity):
            raise ValueError(f"Situation severity must be SituationSeverity, got {type(self.severity)}")
        object.__setattr__(self, "severity", sev)

        stat = self.status
        if isinstance(stat, str):
            stat = SituationStatus.from_str(stat)
        elif not isinstance(stat, SituationStatus):
            raise ValueError(f"Situation status must be SituationStatus, got {type(self.status)}")
        object.__setattr__(self, "status", stat)

        conf = float(self.confidence)
        if not (0.0 <= conf <= 1.0):
            raise ValueError(f"Situation confidence must be in [0.0, 1.0], got {conf}")
        object.__setattr__(self, "confidence", conf)

        c_at = float(self.created_at)
        u_at = float(self.updated_at)
        if c_at <= 0.0:
            raise ValueError(f"Situation created_at must be positive, got {c_at}")
        if u_at < c_at:
            raise ValueError(f"Situation updated_at ({u_at}) cannot be before created_at ({c_at})")
        object.__setattr__(self, "created_at", c_at)
        object.__setattr__(self, "updated_at", u_at)

        if self.valid_until is not None:
            vu = float(self.valid_until)
            if vu < c_at:
                raise ValueError(f"Situation valid_until ({vu}) cannot be before created_at ({c_at})")
            object.__setattr__(self, "valid_until", vu)

        cid = self.correlation_id if self.correlation_id else self.situation_id
        object.__setattr__(self, "correlation_id", cid)

        # Ensure collections are immutable tuples
        entities = tuple(self.involved_entities) if self.involved_entities else ()
        object.__setattr__(self, "involved_entities", entities)

        evidence_list = []
        if self.supporting_evidence:
            for item in self.supporting_evidence:
                if isinstance(item, dict):
                    evidence_list.append(SituationEvidence.from_dict(item))
                elif isinstance(item, SituationEvidence):
                    evidence_list.append(item)
                else:
                    raise ValueError(f"Invalid evidence type in supporting_evidence: {type(item)}")
        object.__setattr__(self, "supporting_evidence", tuple(evidence_list))

        r_events = tuple(self.related_event_ids) if self.related_event_ids else ()
        object.__setattr__(self, "related_event_ids", r_events)

        r_goals = tuple(self.related_goal_ids) if self.related_goal_ids else ()
        object.__setattr__(self, "related_goal_ids", r_goals)

        if self.location is not None and isinstance(self.location, dict):
            object.__setattr__(self, "location", GeoLocation.from_dict(self.location))

        if self.metadata is not None:
            object.__setattr__(self, "metadata", dict(self.metadata))

    def is_valid(self, now: Optional[float] = None) -> bool:
        """
        Check if situation is active and within its valid time window.
        """
        current = now if now is not None else time.time()
        if self.status in (SituationStatus.RESOLVED, SituationStatus.DISMISSED, SituationStatus.EXPIRED):
            return False
        if self.valid_until is not None and current > self.valid_until:
            return False
        return True

    def to_autonomy_event(self, priority: Optional[EventPriority] = None) -> Event:
        """
        Bridge Situation into a Phase 4.5 Autonomy Event.
        Preserves complete correlation and provenance chains.
        """
        if priority is not None:
            ev_priority = priority
        else:
            prio_map = {
                SituationSeverity.CRITICAL: EventPriority.CRITICAL,
                SituationSeverity.HIGH: EventPriority.HIGH,
                SituationSeverity.MEDIUM: EventPriority.NORMAL,
                SituationSeverity.LOW: EventPriority.LOW,
                SituationSeverity.INFO: EventPriority.LOW,
            }
            ev_priority = prio_map.get(self.severity, EventPriority.NORMAL)

        payload = {
            "situation_id": self.situation_id,
            "title": self.title,
            "description": self.description,
            "category": self.category.value,
            "severity": self.severity.value,
            "confidence": self.confidence,
            "status": self.status.value,
            "involved_entities": list(self.involved_entities),
            "evidence_count": len(self.supporting_evidence),
        }

        provenance = EventProvenance(
            source_id=self.situation_id,
            source_type=EventSource.SITUATION,
            origin_timestamp=self.created_at,
            correlation_id=self.correlation_id,
            metadata={
                "causation_id": self.causation_id,
                "situation_category": self.category.value,
            },
        )

        return Event(
            event_id=f"ev_sit_{self.situation_id}",
            source=EventSource.SITUATION,
            event_type=f"SITUATION_{self.category.value}",
            priority=ev_priority,
            timestamp=self.updated_at,
            payload=payload,
            provenance=provenance,
            correlation_id=self.correlation_id,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Deterministic, JSON-safe representation."""
        data: Dict[str, Any] = {
            "situation_id": self.situation_id,
            "category": self.category.value,
            "title": self.title,
            "description": self.description,
            "severity": self.severity.value,
            "confidence": self.confidence,
            "status": self.status.value,
            "involved_entities": list(self.involved_entities),
            "supporting_evidence": [e.to_dict() for e in self.supporting_evidence],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "correlation_id": self.correlation_id,
        }
        if self.location is not None:
            data["location"] = self.location.to_dict()
        if self.valid_until is not None:
            data["valid_until"] = self.valid_until
        if self.related_event_ids:
            data["related_event_ids"] = list(self.related_event_ids)
        if self.related_goal_ids:
            data["related_goal_ids"] = list(self.related_goal_ids)
        if self.causation_id is not None:
            data["causation_id"] = self.causation_id
        if self.metadata:
            data["metadata"] = dict(self.metadata)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Situation":
        loc = GeoLocation.from_dict(data["location"]) if data.get("location") is not None else None
        evidence = tuple(
            SituationEvidence.from_dict(e) for e in data.get("supporting_evidence", [])
        )
        return cls(
            situation_id=data["situation_id"],
            category=SituationCategory.from_str(data["category"]),
            title=data["title"],
            description=data.get("description", ""),
            severity=SituationSeverity.from_str(data["severity"]),
            confidence=float(data["confidence"]),
            status=SituationStatus.from_str(data["status"]),
            involved_entities=tuple(data.get("involved_entities", ())),
            supporting_evidence=evidence,
            location=loc,
            created_at=float(data.get("created_at", time.time())),
            updated_at=float(data.get("updated_at", time.time())),
            valid_until=float(data["valid_until"]) if data.get("valid_until") is not None else None,
            related_event_ids=tuple(data.get("related_event_ids", ())),
            related_goal_ids=tuple(data.get("related_goal_ids", ())),
            correlation_id=data.get("correlation_id", ""),
            causation_id=data.get("causation_id"),
            metadata=dict(data.get("metadata", {})),
        )


class DeviceType(str, Enum):
    """
    Hardware-independent classification of edge participants.
    """
    EDGE_COMPUTE = "EDGE_COMPUTE"
    MICROCONTROLLER = "MICROCONTROLLER"
    SMART_GLASSES = "SMART_GLASSES"
    ROBOT_GROUND = "ROBOT_GROUND"
    DRONE_AERIAL = "DRONE_AERIAL"
    STATIONARY_SENSOR = "STATIONARY_SENSOR"
    ACTUATOR = "ACTUATOR"
    MOBILE_DEVICE = "MOBILE_DEVICE"
    VIRTUAL_SIMULATOR = "VIRTUAL_SIMULATOR"
    UNKNOWN = "UNKNOWN"

    @classmethod
    def from_str(cls, val: Any) -> "DeviceType":
        if isinstance(val, cls):
            return val
        s = str(val).strip().upper()
        for member in cls:
            if member.value == s or member.name == s:
                return member
        return cls.UNKNOWN


class ConnectivityStatus(str, Enum):
    """
    Standard device connectivity status.
    """
    ONLINE = "ONLINE"
    OFFLINE = "OFFLINE"
    DEGRADED = "DEGRADED"
    CONNECTING = "CONNECTING"
    DISCONNECTED = "DISCONNECTED"
    UNKNOWN = "UNKNOWN"

    @classmethod
    def from_str(cls, val: Any) -> "ConnectivityStatus":
        if isinstance(val, cls):
            return val
        s = str(val).strip().upper()
        for member in cls:
            if member.value == s or member.name == s:
                return member
        return cls.UNKNOWN


@dataclass(frozen=True)
class DeviceCapabilityDescriptor:
    """
    Immutable descriptor of a capability claimed by an edge device.

    CRITICAL ARCHITECTURAL BOUNDARY:
    This descriptor ONLY declares capability schema and operational constraints.
    It contains ZERO execution logic, ZERO hardware bindings, and CANNOT execute actions.
    Execution authority remains exclusively with ToolOrchestrator.
    """
    capability_name: str
    action_name: str
    parameters_schema: Optional[Dict[str, Any]] = None
    is_reversible: bool = False
    requires_confirmation: bool = False
    rate_limit_hz: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.capability_name or not isinstance(self.capability_name, str):
            raise ValueError("DeviceCapabilityDescriptor capability_name must be a non-empty string.")
        if not self.action_name or not isinstance(self.action_name, str):
            raise ValueError("DeviceCapabilityDescriptor action_name must be a non-empty string.")
        if self.rate_limit_hz is not None:
            rl = float(self.rate_limit_hz)
            if rl <= 0.0:
                raise ValueError(f"DeviceCapabilityDescriptor rate_limit_hz must be positive, got {rl}")
            object.__setattr__(self, "rate_limit_hz", rl)
        if self.parameters_schema is not None:
            object.__setattr__(self, "parameters_schema", dict(self.parameters_schema))
        if self.metadata is not None:
            object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "capability_name": self.capability_name,
            "action_name": self.action_name,
            "is_reversible": self.is_reversible,
            "requires_confirmation": self.requires_confirmation,
        }
        if self.parameters_schema:
            data["parameters_schema"] = dict(self.parameters_schema)
        if self.rate_limit_hz is not None:
            data["rate_limit_hz"] = self.rate_limit_hz
        if self.metadata:
            data["metadata"] = dict(self.metadata)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DeviceCapabilityDescriptor":
        return cls(
            capability_name=data["capability_name"],
            action_name=data["action_name"],
            parameters_schema=dict(data.get("parameters_schema", {})) if data.get("parameters_schema") else None,
            is_reversible=bool(data.get("is_reversible", False)),
            requires_confirmation=bool(data.get("requires_confirmation", False)),
            rate_limit_hz=float(data["rate_limit_hz"]) if data.get("rate_limit_hz") is not None else None,
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class DeviceIdentity:
    """
    Immutable, hardware-independent identity and capability profile of an edge participant.

    DOES NOT embed:
    - Serial protocol / UART configuration / baud rates
    - GPIO pins / registers / memory offsets
    - MAVLink / ROS2 / ESP32 register specifics
    - Direct hardware drivers
    """
    device_id: str
    device_type: DeviceType
    display_name: str
    firmware_version: Optional[str] = None
    capabilities: Tuple[DeviceCapabilityDescriptor, ...] = ()
    home_location: Optional[GeoLocation] = None
    is_simulation: bool = False
    registered_at: float = field(default_factory=time.time)
    last_heartbeat: Optional[float] = None
    connectivity_status: ConnectivityStatus = ConnectivityStatus.UNKNOWN
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.device_id or not isinstance(self.device_id, str):
            raise ValueError("DeviceIdentity device_id must be a non-empty string.")
        if not self.display_name or not isinstance(self.display_name, str):
            raise ValueError("DeviceIdentity display_name must be a non-empty string.")

        dt = self.device_type
        if isinstance(dt, str):
            dt = DeviceType.from_str(dt)
        elif not isinstance(dt, DeviceType):
            raise ValueError(f"DeviceIdentity device_type must be DeviceType, got {type(self.device_type)}")
        object.__setattr__(self, "device_type", dt)

        cs = self.connectivity_status
        if isinstance(cs, str):
            cs = ConnectivityStatus.from_str(cs)
        elif not isinstance(cs, ConnectivityStatus):
            raise ValueError(f"DeviceIdentity connectivity_status must be ConnectivityStatus, got {type(self.connectivity_status)}")
        object.__setattr__(self, "connectivity_status", cs)

        r_at = float(self.registered_at)
        if r_at <= 0.0:
            raise ValueError(f"DeviceIdentity registered_at must be positive, got {r_at}")
        object.__setattr__(self, "registered_at", r_at)

        if self.last_heartbeat is not None:
            hb = float(self.last_heartbeat)
            if hb < 0.0:
                raise ValueError(f"DeviceIdentity last_heartbeat must be non-negative, got {hb}")
            object.__setattr__(self, "last_heartbeat", hb)

        # Capabilities as immutable tuple
        caps = []
        if self.capabilities:
            for item in self.capabilities:
                if isinstance(item, dict):
                    caps.append(DeviceCapabilityDescriptor.from_dict(item))
                elif isinstance(item, DeviceCapabilityDescriptor):
                    caps.append(item)
                else:
                    raise ValueError(f"Invalid capability type in capabilities: {type(item)}")
        object.__setattr__(self, "capabilities", tuple(caps))

        if self.home_location is not None and isinstance(self.home_location, dict):
            object.__setattr__(self, "home_location", GeoLocation.from_dict(self.home_location))

        if self.metadata is not None:
            object.__setattr__(self, "metadata", dict(self.metadata))

    def has_capability(self, capability_name: str) -> bool:
        """Check if device declares a specific capability."""
        return any(c.capability_name == capability_name for c in self.capabilities)

    def get_capability(self, capability_name: str) -> Optional[DeviceCapabilityDescriptor]:
        """Retrieve capability descriptor by name if declared."""
        for c in self.capabilities:
            if c.capability_name == capability_name:
                return c
        return None

    def is_online(self) -> bool:
        """Check if device connectivity status is ONLINE."""
        return self.connectivity_status == ConnectivityStatus.ONLINE

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "device_id": self.device_id,
            "device_type": self.device_type.value,
            "display_name": self.display_name,
            "is_simulation": self.is_simulation,
            "registered_at": self.registered_at,
            "connectivity_status": self.connectivity_status.value,
            "capabilities": [c.to_dict() for c in self.capabilities],
        }
        if self.firmware_version is not None:
            data["firmware_version"] = self.firmware_version
        if self.home_location is not None:
            data["home_location"] = self.home_location.to_dict()
        if self.last_heartbeat is not None:
            data["last_heartbeat"] = self.last_heartbeat
        if self.metadata:
            data["metadata"] = dict(self.metadata)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DeviceIdentity":
        loc = GeoLocation.from_dict(data["home_location"]) if data.get("home_location") is not None else None
        caps = tuple(
            DeviceCapabilityDescriptor.from_dict(c) for c in data.get("capabilities", [])
        )
        return cls(
            device_id=data["device_id"],
            device_type=DeviceType.from_str(data["device_type"]),
            display_name=data["display_name"],
            firmware_version=data.get("firmware_version"),
            capabilities=caps,
            home_location=loc,
            is_simulation=bool(data.get("is_simulation", False)),
            registered_at=float(data.get("registered_at", time.time())),
            last_heartbeat=float(data["last_heartbeat"]) if data.get("last_heartbeat") is not None else None,
            connectivity_status=ConnectivityStatus.from_str(data.get("connectivity_status", "UNKNOWN")),
            metadata=dict(data.get("metadata", {})),
        )
