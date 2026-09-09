"""ATLAS Phase 6.5c — Spatial & Telemetry Perception Domain Models.

Establishes transport-neutral, immutable perception models for:
- Spatial position (WGS-84 coordinates, altitude, accuracy, heading, speed)
- Relative position and spatial relationships (distance, bearing, relative altitude)
- Navigation observations (movement state, path references)
- Device telemetry metrics (battery, temperature, signal quality, etc.)
- Device health assessment and telemetry quality/freshness evaluation
- Resource limits and safety bounds

CRITICAL INVARIANTS:
1. Model-neutral & hardware-neutral: ZERO ROS2, MAVLink, GPIO, or serial dependencies.
2. Perception does NOT reason, plan missions, create goals, mutate WorldState, or execute tools.
3. Models are frozen/immutable dataclasses with deterministic serialization and validation.
4. Confidence is strictly bounded in [0.0, 1.0] and rejects NaN/Infinity.
5. All observations MUST carry complete provenance.
6. Secrets, tokens, and credentials are NEVER serialized or leaked.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
import math
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

from core.models.device_contract import (
    ConnectivityStatus,
    DeviceHealthStatus,
    sanitize_contract_metadata,
)
from core.models.orchestration import GeoLocation, ModalityType
from core.models.perception import (
    PerceptionLimits,
    REQUIRED_PROVENANCE_KEYS,
)


# ============================================================================
# 1. Enums
# ============================================================================

class TelemetryQuality(str, Enum):
    """
    Quality classification for telemetry data, independent of confidence.
    Distinguishes whether data is fresh, degraded, incomplete, or corrupted.
    """
    VALID = "VALID"
    STALE = "STALE"
    PARTIAL = "PARTIAL"
    INVALID = "INVALID"

    @classmethod
    def from_str(cls, val: Any) -> "TelemetryQuality":
        if isinstance(val, cls):
            return val
        norm = str(val or "").strip().upper()
        for member in cls:
            if member.value == norm or member.name == norm:
                return member
        return cls.INVALID


class SpatialRelationType(str, Enum):
    """
    Semantic spatial relationships between entities or regions.
    """
    SAME_LOCATION = "SAME_LOCATION"
    NEAR = "NEAR"
    NEARBY = "NEARBY"
    VICINITY = "VICINITY"
    DISTANT = "DISTANT"
    REMOTE = "REMOTE"
    FAR = "FAR"
    AHEAD = "AHEAD"
    BEHIND = "BEHIND"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    ABOVE = "ABOVE"
    BELOW = "BELOW"

    @classmethod
    def from_str(cls, val: Any) -> "SpatialRelationType":
        if isinstance(val, cls):
            return val
        norm = str(val or "").strip().upper()
        for member in cls:
            if member.value == norm or member.name == norm:
                return member
        raise ValueError(f"Unknown SpatialRelationType: '{val}'")


class GeofenceRelationType(str, Enum):
    """
    Geometric relationship of a coordinate to a defined geofenced area.
    """
    INSIDE = "INSIDE"
    OUTSIDE = "OUTSIDE"
    BOUNDARY = "BOUNDARY"
    APPROACHING = "APPROACHING"

    @classmethod
    def from_str(cls, val: Any) -> "GeofenceRelationType":
        if isinstance(val, cls):
            return val
        norm = str(val or "").strip().upper()
        for member in cls:
            if member.value == norm or member.name == norm:
                return member
        return cls.OUTSIDE


class MovementState(str, Enum):
    """
    Kinematic movement state of an observed entity.
    """
    STATIONARY = "STATIONARY"
    MOVING = "MOVING"
    ACCELERATING = "ACCELERATING"
    DECELERATING = "DECELERATING"
    UNKNOWN = "UNKNOWN"

    @classmethod
    def from_str(cls, val: Any) -> "MovementState":
        if isinstance(val, cls):
            return val
        norm = str(val or "").strip().upper()
        for member in cls:
            if member.value == norm or member.name == norm:
                return member
        return cls.UNKNOWN


# Standard supported unit vocabulary
VALID_TELEMETRY_UNITS: Set[str] = {
    "percent", "%",
    "celsius", "C",
    "mps", "m/s",
    "deg", "degrees",
    "meters", "m",
    "ratio",
    "volts", "V",
    "amperes", "A",
    "dbm", "dBm",
    "hpa", "hPa",
    "count",
    "none",
    "fps",
}


# ============================================================================
# 2. Limits & Safety Bounds
# ============================================================================

@dataclass(frozen=True)
class SpatialTelemetryLimits:
    """
    Operational resource and payload bounds for spatial and telemetry perception.
    Prevents unbounded memory growth and buffer flooding.
    """
    max_metrics_per_result: int = 50
    max_attributes_per_metric: int = 20
    max_spatial_relations: int = 20
    max_geofences: int = 20
    max_tracked_sources: int = 50
    max_history_entries: int = 10
    max_staleness_seconds: float = 300.0
    max_batch_size: int = 50
    max_string_length: int = 128
    max_route_ref_length: int = 256

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_metrics_per_result": self.max_metrics_per_result,
            "max_attributes_per_metric": self.max_attributes_per_metric,
            "max_spatial_relations": self.max_spatial_relations,
            "max_geofences": self.max_geofences,
            "max_tracked_sources": self.max_tracked_sources,
            "max_history_entries": self.max_history_entries,
            "max_staleness_seconds": self.max_staleness_seconds,
            "max_batch_size": self.max_batch_size,
            "max_string_length": self.max_string_length,
            "max_route_ref_length": self.max_route_ref_length,
        }

    model_dump = to_dict

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SpatialTelemetryLimits":
        return cls(
            max_metrics_per_result=int(data.get("max_metrics_per_result", 50)),
            max_attributes_per_metric=int(data.get("max_attributes_per_metric", 20)),
            max_spatial_relations=int(data.get("max_spatial_relations", 20)),
            max_geofences=int(data.get("max_geofences", 20)),
            max_tracked_sources=int(data.get("max_tracked_sources", 50)),
            max_history_entries=int(data.get("max_history_entries", 10)),
            max_staleness_seconds=float(data.get("max_staleness_seconds", 300.0)),
            max_batch_size=int(data.get("max_batch_size", 50)),
            max_string_length=int(data.get("max_string_length", 128)),
            max_route_ref_length=int(data.get("max_route_ref_length", 256)),
        )

    model_validate = from_dict


# ============================================================================
# 3. Telemetry Metric & Health Models
# ============================================================================

@dataclass(frozen=True)
class TelemetryMetric:
    """
    Immutable representation of a single normalized telemetry measurement.
    """
    name: str
    value: Union[float, int, str, bool]
    unit: str = ""
    raw_name: Optional[str] = None
    quality: TelemetryQuality = TelemetryQuality.VALID
    confidence: float = 1.0
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self):
        if not self.name or not isinstance(self.name, str):
            raise ValueError("TelemetryMetric name must be a non-empty string.")
        if len(self.name) > 64:
            raise ValueError(f"TelemetryMetric name exceeds max length (64): {self.name}")

        conf = float(self.confidence)
        if math.isnan(conf) or math.isinf(conf) or not (0.0 <= conf <= 1.0):
            raise ValueError(f"TelemetryMetric confidence must be a finite number in [0.0, 1.0], got {conf}")
        object.__setattr__(self, "confidence", round(conf, 4))

        ts = float(self.timestamp)
        if ts <= 0.0 or math.isnan(ts) or math.isinf(ts):
            raise ValueError(f"TelemetryMetric timestamp must be positive and finite, got {ts}")
        object.__setattr__(self, "timestamp", ts)

        q = self.quality
        if isinstance(q, str):
            q = TelemetryQuality.from_str(q)
        object.__setattr__(self, "quality", q)

        # Validate finite numeric values
        if isinstance(self.value, (int, float)) and not isinstance(self.value, bool):
            val_f = float(self.value)
            if math.isnan(val_f) or math.isinf(val_f):
                raise ValueError(f"TelemetryMetric value cannot be NaN or Inf, got {val_f}")

        # Standardize unit
        norm_unit = str(self.unit or "").strip()
        object.__setattr__(self, "unit", norm_unit)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "name": self.name,
            "value": round(self.value, 4) if isinstance(self.value, float) else self.value,
            "unit": self.unit,
            "quality": self.quality.value,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
        }
        if self.raw_name is not None:
            d["raw_name"] = self.raw_name
        return d

    def model_dump(self) -> Dict[str, Any]:
        return self.to_dict()

    def model_dump_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TelemetryMetric":
        return cls(
            name=str(data["name"]),
            value=data["value"],
            unit=str(data.get("unit", "")),
            raw_name=data.get("raw_name"),
            quality=TelemetryQuality.from_str(data.get("quality", "VALID")),
            confidence=float(data.get("confidence", 1.0)),
            timestamp=float(data.get("timestamp", time.time())),
        )

    @classmethod
    def model_validate(cls, obj: Any) -> "TelemetryMetric":
        if isinstance(obj, cls):
            return obj
        if isinstance(obj, dict):
            return cls.from_dict(obj)
        raise TypeError(f"Cannot validate {type(obj)} as TelemetryMetric")

    @classmethod
    def model_validate_json(cls, json_str: str) -> "TelemetryMetric":
        return cls.from_dict(json.loads(json_str))


@dataclass(frozen=True)
class TelemetryHealth:
    """
    Immutable semantic health assessment for an edge device.
    Summarizes connectivity, status, battery, temperature, cpu usage, and signal quality.
    """
    health_status: Union[str, DeviceHealthStatus] = DeviceHealthStatus.HEALTHY
    connectivity: Union[str, ConnectivityStatus] = ConnectivityStatus.ONLINE
    battery_percent: Optional[float] = None
    temperature_celsius: Optional[float] = None
    signal_quality: Optional[float] = None
    cpu_usage_percent: Optional[float] = None
    evaluated_at: float = field(default_factory=time.time)

    def __post_init__(self):
        stat = self.health_status
        if isinstance(stat, str):
            stat = stat.upper()
            try:
                stat = DeviceHealthStatus.from_str(stat)
            except Exception:
                pass
        object.__setattr__(self, "health_status", stat.value if hasattr(stat, "value") else str(stat))

        conn = self.connectivity
        if isinstance(conn, str):
            conn = conn.upper()
            try:
                conn = ConnectivityStatus.from_str(conn)
            except Exception:
                pass
        object.__setattr__(self, "connectivity", conn.value if hasattr(conn, "value") else str(conn))

        if self.battery_percent is not None:
            bp = float(self.battery_percent)
            if math.isnan(bp) or math.isinf(bp) or not (0.0 <= bp <= 100.0):
                raise ValueError(f"TelemetryHealth battery_percent must be in [0.0, 100.0], got {bp}")
            object.__setattr__(self, "battery_percent", round(bp, 2))

        if self.temperature_celsius is not None:
            tc = float(self.temperature_celsius)
            if math.isnan(tc) or math.isinf(tc):
                raise ValueError(f"TelemetryHealth temperature_celsius must be finite, got {tc}")
            object.__setattr__(self, "temperature_celsius", round(tc, 2))

        if self.signal_quality is not None:
            sq = float(self.signal_quality)
            if math.isnan(sq) or math.isinf(sq) or not (0.0 <= sq <= 100.0):
                raise ValueError(f"TelemetryHealth signal_quality must be in [0.0, 100.0], got {sq}")
            object.__setattr__(self, "signal_quality", round(sq, 2))

        if self.cpu_usage_percent is not None:
            cp = float(self.cpu_usage_percent)
            if math.isnan(cp) or math.isinf(cp) or not (0.0 <= cp <= 100.0):
                raise ValueError(f"TelemetryHealth cpu_usage_percent must be in [0.0, 100.0], got {cp}")
            object.__setattr__(self, "cpu_usage_percent", round(cp, 2))

        ev_ts = float(self.evaluated_at)
        if ev_ts <= 0.0 or math.isnan(ev_ts) or math.isinf(ev_ts):
            raise ValueError(f"TelemetryHealth evaluated_at must be positive and finite, got {ev_ts}")
        object.__setattr__(self, "evaluated_at", ev_ts)

    @property
    def status(self) -> str:
        return self.health_status

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "health_status": self.health_status,
            "connectivity": self.connectivity,
            "evaluated_at": self.evaluated_at,
        }
        if self.battery_percent is not None:
            data["battery_percent"] = self.battery_percent
        if self.temperature_celsius is not None:
            data["temperature_celsius"] = self.temperature_celsius
        if self.signal_quality is not None:
            data["signal_quality"] = self.signal_quality
        if self.cpu_usage_percent is not None:
            data["cpu_usage_percent"] = self.cpu_usage_percent
        return data

    def model_dump(self) -> Dict[str, Any]:
        return self.to_dict()

    def model_dump_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TelemetryHealth":
        return cls(
            health_status=data.get("health_status", data.get("status", "HEALTHY")),
            connectivity=data.get("connectivity", "ONLINE"),
            battery_percent=float(data["battery_percent"]) if data.get("battery_percent") is not None else None,
            temperature_celsius=float(data["temperature_celsius"]) if data.get("temperature_celsius") is not None else None,
            signal_quality=float(data["signal_quality"]) if data.get("signal_quality") is not None else None,
            cpu_usage_percent=float(data["cpu_usage_percent"]) if data.get("cpu_usage_percent") is not None else None,
            evaluated_at=float(data.get("evaluated_at", time.time())),
        )

    @classmethod
    def model_validate(cls, obj: Any) -> "TelemetryHealth":
        if isinstance(obj, cls):
            return obj
        if isinstance(obj, dict):
            return cls.from_dict(obj)
        raise TypeError(f"Cannot validate {type(obj)} as TelemetryHealth")

    @classmethod
    def model_validate_json(cls, json_str: str) -> "TelemetryHealth":
        return cls.from_dict(json.loads(json_str))


@dataclass(frozen=True)
class TelemetryObservation:
    """
    Immutable structured telemetry observation for perception-level reporting.
    """
    device_id: str = "default_device"
    product_type: str = "general"
    health: TelemetryHealth = field(default_factory=TelemetryHealth)
    metrics: Sequence[TelemetryMetric] = field(default_factory=tuple)
    quality: TelemetryQuality = TelemetryQuality.VALID
    age_seconds: float = 0.0
    is_stale: bool = False
    source_id: Optional[str] = None
    telemetry_type: str = "telemetry"
    timestamp: float = field(default_factory=time.time)
    provenance: Dict[str, Any] = field(default_factory=dict)
    correlation_id: str = ""
    causation_id: Optional[str] = None

    def __post_init__(self):
        s_id = self.source_id or self.device_id
        if not s_id or not isinstance(s_id, str):
            raise ValueError("TelemetryObservation source_id/device_id must be a non-empty string.")
        object.__setattr__(self, "source_id", s_id)
        object.__setattr__(self, "device_id", s_id)

        ts = float(self.timestamp)
        if ts <= 0.0 or math.isnan(ts) or math.isinf(ts):
            raise ValueError(f"TelemetryObservation timestamp must be positive and finite, got {ts}")
        object.__setattr__(self, "timestamp", ts)

        q = self.quality
        if isinstance(q, str):
            q = TelemetryQuality.from_str(q)
        object.__setattr__(self, "quality", q)

        limits = SpatialTelemetryLimits()
        if len(self.metrics) > limits.max_metrics_per_result:
            raise ValueError(f"TelemetryObservation metrics count ({len(self.metrics)}) exceeds limit ({limits.max_metrics_per_result})")

        object.__setattr__(self, "metrics", tuple(self.metrics))
        object.__setattr__(self, "provenance", sanitize_contract_metadata(self.provenance))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "device_id": self.device_id,
            "product_type": self.product_type,
            "source_id": self.source_id,
            "telemetry_type": self.telemetry_type,
            "metrics": [m.to_dict() for m in self.metrics],
            "health": self.health.to_dict(),
            "quality": self.quality.value,
            "age_seconds": self.age_seconds,
            "is_stale": self.is_stale,
            "timestamp": self.timestamp,
            "provenance": sanitize_contract_metadata(self.provenance),
            "correlation_id": self.correlation_id,
            "causation_id": self.causation_id,
        }

    def model_dump(self) -> Dict[str, Any]:
        return self.to_dict()

    def model_dump_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TelemetryObservation":
        metrics_list = tuple(TelemetryMetric.from_dict(m) for m in data.get("metrics", ()))
        hlth = TelemetryHealth.from_dict(data["health"]) if "health" in data else TelemetryHealth()
        return cls(
            device_id=str(data.get("device_id", data.get("source_id", "default_device"))),
            product_type=str(data.get("product_type", "general")),
            source_id=str(data.get("source_id", data.get("device_id", "default_device"))),
            telemetry_type=str(data.get("telemetry_type", "telemetry")),
            metrics=metrics_list,
            health=hlth,
            quality=TelemetryQuality.from_str(data.get("quality", "VALID")),
            age_seconds=float(data.get("age_seconds", 0.0)),
            is_stale=bool(data.get("is_stale", False)),
            timestamp=float(data.get("timestamp", time.time())),
            provenance=dict(data.get("provenance", {})),
            correlation_id=str(data.get("correlation_id", "")),
            causation_id=str(data["causation_id"]) if data.get("causation_id") else None,
        )

    @classmethod
    def model_validate(cls, obj: Any) -> "TelemetryObservation":
        if isinstance(obj, cls):
            return obj
        if isinstance(obj, dict):
            return cls.from_dict(obj)
        raise TypeError(f"Cannot validate {type(obj)} as TelemetryObservation")

    @classmethod
    def model_validate_json(cls, json_str: str) -> "TelemetryObservation":
        return cls.from_dict(json.loads(json_str))


# ============================================================================
# 4. Spatial Position & Navigation Models
# ============================================================================

@dataclass(frozen=True)
class PositionObservation:
    """
    Immutable geographic position observation conforming to WGS-84 standard.
    """
    latitude: float
    longitude: float
    altitude: Optional[float] = None
    accuracy: Optional[float] = None  # Horizontal accuracy in meters
    vertical_accuracy: Optional[float] = None  # Vertical accuracy in meters
    heading: Optional[float] = None  # Degrees [0.0, 360.0)
    speed: Optional[float] = None  # m/s
    source_id: str = "default_source"
    captured_at: float = field(default_factory=time.time)
    observed_at: float = field(default_factory=time.time)
    provenance: Dict[str, Any] = field(default_factory=dict)
    correlation_id: str = ""
    causation_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.source_id or not isinstance(self.source_id, str):
            raise ValueError("PositionObservation source_id must be a non-empty string.")

        lat = float(self.latitude)
        lon = float(self.longitude)
        if math.isnan(lat) or math.isinf(lat) or not (-90.0 <= lat <= 90.0):
            raise ValueError(f"PositionObservation latitude must be finite and in [-90.0, 90.0], got {lat}")
        if math.isnan(lon) or math.isinf(lon) or not (-180.0 <= lon <= 180.0):
            raise ValueError(f"PositionObservation longitude must be finite and in [-180.0, 180.0], got {lon}")
        object.__setattr__(self, "latitude", round(lat, 7))
        object.__setattr__(self, "longitude", round(lon, 7))

        if self.altitude is not None:
            alt = float(self.altitude)
            if math.isnan(alt) or math.isinf(alt):
                raise ValueError(f"PositionObservation altitude must be finite, got {alt}")
            if not (-1000.0 <= alt <= 100000.0):
                raise ValueError(f"Altitude out of bounds [-1000.0, 100000.0], got {alt}")
            object.__setattr__(self, "altitude", round(alt, 2))

        if self.accuracy is not None:
            acc = float(self.accuracy)
            if math.isnan(acc) or math.isinf(acc) or acc < 0.0:
                raise ValueError(f"PositionObservation accuracy must be finite and non-negative, got {acc}")
            object.__setattr__(self, "accuracy", round(acc, 2))

        if self.vertical_accuracy is not None:
            vacc = float(self.vertical_accuracy)
            if math.isnan(vacc) or math.isinf(vacc) or vacc < 0.0:
                raise ValueError(f"PositionObservation vertical_accuracy must be finite and non-negative, got {vacc}")
            object.__setattr__(self, "vertical_accuracy", round(vacc, 2))

        if self.heading is not None:
            hdg = float(self.heading)
            if math.isnan(hdg) or math.isinf(hdg) or not (0.0 <= hdg < 360.0):
                raise ValueError(f"PositionObservation heading must be in [0.0, 360.0), got {hdg}")
            object.__setattr__(self, "heading", round(hdg, 2))

        if self.speed is not None:
            spd = float(self.speed)
            if math.isnan(spd) or math.isinf(spd) or spd < 0.0:
                raise ValueError(f"PositionObservation speed must be finite and non-negative, got {spd}")
            object.__setattr__(self, "speed", round(spd, 2))

        cap_ts = float(self.captured_at)
        obs_ts = float(self.observed_at)
        if cap_ts <= 0.0 or math.isnan(cap_ts) or math.isinf(cap_ts):
            raise ValueError(f"PositionObservation captured_at must be positive and finite, got {cap_ts}")
        if obs_ts <= 0.0 or math.isnan(obs_ts) or math.isinf(obs_ts):
            raise ValueError(f"PositionObservation observed_at must be positive and finite, got {obs_ts}")
        object.__setattr__(self, "captured_at", cap_ts)
        object.__setattr__(self, "observed_at", obs_ts)

        object.__setattr__(self, "provenance", sanitize_contract_metadata(self.provenance))
        object.__setattr__(self, "metadata", sanitize_contract_metadata(self.metadata))

    def to_geo_location(self) -> GeoLocation:
        """Convert into canonical Phase 5 GeoLocation model."""
        return GeoLocation(
            latitude=self.latitude,
            longitude=self.longitude,
            altitude=self.altitude,
            accuracy=self.accuracy,
        )

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "source_id": self.source_id,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "captured_at": self.captured_at,
            "observed_at": self.observed_at,
            "provenance": sanitize_contract_metadata(self.provenance),
            "correlation_id": self.correlation_id,
            "causation_id": self.causation_id,
            "metadata": sanitize_contract_metadata(self.metadata),
        }
        if self.altitude is not None:
            data["altitude"] = self.altitude
        if self.accuracy is not None:
            data["accuracy"] = self.accuracy
        if self.vertical_accuracy is not None:
            data["vertical_accuracy"] = self.vertical_accuracy
        if self.heading is not None:
            data["heading"] = self.heading
        if self.speed is not None:
            data["speed"] = self.speed
        return data

    def model_dump(self) -> Dict[str, Any]:
        return self.to_dict()

    def model_dump_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PositionObservation":
        return cls(
            latitude=float(data["latitude"]),
            longitude=float(data["longitude"]),
            altitude=float(data["altitude"]) if data.get("altitude") is not None else None,
            accuracy=float(data["accuracy"]) if data.get("accuracy") is not None else None,
            vertical_accuracy=float(data["vertical_accuracy"]) if data.get("vertical_accuracy") is not None else None,
            heading=float(data["heading"]) if data.get("heading") is not None else None,
            speed=float(data["speed"]) if data.get("speed") is not None else None,
            source_id=str(data.get("source_id", "default_source")),
            captured_at=float(data.get("captured_at", time.time())),
            observed_at=float(data.get("observed_at", time.time())),
            provenance=dict(data.get("provenance", {})),
            correlation_id=str(data.get("correlation_id", "")),
            causation_id=str(data["causation_id"]) if data.get("causation_id") else None,
            metadata=dict(data.get("metadata", {})),
        )

    @classmethod
    def model_validate(cls, obj: Any) -> "PositionObservation":
        if isinstance(obj, cls):
            return obj
        if isinstance(obj, dict):
            return cls.from_dict(obj)
        raise TypeError(f"Cannot validate {type(obj)} as PositionObservation")

    @classmethod
    def model_validate_json(cls, json_str: str) -> "PositionObservation":
        return cls.from_dict(json.loads(json_str))


@dataclass(frozen=True)
class RelativePosition:
    """
    Immutable semantic spatial relationship between two observed entities.
    """
    reference_entity_id: str = "self"
    target_entity_id: str = "target"
    distance: float = 0.0  # In meters
    bearing: float = 0.0  # Degrees [0.0, 360.0)
    relative_altitude: Optional[float] = None  # Target altitude - Reference altitude (meters)
    relation: SpatialRelationType = SpatialRelationType.NEAR
    confidence: float = 1.0
    timestamp: float = field(default_factory=time.time)
    provenance: Dict[str, Any] = field(default_factory=dict)
    # Aliases
    target_id: Optional[str] = None
    reference_id: Optional[str] = None
    distance_meters: Optional[float] = None
    bearing_degrees: Optional[float] = None

    def __post_init__(self):
        t_id = self.target_id or self.target_entity_id
        r_id = self.reference_id or self.reference_entity_id
        if not t_id or not isinstance(t_id, str):
            raise ValueError("RelativePosition target_id must be a non-empty string.")
        if not r_id or not isinstance(r_id, str):
            raise ValueError("RelativePosition reference_id must be a non-empty string.")
        object.__setattr__(self, "target_entity_id", t_id)
        object.__setattr__(self, "reference_entity_id", r_id)
        object.__setattr__(self, "target_id", t_id)
        object.__setattr__(self, "reference_id", r_id)

        d = float(self.distance_meters if self.distance_meters is not None else self.distance)
        if math.isnan(d) or math.isinf(d) or d < 0.0:
            raise ValueError(f"RelativePosition distance must be finite and non-negative, got {d}")
        object.__setattr__(self, "distance", round(d, 2))
        object.__setattr__(self, "distance_meters", round(d, 2))

        b = float(self.bearing_degrees if self.bearing_degrees is not None else self.bearing)
        if math.isnan(b) or math.isinf(b) or not (0.0 <= b < 360.0):
            raise ValueError(f"RelativePosition bearing must be in [0.0, 360.0), got {b}")
        object.__setattr__(self, "bearing", round(b, 2))
        object.__setattr__(self, "bearing_degrees", round(b, 2))

        if self.relative_altitude is not None:
            ra = float(self.relative_altitude)
            if math.isnan(ra) or math.isinf(ra):
                raise ValueError(f"RelativePosition relative_altitude must be finite, got {ra}")
            object.__setattr__(self, "relative_altitude", round(ra, 2))

        rel = self.relation
        if isinstance(rel, str):
            rel = SpatialRelationType.from_str(rel)
        object.__setattr__(self, "relation", rel)

        conf = float(self.confidence)
        if math.isnan(conf) or math.isinf(conf) or not (0.0 <= conf <= 1.0):
            raise ValueError(f"RelativePosition confidence must be in [0.0, 1.0], got {conf}")
        object.__setattr__(self, "confidence", round(conf, 4))

        ts = float(self.timestamp)
        if ts <= 0.0 or math.isnan(ts) or math.isinf(ts):
            raise ValueError(f"RelativePosition timestamp must be positive and finite, got {ts}")
        object.__setattr__(self, "timestamp", ts)

        object.__setattr__(self, "provenance", sanitize_contract_metadata(self.provenance))

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "reference_entity_id": self.reference_entity_id,
            "target_entity_id": self.target_entity_id,
            "target_id": self.target_id,
            "reference_id": self.reference_id,
            "distance": self.distance,
            "distance_meters": self.distance_meters,
            "bearing": self.bearing,
            "bearing_degrees": self.bearing_degrees,
            "relation": self.relation.value,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "provenance": sanitize_contract_metadata(self.provenance),
        }
        if self.relative_altitude is not None:
            data["relative_altitude"] = self.relative_altitude
        return data

    def model_dump(self) -> Dict[str, Any]:
        return self.to_dict()

    def model_dump_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RelativePosition":
        return cls(
            reference_entity_id=str(data.get("reference_entity_id", data.get("reference_id", "self"))),
            target_entity_id=str(data.get("target_entity_id", data.get("target_id", "target"))),
            target_id=str(data.get("target_id", data.get("target_entity_id", "target"))),
            reference_id=str(data.get("reference_id", data.get("reference_entity_id", "self"))),
            distance=float(data.get("distance", data.get("distance_meters", 0.0))),
            distance_meters=float(data.get("distance_meters", data.get("distance", 0.0))),
            bearing=float(data.get("bearing", data.get("bearing_degrees", 0.0))),
            bearing_degrees=float(data.get("bearing_degrees", data.get("bearing", 0.0))),
            relative_altitude=float(data["relative_altitude"]) if data.get("relative_altitude") is not None else None,
            relation=SpatialRelationType.from_str(data.get("relation", "NEAR")),
            confidence=float(data.get("confidence", 1.0)),
            timestamp=float(data.get("timestamp", time.time())),
            provenance=dict(data.get("provenance", {})),
        )

    @classmethod
    def model_validate(cls, obj: Any) -> "RelativePosition":
        if isinstance(obj, cls):
            return obj
        if isinstance(obj, dict):
            return cls.from_dict(obj)
        raise TypeError(f"Cannot validate {type(obj)} as RelativePosition")

    @classmethod
    def model_validate_json(cls, json_str: str) -> "RelativePosition":
        return cls.from_dict(json.loads(json_str))


@dataclass(frozen=True)
class NavigationObservation:
    """
    Immutable observation of an entity's kinematic and navigational state.
    """
    position: Optional[PositionObservation] = None
    current_position: Optional[PositionObservation] = None
    target_latitude: Optional[float] = None
    target_longitude: Optional[float] = None
    distance_to_target: Optional[float] = None
    bearing_to_target: Optional[float] = None
    movement_state: MovementState = MovementState.UNKNOWN
    heading: Optional[float] = None
    speed: Optional[float] = None
    route_reference: Optional[str] = None
    captured_at: float = field(default_factory=time.time)
    provenance: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        pos = self.current_position or self.position
        if pos is not None and not isinstance(pos, PositionObservation):
            raise TypeError(f"NavigationObservation position must be PositionObservation, got {type(pos)}")
        object.__setattr__(self, "position", pos)
        object.__setattr__(self, "current_position", pos)

        mstate = self.movement_state
        if isinstance(mstate, str):
            mstate = MovementState.from_str(mstate)
        object.__setattr__(self, "movement_state", mstate)

        hdg = self.heading if self.heading is not None else (pos.heading if pos else None)
        if hdg is not None:
            hdg_f = float(hdg)
            if math.isnan(hdg_f) or math.isinf(hdg_f) or not (0.0 <= hdg_f < 360.0):
                raise ValueError(f"NavigationObservation heading must be in [0.0, 360.0), got {hdg_f}")
            object.__setattr__(self, "heading", round(hdg_f, 2))

        spd = self.speed if self.speed is not None else (pos.speed if pos else None)
        if spd is not None:
            spd_f = float(spd)
            if math.isnan(spd_f) or math.isinf(spd_f) or spd_f < 0.0:
                raise ValueError(f"NavigationObservation speed must be non-negative, got {spd_f}")
            object.__setattr__(self, "speed", round(spd_f, 2))

        ts = float(self.captured_at)
        if ts <= 0.0 or math.isnan(ts) or math.isinf(ts):
            raise ValueError(f"NavigationObservation captured_at must be positive, got {ts}")
        object.__setattr__(self, "captured_at", ts)

        object.__setattr__(self, "provenance", sanitize_contract_metadata(self.provenance))

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "position": self.position.to_dict() if self.position else None,
            "current_position": self.current_position.to_dict() if self.current_position else None,
            "movement_state": self.movement_state.value,
            "captured_at": self.captured_at,
            "provenance": sanitize_contract_metadata(self.provenance),
        }
        if self.target_latitude is not None:
            data["target_latitude"] = self.target_latitude
        if self.target_longitude is not None:
            data["target_longitude"] = self.target_longitude
        if self.distance_to_target is not None:
            data["distance_to_target"] = self.distance_to_target
        if self.bearing_to_target is not None:
            data["bearing_to_target"] = self.bearing_to_target
        if self.heading is not None:
            data["heading"] = self.heading
        if self.speed is not None:
            data["speed"] = self.speed
        if self.route_reference is not None:
            data["route_reference"] = self.route_reference
        return data

    def model_dump(self) -> Dict[str, Any]:
        return self.to_dict()

    def model_dump_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NavigationObservation":
        pos_data = data.get("current_position") or data.get("position")
        pos = PositionObservation.from_dict(pos_data) if pos_data else None
        return cls(
            position=pos,
            current_position=pos,
            target_latitude=data.get("target_latitude"),
            target_longitude=data.get("target_longitude"),
            distance_to_target=data.get("distance_to_target"),
            bearing_to_target=data.get("bearing_to_target"),
            movement_state=MovementState.from_str(data.get("movement_state", "UNKNOWN")),
            heading=float(data["heading"]) if data.get("heading") is not None else None,
            speed=float(data["speed"]) if data.get("speed") is not None else None,
            route_reference=data.get("route_reference"),
            captured_at=float(data.get("captured_at", time.time())),
            provenance=dict(data.get("provenance", {})),
        )

    @classmethod
    def model_validate(cls, obj: Any) -> "NavigationObservation":
        if isinstance(obj, cls):
            return obj
        if isinstance(obj, dict):
            return cls.from_dict(obj)
        raise TypeError(f"Cannot validate {type(obj)} as NavigationObservation")

    @classmethod
    def model_validate_json(cls, json_str: str) -> "NavigationObservation":
        return cls.from_dict(json.loads(json_str))


@dataclass(frozen=True)
class GeofenceArea:
    """
    Immutable representation of a circular or bounding-box geofence.
    """
    name: str = "geofence"
    geofence_id: Optional[str] = None
    center: Optional[GeoLocation] = None
    center_latitude: Optional[float] = None
    center_longitude: Optional[float] = None
    radius_meters: Optional[float] = None
    tolerance_meters: float = 2.0
    min_latitude: Optional[float] = None
    max_latitude: Optional[float] = None
    min_longitude: Optional[float] = None
    max_longitude: Optional[float] = None

    def __post_init__(self):
        gid = self.geofence_id or self.name or "fence"
        object.__setattr__(self, "geofence_id", gid)

        # Reconcile center and lat/lon
        c = self.center
        if c is None and self.center_latitude is not None and self.center_longitude is not None:
            c = GeoLocation(latitude=self.center_latitude, longitude=self.center_longitude)
            object.__setattr__(self, "center", c)
        elif c is not None:
            object.__setattr__(self, "center_latitude", c.latitude)
            object.__setattr__(self, "center_longitude", c.longitude)

        if self.radius_meters is not None:
            r = float(self.radius_meters)
            if math.isnan(r) or math.isinf(r) or r <= 0.0:
                raise ValueError(f"GeofenceArea radius_meters must be positive and finite, got {r}")
            object.__setattr__(self, "radius_meters", round(r, 2))

        tol = float(self.tolerance_meters)
        if math.isnan(tol) or math.isinf(tol) or tol < 0.0:
            raise ValueError(f"GeofenceArea tolerance_meters must be non-negative, got {tol}")
        object.__setattr__(self, "tolerance_meters", round(tol, 2))

    def evaluate(self, point: Union[GeoLocation, Tuple[float, float]]) -> GeofenceRelationType:
        """Deterministically evaluate if point is INSIDE, OUTSIDE, or on the BOUNDARY."""
        if isinstance(point, tuple):
            lat, lon = point
            geo = GeoLocation(latitude=lat, longitude=lon)
        elif isinstance(point, GeoLocation):
            lat, lon = point.latitude, point.longitude
            geo = point
        else:
            lat, lon = point.latitude, point.longitude
            geo = GeoLocation(latitude=lat, longitude=lon)

        # Bounding box evaluation if defined
        if (
            self.min_latitude is not None
            and self.max_latitude is not None
            and self.min_longitude is not None
            and self.max_longitude is not None
        ):
            if (
                self.min_latitude <= lat <= self.max_latitude
                and self.min_longitude <= lon <= self.max_longitude
            ):
                return GeofenceRelationType.INSIDE
            return GeofenceRelationType.OUTSIDE

        # Circular evaluation
        if self.center is not None and self.radius_meters is not None:
            dist = self.center.distance_to(geo)
            if abs(dist - self.radius_meters) <= self.tolerance_meters:
                return GeofenceRelationType.BOUNDARY
            elif dist < self.radius_meters:
                return GeofenceRelationType.INSIDE
            elif dist < self.radius_meters * 1.2:
                return GeofenceRelationType.APPROACHING
            else:
                return GeofenceRelationType.OUTSIDE

        return GeofenceRelationType.OUTSIDE

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "geofence_id": self.geofence_id,
            "name": self.name,
            "tolerance_meters": self.tolerance_meters,
        }
        if self.center:
            d["center"] = self.center.to_dict()
        if self.center_latitude is not None:
            d["center_latitude"] = self.center_latitude
        if self.center_longitude is not None:
            d["center_longitude"] = self.center_longitude
        if self.radius_meters is not None:
            d["radius_meters"] = self.radius_meters
        if self.min_latitude is not None:
            d["min_latitude"] = self.min_latitude
            d["max_latitude"] = self.max_latitude
            d["min_longitude"] = self.min_longitude
            d["max_longitude"] = self.max_longitude
        return d

    def model_dump(self) -> Dict[str, Any]:
        return self.to_dict()

    def model_dump_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GeofenceArea":
        c = GeoLocation.from_dict(data["center"]) if "center" in data and data["center"] else None
        return cls(
            name=str(data.get("name", "")),
            geofence_id=data.get("geofence_id"),
            center=c,
            center_latitude=data.get("center_latitude"),
            center_longitude=data.get("center_longitude"),
            radius_meters=float(data["radius_meters"]) if data.get("radius_meters") is not None else None,
            tolerance_meters=float(data.get("tolerance_meters", 2.0)),
            min_latitude=data.get("min_latitude"),
            max_latitude=data.get("max_latitude"),
            min_longitude=data.get("min_longitude"),
            max_longitude=data.get("max_longitude"),
        )

    @classmethod
    def model_validate(cls, obj: Any) -> "GeofenceArea":
        if isinstance(obj, cls):
            return obj
        if isinstance(obj, dict):
            return cls.from_dict(obj)
        raise TypeError(f"Cannot validate {type(obj)} as GeofenceArea")

    @classmethod
    def model_validate_json(cls, json_str: str) -> "GeofenceArea":
        return cls.from_dict(json.loads(json_str))


@dataclass(frozen=True)
class SpatialObservation:
    """
    Immutable aggregate spatial observation covering positions, relative targets, and geofences.
    """
    position: Optional[PositionObservation] = None
    navigation: Optional[NavigationObservation] = None
    geofences: Sequence[GeofenceArea] = field(default_factory=tuple)
    spatial_relations: Sequence[RelativePosition] = field(default_factory=tuple)
    source_id: str = "default_source"
    relative_positions: Tuple[RelativePosition, ...] = field(default_factory=tuple)
    geofence_relations: Tuple[Dict[str, Any], ...] = field(default_factory=tuple)
    timestamp: float = field(default_factory=time.time)
    provenance: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        ts = float(self.timestamp)
        if ts <= 0.0 or math.isnan(ts) or math.isinf(ts):
            raise ValueError(f"SpatialObservation timestamp must be positive, got {ts}")
        object.__setattr__(self, "timestamp", ts)

        limits = SpatialTelemetryLimits()
        rels = list(self.relative_positions) or list(self.spatial_relations)
        if len(rels) > limits.max_spatial_relations:
            raise ValueError(f"relative_positions count ({len(rels)}) exceeds limit ({limits.max_spatial_relations})")

        gf_list = list(self.geofence_relations) or [g.to_dict() for g in self.geofences]
        if len(gf_list) > limits.max_geofences:
            raise ValueError(f"geofence_relations count ({len(gf_list)}) exceeds limit ({limits.max_geofences})")

        object.__setattr__(self, "relative_positions", tuple(rels))
        object.__setattr__(self, "spatial_relations", tuple(rels))
        object.__setattr__(self, "geofence_relations", tuple(gf_list))
        object.__setattr__(self, "geofences", tuple(self.geofences))
        object.__setattr__(self, "provenance", sanitize_contract_metadata(self.provenance))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "position": self.position.to_dict() if self.position else None,
            "navigation": self.navigation.to_dict() if self.navigation else None,
            "geofences": [g.to_dict() for g in self.geofences],
            "spatial_relations": [r.to_dict() for r in self.spatial_relations],
            "relative_positions": [rp.to_dict() for rp in self.relative_positions],
            "geofence_relations": list(self.geofence_relations),
            "timestamp": self.timestamp,
            "provenance": sanitize_contract_metadata(self.provenance),
        }

    def model_dump(self) -> Dict[str, Any]:
        return self.to_dict()

    def model_dump_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SpatialObservation":
        pos = PositionObservation.from_dict(data["position"]) if data.get("position") else None
        nav = NavigationObservation.from_dict(data["navigation"]) if data.get("navigation") else None
        gfs = tuple(GeofenceArea.from_dict(g) for g in data.get("geofences", ()))
        rps = tuple(RelativePosition.from_dict(rp) for rp in data.get("relative_positions", data.get("spatial_relations", ())))
        return cls(
            source_id=str(data.get("source_id", "default_source")),
            position=pos,
            navigation=nav,
            geofences=gfs,
            spatial_relations=rps,
            relative_positions=rps,
            geofence_relations=tuple(data.get("geofence_relations", ())),
            timestamp=float(data.get("timestamp", time.time())),
            provenance=dict(data.get("provenance", {})),
        )

    @classmethod
    def model_validate(cls, obj: Any) -> "SpatialObservation":
        if isinstance(obj, cls):
            return obj
        if isinstance(obj, dict):
            return cls.from_dict(obj)
        raise TypeError(f"Cannot validate {type(obj)} as SpatialObservation")

    @classmethod
    def model_validate_json(cls, json_str: str) -> "SpatialObservation":
        return cls.from_dict(json.loads(json_str))
