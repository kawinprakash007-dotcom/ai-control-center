"""
ATLAS Phase 6.3 — Digital Twin & Simulation Domain Models.

Provides domain models, configurations, telemetry, states, fault models,
and bounds for digital twins across all ATLAS edge products (Vision, Glass, Drone, Rover).

ARCHITECTURAL RULES:
1. Purely transport-neutral: ZERO hardware, serial, MAVLink, ROS2, MQTT, BLE, or GPIO specifics.
2. Domain-safe, frozen/immutable where appropriate, with deterministic serialization.
3. Credentials, secrets, private keys, passwords, and API tokens are NEVER serialized.
4. Strict bounds on history, telemetry records, faults, and collections.
"""

from dataclasses import dataclass, field
from enum import Enum
import math
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from core.models.device_contract import (
    DeviceErrorCode,
    DeviceHealthStatus,
    DeviceTelemetry,
    ProductRole,
    ProductType,
    sanitize_contract_metadata,
)
from core.models.orchestration import (
    ConnectivityStatus,
    GeoLocation,
    ModalityType,
    MultimodalObservation,
)


# ============================================================================
# Enums
# ============================================================================

class TwinSimulationStatus(str, Enum):
    """Execution status of a digital twin within the simulation."""
    INITIALIZING = "INITIALIZING"
    ACTIVE = "ACTIVE"
    PAUSED = "PAUSED"
    ERROR = "ERROR"
    OFFLINE = "OFFLINE"
    STOPPED = "STOPPED"

    @classmethod
    def from_str(cls, val: Union[str, "TwinSimulationStatus"]) -> "TwinSimulationStatus":
        if isinstance(val, cls):
            return val
        norm = str(val).strip().upper()
        for member in cls:
            if member.value == norm:
                return member
        return cls.ACTIVE


class TwinFaultType(str, Enum):
    """Standardized fault classes for controlled fault injection."""
    OFFLINE = "OFFLINE"
    LOW_BATTERY = "LOW_BATTERY"
    GPS_LOSS = "GPS_LOSS"
    SENSOR_FAILURE = "SENSOR_FAILURE"
    TELEMETRY_STALE = "TELEMETRY_STALE"
    COMMAND_TIMEOUT = "COMMAND_TIMEOUT"
    COMMAND_FAILURE = "COMMAND_FAILURE"
    DUPLICATE_OBSERVATION = "DUPLICATE_OBSERVATION"
    CONFLICTING_OBSERVATION = "CONFLICTING_OBSERVATION"

    @classmethod
    def from_str(cls, val: Union[str, "TwinFaultType"]) -> "TwinFaultType":
        if isinstance(val, cls):
            return val
        norm = str(val).strip().upper()
        for member in cls:
            if member.value == norm:
                return member
        return cls.COMMAND_FAILURE


# ============================================================================
# Limits and Capacity Bounds
# ============================================================================

@dataclass(frozen=True)
class SimulationLimits:
    """Explicit bounds preventing runaway resource consumption."""
    max_twins: int = 50
    max_entities: int = 100
    max_scenario_steps: int = 500
    max_pending_observations: int = 200
    max_pending_events: int = 200
    max_scenario_history: int = 1000
    max_fault_records: int = 100
    max_telemetry_records: int = 500
    max_simulation_duration: float = 86400.0  # 24 hours


# ============================================================================
# Digital Twin Position & Configuration
# ============================================================================

@dataclass(frozen=True)
class TwinPosition:
    """
    3D spatial position and movement heading for a digital twin.
    """
    latitude: float
    longitude: float
    altitude: float = 0.0
    heading: float = 0.0  # Degrees 0-360
    speed: float = 0.0    # m/s

    def to_geo_location(self) -> GeoLocation:
        return GeoLocation(
            latitude=self.latitude,
            longitude=self.longitude,
            altitude=self.altitude,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "altitude": self.altitude,
            "heading": self.heading,
            "speed": self.speed,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TwinPosition":
        return cls(
            latitude=float(data.get("latitude", data.get("x", 0.0))),
            longitude=float(data.get("longitude", data.get("y", 0.0))),
            altitude=float(data.get("altitude", data.get("z", 0.0))),
            heading=float(data.get("heading", 0.0)),
            speed=float(data.get("speed", 0.0)),
        )

    def distance_to(self, other: "TwinPosition") -> float:
        """Approximate Euclidean ground distance in meters (flat earth approx for sim)."""
        dlat = (self.latitude - other.latitude) * 111000.0
        dlon = (self.longitude - other.longitude) * 111000.0 * math.cos(math.radians(self.latitude))
        dalt = self.altitude - other.altitude
        return math.sqrt(dlat * dlat + dlon * dlon + dalt * dalt)


@dataclass(frozen=True)
class TwinConfiguration:
    """
    Immutable specification and baseline parameters for initializing a digital twin.
    """
    twin_id: str
    product_type: ProductType
    product_role: ProductRole
    initial_position: Optional[TwinPosition] = None
    initial_battery_pct: float = 100.0
    telemetry_interval_seconds: float = 1.0
    capabilities: Tuple[str, ...] = ()
    parameters: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.twin_id or not isinstance(self.twin_id, str) or not self.twin_id.strip():
            raise ValueError("twin_id must be a non-empty string")
        if self.initial_battery_pct < 0.0 or self.initial_battery_pct > 100.0:
            raise ValueError(f"initial_battery_pct must be between 0.0 and 100.0, got {self.initial_battery_pct}")

    def to_dict(self) -> Dict[str, Any]:
        clean_meta = sanitize_contract_metadata(self.metadata)
        clean_params = sanitize_contract_metadata(self.parameters)
        return {
            "twin_id": self.twin_id,
            "product_type": self.product_type.value,
            "product_role": self.product_role.value,
            "initial_position": self.initial_position.to_dict() if self.initial_position else None,
            "initial_battery_pct": self.initial_battery_pct,
            "telemetry_interval_seconds": self.telemetry_interval_seconds,
            "capabilities": list(self.capabilities),
            "parameters": clean_params,
            "metadata": clean_meta,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TwinConfiguration":
        pos_data = data.get("initial_position")
        pos = TwinPosition.from_dict(pos_data) if pos_data else None
        return cls(
            twin_id=str(data["twin_id"]),
            product_type=ProductType.from_str(data.get("product_type", "UNKNOWN")),
            product_role=ProductRole.from_str(data.get("product_role", "OBSERVATION_SOURCE")),
            initial_position=pos,
            initial_battery_pct=float(data.get("initial_battery_pct", 100.0)),
            telemetry_interval_seconds=float(data.get("telemetry_interval_seconds", 1.0)),
            capabilities=tuple(data.get("capabilities", ())),
            parameters=dict(data.get("parameters", {})),
            metadata=dict(data.get("metadata", {})),
        )


# ============================================================================
# Twin Fault Model
# ============================================================================

@dataclass(frozen=True)
class TwinFault:
    """
    Explicit, observable, and removable fault record injected into a twin.
    """
    fault_id: str
    twin_id: str
    fault_type: TwinFaultType
    parameters: Dict[str, Any] = field(default_factory=dict)
    injected_at: float = 0.0
    duration_seconds: Optional[float] = None  # None = indefinite until explicitly cleared
    is_active: bool = True

    def is_expired(self, current_time: float) -> bool:
        if self.duration_seconds is None:
            return False
        return (current_time - self.injected_at) >= self.duration_seconds

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fault_id": self.fault_id,
            "twin_id": self.twin_id,
            "fault_type": self.fault_type.value,
            "parameters": sanitize_contract_metadata(self.parameters),
            "injected_at": self.injected_at,
            "duration_seconds": self.duration_seconds,
            "is_active": self.is_active,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TwinFault":
        return cls(
            fault_id=str(data["fault_id"]),
            twin_id=str(data["twin_id"]),
            fault_type=TwinFaultType.from_str(data.get("fault_type", "COMMAND_FAILURE")),
            parameters=dict(data.get("parameters", {})),
            injected_at=float(data.get("injected_at", 0.0)),
            duration_seconds=float(data["duration_seconds"]) if data.get("duration_seconds") is not None else None,
            is_active=bool(data.get("is_active", True)),
        )


# ============================================================================
# Digital Twin Telemetry
# ============================================================================

@dataclass(frozen=True)
class TwinTelemetry:
    """
    Deterministic telemetry snapshot emitted by a digital twin.
    """
    telemetry_id: str
    twin_id: str
    product_type: ProductType
    timestamp: float
    battery_level: float
    connectivity: ConnectivityStatus = ConnectivityStatus.ONLINE
    health_status: DeviceHealthStatus = DeviceHealthStatus.HEALTHY
    position: Optional[TwinPosition] = None
    temperature_celsius: Optional[float] = 25.0
    operating_mode: str = "NORMAL"
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "telemetry_id": self.telemetry_id,
            "twin_id": self.twin_id,
            "product_type": self.product_type.value,
            "timestamp": self.timestamp,
            "battery_level": self.battery_level,
            "connectivity": self.connectivity.value,
            "health_status": self.health_status.value,
            "position": self.position.to_dict() if self.position else None,
            "temperature_celsius": self.temperature_celsius,
            "operating_mode": self.operating_mode,
            "metrics": sanitize_contract_metadata(self.metrics),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TwinTelemetry":
        pos_data = data.get("position")
        pos = TwinPosition.from_dict(pos_data) if pos_data else None
        return cls(
            telemetry_id=str(data["telemetry_id"]),
            twin_id=str(data["twin_id"]),
            product_type=ProductType.from_str(data.get("product_type", "UNKNOWN")),
            timestamp=float(data.get("timestamp", 0.0)),
            battery_level=float(data.get("battery_level", 100.0)),
            connectivity=ConnectivityStatus.from_str(data.get("connectivity", "ONLINE")),
            health_status=DeviceHealthStatus.from_str(data.get("health_status", "HEALTHY")),
            position=pos,
            temperature_celsius=float(data["temperature_celsius"]) if data.get("temperature_celsius") is not None else None,
            operating_mode=str(data.get("operating_mode", "NORMAL")),
            metrics=dict(data.get("metrics", {})),
        )

    def to_device_telemetry(self) -> DeviceTelemetry:
        """Convert into canonical Phase 6.2 DeviceTelemetry model."""
        geo = self.position.to_geo_location() if self.position else None
        ts = self.timestamp if self.timestamp > 0.0 else 1000000.0
        return DeviceTelemetry(
            device_id=self.twin_id,
            timestamp=ts,
            location=geo,
            battery_pct=self.battery_level,
            connectivity=self.connectivity,
            health=self.health_status,
            metrics={
                "temperature_celsius": self.temperature_celsius,
                "heading": self.position.heading if self.position else 0.0,
                "speed": self.position.speed if self.position else 0.0,
                **self.metrics,
            },
            source=self.twin_id,
            extension_data={"operating_mode": self.operating_mode, "product_type": self.product_type.value},
        )

    def to_multimodal_observation(self) -> MultimodalObservation:
        """Direct conversion to central nervous system observation."""
        return self.to_device_telemetry().to_multimodal_observation()


# ============================================================================
# Digital Twin State
# ============================================================================

@dataclass(frozen=True)
class TwinState:
    """
    Immutable representation of complete digital twin state at an instant.
    """
    twin_id: str
    product_id: str
    product_type: ProductType
    product_role: ProductRole
    connectivity: ConnectivityStatus = ConnectivityStatus.ONLINE
    health: DeviceHealthStatus = DeviceHealthStatus.HEALTHY
    battery: float = 100.0
    position: Optional[TwinPosition] = None
    active_capabilities: Tuple[str, ...] = ()
    simulation_status: TwinSimulationStatus = TwinSimulationStatus.ACTIVE
    last_update: float = 0.0
    schema_version: str = "1.0"
    internal_state: Dict[str, Any] = field(default_factory=dict)
    active_faults: Tuple[TwinFault, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "twin_id": self.twin_id,
            "product_id": self.product_id,
            "product_type": self.product_type.value,
            "product_role": self.product_role.value,
            "connectivity": self.connectivity.value,
            "health": self.health.value,
            "battery": self.battery,
            "position": self.position.to_dict() if self.position else None,
            "active_capabilities": list(self.active_capabilities),
            "simulation_status": self.simulation_status.value,
            "last_update": self.last_update,
            "schema_version": self.schema_version,
            "internal_state": sanitize_contract_metadata(self.internal_state),
            "active_faults": [f.to_dict() for f in self.active_faults],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TwinState":
        pos_data = data.get("position")
        pos = TwinPosition.from_dict(pos_data) if pos_data else None
        faults = tuple(TwinFault.from_dict(f) for f in data.get("active_faults", []))
        return cls(
            twin_id=str(data["twin_id"]),
            product_id=str(data.get("product_id", data["twin_id"])),
            product_type=ProductType.from_str(data.get("product_type", "UNKNOWN")),
            product_role=ProductRole.from_str(data.get("product_role", "OBSERVATION_SOURCE")),
            connectivity=ConnectivityStatus.from_str(data.get("connectivity", "ONLINE")),
            health=DeviceHealthStatus.from_str(data.get("health", "HEALTHY")),
            battery=float(data.get("battery", 100.0)),
            position=pos,
            active_capabilities=tuple(data.get("active_capabilities", ())),
            simulation_status=TwinSimulationStatus.from_str(data.get("simulation_status", "ACTIVE")),
            last_update=float(data.get("last_update", 0.0)),
            schema_version=str(data.get("schema_version", "1.0")),
            internal_state=dict(data.get("internal_state", {})),
            active_faults=faults,
        )
