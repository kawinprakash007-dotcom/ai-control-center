"""
ATLAS Phase 6.2 — Unified Edge/Product Contract Layer.

Transport-neutral, immutable domain contracts for edge participants and products:
- ATLAS Vision (Home surveillance / persistent environmental sensing)
- ATLAS Glass (Wearable perception + HUD / user interaction)
- ATLAS Drone (Aerial sensing + intervention)
- ATLAS Rover (Ground sensing + intervention)

CRITICAL ARCHITECTURAL RULES:
1. Transport-neutral: ZERO hardware, serial, MAVLink, ROS2, MQTT, BLE, or GPIO specifics.
2. Models are frozen/immutable dataclasses where appropriate with deterministic serialization.
3. Credentials, secrets, private keys, passwords, and API tokens are NEVER serialized.
4. Telemetry converts cleanly into MultimodalObservation before central ingestion.
5. Command acknowledgements are strictly distinguished from command completions.
"""

from dataclasses import dataclass, field
from enum import Enum
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

from core.models.orchestration import (
    ConnectivityStatus,
    GeoLocation,
    ModalityType,
    MultimodalObservation,
)
from core.models.result import Result


# ============================================================================
# 1. Product & Ecosystem Enums
# ============================================================================

class ProductType(str, Enum):
    """
    First-class ATLAS edge product ecosystem classification.
    """
    VISION = "VISION"
    GLASS = "GLASS"
    DRONE = "DRONE"
    ROVER = "ROVER"
    UNKNOWN = "UNKNOWN"

    @classmethod
    def from_str(cls, val: Any) -> "ProductType":
        """Deterministic mapping with safe fallback to UNKNOWN."""
        if isinstance(val, cls):
            return val
        s = str(val or "").strip().upper()
        for member in cls:
            if member.value == s or member.name == s:
                return member
        return cls.UNKNOWN


class ProductRole(str, Enum):
    """
    Semantic operational role of an edge product.
    Used exclusively as capability and descriptive metadata; NOT as execution authority.
    """
    OBSERVATION_SOURCE = "OBSERVATION_SOURCE"
    ACTUATOR = "ACTUATOR"
    HYBRID = "HYBRID"

    @classmethod
    def from_str(cls, val: Any) -> "ProductRole":
        """Deterministic mapping with fallback to HYBRID."""
        if isinstance(val, cls):
            return val
        s = str(val or "").strip().upper()
        for member in cls:
            if member.value == s or member.name == s:
                return member
        return cls.HYBRID

    @classmethod
    def get_default_role(cls, product_type: ProductType) -> "ProductRole":
        """Canonical mapping of ATLAS products to their default semantic role."""
        if product_type == ProductType.VISION:
            return cls.OBSERVATION_SOURCE
        elif product_type in (ProductType.GLASS, ProductType.DRONE, ProductType.ROVER):
            return cls.HYBRID
        return cls.HYBRID


# ============================================================================
# 2. Command States & Acknowledgement
# ============================================================================

class CommandState(str, Enum):
    """
    Semantic command progression states.
    Distinguishes dispatch/acceptance from asynchronous completion.
    """
    REQUESTED = "REQUESTED"
    VALIDATED = "VALIDATED"
    DISPATCHED = "DISPATCHED"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    EXECUTING = "EXECUTING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"

    @classmethod
    def from_str(cls, val: Any) -> "CommandState":
        if isinstance(val, cls):
            return val
        s = str(val or "").strip().upper()
        for member in cls:
            if member.value == s or member.name == s:
                return member
        return cls.REQUESTED


class AcknowledgementStatus(str, Enum):
    """
    Transport-neutral command acknowledgement status.
    Indicates whether a device accepted or rejected receipt of a command for processing.
    """
    ACCEPTED = "ACCEPTED"
    QUEUED = "QUEUED"
    REJECTED = "REJECTED"
    EXECUTING = "EXECUTING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"

    @classmethod
    def from_str(cls, val: Any) -> "AcknowledgementStatus":
        if isinstance(val, cls):
            return val
        s = str(val or "").strip().upper()
        for member in cls:
            if member.value == s or member.name == s:
                return member
        return cls.ACCEPTED


# ============================================================================
# 3. Health & Error Models
# ============================================================================

class DeviceHealthStatus(str, Enum):
    """
    Standardized semantic device health ranking.
    """
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    UNHEALTHY = "UNHEALTHY"
    UNKNOWN = "UNKNOWN"

    @classmethod
    def from_str(cls, val: Any) -> "DeviceHealthStatus":
        if isinstance(val, cls):
            return val
        s = str(val or "").strip().upper()
        for member in cls:
            if member.value == s or member.name == s:
                return member
        return cls.UNKNOWN


class DeviceErrorCode(str, Enum):
    """
    Canonical, deterministic error classifications across all edge products.
    """
    UNKNOWN_DEVICE = "UNKNOWN_DEVICE"
    OFFLINE_DEVICE = "OFFLINE_DEVICE"
    UNSUPPORTED_CAPABILITY = "UNSUPPORTED_CAPABILITY"
    UNSUPPORTED_ACTION = "UNSUPPORTED_ACTION"
    INVALID_PARAMETERS = "INVALID_PARAMETERS"
    COMMAND_REJECTED = "COMMAND_REJECTED"
    COMMAND_TIMEOUT = "COMMAND_TIMEOUT"
    TRANSPORT_ERROR = "TRANSPORT_ERROR"
    DEVICE_ERROR = "DEVICE_ERROR"
    LOW_BATTERY = "LOW_BATTERY"
    SAFETY_REJECTION = "SAFETY_REJECTION"
    DUPLICATE_COMMAND = "DUPLICATE_COMMAND"
    CONTRACT_VERSION_UNSUPPORTED = "CONTRACT_VERSION_UNSUPPORTED"

    @classmethod
    def from_str(cls, val: Any) -> "DeviceErrorCode":
        if isinstance(val, cls):
            return val
        s = str(val or "").strip().upper()
        for member in cls:
            if member.value == s or member.name == s:
                return member
        return cls.DEVICE_ERROR


# Sensitive key filter for sanitizing serialized payloads and metadata
_SENSITIVE_KEYS: Set[str] = {
    "password", "secret", "token", "credential", "api_key", "private_key",
    "auth", "auth_token", "ssh_key", "pin", "access_key", "secret_key"
}


def sanitize_contract_metadata(meta: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Sanitize metadata dictionary to ensure zero credential/secret serialization."""
    if not meta:
        return {}
    clean: Dict[str, Any] = {}
    for k, v in meta.items():
        if any(bad in k.lower() for bad in _SENSITIVE_KEYS):
            clean[k] = "[REDACTED]"
        elif isinstance(v, dict):
            clean[k] = sanitize_contract_metadata(v)
        else:
            clean[k] = v
    return clean


@dataclass(frozen=True)
class DeviceError:
    """
    Immutable representation of an error occurring during edge command or lifecycle operations.
    """
    code: DeviceErrorCode
    message: str
    device_id: str
    command_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    details: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        c = self.code
        if isinstance(c, str):
            object.__setattr__(self, "code", DeviceErrorCode.from_str(c))
        elif not isinstance(c, DeviceErrorCode):
            raise ValueError(f"DeviceError code must be DeviceErrorCode, got {type(c)}")

        object.__setattr__(self, "timestamp", float(self.timestamp))
        if self.details:
            object.__setattr__(self, "details", sanitize_contract_metadata(self.details))

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "code": self.code.value,
            "message": self.message,
            "device_id": self.device_id,
            "timestamp": self.timestamp,
        }
        if self.command_id is not None:
            data["command_id"] = self.command_id
        if self.details:
            data["details"] = sanitize_contract_metadata(self.details)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DeviceError":
        return cls(
            code=DeviceErrorCode.from_str(data["code"]),
            message=data["message"],
            device_id=data["device_id"],
            command_id=data.get("command_id"),
            timestamp=float(data.get("timestamp", time.time())),
            details=dict(data.get("details", {})),
        )


# ============================================================================
# 4. Heartbeat & Health Contracts
# ============================================================================

@dataclass(frozen=True)
class DeviceHeartbeat:
    """
    Immutable heartbeat beacon emitted or recorded for an edge device.
    Strictly bounded; does NOT create background daemon watchdogs.
    """
    device_id: str
    timestamp: float = field(default_factory=time.time)
    sequence_number: int = 0
    connectivity_state: ConnectivityStatus = ConnectivityStatus.ONLINE
    health_summary: DeviceHealthStatus = DeviceHealthStatus.HEALTHY
    metrics: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.device_id or not isinstance(self.device_id, str):
            raise ValueError("DeviceHeartbeat device_id must be a non-empty string.")

        cs = self.connectivity_state
        if isinstance(cs, str):
            object.__setattr__(self, "connectivity_state", ConnectivityStatus.from_str(cs))

        hs = self.health_summary
        if isinstance(hs, str):
            object.__setattr__(self, "health_summary", DeviceHealthStatus.from_str(hs))

        object.__setattr__(self, "timestamp", float(self.timestamp))
        object.__setattr__(self, "sequence_number", int(self.sequence_number))
        if self.metrics:
            object.__setattr__(self, "metrics", sanitize_contract_metadata(self.metrics))

    def get_age(self, now: Optional[float] = None) -> float:
        """Calculate elapsed seconds since this heartbeat."""
        current = now if now is not None else time.time()
        return max(0.0, current - self.timestamp)

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "device_id": self.device_id,
            "timestamp": self.timestamp,
            "sequence_number": self.sequence_number,
            "connectivity_state": self.connectivity_state.value,
            "health_summary": self.health_summary.value,
        }
        if self.metrics:
            data["metrics"] = sanitize_contract_metadata(self.metrics)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DeviceHeartbeat":
        return cls(
            device_id=data["device_id"],
            timestamp=float(data.get("timestamp", time.time())),
            sequence_number=int(data.get("sequence_number", 0)),
            connectivity_state=ConnectivityStatus.from_str(data.get("connectivity_state", "ONLINE")),
            health_summary=DeviceHealthStatus.from_str(data.get("health_summary", "HEALTHY")),
            metrics=dict(data.get("metrics", {})),
        )


@dataclass(frozen=True)
class DeviceHealth:
    """
    Immutable semantic health assessment of an edge product.
    Summarizes connectivity, battery, capability availability, heartbeat age, and recent error.
    Contains ZERO credentials or private state.
    """
    device_id: str
    status: DeviceHealthStatus
    connectivity: ConnectivityStatus
    battery_pct: Optional[float] = None
    capability_availability: Dict[str, bool] = field(default_factory=dict)
    last_heartbeat: Optional[float] = None
    last_successful_command: Optional[float] = None
    latest_normalized_error: Optional[str] = None
    evaluated_at: float = field(default_factory=time.time)

    def __post_init__(self):
        if not self.device_id or not isinstance(self.device_id, str):
            raise ValueError("DeviceHealth device_id must be a non-empty string.")

        stat = self.status
        if isinstance(stat, str):
            object.__setattr__(self, "status", DeviceHealthStatus.from_str(stat))

        conn = self.connectivity
        if isinstance(conn, str):
            object.__setattr__(self, "connectivity", ConnectivityStatus.from_str(conn))

        if self.battery_pct is not None:
            bp = float(self.battery_pct)
            if not (0.0 <= bp <= 100.0):
                raise ValueError(f"battery_pct must be in [0.0, 100.0], got {bp}")
            object.__setattr__(self, "battery_pct", bp)

        object.__setattr__(self, "evaluated_at", float(self.evaluated_at))
        if self.capability_availability:
            object.__setattr__(self, "capability_availability", dict(self.capability_availability))

    def is_healthy(self) -> bool:
        return self.status == DeviceHealthStatus.HEALTHY

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "device_id": self.device_id,
            "status": self.status.value,
            "connectivity": self.connectivity.value,
            "evaluated_at": self.evaluated_at,
        }
        if self.battery_pct is not None:
            data["battery_pct"] = round(self.battery_pct, 1)
        if self.capability_availability:
            data["capability_availability"] = dict(self.capability_availability)
        if self.last_heartbeat is not None:
            data["last_heartbeat"] = self.last_heartbeat
        if self.last_successful_command is not None:
            data["last_successful_command"] = self.last_successful_command
        if self.latest_normalized_error is not None:
            data["latest_normalized_error"] = self.latest_normalized_error
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DeviceHealth":
        return cls(
            device_id=data["device_id"],
            status=DeviceHealthStatus.from_str(data["status"]),
            connectivity=ConnectivityStatus.from_str(data["connectivity"]),
            battery_pct=float(data["battery_pct"]) if data.get("battery_pct") is not None else None,
            capability_availability=dict(data.get("capability_availability", {})),
            last_heartbeat=float(data["last_heartbeat"]) if data.get("last_heartbeat") is not None else None,
            last_successful_command=float(data["last_successful_command"]) if data.get("last_successful_command") is not None else None,
            latest_normalized_error=data.get("latest_normalized_error"),
            evaluated_at=float(data.get("evaluated_at", time.time())),
        )


# ============================================================================
# 5. Command Contracts & Lineage
# ============================================================================

@dataclass(frozen=True)
class DeviceCommandRequest:
    """
    Immutable, transport-neutral command request sent to an edge device adapter.
    Preserves complete causal lineage across goal, turn, and objective scopes.
    """
    command_id: str
    device_id: str
    capability: str
    action: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    correlation_id: str = ""
    causation_id: Optional[str] = None
    turn_id: Optional[str] = None
    goal_id: Optional[str] = None
    objective_id: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    contract_version: str = "1.0"

    def __post_init__(self):
        if not self.command_id or not isinstance(self.command_id, str):
            raise ValueError("DeviceCommandRequest command_id must be a non-empty string.")
        if not self.device_id or not isinstance(self.device_id, str):
            raise ValueError("DeviceCommandRequest device_id must be a non-empty string.")
        if not self.capability or not isinstance(self.capability, str):
            raise ValueError("DeviceCommandRequest capability must be a non-empty string.")
        if not self.action or not isinstance(self.action, str):
            raise ValueError("DeviceCommandRequest action must be a non-empty string.")

        cid = self.correlation_id if self.correlation_id else self.command_id
        object.__setattr__(self, "correlation_id", cid)
        object.__setattr__(self, "created_at", float(self.created_at))
        if self.parameters:
            object.__setattr__(self, "parameters", sanitize_contract_metadata(self.parameters))

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "command_id": self.command_id,
            "device_id": self.device_id,
            "capability": self.capability,
            "action": self.action,
            "parameters": sanitize_contract_metadata(self.parameters),
            "correlation_id": self.correlation_id,
            "created_at": self.created_at,
            "contract_version": self.contract_version,
        }
        if self.causation_id is not None:
            data["causation_id"] = self.causation_id
        if self.turn_id is not None:
            data["turn_id"] = self.turn_id
        if self.goal_id is not None:
            data["goal_id"] = self.goal_id
        if self.objective_id is not None:
            data["objective_id"] = self.objective_id
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DeviceCommandRequest":
        return cls(
            command_id=data["command_id"],
            device_id=data["device_id"],
            capability=data["capability"],
            action=data["action"],
            parameters=dict(data.get("parameters", {})),
            correlation_id=data.get("correlation_id", ""),
            causation_id=data.get("causation_id"),
            turn_id=data.get("turn_id"),
            goal_id=data.get("goal_id"),
            objective_id=data.get("objective_id"),
            created_at=float(data.get("created_at", time.time())),
            contract_version=data.get("contract_version", "1.0"),
        )


@dataclass(frozen=True)
class DeviceCommandAcknowledgement:
    """
    Immutable acknowledgement confirming or rejecting command receipt.
    Distinguishes command ingestion/dispatch from command execution completion.
    """
    command_id: str
    device_id: str
    status: AcknowledgementStatus
    timestamp: float = field(default_factory=time.time)
    message: str = ""
    error_code: Optional[DeviceErrorCode] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.command_id or not isinstance(self.command_id, str):
            raise ValueError("DeviceCommandAcknowledgement command_id must be a non-empty string.")
        if not self.device_id or not isinstance(self.device_id, str):
            raise ValueError("DeviceCommandAcknowledgement device_id must be a non-empty string.")

        st = self.status
        if isinstance(st, str):
            object.__setattr__(self, "status", AcknowledgementStatus.from_str(st))

        ec = self.error_code
        if ec is not None and isinstance(ec, str):
            object.__setattr__(self, "error_code", DeviceErrorCode.from_str(ec))

        object.__setattr__(self, "timestamp", float(self.timestamp))
        if self.metadata:
            object.__setattr__(self, "metadata", sanitize_contract_metadata(self.metadata))

    def is_accepted(self) -> bool:
        return self.status in (AcknowledgementStatus.ACCEPTED, AcknowledgementStatus.QUEUED, AcknowledgementStatus.EXECUTING, AcknowledgementStatus.COMPLETED)

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "command_id": self.command_id,
            "device_id": self.device_id,
            "status": self.status.value,
            "timestamp": self.timestamp,
            "message": self.message,
        }
        if self.error_code is not None:
            data["error_code"] = self.error_code.value
        if self.metadata:
            data["metadata"] = sanitize_contract_metadata(self.metadata)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DeviceCommandAcknowledgement":
        return cls(
            command_id=data["command_id"],
            device_id=data["device_id"],
            status=AcknowledgementStatus.from_str(data["status"]),
            timestamp=float(data.get("timestamp", time.time())),
            message=data.get("message", ""),
            error_code=DeviceErrorCode.from_str(data["error_code"]) if data.get("error_code") else None,
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class DeviceCommandResult:
    """
    Immutable result of an executed edge command.
    Bridges cleanly to Result, preserving observations and lineage.
    """
    command_id: str
    device_id: str
    state: CommandState
    success: bool
    message: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    observations: Tuple[MultimodalObservation, ...] = ()
    error: Optional[DeviceError] = None
    correlation_id: str = ""
    causation_id: Optional[str] = None
    completed_at: float = field(default_factory=time.time)
    duration_seconds: float = 0.0

    def __post_init__(self):
        if not self.command_id or not isinstance(self.command_id, str):
            raise ValueError("DeviceCommandResult command_id must be a non-empty string.")
        if not self.device_id or not isinstance(self.device_id, str):
            raise ValueError("DeviceCommandResult device_id must be a non-empty string.")

        st = self.state
        if isinstance(st, str):
            object.__setattr__(self, "state", CommandState.from_str(st))

        object.__setattr__(self, "completed_at", float(self.completed_at))
        object.__setattr__(self, "duration_seconds", max(0.0, float(self.duration_seconds)))

        obs_list = []
        if self.observations:
            for o in self.observations:
                if isinstance(o, dict):
                    obs_list.append(MultimodalObservation.from_dict(o))
                elif isinstance(o, MultimodalObservation):
                    obs_list.append(o)
                else:
                    raise ValueError(f"Invalid observation type: {type(o)}")
        object.__setattr__(self, "observations", tuple(obs_list))

        if self.error is not None and isinstance(self.error, dict):
            object.__setattr__(self, "error", DeviceError.from_dict(self.error))

        if self.data:
            object.__setattr__(self, "data", sanitize_contract_metadata(self.data))

    def to_result(self, capability: str = "device_gateway", action: str = "") -> Result:
        """Bridge into standard Result execution model."""
        res_data = dict(self.data)
        if self.observations:
            res_data["observations"] = list(self.observations)
        if self.correlation_id:
            res_data["correlation_id"] = self.correlation_id
        if self.error is not None:
            res_data["error_code"] = self.error.code.value
            res_data["error_details"] = self.error.to_dict()

        if self.success:
            return Result.ok(
                message=self.message or f"Command '{self.command_id}' completed successfully on '{self.device_id}'.",
                data=res_data,
                capability=capability,
                action=action,
                call_id=self.command_id,
            )
        else:
            return Result.fail(
                message=self.message or (self.error.message if self.error else f"Command '{self.command_id}' failed on '{self.device_id}'."),
                data=res_data,
                capability=capability,
                action=action,
                call_id=self.command_id,
            )

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "command_id": self.command_id,
            "device_id": self.device_id,
            "state": self.state.value,
            "success": self.success,
            "message": self.message,
            "completed_at": self.completed_at,
            "duration_seconds": round(self.duration_seconds, 4),
        }
        if self.data:
            data["data"] = sanitize_contract_metadata(self.data)
        if self.observations:
            data["observations"] = [o.to_dict() for o in self.observations]
        if self.error is not None:
            data["error"] = self.error.to_dict()
        if self.correlation_id:
            data["correlation_id"] = self.correlation_id
        if self.causation_id is not None:
            data["causation_id"] = self.causation_id
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DeviceCommandResult":
        obs = tuple(
            MultimodalObservation.from_dict(o) for o in data.get("observations", [])
        )
        err = DeviceError.from_dict(data["error"]) if data.get("error") else None
        return cls(
            command_id=data["command_id"],
            device_id=data["device_id"],
            state=CommandState.from_str(data["state"]),
            success=bool(data["success"]),
            message=data.get("message", ""),
            data=dict(data.get("data", {})),
            observations=obs,
            error=err,
            correlation_id=data.get("correlation_id", ""),
            causation_id=data.get("causation_id"),
            completed_at=float(data.get("completed_at", time.time())),
            duration_seconds=float(data.get("duration_seconds", 0.0)),
        )


# ============================================================================
# 6. Canonical Telemetry Contract
# ============================================================================

@dataclass(frozen=True)
class DeviceTelemetry:
    """
    Immutable canonical telemetry emitted by edge devices.
    Converts directly to MultimodalObservation before CentralInputGateway ingress.
    Does NOT directly mutate WorldState.
    """
    device_id: str
    timestamp: float = field(default_factory=time.time)
    location: Optional[GeoLocation] = None
    battery_pct: Optional[float] = None
    connectivity: ConnectivityStatus = ConnectivityStatus.ONLINE
    health: DeviceHealthStatus = DeviceHealthStatus.HEALTHY
    metrics: Dict[str, Any] = field(default_factory=dict)
    source: str = ""
    correlation_id: str = ""
    causation_id: Optional[str] = None
    extension_data: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.device_id or not isinstance(self.device_id, str):
            raise ValueError("DeviceTelemetry device_id must be a non-empty string.")

        object.__setattr__(self, "timestamp", float(self.timestamp))
        if self.battery_pct is not None:
            bp = float(self.battery_pct)
            if not (0.0 <= bp <= 100.0):
                raise ValueError(f"battery_pct must be in [0.0, 100.0], got {bp}")
            object.__setattr__(self, "battery_pct", bp)

        conn = self.connectivity
        if isinstance(conn, str):
            object.__setattr__(self, "connectivity", ConnectivityStatus.from_str(conn))

        hlth = self.health
        if isinstance(hlth, str):
            object.__setattr__(self, "health", DeviceHealthStatus.from_str(hlth))

        if self.location is not None and isinstance(self.location, dict):
            object.__setattr__(self, "location", GeoLocation.from_dict(self.location))

        src = self.source or self.device_id
        object.__setattr__(self, "source", src)

        cid = self.correlation_id or f"telem_{self.device_id}_{int(self.timestamp * 1000)}"
        object.__setattr__(self, "correlation_id", cid)

        if self.metrics:
            object.__setattr__(self, "metrics", sanitize_contract_metadata(self.metrics))
        if self.extension_data:
            object.__setattr__(self, "extension_data", sanitize_contract_metadata(self.extension_data))

    def to_multimodal_observation(self, observation_id: Optional[str] = None) -> MultimodalObservation:
        """
        Convert canonical telemetry into an immutable MultimodalObservation
        ready for CentralInputGateway ingress.
        """
        obs_id = observation_id or f"obs_telem_{self.device_id}_{int(self.timestamp * 1000)}"
        payload = {
            "battery_pct": self.battery_pct,
            "connectivity": self.connectivity.value,
            "health": self.health.value,
            "metrics": dict(self.metrics),
            **(self.extension_data),
        }
        return MultimodalObservation(
            observation_id=obs_id,
            source_id=self.device_id,
            source_type="DEVICE_TELEMETRY",
            modality=ModalityType.TELEMETRY,
            timestamp=self.timestamp,
            payload=payload,
            location=self.location,
            device_id=self.device_id,
            correlation_id=self.correlation_id,
            causation_id=self.causation_id,
            metadata={"product_source": self.source},
        )

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "device_id": self.device_id,
            "timestamp": self.timestamp,
            "connectivity": self.connectivity.value,
            "health": self.health.value,
            "source": self.source,
            "correlation_id": self.correlation_id,
        }
        if self.location is not None:
            data["location"] = self.location.to_dict()
        if self.battery_pct is not None:
            data["battery_pct"] = round(self.battery_pct, 1)
        if self.metrics:
            data["metrics"] = sanitize_contract_metadata(self.metrics)
        if self.causation_id is not None:
            data["causation_id"] = self.causation_id
        if self.extension_data:
            data["extension_data"] = sanitize_contract_metadata(self.extension_data)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DeviceTelemetry":
        loc = GeoLocation.from_dict(data["location"]) if data.get("location") is not None else None
        return cls(
            device_id=data["device_id"],
            timestamp=float(data.get("timestamp", time.time())),
            location=loc,
            battery_pct=float(data["battery_pct"]) if data.get("battery_pct") is not None else None,
            connectivity=ConnectivityStatus.from_str(data.get("connectivity", "ONLINE")),
            health=DeviceHealthStatus.from_str(data.get("health", "HEALTHY")),
            metrics=dict(data.get("metrics", {})),
            source=data.get("source", ""),
            correlation_id=data.get("correlation_id", ""),
            causation_id=data.get("causation_id"),
            extension_data=dict(data.get("extension_data", {})),
        )


# ============================================================================
# 7. Device Contract Declaration
# ============================================================================

@dataclass(frozen=True)
class DeviceContract:
    """
    Top-level immutable semantic contract for an ATLAS edge product.
    Describes product type, role, capabilities, and version constraints.
    """
    device_id: str
    product_type: ProductType
    role: ProductRole
    contract_version: str = "1.0"
    supported_schema_versions: Tuple[str, ...] = ("1.0",)
    capabilities: Tuple[str, ...] = ()
    is_simulation: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.device_id or not isinstance(self.device_id, str):
            raise ValueError("DeviceContract device_id must be a non-empty string.")

        pt = self.product_type
        if isinstance(pt, str):
            object.__setattr__(self, "product_type", ProductType.from_str(pt))

        pr = self.role
        if isinstance(pr, str):
            object.__setattr__(self, "role", ProductRole.from_str(pr))

        caps = tuple(self.capabilities) if self.capabilities else ()
        object.__setattr__(self, "capabilities", caps)

        schemas = tuple(self.supported_schema_versions) if self.supported_schema_versions else ("1.0",)
        object.__setattr__(self, "supported_schema_versions", schemas)

        if self.metadata:
            object.__setattr__(self, "metadata", sanitize_contract_metadata(self.metadata))

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "device_id": self.device_id,
            "product_type": self.product_type.value,
            "role": self.role.value,
            "contract_version": self.contract_version,
            "supported_schema_versions": list(self.supported_schema_versions),
            "capabilities": list(self.capabilities),
            "is_simulation": self.is_simulation,
        }
        if self.metadata:
            data["metadata"] = sanitize_contract_metadata(self.metadata)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DeviceContract":
        return cls(
            device_id=data["device_id"],
            product_type=ProductType.from_str(data["product_type"]),
            role=ProductRole.from_str(data["role"]),
            contract_version=data.get("contract_version", "1.0"),
            supported_schema_versions=tuple(data.get("supported_schema_versions", ["1.0"])),
            capabilities=tuple(data.get("capabilities", [])),
            is_simulation=bool(data.get("is_simulation", False)),
            metadata=dict(data.get("metadata", {})),
        )
