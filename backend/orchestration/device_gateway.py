import collections
from dataclasses import dataclass, field
import hashlib
import json
import logging
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

from core.interfaces.orchestration_interface import (
    DeviceAdapterInterface,
    DeviceGatewayInterface,
)
from core.interfaces.runtime_interface import CognitiveEventSinkInterface
from core.models.orchestration import (
    ConnectivityStatus,
    DeviceCapabilityDescriptor,
    DeviceIdentity,
    DeviceType,
    MultimodalObservation,
)
from core.models.device_contract import (
    AcknowledgementStatus,
    CommandState,
    DeviceCommandAcknowledgement,
    DeviceCommandRequest,
    DeviceCommandResult,
    DeviceContract,
    DeviceError,
    DeviceErrorCode,
    DeviceHealth,
    DeviceHealthStatus,
    DeviceHeartbeat,
    DeviceTelemetry,
    ProductRole,
    ProductType,
    sanitize_contract_metadata,
)
from core.models.result import Result
from core.models.runtime import (
    CognitiveEvent,
    CognitiveEventType,
    CognitiveStage,
    sanitize_event_metadata,
)
from core.models.tool_call import ToolCall

logger = logging.getLogger("atlas.orchestration.device_gateway")

SUPPORTED_CONTRACT_VERSIONS: Set[str] = {"1.0"}
SUPPORTED_CAPABILITY_SCHEMA_VERSIONS: Set[str] = {"1.0"}
MAX_CAPABILITIES_PER_DEVICE: int = 100
MAX_METADATA_BYTES: int = 50_000


# ============================================================================
# Semantic Device Command Model
# ============================================================================

@dataclass(frozen=True)
class DeviceCommand:
    """
    Semantic, hardware-independent command dispatched to an edge device adapter.
    Preserves complete lineage: goal, turn, and objective identifiers.
    """
    dispatch_id: str
    device_id: str
    capability: str
    action: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    correlation_id: str = ""
    causation_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    turn_id: Optional[str] = None
    goal_id: Optional[str] = None
    objective_id: Optional[str] = None
    contract_version: str = "1.0"

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "dispatch_id": self.dispatch_id,
            "device_id": self.device_id,
            "capability": self.capability,
            "action": self.action,
            "parameters": dict(self.parameters),
            "correlation_id": self.correlation_id,
            "causation_id": self.causation_id,
            "timestamp": self.timestamp,
            "contract_version": self.contract_version,
        }
        if self.turn_id is not None:
            data["turn_id"] = self.turn_id
        if self.goal_id is not None:
            data["goal_id"] = self.goal_id
        if self.objective_id is not None:
            data["objective_id"] = self.objective_id
        return data

    def to_request(self) -> DeviceCommandRequest:
        """Convert to domain DeviceCommandRequest."""
        return DeviceCommandRequest(
            command_id=self.dispatch_id,
            device_id=self.device_id,
            capability=self.capability,
            action=self.action,
            parameters=dict(self.parameters),
            correlation_id=self.correlation_id,
            causation_id=self.causation_id,
            turn_id=self.turn_id,
            goal_id=self.goal_id,
            objective_id=self.objective_id,
            created_at=self.timestamp,
            contract_version=self.contract_version,
        )


# ============================================================================
# Parameter Validation Helper
# ============================================================================

def validate_parameters_against_schema(
    parameters: Dict[str, Any],
    schema: Optional[Dict[str, Any]],
) -> Tuple[bool, Optional[str]]:
    """
    Deterministic validation of command parameters against capability parameters_schema.
    Supports:
      - required: list of required parameter names
      - properties: dict of param_name -> {type, minimum, maximum, min, max, enum}
      - type enforcement: str, int, float, bool, dict, list
    """
    if not schema:
        return True, None

    # Check required fields
    required = schema.get("required", [])
    for req in required:
        if req not in parameters:
            return False, f"Missing required parameter: '{req}'"

    # Check properties
    properties = schema.get("properties", {})
    for param_name, param_val in parameters.items():
        if param_name not in properties:
            if schema.get("additionalProperties") is False:
                return False, f"Unknown parameter '{param_name}' not allowed by schema"
            continue

        rules = properties[param_name]
        expected_type = rules.get("type")

        # Type checking
        if expected_type:
            t = expected_type.lower()
            if t in ("string", "str"):
                if not isinstance(param_val, str):
                    return False, f"Parameter '{param_name}' must be a string, got {type(param_val).__name__}"
            elif t in ("integer", "int"):
                if not isinstance(param_val, int) or isinstance(param_val, bool):
                    return False, f"Parameter '{param_name}' must be an integer, got {type(param_val).__name__}"
            elif t in ("number", "float"):
                if not isinstance(param_val, (int, float)) or isinstance(param_val, bool):
                    return False, f"Parameter '{param_name}' must be a number, got {type(param_val).__name__}"
            elif t in ("boolean", "bool"):
                if not isinstance(param_val, bool):
                    return False, f"Parameter '{param_name}' must be a boolean, got {type(param_val).__name__}"
            elif t in ("dict", "object"):
                if not isinstance(param_val, dict):
                    return False, f"Parameter '{param_name}' must be a dictionary, got {type(param_val).__name__}"
            elif t in ("list", "array"):
                if not isinstance(param_val, (list, tuple)):
                    return False, f"Parameter '{param_name}' must be a list, got {type(param_val).__name__}"

        # Numeric bounds
        min_val = rules.get("minimum", rules.get("min"))
        if min_val is not None and isinstance(param_val, (int, float)):
            if param_val < min_val:
                return False, f"Parameter '{param_name}' value {param_val} is less than minimum {min_val}"

        max_val = rules.get("maximum", rules.get("max"))
        if max_val is not None and isinstance(param_val, (int, float)):
            if param_val > max_val:
                return False, f"Parameter '{param_name}' value {param_val} exceeds maximum {max_val}"

        # Enum check
        allowed_enum = rules.get("enum")
        if allowed_enum is not None and param_val not in allowed_enum:
            return False, f"Parameter '{param_name}' value '{param_val}' must be one of {allowed_enum}"

    return True, None


# ============================================================================
# DeviceGateway Implementation
# ============================================================================

class DeviceGateway(DeviceGatewayInterface):
    """
    Central Device & Product Gateway for ATLAS Central Orchestration Layer (Phase 6.2).

    Responsibilities:
    - Maintains in-memory registry of edge participant DeviceIdentity profiles across
      four core products: ATLAS Vision, ATLAS Glass, ATLAS Drone, ATLAS Rover.
    - Manages capability declarations and validates parameters against schema and version contracts.
    - Resolves transport-specific or virtual DeviceAdapters by device_id or product_type/device_type.
    - Tracks operational health, heartbeats, and connectivity on demand without daemon watchdog loops.
    - Enforces idempotency via bounded LRU duplicate command suppression.
    - Emits structured cognitive events for all lifecycle transitions.

    CRITICAL ARCHITECTURAL BOUNDARIES:
    - Zero direct hardware, serial, MAVLink, ROS2, MQTT, BLE, socket, or network execution.
    - Zero autonomous reasoning or LLM routing.
    - Zero direct world-state or goal-store mutations.
    - Never bypasses ToolOrchestrator: execution requests arrive via governed capability paths.
    - PolicyEngine remains the authorization authority; gateway validates capability & dispatchability.
    """

    def __init__(
        self,
        event_sink: Optional[CognitiveEventSinkInterface] = None,
        clock: Optional[Callable[[], float]] = None,
        heartbeat_timeout_seconds: float = 60.0,
        max_devices: int = 1000,
        max_dispatch_history: int = 5000,
        max_capabilities_per_device: int = MAX_CAPABILITIES_PER_DEVICE,
        max_metadata_bytes: int = MAX_METADATA_BYTES,
    ):
        self.event_sink = event_sink
        self.clock = clock or time.time
        self.heartbeat_timeout_seconds = heartbeat_timeout_seconds
        self.max_devices = max_devices
        self.max_dispatch_history = max_dispatch_history
        self.max_capabilities_per_device = max_capabilities_per_device
        self.max_metadata_bytes = max_metadata_bytes

        self._lock = threading.RLock()
        self._devices: Dict[str, DeviceIdentity] = {}
        self._type_adapters: Dict[DeviceType, DeviceAdapterInterface] = {}
        self._product_adapters: Dict[str, DeviceAdapterInterface] = {}
        self._device_adapters: Dict[str, DeviceAdapterInterface] = {}
        self._seen_dispatch_ids: collections.deque = collections.deque(maxlen=max_dispatch_history)
        self._seen_dispatch_set: Set[str] = set()
        self._dispatch_results: Dict[str, Result] = {}

        # Health, telemetry, and error tracking
        self._last_successful_command: Dict[str, float] = {}
        self._latest_errors: Dict[str, str] = {}
        self._heartbeat_sequences: Dict[str, int] = {}
        self._latest_heartbeats: Dict[str, DeviceHeartbeat] = {}
        self._latest_telemetry: Dict[str, DeviceTelemetry] = {}
        self._telemetry_history: Dict[str, collections.deque] = {}
        self._latest_acknowledgements: Dict[str, DeviceCommandAcknowledgement] = {}

    # ========================================================================
    # Device Registry Operations
    # ========================================================================

    def register_device(self, device: DeviceIdentity) -> None:
        """
        Register a device identity profile.
        Rejects duplicate device IDs deterministically to prevent silent overwrites.
        Enforces bounds on metadata size and capability count.
        """
        if not isinstance(device, DeviceIdentity):
            raise TypeError(f"Expected DeviceIdentity instance, got {type(device).__name__}")

        with self._lock:
            if device.device_id in self._devices:
                raise ValueError(f"Device '{device.device_id}' is already registered.")

            if len(self._devices) >= self.max_devices:
                raise ValueError(f"Maximum registered devices limit ({self.max_devices}) reached.")

            if len(device.capabilities) > self.max_capabilities_per_device:
                raise ValueError(
                    f"Device '{device.device_id}' declares {len(device.capabilities)} capabilities, "
                    f"exceeding limit of {self.max_capabilities_per_device}."
                )

            # Metadata size check
            if device.metadata:
                try:
                    meta_bytes = len(json.dumps(device.metadata))
                    if meta_bytes > self.max_metadata_bytes:
                        raise ValueError(
                            f"Device '{device.device_id}' metadata size ({meta_bytes} bytes) "
                            f"exceeds maximum allowed ({self.max_metadata_bytes} bytes)."
                        )
                except (TypeError, OverflowError):
                    pass

            # Validate capability schema versions
            for cap in device.capabilities:
                schema_ver = getattr(cap, "schema_version", "1.0")
                if schema_ver not in SUPPORTED_CAPABILITY_SCHEMA_VERSIONS:
                    raise ValueError(
                        f"Capability '{cap.capability_name}' declares unsupported schema_version '{schema_ver}'."
                    )

            self._devices[device.device_id] = device

        self._emit_event(
            CognitiveEventType.DEVICE_REGISTERED,
            device_id=device.device_id,
            metadata={
                "device_type": device.device_type.value,
                "product_type": getattr(device, "product_type", "UNKNOWN"),
                "product_role": getattr(device, "product_role", "HYBRID"),
                "display_name": device.display_name,
                "is_simulation": device.is_simulation,
            },
        )

    def unregister_device(self, device_id: str) -> bool:
        """
        Unregister an edge participant device.
        """
        with self._lock:
            removed = self._devices.pop(device_id, None)
            self._device_adapters.pop(device_id, None)
            self._last_successful_command.pop(device_id, None)
            self._latest_errors.pop(device_id, None)
            self._latest_heartbeats.pop(device_id, None)
            self._latest_telemetry.pop(device_id, None)
            self._telemetry_history.pop(device_id, None)

        if removed:
            self._emit_event(
                CognitiveEventType.DEVICE_UNREGISTERED,
                device_id=device_id,
                metadata={"device_type": removed.device_type.value},
            )
            return True
        return False

    def get_device(self, device_id: str) -> Optional[DeviceIdentity]:
        """Retrieve registered device identity by device_id."""
        with self._lock:
            return self._devices.get(device_id)

    def list_devices(
        self,
        device_type: Optional[DeviceType] = None,
        product_type: Optional[ProductType] = None,
    ) -> Sequence[DeviceIdentity]:
        """List registered devices, optionally filtered by DeviceType or ProductType."""
        with self._lock:
            devices = list(self._devices.values())

        if device_type is not None:
            norm_type = (
                device_type
                if isinstance(device_type, DeviceType)
                else DeviceType.from_str(device_type)
            )
            devices = [d for d in devices if d.device_type == norm_type]

        if product_type is not None:
            norm_prod = (
                product_type.value
                if isinstance(product_type, ProductType)
                else str(product_type).upper()
            )
            devices = [d for d in devices if getattr(d, "product_type", "") == norm_prod]

        # Deterministic sorting by device_id
        return tuple(sorted(devices, key=lambda d: d.device_id))

    def update_device_status(
        self,
        device_id: str,
        status: ConnectivityStatus,
        timestamp: Optional[float] = None,
    ) -> bool:
        """
        Update the connectivity status of a registered device.
        """
        now = timestamp if timestamp is not None else self.clock()
        norm_status = (
            status
            if isinstance(status, ConnectivityStatus)
            else ConnectivityStatus.from_str(status)
        )

        with self._lock:
            dev = self._devices.get(device_id)
            if not dev:
                return False

            updated = DeviceIdentity(
                device_id=dev.device_id,
                device_type=dev.device_type,
                display_name=dev.display_name,
                firmware_version=dev.firmware_version,
                capabilities=dev.capabilities,
                home_location=dev.home_location,
                is_simulation=dev.is_simulation,
                registered_at=dev.registered_at,
                last_heartbeat=now,
                connectivity_status=norm_status,
                product_type=getattr(dev, "product_type", None),
                product_role=getattr(dev, "product_role", None),
                vendor=getattr(dev, "vendor", None),
                model=getattr(dev, "model", None),
                contract_version=getattr(dev, "contract_version", "1.0"),
                metadata=dict(dev.metadata),
            )
            self._devices[device_id] = updated

        self._emit_event(
            CognitiveEventType.DEVICE_STATUS_UPDATED,
            device_id=device_id,
            metadata={"status": norm_status.value, "timestamp": now},
        )
        return True

    # ========================================================================
    # Heartbeat & Health Operations
    # ========================================================================

    def record_heartbeat(
        self,
        device_id: str,
        timestamp: Optional[float] = None,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Record a heartbeat timestamp and beacon for an edge device.
        Automatically marks connectivity as ONLINE if previously disconnected.
        Increments deterministic sequence counter.
        """
        now = timestamp if timestamp is not None else self.clock()
        with self._lock:
            dev = self._devices.get(device_id)
            if not dev:
                return False

            seq = self._heartbeat_sequences.get(device_id, 0) + 1
            self._heartbeat_sequences[device_id] = seq

            new_status = dev.connectivity_status
            if new_status in (
                ConnectivityStatus.DISCONNECTED,
                ConnectivityStatus.OFFLINE,
                ConnectivityStatus.UNKNOWN,
            ):
                new_status = ConnectivityStatus.ONLINE

            updated = DeviceIdentity(
                device_id=dev.device_id,
                device_type=dev.device_type,
                display_name=dev.display_name,
                firmware_version=dev.firmware_version,
                capabilities=dev.capabilities,
                home_location=dev.home_location,
                is_simulation=dev.is_simulation,
                registered_at=dev.registered_at,
                last_heartbeat=now,
                connectivity_status=new_status,
                product_type=getattr(dev, "product_type", None),
                product_role=getattr(dev, "product_role", None),
                vendor=getattr(dev, "vendor", None),
                model=getattr(dev, "model", None),
                contract_version=getattr(dev, "contract_version", "1.0"),
                metadata=dict(dev.metadata),
            )
            self._devices[device_id] = updated

            hb = DeviceHeartbeat(
                device_id=device_id,
                timestamp=now,
                sequence_number=seq,
                connectivity_state=new_status,
                health_summary=DeviceHealthStatus.HEALTHY,
                metrics=metrics or {},
            )
            self._latest_heartbeats[device_id] = hb

        return True

    def query_device_status(self, device_id: str, now: Optional[float] = None) -> ConnectivityStatus:
        """
        Evaluate and return the connectivity status of a device.
        Derives DISCONNECTED if last_heartbeat exceeds heartbeat_timeout_seconds.
        """
        current_time = now if now is not None else self.clock()
        with self._lock:
            dev = self._devices.get(device_id)
            if not dev:
                return ConnectivityStatus.UNKNOWN

            if dev.last_heartbeat is not None:
                elapsed = current_time - dev.last_heartbeat
                if elapsed > self.heartbeat_timeout_seconds:
                    return ConnectivityStatus.DISCONNECTED

            return dev.connectivity_status

    def evaluate_heartbeat(self, device_id: str, now: Optional[float] = None) -> Dict[str, Any]:
        """
        Evaluate heartbeat freshness and elapsed age deterministically without watchdogs.
        """
        current_time = now if now is not None else self.clock()
        with self._lock:
            dev = self._devices.get(device_id)
            if not dev:
                return {"device_id": device_id, "status": "UNKNOWN", "exists": False}

            last_hb = dev.last_heartbeat
            age = (current_time - last_hb) if last_hb is not None else None
            status = self.query_device_status(device_id, now=current_time)
            seq = self._heartbeat_sequences.get(device_id, 0)
            is_expired = (age > self.heartbeat_timeout_seconds) if age is not None else True

            return {
                "device_id": device_id,
                "last_heartbeat": last_hb,
                "age_seconds": round(age, 4) if age is not None else None,
                "sequence_number": seq,
                "connectivity_status": status.value,
                "is_expired": is_expired,
            }

    def get_device_health(self, device_id: str, now: Optional[float] = None) -> DeviceHealth:
        """
        Compute on-demand semantic health assessment of a registered edge device.
        Summarizes connectivity, battery, capability availability, heartbeat age, and recent errors.
        """
        current_time = now if now is not None else self.clock()
        with self._lock:
            dev = self._devices.get(device_id)
            if not dev:
                return DeviceHealth(
                    device_id=device_id,
                    status=DeviceHealthStatus.UNKNOWN,
                    connectivity=ConnectivityStatus.UNKNOWN,
                    evaluated_at=current_time,
                )

            conn = self.query_device_status(device_id, now=current_time)
            last_hb = dev.last_heartbeat
            last_cmd = self._last_successful_command.get(device_id)
            last_err = self._latest_errors.get(device_id)
            telem = self._latest_telemetry.get(device_id)
            battery = telem.battery_pct if telem and telem.battery_pct is not None else None
            if battery is None and device_id in self._device_adapters:
                adapter = self._device_adapters[device_id]
                if hasattr(adapter, "get_health"):
                    try:
                        adapter_health = adapter.get_health(device_id)
                        battery = getattr(adapter_health, "battery_pct", None)
                    except Exception:
                        pass
                if battery is None and hasattr(adapter, "battery_pct"):
                    battery = getattr(adapter, "battery_pct", None)

            # Capability availability map
            cap_avail = {c.capability_name: True for c in dev.capabilities}

            # Evaluate health status
            if conn in (ConnectivityStatus.OFFLINE, ConnectivityStatus.DISCONNECTED):
                health_stat = DeviceHealthStatus.UNHEALTHY
            elif battery is not None and battery <= 5.0:
                health_stat = DeviceHealthStatus.UNHEALTHY
            elif conn == ConnectivityStatus.DEGRADED:
                health_stat = DeviceHealthStatus.DEGRADED
            elif battery is not None and battery <= 20.0:
                health_stat = DeviceHealthStatus.DEGRADED
            elif last_hb is not None and (current_time - last_hb) > (self.heartbeat_timeout_seconds * 0.8):
                health_stat = DeviceHealthStatus.DEGRADED
            elif conn == ConnectivityStatus.ONLINE:
                health_stat = DeviceHealthStatus.HEALTHY
            else:
                health_stat = DeviceHealthStatus.UNKNOWN

            return DeviceHealth(
                device_id=device_id,
                status=health_stat,
                connectivity=conn,
                battery_pct=battery,
                capability_availability=cap_avail,
                last_heartbeat=last_hb,
                last_successful_command=last_cmd,
                latest_normalized_error=last_err,
                evaluated_at=current_time,
            )

    def record_telemetry(self, telemetry: DeviceTelemetry) -> MultimodalObservation:
        """
        Record canonical telemetry from an edge device into bounded history.
        Returns converted MultimodalObservation ready for CentralInputGateway.
        """
        if not isinstance(telemetry, DeviceTelemetry):
            raise TypeError(f"Expected DeviceTelemetry, got {type(telemetry).__name__}")

        with self._lock:
            dev = self._devices.get(telemetry.device_id)
            if not dev:
                raise KeyError(f"Cannot record telemetry for unknown device '{telemetry.device_id}'.")

            self._latest_telemetry[telemetry.device_id] = telemetry
            if telemetry.device_id not in self._telemetry_history:
                self._telemetry_history[telemetry.device_id] = collections.deque(maxlen=100)
            self._telemetry_history[telemetry.device_id].append(telemetry)

        return telemetry.to_multimodal_observation()

    def list_device_capabilities(self, device_id: str) -> Sequence[DeviceCapabilityDescriptor]:
        """List declared capability descriptors of a device."""
        with self._lock:
            dev = self._devices.get(device_id)
            if not dev:
                return ()
            return dev.capabilities

    # ========================================================================
    # Adapter Registration & Resolution
    # ========================================================================

    def register_adapter(
        self,
        adapter: DeviceAdapterInterface,
        device_type: Optional[DeviceType] = None,
        product_type: Optional[ProductType] = None,
        device_id: Optional[str] = None,
    ) -> None:
        """
        Register a DeviceAdapter for an exact device_id, ProductType, or DeviceType.
        """
        if not isinstance(adapter, DeviceAdapterInterface):
            raise TypeError(f"Expected DeviceAdapterInterface, got {type(adapter).__name__}")

        if device_id is None and device_type is None and product_type is None:
            raise ValueError("Must specify at least one of device_id, product_type, or device_type.")

        with self._lock:
            if device_id is not None:
                self._device_adapters[device_id] = adapter
            if product_type is not None:
                norm_prod = product_type.value if isinstance(product_type, ProductType) else str(product_type).upper()
                self._product_adapters[norm_prod] = adapter
            if device_type is not None:
                norm_type = (
                    device_type
                    if isinstance(device_type, DeviceType)
                    else DeviceType.from_str(device_type)
                )
                self._type_adapters[norm_type] = adapter

    def resolve_adapter(self, device_id: str) -> DeviceAdapterInterface:
        """
        Deterministically resolve protocol adapter for device_id.
        Priority: 1) Device-specific adapter, 2) ProductType adapter, 3) DeviceType adapter.
        """
        with self._lock:
            # 1. Device-specific adapter
            if device_id in self._device_adapters:
                return self._device_adapters[device_id]

            dev = self._devices.get(device_id)
            if dev:
                # 2. Product-specific adapter
                prod_type = getattr(dev, "product_type", None)
                if prod_type and prod_type in self._product_adapters:
                    return self._product_adapters[prod_type]

                # 3. Type-specific adapter
                if dev.device_type in self._type_adapters:
                    return self._type_adapters[dev.device_type]

        raise KeyError(f"No registered adapter found for device '{device_id}'.")

    # ========================================================================
    # Bounded Semantic Command Dispatch
    # ========================================================================

    def dispatch_to_device(
        self,
        device_id: str,
        capability: str,
        action: str,
        parameters: Dict[str, Any],
        dispatch_id: Optional[str] = None,
        correlation_id: str = "",
        causation_id: Optional[str] = None,
        now: Optional[float] = None,
        contract_version: str = "1.0",
    ) -> Result:
        """
        Orchestrate capability validation and route command to the resolved adapter.
        Guarantees:
        - Version support check (rejection of unsupported contract versions).
        - Device existence and connectivity check.
        - Declarative capability and action matching.
        - Strict parameter schema validation.
        - Duplicate dispatch_id suppression with bounded LRU.
        - Transport-neutral command acknowledgement.
        - Complete causal lineage preservation.
        """
        current_time = now if now is not None else self.clock()
        norm_cap = str(capability or "").strip().lower()
        norm_act = str(action or "").strip().lower()

        # 0. Contract version validation
        if contract_version not in SUPPORTED_CONTRACT_VERSIONS:
            err_msg = (
                f"Contract version '{contract_version}' is unsupported. "
                f"Supported versions: {sorted(SUPPORTED_CONTRACT_VERSIONS)}."
            )
            self._emit_dispatch_event(
                CognitiveEventType.DEVICE_DISPATCH_FAILED,
                device_id=device_id,
                capability=norm_cap,
                action=norm_act,
                reason=err_msg,
                now=current_time,
            )
            return Result.fail(
                message=err_msg,
                capability=norm_cap,
                action=norm_act,
                data={"error_code": DeviceErrorCode.CONTRACT_VERSION_UNSUPPORTED.value},
            )

        # 1. Device existence check
        with self._lock:
            dev = self._devices.get(device_id)

        if not dev:
            self._emit_dispatch_event(
                CognitiveEventType.DEVICE_DISPATCH_FAILED,
                device_id=device_id,
                capability=norm_cap,
                action=norm_act,
                reason=f"Unknown device '{device_id}'.",
                now=current_time,
            )
            return Result.fail(
                message=f"Device '{device_id}' is not registered in DeviceGateway.",
                capability=norm_cap,
                action=norm_act,
                data={"error_code": DeviceErrorCode.UNKNOWN_DEVICE.value},
            )

        # 2. Connectivity check
        status = self.query_device_status(device_id, now=current_time)
        if status in (ConnectivityStatus.OFFLINE, ConnectivityStatus.DISCONNECTED):
            self._emit_dispatch_event(
                CognitiveEventType.DEVICE_DISPATCH_FAILED,
                device_id=device_id,
                capability=norm_cap,
                action=norm_act,
                reason=f"Device '{device_id}' is offline ({status.value}).",
                now=current_time,
            )
            return Result.fail(
                message=f"Device '{device_id}' is unavailable: connectivity status is {status.value}.",
                capability=norm_cap,
                action=norm_act,
                data={"error_code": DeviceErrorCode.OFFLINE_DEVICE.value},
            )

        # 3. Capability and action matching
        matched_cap: Optional[DeviceCapabilityDescriptor] = None
        cap_found_by_name = False
        for cap_desc in dev.capabilities:
            c_name = cap_desc.capability_name.strip().lower()
            c_id = getattr(cap_desc, "capability_id", "").strip().lower()
            if norm_cap in (c_name, c_id):
                cap_found_by_name = True
                # Check action support
                has_action = getattr(cap_desc, "supports_action", None)
                if has_action and has_action(norm_act):
                    matched_cap = cap_desc
                    break
                elif cap_desc.action_name.strip().lower() == norm_act:
                    matched_cap = cap_desc
                    break

        if not cap_found_by_name:
            avail = [f"{c.capability_name}:{c.action_name}" for c in dev.capabilities]
            err_msg = (
                f"Device '{device_id}' does not support action '{norm_act}' on capability '{norm_cap}'. "
                f"Available capabilities: {avail}"
            )
            self._emit_dispatch_event(
                CognitiveEventType.DEVICE_DISPATCH_FAILED,
                device_id=device_id,
                capability=norm_cap,
                action=norm_act,
                reason=err_msg,
                now=current_time,
            )
            return Result.fail(
                message=err_msg,
                capability=norm_cap,
                action=norm_act,
                data={"error_code": DeviceErrorCode.UNSUPPORTED_CAPABILITY.value},
            )

        if not matched_cap:
            avail_actions = []
            for c in dev.capabilities:
                if norm_cap in (c.capability_name.strip().lower(), getattr(c, "capability_id", "").strip().lower()):
                    avail_actions.extend(getattr(c, "supported_actions", [c.action_name]))
            err_msg = (
                f"Device '{device_id}' does not support action '{norm_act}' on capability '{norm_cap}'. "
                f"Supported actions: {avail_actions}"
            )
            self._emit_dispatch_event(
                CognitiveEventType.DEVICE_DISPATCH_FAILED,
                device_id=device_id,
                capability=norm_cap,
                action=norm_act,
                reason=err_msg,
                now=current_time,
            )
            return Result.fail(
                message=err_msg,
                capability=norm_cap,
                action=norm_act,
                data={"error_code": DeviceErrorCode.UNSUPPORTED_ACTION.value},
            )

        # 4. Parameter validation against capability schema
        is_valid, param_err = validate_parameters_against_schema(
            parameters, matched_cap.parameters_schema
        )
        if not is_valid:
            self._emit_dispatch_event(
                CognitiveEventType.DEVICE_DISPATCH_FAILED,
                device_id=device_id,
                capability=norm_cap,
                action=norm_act,
                reason=param_err or "Parameter validation failed.",
                now=current_time,
            )
            return Result.fail(
                message=f"Parameter validation failed for '{norm_cap}:{norm_act}': {param_err}",
                capability=norm_cap,
                action=norm_act,
                data={"error_code": DeviceErrorCode.INVALID_PARAMETERS.value},
            )

        # 5. Dispatch ID assignment & Duplicate check
        effective_dispatch_id = dispatch_id or f"disp_{device_id}_{int(current_time * 1000)}"
        with self._lock:
            if effective_dispatch_id in self._seen_dispatch_set:
                err_dup = f"Duplicate dispatch_id '{effective_dispatch_id}' rejected."
                prev_res = self._dispatch_results.get(effective_dispatch_id)
                res_data = {"error_code": DeviceErrorCode.DUPLICATE_COMMAND.value}
                if prev_res and prev_res.data:
                    res_data.update(prev_res.data)
                return Result.fail(
                    message=err_dup,
                    capability=norm_cap,
                    action=norm_act,
                    call_id=effective_dispatch_id,
                    data=res_data,
                )

            # Record dispatch ID in bounded LRU cache
            if len(self._seen_dispatch_ids) >= self.max_dispatch_history:
                oldest = self._seen_dispatch_ids.popleft()
                self._seen_dispatch_set.discard(oldest)
            self._seen_dispatch_ids.append(effective_dispatch_id)
            self._seen_dispatch_set.add(effective_dispatch_id)

        # 6. Record Command Acknowledgement
        ack = DeviceCommandAcknowledgement(
            command_id=effective_dispatch_id,
            device_id=device_id,
            status=AcknowledgementStatus.ACCEPTED,
            timestamp=current_time,
            message=f"Command '{effective_dispatch_id}' accepted for dispatch.",
        )
        with self._lock:
            self._latest_acknowledgements[effective_dispatch_id] = ack

        # 7. Adapter resolution
        try:
            adapter = self.resolve_adapter(device_id)
        except KeyError as e:
            self._emit_dispatch_event(
                CognitiveEventType.DEVICE_ADAPTER_ERROR,
                device_id=device_id,
                capability=norm_cap,
                action=norm_act,
                reason=str(e),
                now=current_time,
            )
            with self._lock:
                self._latest_errors[device_id] = str(e)
            return Result.fail(
                message=str(e),
                capability=norm_cap,
                action=norm_act,
                call_id=effective_dispatch_id,
                data={"error_code": DeviceErrorCode.DEVICE_ERROR.value},
            )

        # 8. Construct command and delegate to adapter
        command = DeviceCommand(
            dispatch_id=effective_dispatch_id,
            device_id=device_id,
            capability=norm_cap,
            action=norm_act,
            parameters=dict(parameters),
            correlation_id=correlation_id or effective_dispatch_id,
            causation_id=causation_id,
            timestamp=current_time,
            contract_version=contract_version,
        )

        self._emit_dispatch_event(
            CognitiveEventType.DEVICE_DISPATCH_REQUESTED,
            device_id=device_id,
            capability=norm_cap,
            action=norm_act,
            reason="Dispatching to adapter",
            now=current_time,
            extra={"dispatch_id": effective_dispatch_id, "adapter": adapter.get_protocol_name()},
        )

        try:
            res = adapter.execute_command(command)
        except Exception as e:
            logger.error(f"Adapter execution failed on device '{device_id}': {e}")
            self._emit_dispatch_event(
                CognitiveEventType.DEVICE_DISPATCH_FAILED,
                device_id=device_id,
                capability=norm_cap,
                action=norm_act,
                reason=f"Adapter error: {e}",
                now=current_time,
            )
            with self._lock:
                self._latest_errors[device_id] = str(e)
            return Result.fail(
                message=f"Adapter execution error on device '{device_id}': {e}",
                capability=norm_cap,
                action=norm_act,
                call_id=effective_dispatch_id,
                data={"error_code": DeviceErrorCode.DEVICE_ERROR.value},
            )

        # 9. Record result and update health markers
        if isinstance(res.data, dict):
            if "correlation_id" not in res.data and command.correlation_id:
                res.data["correlation_id"] = command.correlation_id

        with self._lock:
            self._dispatch_results[effective_dispatch_id] = res
            if res.success:
                self._last_successful_command[device_id] = current_time
            else:
                self._latest_errors[device_id] = res.message
                if isinstance(res.data, dict) and "error_code" not in res.data:
                    res.data["error_code"] = DeviceErrorCode.DEVICE_ERROR.value

        if res.success:
            self._emit_dispatch_event(
                CognitiveEventType.DEVICE_DISPATCH_ACCEPTED,
                device_id=device_id,
                capability=norm_cap,
                action=norm_act,
                reason="Adapter executed successfully",
                now=current_time,
                extra={"dispatch_id": effective_dispatch_id},
            )
        else:
            self._emit_dispatch_event(
                CognitiveEventType.DEVICE_DISPATCH_FAILED,
                device_id=device_id,
                capability=norm_cap,
                action=norm_act,
                reason=res.message,
                now=current_time,
                extra={"dispatch_id": effective_dispatch_id},
            )

        return res

    # ========================================================================
    # Observability Helper
    # ========================================================================

    def _emit_event(
        self,
        event_type: CognitiveEventType,
        device_id: str,
        metadata: Dict[str, Any],
    ) -> None:
        """Emit structured lifecycle event via event_sink."""
        if not self.event_sink:
            return

        now = self.clock()
        safe_meta = sanitize_event_metadata({"device_id": device_id, **metadata})
        try:
            evt = CognitiveEvent(
                event_id=f"dev_evt_{device_id}_{int(now * 1000)}",
                turn_id=device_id,
                session_id="device_gateway",
                stage=CognitiveStage.EXECUTION,
                event_type=event_type,
                timestamp=now,
                metadata=safe_meta,
            )
            if hasattr(self.event_sink, "publish"):
                self.event_sink.publish(evt)
            elif hasattr(self.event_sink, "record_event"):
                self.event_sink.record_event(evt)
        except Exception as e:
            logger.debug(f"Failed to record device cognitive event: {e}")

    def _emit_dispatch_event(
        self,
        event_type: CognitiveEventType,
        device_id: str,
        capability: str,
        action: str,
        reason: str,
        now: float,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Emit structured command dispatch event via event_sink."""
        if not self.event_sink:
            return

        meta = {
            "device_id": device_id,
            "capability": capability,
            "action": action,
            "reason": reason,
            **(extra or {}),
        }
        safe_meta = sanitize_event_metadata(meta)
        try:
            evt = CognitiveEvent(
                event_id=f"disp_evt_{device_id}_{int(now * 1000)}",
                turn_id=extra.get("dispatch_id", device_id) if extra else device_id,
                session_id="device_gateway",
                stage=CognitiveStage.EXECUTION,
                event_type=event_type,
                timestamp=now,
                metadata=safe_meta,
            )
            if hasattr(self.event_sink, "publish"):
                self.event_sink.publish(evt)
            elif hasattr(self.event_sink, "record_event"):
                self.event_sink.record_event(evt)
        except Exception as e:
            logger.debug(f"Failed to record dispatch cognitive event: {e}")


# ============================================================================
# ToolOrchestrator Capability Bridge
# ============================================================================

class DeviceGatewayCapability:
    """
    Exposes DeviceGateway as an authorized capability to ToolOrchestrator.
    Conforms to standard Capability callable interface: __call__(task) -> Result.
    """

    def __init__(self, gateway: DeviceGatewayInterface):
        self.gateway = gateway

    def __call__(self, task: Any) -> Result:
        """
        Execute governed device operations requested through ToolOrchestrator.
        """
        action = getattr(task, "action", "") or ""
        params = getattr(task, "parameters", {}) or {}
        call_id = getattr(task, "call_id", None) or params.get("call_id")

        if action == "dispatch_capability":
            device_id = str(params.get("device_id", ""))
            capability = str(params.get("capability", ""))
            act = str(
                params.get("device_action")
                or (params.get("parameters", {}).get("action") if isinstance(params.get("parameters"), dict) else None)
                or (params.get("action") if params.get("action") != "dispatch_capability" else "")
                or ""
            )
            action_params = dict(params.get("parameters", {}))
            correlation_id = str(params.get("correlation_id", call_id or ""))
            causation_id = params.get("causation_id")
            contract_ver = str(params.get("contract_version", "1.0"))

            return self.gateway.dispatch_to_device(
                device_id=device_id,
                capability=capability,
                action=act,
                parameters=action_params,
                dispatch_id=call_id,
                correlation_id=correlation_id,
                causation_id=causation_id,
                contract_version=contract_ver,
            )

        elif action == "query_status":
            device_id = str(params.get("device_id", ""))
            status = self.gateway.query_device_status(device_id)
            return Result.ok(
                message=f"Device '{device_id}' status is {status.value}.",
                data={"device_id": device_id, "status": status.value},
                capability="device_gateway",
                action="query_status",
                call_id=call_id,
            )

        elif action == "get_health":
            device_id = str(params.get("device_id", ""))
            health = getattr(self.gateway, "get_device_health", lambda d: None)(device_id)
            if health:
                return Result.ok(
                    message=f"Device '{device_id}' health is {health.status.value}.",
                    data=health.to_dict(),
                    capability="device_gateway",
                    action="get_health",
                    call_id=call_id,
                )
            return Result.fail(
                message=f"Health query unsupported on gateway for '{device_id}'.",
                capability="device_gateway",
                action="get_health",
                call_id=call_id,
            )

        elif action == "get_telemetry":
            device_id = str(params.get("device_id", ""))
            health = getattr(self.gateway, "get_device_health", lambda d: None)(device_id)
            telem = getattr(self.gateway, "_latest_telemetry", {}).get(device_id)
            data = {}
            if telem:
                data = telem.to_dict() if hasattr(telem, "to_dict") else dict(telem)
            elif health:
                data = {
                    "battery_pct": health.battery_pct,
                    "connectivity": health.connectivity.value if hasattr(health.connectivity, "value") else str(health.connectivity),
                }
            return Result.ok(
                message=f"Device '{device_id}' telemetry retrieved.",
                data=data,
                capability="device_gateway",
                action="get_telemetry",
                call_id=call_id,
            )

        elif action == "record_heartbeat":
            device_id = str(params.get("device_id", ""))
            metrics = params.get("metrics")
            ok = getattr(self.gateway, "record_heartbeat", lambda d, m: False)(device_id, metrics=metrics)
            if ok:
                return Result.ok(
                    message=f"Heartbeat recorded for '{device_id}'.",
                    data={"device_id": device_id, "status": "RECORDED"},
                    capability="device_gateway",
                    action="record_heartbeat",
                    call_id=call_id,
                )
            return Result.fail(
                message=f"Failed to record heartbeat for '{device_id}'.",
                capability="device_gateway",
                action="record_heartbeat",
                call_id=call_id,
            )

        elif action == "list_devices":
            devices = self.gateway.list_devices()
            data = [d.to_dict() for d in devices]
            return Result.ok(
                message=f"Found {len(devices)} registered devices.",
                data={"devices": data, "count": len(devices)},
                capability="device_gateway",
                action="list_devices",
                call_id=call_id,
            )

        return Result.fail(
            message=f"Unsupported device_gateway action: '{action}'.",
            capability="device_gateway",
            action=action,
            call_id=call_id,
        )
