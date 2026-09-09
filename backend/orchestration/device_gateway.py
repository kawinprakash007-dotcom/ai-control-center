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


# ============================================================================
# Semantic Device Command Model
# ============================================================================

@dataclass(frozen=True)
class DeviceCommand:
    """
    Semantic, hardware-independent command dispatched to an edge device adapter.
    """
    dispatch_id: str
    device_id: str
    capability: str
    action: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    correlation_id: str = ""
    causation_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dispatch_id": self.dispatch_id,
            "device_id": self.device_id,
            "capability": self.capability,
            "action": self.action,
            "parameters": dict(self.parameters),
            "correlation_id": self.correlation_id,
            "causation_id": self.causation_id,
            "timestamp": self.timestamp,
        }


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
    Central Device & Agent Gateway for ATLAS Central Orchestration Layer.

    Responsibilities:
    - Maintains the in-memory registry of known edge participant DeviceIdentity profiles.
    - Manages capability declarations and validates command parameters against schemas.
    - Resolves transport-specific or virtual DeviceAdapters by device_id or device_type.
    - Validates device connectivity and health before delegating execution.
    - Dispatches semantic commands with deterministic dispatch_id and correlation preservation.
    - Exposes operational queries (device status, capabilities, heartbeats).

    CRITICAL ARCHITECTURAL BOUNDARIES:
    - Zero direct hardware, serial, socket, or network execution.
    - Zero autonomous reasoning, LLM/VLM calls, or model routing.
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
    ):
        self.event_sink = event_sink
        self.clock = clock or time.time
        self.heartbeat_timeout_seconds = heartbeat_timeout_seconds
        self.max_devices = max_devices
        self.max_dispatch_history = max_dispatch_history

        self._lock = threading.Lock()
        self._devices: Dict[str, DeviceIdentity] = {}
        self._type_adapters: Dict[DeviceType, DeviceAdapterInterface] = {}
        self._device_adapters: Dict[str, DeviceAdapterInterface] = {}
        self._seen_dispatch_ids: collections.deque = collections.deque(maxlen=max_dispatch_history)
        self._seen_dispatch_set: Set[str] = set()
        self._dispatch_results: Dict[str, Result] = {}

    # ========================================================================
    # Device Registry Operations
    # ========================================================================

    def register_device(self, device: DeviceIdentity) -> None:
        """
        Register a device identity profile.
        Rejects duplicate device IDs deterministically to prevent silent overwrites.
        """
        if not isinstance(device, DeviceIdentity):
            raise TypeError(f"Expected DeviceIdentity instance, got {type(device).__name__}")

        with self._lock:
            if device.device_id in self._devices:
                raise ValueError(f"Device '{device.device_id}' is already registered.")

            if len(self._devices) >= self.max_devices:
                raise ValueError(f"Maximum registered devices limit ({self.max_devices}) reached.")

            self._devices[device.device_id] = device

        self._emit_event(
            CognitiveEventType.DEVICE_REGISTERED,
            device_id=device.device_id,
            metadata={
                "device_type": device.device_type.value,
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

    def list_devices(self, device_type: Optional[DeviceType] = None) -> Sequence[DeviceIdentity]:
        """List registered devices, optionally filtered by DeviceType."""
        with self._lock:
            devices = list(self._devices.values())

        if device_type is not None:
            norm_type = (
                device_type
                if isinstance(device_type, DeviceType)
                else DeviceType.from_str(device_type)
            )
            devices = [d for d in devices if d.device_type == norm_type]

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
                metadata=dict(dev.metadata),
            )
            self._devices[device_id] = updated

        self._emit_event(
            CognitiveEventType.DEVICE_STATUS_UPDATED,
            device_id=device_id,
            metadata={"status": norm_status.value, "timestamp": now},
        )
        return True

    def record_heartbeat(self, device_id: str, timestamp: Optional[float] = None) -> bool:
        """
        Record a heartbeat timestamp for an edge device.
        Automatically marks connectivity as ONLINE if previously disconnected.
        """
        now = timestamp if timestamp is not None else self.clock()
        with self._lock:
            dev = self._devices.get(device_id)
            if not dev:
                return False

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
                metadata=dict(dev.metadata),
            )
            self._devices[device_id] = updated

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
        device_id: Optional[str] = None,
    ) -> None:
        """
        Register a DeviceAdapter for a specific DeviceType or an exact device_id.
        """
        if not isinstance(adapter, DeviceAdapterInterface):
            raise TypeError(f"Expected DeviceAdapterInterface, got {type(adapter).__name__}")

        if device_id is None and device_type is None:
            raise ValueError("Must specify at least one of device_id or device_type.")

        with self._lock:
            if device_id is not None:
                self._device_adapters[device_id] = adapter
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
        Priority: 1) Device-specific adapter, 2) DeviceType adapter.
        """
        with self._lock:
            # 1. Device-specific adapter
            if device_id in self._device_adapters:
                return self._device_adapters[device_id]

            # 2. Type-specific adapter
            dev = self._devices.get(device_id)
            if dev and dev.device_type in self._type_adapters:
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
    ) -> Result:
        """
        Orchestrate capability validation and route command to the resolved adapter.
        Guarantees:
        - Device existence and connectivity check.
        - Declarative capability and action matching.
        - Strict parameter schema validation.
        - Duplicate dispatch_id suppression.
        - Preserves correlation and causation identifiers.
        """
        current_time = now if now is not None else self.clock()
        norm_cap = str(capability or "").strip().lower()
        norm_act = str(action or "").strip().lower()

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
            )

        # 3. Capability and action matching
        matched_cap: Optional[DeviceCapabilityDescriptor] = None
        for cap_desc in dev.capabilities:
            if (
                cap_desc.capability_name.strip().lower() == norm_cap
                and cap_desc.action_name.strip().lower() == norm_act
            ):
                matched_cap = cap_desc
                break

        if not matched_cap:
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
            )

        # 5. Dispatch ID assignment & Duplicate check
        effective_dispatch_id = dispatch_id or f"disp_{device_id}_{int(current_time * 1000)}"
        with self._lock:
            if effective_dispatch_id in self._seen_dispatch_set:
                err_dup = f"Duplicate dispatch_id '{effective_dispatch_id}' rejected."
                return Result.fail(
                    message=err_dup,
                    capability=norm_cap,
                    action=norm_act,
                    call_id=effective_dispatch_id,
                )

            # Record dispatch ID in bounded LRU cache
            if len(self._seen_dispatch_ids) >= self.max_dispatch_history:
                oldest = self._seen_dispatch_ids.popleft()
                self._seen_dispatch_set.discard(oldest)
            self._seen_dispatch_ids.append(effective_dispatch_id)
            self._seen_dispatch_set.add(effective_dispatch_id)

        # 6. Adapter resolution
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
            return Result.fail(
                message=str(e),
                capability=norm_cap,
                action=norm_act,
                call_id=effective_dispatch_id,
            )

        # 7. Construct command and delegate to adapter
        command = DeviceCommand(
            dispatch_id=effective_dispatch_id,
            device_id=device_id,
            capability=norm_cap,
            action=norm_act,
            parameters=dict(parameters),
            correlation_id=correlation_id or effective_dispatch_id,
            causation_id=causation_id,
            timestamp=current_time,
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
            return Result.fail(
                message=f"Adapter execution error on device '{device_id}': {e}",
                capability=norm_cap,
                action=norm_act,
                call_id=effective_dispatch_id,
            )

        # 8. Record result
        if isinstance(res.data, dict) and "correlation_id" not in res.data and command.correlation_id:
            res.data["correlation_id"] = command.correlation_id

        with self._lock:
            self._dispatch_results[effective_dispatch_id] = res

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

            return self.gateway.dispatch_to_device(
                device_id=device_id,
                capability=capability,
                action=act,
                parameters=action_params,
                dispatch_id=call_id,
                correlation_id=correlation_id,
                causation_id=causation_id,
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
