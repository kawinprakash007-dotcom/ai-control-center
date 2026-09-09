"""
ATLAS Phase 6.3 — Product Digital Twins.

Provides first-class digital twin implementations for:
1. ATLAS Vision (VisionDigitalTwin) — persistent observer (OBSERVATION_SOURCE)
2. ATLAS Glass (GlassDigitalTwin) — wearable perception & HUD (HYBRID)
3. ATLAS Drone (DroneDigitalTwin) — aerial sensing & intervention (HYBRID)
4. ATLAS Rover (RoverDigitalTwin) — ground sensing & intervention (HYBRID)

ARCHITECTURAL RULES:
1. Transport-neutral: ZERO hardware, serial, MAVLink, ROS2, MQTT, BLE, or GPIO specifics.
2. Emits canonical MultimodalObservation instances; NEVER directly mutates WorldState.
3. Completely deterministic execution driven by SimulationClock.
4. Bounded telemetry and history buffers.
"""

from collections import deque
import math
import threading
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from core.interfaces.simulation_interface import DigitalTwinInterface
from core.models.device_contract import (
    DeviceErrorCode,
    DeviceHealthStatus,
    ProductRole,
    ProductType,
    sanitize_contract_metadata,
)
from core.models.orchestration import (
    ConnectivityStatus,
    DeviceCapabilityDescriptor,
    DeviceIdentity,
    GeoLocation,
    ModalityType,
    MultimodalObservation,
)
from core.models.result import Result
from core.models.simulation import (
    SimulationLimits,
    TwinConfiguration,
    TwinFault,
    TwinFaultType,
    TwinPosition,
    TwinSimulationStatus,
    TwinState,
    TwinTelemetry,
)
from orchestration.device_gateway import DeviceCommand
from simulation.fault_injection import FaultInjectionManager


# ============================================================================
# Base Digital Twin
# ============================================================================

class BaseDigitalTwin(DigitalTwinInterface):
    """
    Common foundational implementation for all ATLAS product digital twins.
    Handles thread safety, state immutability, battery models, fault injection,
    and bounded telemetry history.
    """

    def __init__(
        self,
        config: TwinConfiguration,
        fault_manager: Optional[FaultInjectionManager] = None,
        limits: Optional[SimulationLimits] = None,
    ):
        self._lock = threading.RLock()
        self.config = config
        self.limits = limits or SimulationLimits()
        self.fault_manager = fault_manager or FaultInjectionManager(self.limits)

        # State attributes
        self._battery: float = float(config.initial_battery_pct)
        self._position: Optional[TwinPosition] = config.initial_position
        self._simulation_status: TwinSimulationStatus = TwinSimulationStatus.ACTIVE
        self._connectivity: ConnectivityStatus = ConnectivityStatus.ONLINE
        self._last_update: float = 1000000.0
        self._step_counter: int = 0
        self._internal_state: Dict[str, Any] = dict(config.parameters)

        self._telemetry_history: deque = deque(maxlen=self.limits.max_telemetry_records)

    @property
    def twin_id(self) -> str:
        return self.config.twin_id

    @property
    def product_type(self) -> ProductType:
        return self.config.product_type

    @property
    def product_role(self) -> ProductRole:
        return self.config.product_role

    def get_state(self) -> TwinState:
        """Retrieve current immutable snapshot of twin state."""
        with self._lock:
            # Check for active faults affecting state
            active_faults = tuple(self.fault_manager.get_active_faults(self.twin_id, self._last_update))
            connectivity = self._connectivity
            health = DeviceHealthStatus.HEALTHY
            battery = self._battery
            pos = self._position

            for f in active_faults:
                if f.fault_type == TwinFaultType.OFFLINE:
                    connectivity = ConnectivityStatus.DISCONNECTED
                    health = DeviceHealthStatus.UNHEALTHY
                elif f.fault_type == TwinFaultType.LOW_BATTERY:
                    battery = min(battery, float(f.parameters.get("battery_pct", 5.0)))
                    health = DeviceHealthStatus.DEGRADED
                elif f.fault_type == TwinFaultType.GPS_LOSS:
                    pos = None
                    health = DeviceHealthStatus.DEGRADED
                elif f.fault_type == TwinFaultType.SENSOR_FAILURE:
                    health = DeviceHealthStatus.DEGRADED

            if battery <= 15.0 and health == DeviceHealthStatus.HEALTHY:
                health = DeviceHealthStatus.DEGRADED

            return TwinState(
                twin_id=self.twin_id,
                product_id=self.twin_id,
                product_type=self.product_type,
                product_role=self.product_role,
                connectivity=connectivity,
                health=health,
                battery=battery,
                position=pos,
                active_capabilities=self.config.capabilities,
                simulation_status=self._simulation_status,
                last_update=self._last_update,
                internal_state=dict(self._internal_state),
                active_faults=active_faults,
            )

    def get_telemetry(self) -> TwinTelemetry:
        """Generate current telemetry snapshot."""
        with self._lock:
            state = self.get_state()
            # Stale telemetry fault freezes the timestamp
            ts = self._last_update
            stale_fault = self.fault_manager.get_fault(self.twin_id, TwinFaultType.TELEMETRY_STALE, ts)
            if stale_fault:
                ts = float(stale_fault.parameters.get("frozen_timestamp", ts - 3600.0))

            telemetry = TwinTelemetry(
                telemetry_id=f"telem_{self.twin_id}_{self._step_counter}",
                twin_id=self.twin_id,
                product_type=self.product_type,
                timestamp=ts,
                battery_level=state.battery,
                connectivity=state.connectivity,
                health_status=state.health,
                position=state.position,
                temperature_celsius=float(self._internal_state.get("temperature_celsius", 25.0)),
                operating_mode=str(self._internal_state.get("operating_mode", "NORMAL")),
                metrics={
                    "step": self._step_counter,
                    **self._internal_state,
                },
            )
            self._telemetry_history.append(telemetry)
            return telemetry

    def get_telemetry_history(self) -> Sequence[TwinTelemetry]:
        """Retrieve recent telemetry records."""
        with self._lock:
            return tuple(self._telemetry_history)

    def inject_fault(self, fault: TwinFault) -> None:
        """Inject a fault explicitly."""
        with self._lock:
            if fault.twin_id != self.twin_id:
                # Ensure correct twin ID
                fault = TwinFault(
                    fault_id=fault.fault_id,
                    twin_id=self.twin_id,
                    fault_type=fault.fault_type,
                    parameters=fault.parameters,
                    injected_at=fault.injected_at or self._last_update,
                    duration_seconds=fault.duration_seconds,
                    is_active=fault.is_active,
                )
            self.fault_manager.inject_fault(fault)

    def remove_fault(self, fault_id: str) -> bool:
        """Remove an injected fault."""
        return self.fault_manager.remove_fault(fault_id)

    def clear_faults(self) -> None:
        """Clear all active faults on this twin."""
        self.fault_manager.clear_faults(self.twin_id)

    def set_simulation_status(self, status: TwinSimulationStatus) -> None:
        """Set the simulation execution status of the twin."""
        with self._lock:
            self._simulation_status = status

    def update_battery(self, battery_pct: float) -> None:
        """Update twin battery percentage."""
        with self._lock:
            self._battery = max(0.0, min(100.0, float(battery_pct)))

    def update_position(self, position: Optional[TwinPosition]) -> None:
        """Update twin 3D spatial position."""
        with self._lock:
            self._position = position

    def get_simulation_status(self) -> TwinSimulationStatus:
        """Get the current simulation execution status."""
        with self._lock:
            return self._simulation_status

    def _check_pre_command_faults(self, command: DeviceCommand) -> Optional[Result]:
        """Verify if any active faults block command execution."""
        now = command.timestamp or self._last_update
        state = self.get_state()

        if state.connectivity != ConnectivityStatus.ONLINE or self.fault_manager.has_fault(self.twin_id, TwinFaultType.OFFLINE, now):
            return Result.failure(
                message=f"Twin '{self.twin_id}' is offline and cannot execute '{command.action}'.",
                capability=command.capability,
                action=command.action,
                call_id=command.dispatch_id,
                error_code=DeviceErrorCode.OFFLINE_DEVICE.value,
            )

        cmd_fault = self.fault_manager.get_fault(self.twin_id, TwinFaultType.COMMAND_FAILURE, now)
        if cmd_fault:
            target_act = cmd_fault.parameters.get("action")
            if target_act is None or str(target_act).lower() == command.action.lower():
                err_code = cmd_fault.parameters.get("error_code", DeviceErrorCode.DEVICE_ERROR.value)
                msg = cmd_fault.parameters.get("message", f"Twin '{self.twin_id}' simulated command failure.")
                return Result.failure(
                    message=msg,
                    capability=command.capability,
                    action=command.action,
                    call_id=command.dispatch_id,
                    error_code=err_code,
                )

        timeout_fault = self.fault_manager.get_fault(self.twin_id, TwinFaultType.COMMAND_TIMEOUT, now)
        if timeout_fault:
            target_act = timeout_fault.parameters.get("action")
            if target_act is None or str(target_act).lower() == command.action.lower():
                return Result.failure(
                    message=f"Command '{command.action}' on twin '{self.twin_id}' timed out.",
                    capability=command.capability,
                    action=command.action,
                    call_id=command.dispatch_id,
                    error_code=DeviceErrorCode.COMMAND_TIMEOUT.value,
                )

        return None


# ============================================================================
# 1. ATLAS Vision Digital Twin
# ============================================================================

_VISION_ACTIONS = (
    "detect_motion",
    "detect_person",
    "detect_anomaly",
    "capture_image",
    "capture_video",
    "get_telemetry",
    "emit_event",
)

class VisionDigitalTwin(BaseDigitalTwin):
    """
    Digital twin for ATLAS Vision.
    Primary role: persistent observer (OBSERVATION_SOURCE).
    Simulates stationary camera/sensor hub, environmental sensing, and multimodal observation generation.
    """

    def __init__(
        self,
        config: Optional[TwinConfiguration] = None,
        fault_manager: Optional[FaultInjectionManager] = None,
        limits: Optional[SimulationLimits] = None,
    ):
        cfg = config or TwinConfiguration(
            twin_id="ATLAS_VISION_01",
            product_type=ProductType.VISION,
            product_role=ProductRole.OBSERVATION_SOURCE,
            initial_position=TwinPosition(latitude=37.7749, longitude=-122.4194, altitude=2.5),
            capabilities=_VISION_ACTIONS,
        )
        super().__init__(config=cfg, fault_manager=fault_manager, limits=limits)
        self._internal_state.setdefault("camera_online", True)
        self._internal_state.setdefault("recording", False)
        self._internal_state.setdefault("last_detection", None)

    def execute_command(self, command: DeviceCommand) -> Result:
        with self._lock:
            fault_res = self._check_pre_command_faults(command)
            if fault_res:
                return fault_res

            action = command.action.lower()
            params = command.parameters
            now = command.timestamp or self._last_update

            # Check for sensor failure fault
            if self.fault_manager.has_fault(self.twin_id, TwinFaultType.SENSOR_FAILURE, now):
                return Result.failure(
                    message=f"Vision camera sensor failure on '{self.twin_id}'.",
                    capability=command.capability,
                    action=action,
                    call_id=command.dispatch_id,
                    error_code=DeviceErrorCode.DEVICE_ERROR.value,
                )

            observations: List[MultimodalObservation] = []

            if action == "detect_motion":
                region = str(params.get("region", "entryway"))
                data = {"motion_detected": True, "region": region, "confidence": 0.95}
                self._internal_state["last_detection"] = data
                obs = MultimodalObservation(
                    observation_id=f"obs_{command.dispatch_id}_motion",
                    source_id=self.twin_id,
                    source_type="VISION",
                    modality=ModalityType.EVENT,
                    timestamp=now,
                    payload=data,
                    location=self._position.to_geo_location() if self._position else None,
                    correlation_id=command.correlation_id,
                    causation_id=command.dispatch_id,
                )
                observations.append(obs)
                return Result.ok(
                    message=f"Vision '{self.twin_id}' detected motion in '{region}'.",
                    data={"motion_detected": True, "region": region, "observations": observations},
                    capability=command.capability,
                    action=action,
                    call_id=command.dispatch_id,
                )

            elif action == "detect_person":
                count = int(params.get("person_count", 1))
                data = {"event": "PERSON_DETECTED", "person_count": count, "confidence": 0.94}
                self._internal_state["last_detection"] = data
                obs = MultimodalObservation(
                    observation_id=f"obs_{command.dispatch_id}_person",
                    source_id=self.twin_id,
                    source_type="VISION",
                    modality=ModalityType.EVENT,
                    timestamp=now,
                    payload=data,
                    location=self._position.to_geo_location() if self._position else None,
                    correlation_id=command.correlation_id,
                    causation_id=command.dispatch_id,
                )
                observations.append(obs)
                return Result.ok(
                    message=f"Vision '{self.twin_id}' detected {count} person(s).",
                    data={"person_count": count, "observations": observations},
                    capability=command.capability,
                    action=action,
                    call_id=command.dispatch_id,
                )

            elif action == "detect_anomaly":
                hazard_type = str(params.get("hazard_type", "smoke"))
                data = {"event": "ANOMALY_DETECTED", "hazard_type": hazard_type, "severity": 0.85}
                self._internal_state["last_detection"] = data
                obs = MultimodalObservation(
                    observation_id=f"obs_{command.dispatch_id}_anomaly",
                    source_id=self.twin_id,
                    source_type="VISION",
                    modality=ModalityType.EVENT,
                    timestamp=now,
                    payload=data,
                    location=self._position.to_geo_location() if self._position else None,
                    correlation_id=command.correlation_id,
                    causation_id=command.dispatch_id,
                )
                observations.append(obs)
                return Result.ok(
                    message=f"Vision '{self.twin_id}' detected anomaly '{hazard_type}'.",
                    data={"anomaly_detected": True, "hazard_type": hazard_type, "observations": observations},
                    capability=command.capability,
                    action=action,
                    call_id=command.dispatch_id,
                )

            elif action == "capture_image":
                data = {
                    "image_url": f"sim://vision/{self.twin_id}/snapshot_{int(now)}.jpg",
                    "resolution": "1080p",
                    "timestamp": now,
                }
                obs = MultimodalObservation(
                    observation_id=f"obs_{command.dispatch_id}_img",
                    source_id=self.twin_id,
                    source_type="VISION",
                    modality=ModalityType.IMAGE,
                    timestamp=now,
                    payload=data,
                    location=self._position.to_geo_location() if self._position else None,
                    correlation_id=command.correlation_id,
                    causation_id=command.dispatch_id,
                )
                observations.append(obs)
                return Result.ok(
                    message=f"Vision '{self.twin_id}' captured image.",
                    data={"image_captured": True, "observations": observations},
                    capability=command.capability,
                    action=action,
                    call_id=command.dispatch_id,
                )

            elif action == "capture_video":
                duration = float(params.get("duration", 5.0))
                self._internal_state["recording"] = True
                data = {
                    "video_url": f"sim://vision/{self.twin_id}/video_{int(now)}.mp4",
                    "duration": duration,
                }
                obs = MultimodalObservation(
                    observation_id=f"obs_{command.dispatch_id}_video",
                    source_id=self.twin_id,
                    source_type="VISION",
                    modality=ModalityType.VIDEO_FRAME,
                    timestamp=now,
                    payload=data,
                    location=self._position.to_geo_location() if self._position else None,
                    correlation_id=command.correlation_id,
                    causation_id=command.dispatch_id,
                )
                observations.append(obs)
                return Result.ok(
                    message=f"Vision '{self.twin_id}' captured {duration}s video clip.",
                    data={"recording": True, "duration": duration, "observations": observations},
                    capability=command.capability,
                    action=action,
                    call_id=command.dispatch_id,
                )

            elif action == "get_telemetry":
                telem = self.get_telemetry()
                return Result.ok(
                    message=f"Vision '{self.twin_id}' telemetry reported.",
                    data=telem.to_dict(),
                    capability=command.capability,
                    action=action,
                    call_id=command.dispatch_id,
                )

            elif action == "emit_event":
                evt_type = str(params.get("event_type", "ENVIRONMENTAL_CHANGE"))
                payload = dict(params.get("payload", {}))
                obs = MultimodalObservation(
                    observation_id=f"obs_{command.dispatch_id}_evt",
                    source_id=self.twin_id,
                    source_type="VISION",
                    modality=ModalityType.EVENT,
                    timestamp=now,
                    payload={"event": evt_type, **payload},
                    location=self._position.to_geo_location() if self._position else None,
                    correlation_id=command.correlation_id,
                    causation_id=command.dispatch_id,
                )
                observations.append(obs)
                return Result.ok(
                    message=f"Vision '{self.twin_id}' emitted event '{evt_type}'.",
                    data={"event_emitted": True, "observations": observations},
                    capability=command.capability,
                    action=action,
                    call_id=command.dispatch_id,
                )

            return Result.failure(
                message=f"Vision twin '{self.twin_id}' does not support action '{action}'.",
                capability=command.capability,
                action=action,
                call_id=command.dispatch_id,
                error_code=DeviceErrorCode.UNSUPPORTED_ACTION.value,
            )

    def generate_observation(self, observation_type: str, **kwargs) -> Optional[MultimodalObservation]:
        with self._lock:
            now = kwargs.get("timestamp", self._last_update)
            obs_id = kwargs.get("observation_id", f"obs_vis_{self.twin_id}_{self._step_counter}_{observation_type.lower()}")
            payload = {"event": observation_type.upper(), **kwargs.get("payload", {})}

            # Check for conflicting observation fault
            if self.fault_manager.has_fault(self.twin_id, TwinFaultType.CONFLICTING_OBSERVATION, now):
                payload["conflicting_reading"] = True
                payload["motion_detected"] = False

            return MultimodalObservation(
                observation_id=obs_id,
                source_id=self.twin_id,
                source_type="VISION",
                modality=ModalityType.EVENT,
                timestamp=now,
                payload=payload,
                location=self._position.to_geo_location() if self._position else None,
            )

    def tick(self, now: float, delta_seconds: float) -> Sequence[MultimodalObservation]:
        with self._lock:
            self._last_update = now
            self._step_counter += 1
            # Stationary vision has very low battery drain
            self._battery = max(0.0, self._battery - (0.001 * delta_seconds))
            return []


# ============================================================================
# 2. ATLAS Glass Digital Twin
# ============================================================================

_GLASS_ACTIONS = (
    "display_hud",
    "show_hud",
    "capture_image",
    "capture_audio",
    "start_recording",
    "stop_recording",
    "get_position",
    "get_location",
    "get_telemetry",
    "send_notification",
)

class GlassDigitalTwin(BaseDigitalTwin):
    """
    Digital twin for ATLAS Glass.
    Primary role: wearable perception and HUD display (HYBRID).
    Simulates egocentric perception, audio capture, and heads-up notifications.
    """

    def __init__(
        self,
        config: Optional[TwinConfiguration] = None,
        fault_manager: Optional[FaultInjectionManager] = None,
        limits: Optional[SimulationLimits] = None,
    ):
        cfg = config or TwinConfiguration(
            twin_id="ATLAS_GLASS_01",
            product_type=ProductType.GLASS,
            product_role=ProductRole.HYBRID,
            initial_position=TwinPosition(latitude=37.7749, longitude=-122.4194, altitude=1.7),
            capabilities=_GLASS_ACTIONS,
        )
        super().__init__(config=cfg, fault_manager=fault_manager, limits=limits)
        self._hud_messages: deque = deque(maxlen=50)
        self._internal_state.setdefault("hud_active", True)
        self._internal_state.setdefault("recording", False)
        self._internal_state.setdefault("wearer_id", "user_01")

    def execute_command(self, command: DeviceCommand) -> Result:
        with self._lock:
            fault_res = self._check_pre_command_faults(command)
            if fault_res:
                return fault_res

            action = command.action.lower()
            params = command.parameters
            now = command.timestamp or self._last_update
            observations: List[MultimodalObservation] = []

            if action in ("display_hud", "show_hud"):
                msg = str(params.get("message", ""))
                duration = float(params.get("duration", 3.0))
                priority = str(params.get("priority", "NORMAL"))
                entry = {"message": msg, "duration": duration, "priority": priority, "timestamp": now}
                self._hud_messages.append(entry)
                return Result.ok(
                    message=f"HUD displayed message: '{msg}'",
                    data={"displayed": True, "hud_entry": entry},
                    capability=command.capability,
                    action=action,
                    call_id=command.dispatch_id,
                )

            elif action == "capture_image":
                data = {
                    "image_url": f"sim://glass/{self.twin_id}/egocentric_{int(now)}.jpg",
                    "perspective": "egocentric",
                    "timestamp": now,
                }
                obs = MultimodalObservation(
                    observation_id=f"obs_{command.dispatch_id}_glass_img",
                    source_id=self.twin_id,
                    source_type="GLASS",
                    modality=ModalityType.IMAGE,
                    timestamp=now,
                    payload=data,
                    location=self._position.to_geo_location() if self._position else None,
                    correlation_id=command.correlation_id,
                    causation_id=command.dispatch_id,
                )
                observations.append(obs)
                return Result.ok(
                    message=f"Glass '{self.twin_id}' captured egocentric snapshot.",
                    data={"image_captured": True, "observations": observations},
                    capability=command.capability,
                    action=action,
                    call_id=command.dispatch_id,
                )

            elif action == "capture_audio":
                duration = float(params.get("duration", 3.0))
                data = {
                    "audio_url": f"sim://glass/{self.twin_id}/audio_{int(now)}.wav",
                    "duration": duration,
                }
                obs = MultimodalObservation(
                    observation_id=f"obs_{command.dispatch_id}_audio",
                    source_id=self.twin_id,
                    source_type="GLASS",
                    modality=ModalityType.AUDIO_EVENT,
                    timestamp=now,
                    payload=data,
                    location=self._position.to_geo_location() if self._position else None,
                    correlation_id=command.correlation_id,
                    causation_id=command.dispatch_id,
                )
                observations.append(obs)
                return Result.ok(
                    message=f"Glass '{self.twin_id}' recorded {duration}s audio.",
                    data={"audio_captured": True, "observations": observations},
                    capability=command.capability,
                    action=action,
                    call_id=command.dispatch_id,
                )

            elif action in ("start_recording", "stop_recording"):
                recording = (action == "start_recording")
                self._internal_state["recording"] = recording
                return Result.ok(
                    message=f"Glass recording set to {recording}.",
                    data={"recording": recording},
                    capability=command.capability,
                    action=action,
                    call_id=command.dispatch_id,
                )

            elif action in ("get_position", "get_location"):
                state = self.get_state()
                return Result.ok(
                    message=f"Glass '{self.twin_id}' position reported.",
                    data={"position": state.position.to_dict() if state.position else None},
                    capability=command.capability,
                    action=action,
                    call_id=command.dispatch_id,
                )

            elif action == "get_telemetry":
                telem = self.get_telemetry()
                return Result.ok(
                    message=f"Glass '{self.twin_id}' telemetry reported.",
                    data=telem.to_dict(),
                    capability=command.capability,
                    action=action,
                    call_id=command.dispatch_id,
                )

            elif action == "send_notification":
                text = str(params.get("text", ""))
                self._hud_messages.append({"message": text, "type": "NOTIFICATION", "timestamp": now})
                return Result.ok(
                    message=f"Notification delivered to Glass: '{text}'",
                    data={"delivered": True},
                    capability=command.capability,
                    action=action,
                    call_id=command.dispatch_id,
                )

            return Result.failure(
                message=f"Glass twin '{self.twin_id}' does not support action '{action}'.",
                capability=command.capability,
                action=action,
                call_id=command.dispatch_id,
                error_code=DeviceErrorCode.UNSUPPORTED_ACTION.value,
            )

    def generate_observation(self, observation_type: str, **kwargs) -> Optional[MultimodalObservation]:
        with self._lock:
            now = kwargs.get("timestamp", self._last_update)
            obs_id = kwargs.get("observation_id", f"obs_glass_{self.twin_id}_{self._step_counter}")
            payload = {"event": observation_type.upper(), **kwargs.get("payload", {})}
            return MultimodalObservation(
                observation_id=obs_id,
                source_id=self.twin_id,
                source_type="GLASS",
                modality=ModalityType.EVENT,
                timestamp=now,
                payload=payload,
                location=self._position.to_geo_location() if self._position else None,
            )

    def tick(self, now: float, delta_seconds: float) -> Sequence[MultimodalObservation]:
        with self._lock:
            self._last_update = now
            self._step_counter += 1
            # Glass battery drain rate: ~0.005% per second
            drain = 0.005 * delta_seconds
            if self._internal_state.get("recording"):
                drain *= 2.0
            self._battery = max(0.0, self._battery - drain)
            return []


# ============================================================================
# 3. ATLAS Drone Digital Twin
# ============================================================================

_DRONE_ACTIONS = (
    "takeoff",
    "land",
    "goto_location",
    "navigate_waypoint",
    "hover",
    "return_to_base",
    "capture_image",
    "capture_video",
    "get_position",
    "get_telemetry",
)

class DroneDigitalTwin(BaseDigitalTwin):
    """
    Digital twin for ATLAS Drone.
    Primary role: aerial perception and intervention (HYBRID).
    Simulates flight dynamics (takeoff, cruising, landing), waypoint navigation,
    battery drain under flight load, and low-battery failsafes.
    """

    def __init__(
        self,
        config: Optional[TwinConfiguration] = None,
        fault_manager: Optional[FaultInjectionManager] = None,
        limits: Optional[SimulationLimits] = None,
    ):
        cfg = config or TwinConfiguration(
            twin_id="ATLAS_DRONE_01",
            product_type=ProductType.DRONE,
            product_role=ProductRole.HYBRID,
            initial_position=TwinPosition(latitude=37.7749, longitude=-122.4194, altitude=0.0),
            capabilities=_DRONE_ACTIONS,
        )
        super().__init__(config=cfg, fault_manager=fault_manager, limits=limits)
        self._flight_state: str = "GROUNDED"
        self._target_position: Optional[TwinPosition] = None
        self._internal_state["flight_state"] = self._flight_state
        self._internal_state["altitude"] = 0.0

    def execute_command(self, command: DeviceCommand) -> Result:
        with self._lock:
            fault_res = self._check_pre_command_faults(command)
            if fault_res:
                return fault_res

            action = command.action.lower()
            params = command.parameters
            now = command.timestamp or self._last_update

            # Battery check failsafe
            state = self.get_state()
            if state.battery < 15.0 and action not in ("land", "get_position", "get_telemetry"):
                return Result.failure(
                    message=f"Drone '{self.twin_id}' battery critical ({state.battery:.1f}%). Action '{action}' rejected.",
                    capability=command.capability,
                    action=action,
                    call_id=command.dispatch_id,
                    error_code=DeviceErrorCode.LOW_BATTERY.value,
                )

            if action == "takeoff":
                target_alt = float(params.get("altitude", 10.0))
                if self._flight_state not in ("GROUNDED", "LANDED"):
                    return Result.failure(
                        message=f"Drone '{self.twin_id}' is already airborne (state: {self._flight_state}).",
                        capability=command.capability,
                        action=action,
                        call_id=command.dispatch_id,
                        error_code=DeviceErrorCode.COMMAND_REJECTED.value,
                    )
                self._flight_state = "HOVERING"
                if self._position:
                    self._position = TwinPosition(
                        latitude=self._position.latitude,
                        longitude=self._position.longitude,
                        altitude=target_alt,
                    )
                self._internal_state["flight_state"] = self._flight_state
                self._internal_state["altitude"] = target_alt
                return Result.ok(
                    message=f"Drone '{self.twin_id}' took off to {target_alt}m altitude.",
                    data={"flight_state": self._flight_state, "altitude": target_alt},
                    capability=command.capability,
                    action=action,
                    call_id=command.dispatch_id,
                )

            elif action == "land":
                self._flight_state = "GROUNDED"
                if self._position:
                    self._position = TwinPosition(
                        latitude=self._position.latitude,
                        longitude=self._position.longitude,
                        altitude=0.0,
                    )
                self._internal_state["flight_state"] = self._flight_state
                self._internal_state["altitude"] = 0.0
                return Result.ok(
                    message=f"Drone '{self.twin_id}' landed successfully.",
                    data={"flight_state": self._flight_state, "altitude": 0.0},
                    capability=command.capability,
                    action=action,
                    call_id=command.dispatch_id,
                )

            elif action in ("goto_location", "navigate_waypoint"):
                lat = float(params.get("latitude", self._position.latitude if self._position else 0.0))
                lon = float(params.get("longitude", self._position.longitude if self._position else 0.0))
                alt = float(params.get("altitude", 15.0))
                target = TwinPosition(latitude=lat, longitude=lon, altitude=alt)
                self._target_position = target
                self._flight_state = "NAVIGATING"
                self._position = target  # Simulates instant or fast arrival for step tests
                self._internal_state["flight_state"] = self._flight_state
                self._internal_state["altitude"] = alt
                return Result.ok(
                    message=f"Drone '{self.twin_id}' arrived at waypoint ({lat:.4f}, {lon:.4f}, {alt}m).",
                    data={"arrived": True, "position": target.to_dict()},
                    capability=command.capability,
                    action=action,
                    call_id=command.dispatch_id,
                )

            elif action == "hover":
                self._flight_state = "HOVERING"
                self._internal_state["flight_state"] = self._flight_state
                return Result.ok(
                    message=f"Drone '{self.twin_id}' is hovering.",
                    data={"flight_state": self._flight_state},
                    capability=command.capability,
                    action=action,
                    call_id=command.dispatch_id,
                )

            elif action == "return_to_base":
                self._flight_state = "GROUNDED"
                if self.config.initial_position:
                    self._position = self.config.initial_position
                self._internal_state["flight_state"] = self._flight_state
                self._internal_state["altitude"] = 0.0
                return Result.ok(
                    message=f"Drone '{self.twin_id}' returned to base.",
                    data={"returned": True, "position": self._position.to_dict() if self._position else None},
                    capability=command.capability,
                    action=action,
                    call_id=command.dispatch_id,
                )

            elif action in ("capture_image", "capture_video"):
                obs = MultimodalObservation(
                    observation_id=f"obs_{command.dispatch_id}_drone_media",
                    source_id=self.twin_id,
                    source_type="DRONE",
                    modality=ModalityType.IMAGE if action == "capture_image" else ModalityType.VIDEO_FRAME,
                    timestamp=now,
                    payload={"perspective": "aerial", "altitude": self._internal_state.get("altitude", 0.0)},
                    location=self._position.to_geo_location() if self._position else None,
                    correlation_id=command.correlation_id,
                    causation_id=command.dispatch_id,
                )
                return Result.ok(
                    message=f"Drone '{self.twin_id}' captured aerial media.",
                    data={"media_captured": True, "observations": [obs]},
                    capability=command.capability,
                    action=action,
                    call_id=command.dispatch_id,
                )

            elif action in ("get_position", "get_telemetry"):
                telem = self.get_telemetry()
                return Result.ok(
                    message=f"Drone '{self.twin_id}' telemetry reported.",
                    data=telem.to_dict(),
                    capability=command.capability,
                    action=action,
                    call_id=command.dispatch_id,
                )

            return Result.failure(
                message=f"Drone twin '{self.twin_id}' does not support action '{action}'.",
                capability=command.capability,
                action=action,
                call_id=command.dispatch_id,
                error_code=DeviceErrorCode.UNSUPPORTED_ACTION.value,
            )

    def generate_observation(self, observation_type: str, **kwargs) -> Optional[MultimodalObservation]:
        with self._lock:
            now = kwargs.get("timestamp", self._last_update)
            obs_id = kwargs.get("observation_id", f"obs_drone_{self.twin_id}_{self._step_counter}")
            payload = {"event": observation_type.upper(), "flight_state": self._flight_state, **kwargs.get("payload", {})}
            return MultimodalObservation(
                observation_id=obs_id,
                source_id=self.twin_id,
                source_type="DRONE",
                modality=ModalityType.EVENT,
                timestamp=now,
                payload=payload,
                location=self._position.to_geo_location() if self._position else None,
            )

    def tick(self, now: float, delta_seconds: float) -> Sequence[MultimodalObservation]:
        with self._lock:
            self._last_update = now
            self._step_counter += 1
            # Drone battery consumption: airborne drain is much higher (~0.05%/s) than grounded (~0.002%/s)
            drain = 0.05 * delta_seconds if self._flight_state != "GROUNDED" else 0.002 * delta_seconds
            self._battery = max(0.0, self._battery - drain)

            # Auto-land if battery hits critical during flight
            if self._battery <= 5.0 and self._flight_state != "GROUNDED":
                self._flight_state = "GROUNDED"
                if self._position:
                    self._position = TwinPosition(
                        latitude=self._position.latitude,
                        longitude=self._position.longitude,
                        altitude=0.0,
                    )
                self._internal_state["flight_state"] = "GROUNDED"
                self._internal_state["altitude"] = 0.0

            return []


# ============================================================================
# 4. ATLAS Rover Digital Twin
# ============================================================================

_ROVER_ACTIONS = (
    "move",
    "navigate_to",
    "goto_location",
    "stop",
    "patrol_zone",
    "dock",
    "capture_image",
    "get_position",
    "get_telemetry",
)

class RoverDigitalTwin(BaseDigitalTwin):
    """
    Digital twin for ATLAS Rover.
    Primary role: ground sensing and intervention (HYBRID).
    Simulates ground locomotion, obstacle detection, emergency stopping, and docking recharge.
    """

    def __init__(
        self,
        config: Optional[TwinConfiguration] = None,
        fault_manager: Optional[FaultInjectionManager] = None,
        limits: Optional[SimulationLimits] = None,
    ):
        cfg = config or TwinConfiguration(
            twin_id="ATLAS_ROVER_01",
            product_type=ProductType.ROVER,
            product_role=ProductRole.HYBRID,
            initial_position=TwinPosition(latitude=37.7749, longitude=-122.4194, altitude=0.0),
            capabilities=_ROVER_ACTIONS,
        )
        super().__init__(config=cfg, fault_manager=fault_manager, limits=limits)
        self._rover_state: str = "STOPPED"
        self._obstacle_detected: bool = False
        self._internal_state["rover_state"] = self._rover_state
        self._internal_state["obstacle_detected"] = False

    def execute_command(self, command: DeviceCommand) -> Result:
        with self._lock:
            fault_res = self._check_pre_command_faults(command)
            if fault_res:
                return fault_res

            action = command.action.lower()
            params = command.parameters
            now = command.timestamp or self._last_update

            if action in ("move", "navigate_to", "goto_location"):
                if self._obstacle_detected:
                    return Result.failure(
                        message=f"Rover '{self.twin_id}' cannot move: obstacle detected ahead.",
                        capability=command.capability,
                        action=action,
                        call_id=command.dispatch_id,
                        error_code=DeviceErrorCode.SAFETY_REJECTION.value,
                    )
                lat = float(params.get("latitude", self._position.latitude if self._position else 0.0))
                lon = float(params.get("longitude", self._position.longitude if self._position else 0.0))
                self._rover_state = "MOVING"
                self._position = TwinPosition(latitude=lat, longitude=lon, altitude=0.0)
                self._internal_state["rover_state"] = self._rover_state
                return Result.ok(
                    message=f"Rover '{self.twin_id}' navigated to ({lat:.4f}, {lon:.4f}).",
                    data={"rover_state": self._rover_state, "position": self._position.to_dict()},
                    capability=command.capability,
                    action=action,
                    call_id=command.dispatch_id,
                )

            elif action == "stop":
                self._rover_state = "STOPPED"
                self._internal_state["rover_state"] = self._rover_state
                return Result.ok(
                    message=f"Rover '{self.twin_id}' stopped.",
                    data={"rover_state": self._rover_state},
                    capability=command.capability,
                    action=action,
                    call_id=command.dispatch_id,
                )

            elif action == "patrol_zone":
                zone = str(params.get("zone", "perimeter_north"))
                self._rover_state = "PATROLLING"
                self._internal_state["rover_state"] = self._rover_state
                self._internal_state["patrol_zone"] = zone
                return Result.ok(
                    message=f"Rover '{self.twin_id}' patrolling zone '{zone}'.",
                    data={"rover_state": self._rover_state, "zone": zone},
                    capability=command.capability,
                    action=action,
                    call_id=command.dispatch_id,
                )

            elif action == "dock":
                self._rover_state = "DOCKED"
                self._internal_state["rover_state"] = self._rover_state
                return Result.ok(
                    message=f"Rover '{self.twin_id}' docked at charging station.",
                    data={"rover_state": self._rover_state, "charging": True},
                    capability=command.capability,
                    action=action,
                    call_id=command.dispatch_id,
                )

            elif action == "capture_image":
                obs = MultimodalObservation(
                    observation_id=f"obs_{command.dispatch_id}_rover_img",
                    source_id=self.twin_id,
                    source_type="ROVER",
                    modality=ModalityType.IMAGE,
                    timestamp=now,
                    payload={"perspective": "ground_level"},
                    location=self._position.to_geo_location() if self._position else None,
                    correlation_id=command.correlation_id,
                    causation_id=command.dispatch_id,
                )
                return Result.ok(
                    message=f"Rover '{self.twin_id}' captured ground snapshot.",
                    data={"image_captured": True, "observations": [obs]},
                    capability=command.capability,
                    action=action,
                    call_id=command.dispatch_id,
                )

            elif action in ("get_position", "get_telemetry"):
                telem = self.get_telemetry()
                return Result.ok(
                    message=f"Rover '{self.twin_id}' telemetry reported.",
                    data=telem.to_dict(),
                    capability=command.capability,
                    action=action,
                    call_id=command.dispatch_id,
                )

            return Result.failure(
                message=f"Rover twin '{self.twin_id}' does not support action '{action}'.",
                capability=command.capability,
                action=action,
                call_id=command.dispatch_id,
                error_code=DeviceErrorCode.UNSUPPORTED_ACTION.value,
            )

    def set_obstacle(self, detected: bool) -> None:
        """Simulate obstacle proximity detection."""
        with self._lock:
            self._obstacle_detected = detected
            self._internal_state["obstacle_detected"] = detected

    def generate_observation(self, observation_type: str, **kwargs) -> Optional[MultimodalObservation]:
        with self._lock:
            now = kwargs.get("timestamp", self._last_update)
            obs_id = kwargs.get("observation_id", f"obs_rover_{self.twin_id}_{self._step_counter}")
            payload = {"event": observation_type.upper(), "rover_state": self._rover_state, **kwargs.get("payload", {})}
            return MultimodalObservation(
                observation_id=obs_id,
                source_id=self.twin_id,
                source_type="ROVER",
                modality=ModalityType.EVENT,
                timestamp=now,
                payload=payload,
                location=self._position.to_geo_location() if self._position else None,
            )

    def tick(self, now: float, delta_seconds: float) -> Sequence[MultimodalObservation]:
        with self._lock:
            self._last_update = now
            self._step_counter += 1
            if self._rover_state == "DOCKED":
                # Recharging: +0.2% per second
                self._battery = min(100.0, self._battery + (0.2 * delta_seconds))
            elif self._rover_state in ("MOVING", "PATROLLING"):
                self._battery = max(0.0, self._battery - (0.02 * delta_seconds))
            else:
                self._battery = max(0.0, self._battery - (0.002 * delta_seconds))
            return []
