"""
ATLAS Phase 6.2 — Unified Virtual Device Adapters.

Simulated adapters for the four first-class ATLAS edge products:
1. ATLAS Vision (VirtualVisionAdapter) — persistent observer (OBSERVATION_SOURCE)
2. ATLAS Glass (VirtualGlassAdapter) — wearable perception & HUD (HYBRID)
3. ATLAS Drone (VirtualDroneAdapter) — aerial sensing & intervention (HYBRID)
4. ATLAS Rover (VirtualRoverAdapter) — ground sensing & intervention (HYBRID)

CRITICAL ARCHITECTURAL RULES:
1. Transport-neutral: ZERO hardware, serial, MAVLink, ROS2, MQTT, BLE, or GPIO specifics.
2. Generates MultimodalObservation instances for CentralInputGateway ingress; NEVER directly mutates WorldState.
3. Implements the full DeviceAdapterInterface: connect, disconnect, get_health, get_status, get_capabilities, heartbeat.
4. Purely deterministic execution for testability and replay reproducibility.
"""

from dataclasses import dataclass, field
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from core.interfaces.orchestration_interface import DeviceAdapterInterface
from core.models.device_contract import (
    DeviceErrorCode,
    DeviceHealth,
    DeviceHealthStatus,
    DeviceHeartbeat,
    ProductRole,
    ProductType,
)
from core.models.orchestration import (
    ConnectivityStatus,
    DeviceCapabilityDescriptor,
    DeviceIdentity,
    DeviceType,
    GeoLocation,
    ModalityType,
    MultimodalObservation,
)
from core.models.result import Result
from core.models.tool_call import ToolCall
from orchestration.device_gateway import DeviceCommand


# ============================================================================
# 1. Virtual Vision Adapter (ATLAS Vision)
# ============================================================================

class VirtualVisionAdapter(DeviceAdapterInterface):
    """
    Deterministic simulated environmental sensor and surveillance adapter for ATLAS Vision.
    Primary role: persistent observer (OBSERVATION_SOURCE).
    Emits multimodal observations into CentralInputGateway without directly mutating WorldState.
    """

    def __init__(
        self,
        device_id: str = "ATLAS_VISION_01",
        home_location: Optional[GeoLocation] = None,
    ):
        self.device_id = device_id
        self.location = home_location or GeoLocation(latitude=37.7749, longitude=-122.4194, altitude=2.5)
        self.battery_pct = 100.0
        self.recording = False
        self.camera_online = True
        self.microphone_online = True
        self.last_detection: Optional[Dict[str, Any]] = None
        self.connectivity = ConnectivityStatus.ONLINE
        self.simulate_failure = False

    def get_protocol_name(self) -> str:
        return "simulated_vision"

    def connect(self, device_id: str = "") -> bool:
        self.connectivity = ConnectivityStatus.ONLINE
        return True

    def disconnect(self, device_id: str = "") -> bool:
        self.connectivity = ConnectivityStatus.DISCONNECTED
        return True

    def get_status(self, device_id: str = "") -> ConnectivityStatus:
        return self.connectivity

    def get_health(self, device_id: str = "") -> DeviceHealth:
        status = DeviceHealthStatus.HEALTHY if self.connectivity == ConnectivityStatus.ONLINE and self.battery_pct > 20.0 else DeviceHealthStatus.DEGRADED
        return DeviceHealth(
            device_id=self.device_id,
            status=status,
            connectivity=self.connectivity,
            battery_pct=self.battery_pct,
            capability_availability={
                "detect_motion": self.camera_online,
                "detect_person": self.camera_online,
                "detect_anomaly": self.camera_online,
                "capture_image": self.camera_online,
                "capture_video": self.camera_online,
                "get_telemetry": True,
                "emit_event": True,
            },
            last_heartbeat=time.time(),
        )

    def heartbeat(self, device_id: str = "") -> DeviceHeartbeat:
        return DeviceHeartbeat(
            device_id=self.device_id,
            timestamp=time.time(),
            connectivity_state=self.connectivity,
            health_summary=DeviceHealthStatus.HEALTHY if self.connectivity == ConnectivityStatus.ONLINE else DeviceHealthStatus.UNHEALTHY,
            metrics={"camera_online": self.camera_online, "microphone_online": self.microphone_online},
        )

    def get_capabilities(self, device_id: str = "") -> Sequence[DeviceCapabilityDescriptor]:
        return _VISION_CAPABILITIES

    def format_command(self, tool_call: ToolCall) -> Dict[str, Any]:
        return {
            "protocol": self.get_protocol_name(),
            "action": tool_call.action,
            "params": tool_call.parameters,
        }

    def execute_command(self, command: DeviceCommand) -> Result:
        if self.simulate_failure:
            return Result.fail(
                message=f"Vision '{self.device_id}' hardware error simulated.",
                capability=command.capability,
                action=command.action,
                call_id=command.dispatch_id,
                data={"error_code": DeviceErrorCode.DEVICE_ERROR.value},
            )

        action = command.action.lower()
        params = command.parameters
        observations: List[MultimodalObservation] = []

        if action == "detect_motion":
            target_region = str(params.get("region", "entryway"))
            motion_data = {
                "motion_detected": True,
                "confidence": 0.95,
                "region": target_region,
                "bounding_box": [0.12, 0.25, 0.45, 0.65],
            }
            self.last_detection = motion_data
            obs = MultimodalObservation(
                observation_id=f"obs_{command.dispatch_id}_motion",
                source_id=self.device_id,
                source_type="VISION",
                modality=ModalityType.EVENT,
                timestamp=command.timestamp,
                payload=motion_data,
                location=self.location,
                correlation_id=command.correlation_id,
                causation_id=command.dispatch_id,
            )
            observations.append(obs)
            return Result.ok(
                message=f"Vision '{self.device_id}' detected motion in region '{target_region}'.",
                data={"motion_detected": True, "confidence": 0.95, "region": target_region, "observations": observations},
                capability=command.capability,
                action=action,
                call_id=command.dispatch_id,
            )

        elif action == "detect_person":
            person_data = {
                "event": "PERSON_DETECTED",
                "confidence": float(params.get("confidence", 0.94)),
                "person_count": int(params.get("person_count", 1)),
                "location": self.location.to_dict(),
            }
            self.last_detection = person_data
            obs = MultimodalObservation(
                observation_id=f"obs_{command.dispatch_id}_person",
                source_id=self.device_id,
                source_type="VISION",
                modality=ModalityType.EVENT,
                timestamp=command.timestamp,
                payload=person_data,
                location=self.location,
                correlation_id=command.correlation_id,
                causation_id=command.dispatch_id,
            )
            observations.append(obs)
            return Result.ok(
                message=f"Vision '{self.device_id}' detected person with confidence {person_data['confidence']}.",
                data={"event": "PERSON_DETECTED", "person_data": person_data, "observations": observations},
                capability=command.capability,
                action=action,
                call_id=command.dispatch_id,
            )

        elif action == "detect_anomaly":
            anomaly_data = {
                "event": "ANOMALY_DETECTED",
                "anomaly_type": str(params.get("type", "UNEXPECTED_MOTION")),
                "confidence": 0.89,
                "description": str(params.get("description", "Unidentified nocturnal activity")),
            }
            self.last_detection = anomaly_data
            obs = MultimodalObservation(
                observation_id=f"obs_{command.dispatch_id}_anomaly",
                source_id=self.device_id,
                source_type="VISION",
                modality=ModalityType.EVENT,
                timestamp=command.timestamp,
                payload=anomaly_data,
                location=self.location,
                correlation_id=command.correlation_id,
                causation_id=command.dispatch_id,
            )
            observations.append(obs)
            return Result.ok(
                message=f"Vision '{self.device_id}' detected anomaly '{anomaly_data['anomaly_type']}'.",
                data={"anomaly": anomaly_data, "observations": observations},
                capability=command.capability,
                action=action,
                call_id=command.dispatch_id,
            )

        elif action == "capture_image":
            artifact_ref = f"artifacts/vision/{self.device_id}/{command.dispatch_id}.jpg"
            obs = MultimodalObservation(
                observation_id=f"obs_{command.dispatch_id}_img",
                source_id=self.device_id,
                source_type="VISION",
                modality=ModalityType.IMAGE,
                timestamp=command.timestamp,
                payload={"description": "High-resolution surveillance capture", "fov": "wide"},
                artifact_reference=artifact_ref,
                location=self.location,
                correlation_id=command.correlation_id,
                causation_id=command.dispatch_id,
            )
            observations.append(obs)
            return Result.ok(
                message=f"Vision '{self.device_id}' captured image.",
                data={"artifact_reference": artifact_ref, "observations": observations},
                capability=command.capability,
                action=action,
                call_id=command.dispatch_id,
            )

        elif action == "capture_video":
            duration = float(params.get("duration_seconds", 10.0))
            self.recording = True
            artifact_ref = f"artifacts/vision/{self.device_id}/{command.dispatch_id}.mp4"
            obs = MultimodalObservation(
                observation_id=f"obs_{command.dispatch_id}_vid",
                source_id=self.device_id,
                source_type="VISION",
                modality=ModalityType.VIDEO_FRAME,
                timestamp=command.timestamp,
                payload={"duration_seconds": duration, "fps": 30},
                artifact_reference=artifact_ref,
                location=self.location,
                correlation_id=command.correlation_id,
                causation_id=command.dispatch_id,
            )
            observations.append(obs)
            self.recording = False
            return Result.ok(
                message=f"Vision '{self.device_id}' recorded {duration}s video clip.",
                data={"artifact_reference": artifact_ref, "duration_seconds": duration, "observations": observations},
                capability=command.capability,
                action=action,
                call_id=command.dispatch_id,
            )

        elif action == "get_telemetry":
            telem_payload = {
                "battery_pct": self.battery_pct,
                "camera_online": self.camera_online,
                "microphone_online": self.microphone_online,
                "recording": self.recording,
                "last_detection": self.last_detection,
            }
            obs = MultimodalObservation(
                observation_id=f"obs_{command.dispatch_id}_telem",
                source_id=self.device_id,
                source_type="VISION",
                modality=ModalityType.TELEMETRY,
                timestamp=command.timestamp,
                payload=telem_payload,
                location=self.location,
                correlation_id=command.correlation_id,
                causation_id=command.dispatch_id,
            )
            observations.append(obs)
            return Result.ok(
                message=f"Vision '{self.device_id}' telemetry retrieved.",
                data={"telemetry": telem_payload, "observations": observations},
                capability=command.capability,
                action=action,
                call_id=command.dispatch_id,
            )

        elif action == "emit_event":
            evt_name = str(params.get("event", "PERSON_DETECTED"))
            evt_payload = {
                "event": evt_name,
                "confidence": float(params.get("confidence", 0.94)),
                "location": self.location.to_dict(),
                "details": dict(params.get("details", {})),
            }
            obs = MultimodalObservation(
                observation_id=f"obs_{command.dispatch_id}_evt",
                source_id=self.device_id,
                source_type="VISION",
                modality=ModalityType.EVENT,
                timestamp=command.timestamp,
                payload=evt_payload,
                location=self.location,
                correlation_id=command.correlation_id,
                causation_id=command.dispatch_id,
            )
            observations.append(obs)
            return Result.ok(
                message=f"Vision '{self.device_id}' emitted event '{evt_name}'.",
                data={"event": evt_name, "observations": observations},
                capability=command.capability,
                action=action,
                call_id=command.dispatch_id,
            )

        return Result.fail(
            message=f"Action '{action}' is not supported by VirtualVisionAdapter.",
            capability=command.capability,
            action=action,
            call_id=command.dispatch_id,
            data={"error_code": DeviceErrorCode.UNSUPPORTED_ACTION.value},
        )


# ============================================================================
# 2. Virtual Drone Adapter (ATLAS Drone)
# ============================================================================

class VirtualDroneAdapter(DeviceAdapterInterface):
    """
    Deterministic simulated flight controller and telemetry adapter for ATLAS Drones.
    Primary role: aerial sensing & intervention (HYBRID).
    """

    def __init__(
        self,
        device_id: str = "ATLAS_DRONE_01",
        home_location: Optional[GeoLocation] = None,
    ):
        self.device_id = device_id
        self.location = home_location or GeoLocation(latitude=37.7749, longitude=-122.4194, altitude=0.0)
        self.altitude = 0.0
        self.battery_pct = 100.0
        self.armed = False
        self.airborne = False
        self.current_waypoint: Optional[GeoLocation] = None
        self.connectivity = ConnectivityStatus.ONLINE
        self.simulate_timeout = False
        self.simulate_failure = False

    def get_protocol_name(self) -> str:
        return "simulated_drone"

    def connect(self, device_id: str = "") -> bool:
        self.connectivity = ConnectivityStatus.ONLINE
        return True

    def disconnect(self, device_id: str = "") -> bool:
        self.connectivity = ConnectivityStatus.DISCONNECTED
        return True

    def get_status(self, device_id: str = "") -> ConnectivityStatus:
        return self.connectivity

    def get_health(self, device_id: str = "") -> DeviceHealth:
        status = DeviceHealthStatus.HEALTHY
        if self.connectivity != ConnectivityStatus.ONLINE:
            status = DeviceHealthStatus.UNHEALTHY
        elif self.battery_pct <= 5.0:
            status = DeviceHealthStatus.UNHEALTHY
        elif self.battery_pct <= 20.0:
            status = DeviceHealthStatus.DEGRADED

        return DeviceHealth(
            device_id=self.device_id,
            status=status,
            connectivity=self.connectivity,
            battery_pct=self.battery_pct,
            capability_availability={
                "takeoff": self.battery_pct > 5.0,
                "land": True,
                "navigate": self.airborne,
                "hover": self.airborne,
                "capture_image": True,
                "get_telemetry": True,
            },
            last_heartbeat=time.time(),
        )

    def heartbeat(self, device_id: str = "") -> DeviceHeartbeat:
        return DeviceHeartbeat(
            device_id=self.device_id,
            timestamp=time.time(),
            connectivity_state=self.connectivity,
            health_summary=DeviceHealthStatus.HEALTHY if self.connectivity == ConnectivityStatus.ONLINE else DeviceHealthStatus.UNHEALTHY,
            metrics={"altitude": self.altitude, "airborne": self.airborne, "battery_pct": self.battery_pct},
        )

    def get_capabilities(self, device_id: str = "") -> Sequence[DeviceCapabilityDescriptor]:
        return _DRONE_CAPABILITIES

    def format_command(self, tool_call: ToolCall) -> Dict[str, Any]:
        return {
            "protocol": self.get_protocol_name(),
            "action": tool_call.action,
            "params": tool_call.parameters,
        }

    def execute_command(self, command: DeviceCommand) -> Result:
        if self.simulate_timeout:
            return Result.fail(
                message=f"Drone '{self.device_id}' command timed out.",
                capability=command.capability,
                action=command.action,
                call_id=command.dispatch_id,
                data={"error_code": DeviceErrorCode.COMMAND_TIMEOUT.value},
            )

        if self.simulate_failure:
            return Result.fail(
                message=f"Drone '{self.device_id}' flight controller hardware failure simulated.",
                capability=command.capability,
                action=command.action,
                call_id=command.dispatch_id,
                data={"error_code": DeviceErrorCode.DEVICE_ERROR.value},
            )

        action = command.action.lower()
        params = command.parameters
        observations: List[MultimodalObservation] = []

        if action == "takeoff":
            if self.battery_pct <= 5.0:
                return Result.fail(
                    message=f"Drone '{self.device_id}' cannot takeoff: battery too low ({self.battery_pct:.1f}%).",
                    capability=command.capability,
                    action=action,
                    call_id=command.dispatch_id,
                    data={"error_code": DeviceErrorCode.LOW_BATTERY.value},
                )
            target_alt = float(params.get("target_altitude", 10.0))
            self.armed = True
            self.airborne = True
            self.altitude = target_alt
            self.battery_pct = max(0.0, self.battery_pct - 2.0)
            self.location = GeoLocation(
                latitude=self.location.latitude,
                longitude=self.location.longitude,
                altitude=self.altitude,
            )

            obs = MultimodalObservation(
                observation_id=f"obs_{command.dispatch_id}_telem",
                source_id=self.device_id,
                source_type="DRONE",
                modality=ModalityType.TELEMETRY,
                timestamp=command.timestamp,
                payload={"altitude": self.altitude, "battery_pct": self.battery_pct, "airborne": True},
                location=self.location,
                correlation_id=command.correlation_id,
                causation_id=command.dispatch_id,
            )
            observations.append(obs)

            return Result.ok(
                message=f"Drone '{self.device_id}' took off successfully to {target_alt}m.",
                data={"altitude": self.altitude, "observations": observations},
                capability=command.capability,
                action=action,
                call_id=command.dispatch_id,
            )

        elif action == "land":
            self.airborne = False
            self.altitude = 0.0
            self.armed = False
            self.battery_pct = max(0.0, self.battery_pct - 1.0)
            self.location = GeoLocation(
                latitude=self.location.latitude,
                longitude=self.location.longitude,
                altitude=0.0,
            )

            obs = MultimodalObservation(
                observation_id=f"obs_{command.dispatch_id}_land",
                source_id=self.device_id,
                source_type="DRONE",
                modality=ModalityType.TELEMETRY,
                timestamp=command.timestamp,
                payload={"altitude": 0.0, "battery_pct": self.battery_pct, "airborne": False},
                location=self.location,
                correlation_id=command.correlation_id,
                causation_id=command.dispatch_id,
            )
            observations.append(obs)

            return Result.ok(
                message=f"Drone '{self.device_id}' landed successfully.",
                data={"altitude": 0.0, "observations": observations},
                capability=command.capability,
                action=action,
                call_id=command.dispatch_id,
            )

        elif action in ("go_to_waypoint", "navigate"):
            lat = float(params["latitude"])
            lon = float(params["longitude"])
            alt = float(params.get("altitude", self.altitude or 10.0))

            target_loc = GeoLocation(latitude=lat, longitude=lon, altitude=alt)
            self.current_waypoint = target_loc
            self.location = target_loc
            self.altitude = alt
            self.battery_pct = max(0.0, self.battery_pct - 3.0)

            obs = MultimodalObservation(
                observation_id=f"obs_{command.dispatch_id}_gps",
                source_id=self.device_id,
                source_type="DRONE",
                modality=ModalityType.GPS,
                timestamp=command.timestamp,
                payload={"status": "WAYPOINT_REACHED", "battery_pct": self.battery_pct},
                location=self.location,
                correlation_id=command.correlation_id,
                causation_id=command.dispatch_id,
            )
            observations.append(obs)

            return Result.ok(
                message=f"Drone '{self.device_id}' navigated to waypoint ({lat:.5f}, {lon:.5f}).",
                data={"location": self.location.to_dict(), "observations": observations},
                capability=command.capability,
                action=action,
                call_id=command.dispatch_id,
            )

        elif action == "hover":
            return Result.ok(
                message=f"Drone '{self.device_id}' hovering at altitude {self.altitude}m.",
                data={"altitude": self.altitude, "location": self.location.to_dict()},
                capability=command.capability,
                action=action,
                call_id=command.dispatch_id,
            )

        elif action == "capture_image":
            target = params.get("target", "field_of_view")
            artifact_ref = f"artifacts/drone/{self.device_id}/{command.dispatch_id}.jpg"
            obs = MultimodalObservation(
                observation_id=f"obs_{command.dispatch_id}_img",
                source_id=self.device_id,
                source_type="DRONE",
                modality=ModalityType.IMAGE,
                timestamp=command.timestamp,
                payload={"target": target, "description": f"Aerial capture at ({self.location.latitude:.4f}, {self.location.longitude:.4f})"},
                artifact_reference=artifact_ref,
                location=self.location,
                correlation_id=command.correlation_id,
                causation_id=command.dispatch_id,
            )
            observations.append(obs)

            return Result.ok(
                message=f"Drone '{self.device_id}' captured image of '{target}'.",
                data={"artifact_reference": artifact_ref, "observations": observations},
                capability=command.capability,
                action=action,
                call_id=command.dispatch_id,
            )

        elif action == "get_telemetry":
            obs = MultimodalObservation(
                observation_id=f"obs_{command.dispatch_id}_telem",
                source_id=self.device_id,
                source_type="DRONE",
                modality=ModalityType.TELEMETRY,
                timestamp=command.timestamp,
                payload={
                    "battery_pct": self.battery_pct,
                    "altitude": self.altitude,
                    "armed": self.armed,
                    "airborne": self.airborne,
                },
                location=self.location,
                correlation_id=command.correlation_id,
                causation_id=command.dispatch_id,
            )
            observations.append(obs)

            return Result.ok(
                message=f"Drone '{self.device_id}' telemetry retrieved.",
                data={
                    "battery_pct": self.battery_pct,
                    "altitude": self.altitude,
                    "airborne": self.airborne,
                    "observations": observations,
                },
                capability=command.capability,
                action=action,
                call_id=command.dispatch_id,
            )

        return Result.fail(
            message=f"Action '{action}' is not supported by VirtualDroneAdapter.",
            capability=command.capability,
            action=action,
            call_id=command.dispatch_id,
            data={"error_code": DeviceErrorCode.UNSUPPORTED_ACTION.value},
        )


# ============================================================================
# 3. Virtual Rover Adapter (ATLAS Rover)
# ============================================================================

class VirtualRoverAdapter(DeviceAdapterInterface):
    """
    Deterministic simulated ground robotics rover adapter.
    Primary role: ground sensing & intervention (HYBRID).
    """

    def __init__(
        self,
        device_id: str = "ATLAS_ROVER_01",
        home_location: Optional[GeoLocation] = None,
    ):
        self.device_id = device_id
        self.location = home_location or GeoLocation(latitude=37.7750, longitude=-122.4195, altitude=0.0)
        self.battery_pct = 100.0
        self.moving = False
        self.current_waypoint: Optional[GeoLocation] = None
        self.connectivity = ConnectivityStatus.ONLINE
        self.simulate_obstacle = False
        self.simulate_failure = False

    def get_protocol_name(self) -> str:
        return "simulated_rover"

    def connect(self, device_id: str = "") -> bool:
        self.connectivity = ConnectivityStatus.ONLINE
        return True

    def disconnect(self, device_id: str = "") -> bool:
        self.connectivity = ConnectivityStatus.DISCONNECTED
        return True

    def get_status(self, device_id: str = "") -> ConnectivityStatus:
        return self.connectivity

    def get_health(self, device_id: str = "") -> DeviceHealth:
        status = DeviceHealthStatus.HEALTHY
        if self.connectivity != ConnectivityStatus.ONLINE or self.battery_pct <= 5.0:
            status = DeviceHealthStatus.UNHEALTHY
        elif self.battery_pct <= 20.0:
            status = DeviceHealthStatus.DEGRADED

        return DeviceHealth(
            device_id=self.device_id,
            status=status,
            connectivity=self.connectivity,
            battery_pct=self.battery_pct,
            capability_availability={
                "navigate": not self.simulate_obstacle and self.battery_pct > 5.0,
                "stop": True,
                "capture_image": True,
                "get_telemetry": True,
            },
            last_heartbeat=time.time(),
        )

    def heartbeat(self, device_id: str = "") -> DeviceHeartbeat:
        return DeviceHeartbeat(
            device_id=self.device_id,
            timestamp=time.time(),
            connectivity_state=self.connectivity,
            health_summary=DeviceHealthStatus.HEALTHY if self.connectivity == ConnectivityStatus.ONLINE else DeviceHealthStatus.UNHEALTHY,
            metrics={"moving": self.moving, "battery_pct": self.battery_pct},
        )

    def get_capabilities(self, device_id: str = "") -> Sequence[DeviceCapabilityDescriptor]:
        return _ROVER_CAPABILITIES

    def format_command(self, tool_call: ToolCall) -> Dict[str, Any]:
        return {
            "protocol": self.get_protocol_name(),
            "action": tool_call.action,
            "params": tool_call.parameters,
        }

    def execute_command(self, command: DeviceCommand) -> Result:
        if self.simulate_failure:
            return Result.fail(
                message=f"Rover '{self.device_id}' ground powertrain failure simulated.",
                capability=command.capability,
                action=command.action,
                call_id=command.dispatch_id,
                data={"error_code": DeviceErrorCode.DEVICE_ERROR.value},
            )

        action = command.action.lower()
        params = command.parameters
        observations: List[MultimodalObservation] = []

        if action in ("go_to_waypoint", "navigate"):
            if self.simulate_obstacle:
                self.moving = False
                return Result.fail(
                    message=f"Rover '{self.device_id}' navigation halted: obstacle detected in path.",
                    capability=command.capability,
                    action=action,
                    call_id=command.dispatch_id,
                    data={"error_code": DeviceErrorCode.SAFETY_REJECTION.value},
                )

            lat = float(params["latitude"])
            lon = float(params["longitude"])
            target_loc = GeoLocation(latitude=lat, longitude=lon, altitude=0.0)
            self.current_waypoint = target_loc
            self.location = target_loc
            self.moving = True
            self.battery_pct = max(0.0, self.battery_pct - 2.0)

            obs = MultimodalObservation(
                observation_id=f"obs_{command.dispatch_id}_gps",
                source_id=self.device_id,
                source_type="ROVER",
                modality=ModalityType.GPS,
                timestamp=command.timestamp,
                payload={"status": "WAYPOINT_REACHED", "moving": True},
                location=self.location,
                correlation_id=command.correlation_id,
                causation_id=command.dispatch_id,
            )
            observations.append(obs)

            return Result.ok(
                message=f"Rover '{self.device_id}' navigated to waypoint ({lat:.5f}, {lon:.5f}).",
                data={"location": self.location.to_dict(), "observations": observations},
                capability=command.capability,
                action=action,
                call_id=command.dispatch_id,
            )

        elif action == "stop":
            self.moving = False
            obs = MultimodalObservation(
                observation_id=f"obs_{command.dispatch_id}_stop",
                source_id=self.device_id,
                source_type="ROVER",
                modality=ModalityType.TELEMETRY,
                timestamp=command.timestamp,
                payload={"status": "STOPPED", "moving": False},
                location=self.location,
                correlation_id=command.correlation_id,
                causation_id=command.dispatch_id,
            )
            observations.append(obs)

            return Result.ok(
                message=f"Rover '{self.device_id}' stopped.",
                data={"moving": False, "observations": observations},
                capability=command.capability,
                action=action,
                call_id=command.dispatch_id,
            )

        elif action == "capture_image":
            target = params.get("target", "ground_view")
            artifact_ref = f"artifacts/rover/{self.device_id}/{command.dispatch_id}.jpg"
            obs = MultimodalObservation(
                observation_id=f"obs_{command.dispatch_id}_img",
                source_id=self.device_id,
                source_type="ROVER",
                modality=ModalityType.IMAGE,
                timestamp=command.timestamp,
                payload={"target": target, "description": f"Ground image at ({self.location.latitude:.4f}, {self.location.longitude:.4f})"},
                artifact_reference=artifact_ref,
                location=self.location,
                correlation_id=command.correlation_id,
                causation_id=command.dispatch_id,
            )
            observations.append(obs)

            return Result.ok(
                message=f"Rover '{self.device_id}' captured image of '{target}'.",
                data={"artifact_reference": artifact_ref, "observations": observations},
                capability=command.capability,
                action=action,
                call_id=command.dispatch_id,
            )

        elif action == "get_telemetry":
            obs = MultimodalObservation(
                observation_id=f"obs_{command.dispatch_id}_telem",
                source_id=self.device_id,
                source_type="ROVER",
                modality=ModalityType.TELEMETRY,
                timestamp=command.timestamp,
                payload={"battery_pct": self.battery_pct, "moving": self.moving},
                location=self.location,
                correlation_id=command.correlation_id,
                causation_id=command.dispatch_id,
            )
            observations.append(obs)

            return Result.ok(
                message=f"Rover '{self.device_id}' telemetry retrieved.",
                data={"battery_pct": self.battery_pct, "moving": self.moving, "observations": observations},
                capability=command.capability,
                action=action,
                call_id=command.dispatch_id,
            )

        return Result.fail(
            message=f"Action '{action}' is not supported by VirtualRoverAdapter.",
            capability=command.capability,
            action=action,
            call_id=command.dispatch_id,
            data={"error_code": DeviceErrorCode.UNSUPPORTED_ACTION.value},
        )


# ============================================================================
# 4. Virtual Glass Adapter (ATLAS Glass)
# ============================================================================

class VirtualGlassAdapter(DeviceAdapterInterface):
    """
    Deterministic simulated smart glasses (HUD & camera) adapter for ATLAS Glass.
    Primary role: wearable perception & user interaction (HYBRID).
    """

    def __init__(
        self,
        device_id: str = "ATLAS_GLASS_01",
        home_location: Optional[GeoLocation] = None,
    ):
        self.device_id = device_id
        self.location = home_location or GeoLocation(latitude=37.7749, longitude=-122.4194, altitude=1.7)
        self.display_text = ""
        self.recording = False
        self.camera_online = True
        self.battery_pct = 100.0
        self.connectivity = ConnectivityStatus.ONLINE
        self.simulate_failure = False

    @property
    def hud_message(self) -> str:
        return self.display_text

    def get_protocol_name(self) -> str:
        return "simulated_glass"

    def connect(self, device_id: str = "") -> bool:
        self.connectivity = ConnectivityStatus.ONLINE
        return True

    def disconnect(self, device_id: str = "") -> bool:
        self.connectivity = ConnectivityStatus.DISCONNECTED
        return True

    def get_status(self, device_id: str = "") -> ConnectivityStatus:
        return self.connectivity

    def get_health(self, device_id: str = "") -> DeviceHealth:
        status = DeviceHealthStatus.HEALTHY if self.connectivity == ConnectivityStatus.ONLINE and self.battery_pct > 20.0 else DeviceHealthStatus.DEGRADED
        return DeviceHealth(
            device_id=self.device_id,
            status=status,
            connectivity=self.connectivity,
            battery_pct=self.battery_pct,
            capability_availability={
                "display_hud": True,
                "capture_image": self.camera_online,
                "capture_audio": True,
                "get_location": True,
                "get_telemetry": True,
            },
            last_heartbeat=time.time(),
        )

    def heartbeat(self, device_id: str = "") -> DeviceHeartbeat:
        return DeviceHeartbeat(
            device_id=self.device_id,
            timestamp=time.time(),
            connectivity_state=self.connectivity,
            health_summary=DeviceHealthStatus.HEALTHY if self.connectivity == ConnectivityStatus.ONLINE else DeviceHealthStatus.UNHEALTHY,
            metrics={"display_text": self.display_text, "battery_pct": self.battery_pct},
        )

    def get_capabilities(self, device_id: str = "") -> Sequence[DeviceCapabilityDescriptor]:
        return _GLASS_CAPABILITIES

    def format_command(self, tool_call: ToolCall) -> Dict[str, Any]:
        return {
            "protocol": self.get_protocol_name(),
            "action": tool_call.action,
            "params": tool_call.parameters,
        }

    def execute_command(self, command: DeviceCommand) -> Result:
        if self.simulate_failure:
            return Result.fail(
                message=f"Glass '{self.device_id}' display/camera bus error simulated.",
                capability=command.capability,
                action=command.action,
                call_id=command.dispatch_id,
                data={"error_code": DeviceErrorCode.DEVICE_ERROR.value},
            )

        action = command.action.lower()
        params = command.parameters
        observations: List[MultimodalObservation] = []

        if action == "display_hud":
            text = str(params.get("text") or params.get("message") or "")
            self.display_text = text
            self.battery_pct = max(0.0, self.battery_pct - 0.5)

            return Result.ok(
                message=f"Glass '{self.device_id}' HUD updated with '{text}'.",
                data={"display_text": text, "message": text},
                capability=command.capability,
                action=action,
                call_id=command.dispatch_id,
            )

        elif action == "capture_image":
            target = params.get("target", "user_perspective")
            artifact_ref = f"artifacts/glass/{self.device_id}/{command.dispatch_id}.jpg"
            obs = MultimodalObservation(
                observation_id=f"obs_{command.dispatch_id}_img",
                source_id=self.device_id,
                source_type="ATLAS_GLASS",
                modality=ModalityType.IMAGE,
                timestamp=command.timestamp,
                payload={"target": target, "description": f"First-person perspective capture"},
                artifact_reference=artifact_ref,
                location=self.location,
                correlation_id=command.correlation_id,
                causation_id=command.dispatch_id,
            )
            observations.append(obs)

            return Result.ok(
                message=f"Glass '{self.device_id}' captured image.",
                data={"artifact_reference": artifact_ref, "observations": observations},
                capability=command.capability,
                action=action,
                call_id=command.dispatch_id,
            )

        elif action == "capture_audio":
            duration = float(params.get("duration_seconds", 5.0))
            artifact_ref = f"artifacts/glass/{self.device_id}/{command.dispatch_id}.wav"
            obs = MultimodalObservation(
                observation_id=f"obs_{command.dispatch_id}_audio",
                source_id=self.device_id,
                source_type="ATLAS_GLASS",
                modality=ModalityType.AUDIO_EVENT,
                timestamp=command.timestamp,
                payload={"duration_seconds": duration, "description": "Ambient mic audio capture"},
                artifact_reference=artifact_ref,
                location=self.location,
                correlation_id=command.correlation_id,
                causation_id=command.dispatch_id,
            )
            observations.append(obs)

            return Result.ok(
                message=f"Glass '{self.device_id}' captured {duration}s audio snippet.",
                data={"artifact_reference": artifact_ref, "duration_seconds": duration, "observations": observations},
                capability=command.capability,
                action=action,
                call_id=command.dispatch_id,
            )

        elif action == "get_location":
            obs = MultimodalObservation(
                observation_id=f"obs_{command.dispatch_id}_loc",
                source_id=self.device_id,
                source_type="ATLAS_GLASS",
                modality=ModalityType.GPS,
                timestamp=command.timestamp,
                payload={"location": self.location.to_dict()},
                location=self.location,
                correlation_id=command.correlation_id,
                causation_id=command.dispatch_id,
            )
            observations.append(obs)

            return Result.ok(
                message=f"Glass '{self.device_id}' location retrieved.",
                data={"location": self.location.to_dict(), "observations": observations},
                capability=command.capability,
                action=action,
                call_id=command.dispatch_id,
            )

        elif action == "get_telemetry":
            obs = MultimodalObservation(
                observation_id=f"obs_{command.dispatch_id}_telem",
                source_id=self.device_id,
                source_type="ATLAS_GLASS",
                modality=ModalityType.TELEMETRY,
                timestamp=command.timestamp,
                payload={"battery_pct": self.battery_pct, "display_text": self.display_text},
                location=self.location,
                correlation_id=command.correlation_id,
                causation_id=command.dispatch_id,
            )
            observations.append(obs)

            return Result.ok(
                message=f"Glass '{self.device_id}' telemetry retrieved.",
                data={"battery_pct": self.battery_pct, "display_text": self.display_text, "observations": observations},
                capability=command.capability,
                action=action,
                call_id=command.dispatch_id,
            )

        return Result.fail(
            message=f"Action '{action}' is not supported by VirtualGlassAdapter.",
            capability=command.capability,
            action=action,
            call_id=command.dispatch_id,
            data={"error_code": DeviceErrorCode.UNSUPPORTED_ACTION.value},
        )


# ============================================================================
# Declarative Capability Suites
# ============================================================================

_VISION_CAPABILITIES: Tuple[DeviceCapabilityDescriptor, ...] = (
    DeviceCapabilityDescriptor(
        capability_name="detect_motion",
        action_name="detect_motion",
        capability_id="detect_motion",
        description="Detect motion within an environmental zone",
        supported_actions=("detect_motion",),
        parameters_schema={
            "properties": {
                "region": {"type": "string"},
            }
        },
    ),
    DeviceCapabilityDescriptor(
        capability_name="detect_person",
        action_name="detect_person",
        capability_id="detect_person",
        description="Detect presence and count of persons in view",
        supported_actions=("detect_person",),
        parameters_schema={
            "properties": {
                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "person_count": {"type": "integer", "minimum": 1},
            }
        },
    ),
    DeviceCapabilityDescriptor(
        capability_name="detect_anomaly",
        action_name="detect_anomaly",
        capability_id="detect_anomaly",
        description="Detect visual or auditory environmental anomalies",
        supported_actions=("detect_anomaly",),
        parameters_schema={
            "properties": {
                "type": {"type": "string"},
                "description": {"type": "string"},
            }
        },
    ),
    DeviceCapabilityDescriptor(
        capability_name="capture_image",
        action_name="capture_image",
        capability_id="capture_image",
        description="Capture high-resolution surveillance still frame",
        supported_actions=("capture_image",),
    ),
    DeviceCapabilityDescriptor(
        capability_name="capture_video",
        action_name="capture_video",
        capability_id="capture_video",
        description="Record environmental video clip",
        supported_actions=("capture_video",),
        parameters_schema={
            "properties": {
                "duration_seconds": {"type": "number", "minimum": 1.0, "maximum": 120.0},
            }
        },
    ),
    DeviceCapabilityDescriptor(
        capability_name="get_telemetry",
        action_name="get_telemetry",
        capability_id="get_telemetry",
        description="Retrieve vision sensor health and status telemetry",
        supported_actions=("get_telemetry",),
    ),
    DeviceCapabilityDescriptor(
        capability_name="emit_event",
        action_name="emit_event",
        capability_id="emit_event",
        description="Emit synthetic or detected surveillance event",
        supported_actions=("emit_event",),
        parameters_schema={
            "properties": {
                "event": {"type": "string"},
                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            }
        },
    ),
)

_DRONE_CAPABILITIES: Tuple[DeviceCapabilityDescriptor, ...] = (
    DeviceCapabilityDescriptor(
        capability_name="navigate",
        action_name="go_to_waypoint",
        capability_id="navigate",
        description="Navigate drone to specified GPS coordinate",
        supported_actions=("go_to_waypoint", "navigate"),
        parameters_schema={
            "required": ["latitude", "longitude"],
            "properties": {
                "latitude": {"type": "number", "minimum": -90.0, "maximum": 90.0},
                "longitude": {"type": "number", "minimum": -180.0, "maximum": 180.0},
                "altitude": {"type": "number", "minimum": 0.0, "maximum": 1000.0},
            },
        },
    ),
    DeviceCapabilityDescriptor(
        capability_name="takeoff",
        action_name="takeoff",
        capability_id="takeoff",
        description="Initiate automated vertical takeoff to target altitude",
        supported_actions=("takeoff",),
        parameters_schema={
            "properties": {
                "target_altitude": {"type": "number", "minimum": 1.0, "maximum": 500.0},
            },
        },
    ),
    DeviceCapabilityDescriptor(
        capability_name="land",
        action_name="land",
        capability_id="land",
        description="Initiate automated landing at current coordinates",
        supported_actions=("land",),
    ),
    DeviceCapabilityDescriptor(
        capability_name="hover",
        action_name="hover",
        capability_id="hover",
        description="Hold position and maintain current altitude",
        supported_actions=("hover",),
    ),
    DeviceCapabilityDescriptor(
        capability_name="capture_image",
        action_name="capture_image",
        capability_id="capture_image",
        description="Capture aerial photograph from drone gimbal",
        supported_actions=("capture_image",),
        parameters_schema={
            "properties": {
                "target": {"type": "string"},
            },
        },
    ),
    DeviceCapabilityDescriptor(
        capability_name="get_telemetry",
        action_name="get_telemetry",
        capability_id="get_telemetry",
        description="Query flight metrics, battery, and GPS altitude",
        supported_actions=("get_telemetry",),
    ),
)

_ROVER_CAPABILITIES: Tuple[DeviceCapabilityDescriptor, ...] = (
    DeviceCapabilityDescriptor(
        capability_name="navigate",
        action_name="go_to_waypoint",
        capability_id="navigate",
        description="Drive ground rover to target coordinates",
        supported_actions=("go_to_waypoint", "navigate"),
        parameters_schema={
            "required": ["latitude", "longitude"],
            "properties": {
                "latitude": {"type": "number", "minimum": -90.0, "maximum": 90.0},
                "longitude": {"type": "number", "minimum": -180.0, "maximum": 180.0},
            },
        },
    ),
    DeviceCapabilityDescriptor(
        capability_name="stop",
        action_name="stop",
        capability_id="stop",
        description="Immediately halt ground rover locomotion",
        supported_actions=("stop",),
    ),
    DeviceCapabilityDescriptor(
        capability_name="capture_image",
        action_name="capture_image",
        capability_id="capture_image",
        description="Capture forward or obstacle camera picture",
        supported_actions=("capture_image",),
        parameters_schema={
            "properties": {
                "target": {"type": "string"},
            },
        },
    ),
    DeviceCapabilityDescriptor(
        capability_name="get_telemetry",
        action_name="get_telemetry",
        capability_id="get_telemetry",
        description="Query rover battery, locomotion state, and sensors",
        supported_actions=("get_telemetry",),
    ),
)

_GLASS_CAPABILITIES: Tuple[DeviceCapabilityDescriptor, ...] = (
    DeviceCapabilityDescriptor(
        capability_name="display_hud",
        action_name="display_hud",
        capability_id="display_hud",
        description="Render text or notification onto user smart glasses HUD",
        supported_actions=("display_hud",),
        parameters_schema={
            "required": ["text"],
            "properties": {
                "text": {"type": "string"},
            },
        },
    ),
    DeviceCapabilityDescriptor(
        capability_name="capture_image",
        action_name="capture_image",
        capability_id="capture_image",
        description="Capture first-person POV photograph",
        supported_actions=("capture_image",),
        parameters_schema={
            "properties": {
                "target": {"type": "string"},
            },
        },
    ),
    DeviceCapabilityDescriptor(
        capability_name="capture_audio",
        action_name="capture_audio",
        capability_id="capture_audio",
        description="Record short ambient audio sample through glasses microphone",
        supported_actions=("capture_audio",),
        parameters_schema={
            "properties": {
                "duration_seconds": {"type": "number", "minimum": 1.0, "maximum": 60.0},
            },
        },
    ),
    DeviceCapabilityDescriptor(
        capability_name="get_location",
        action_name="get_location",
        capability_id="get_location",
        description="Query current GPS coordinates of smart glasses wearer",
        supported_actions=("get_location",),
    ),
    DeviceCapabilityDescriptor(
        capability_name="get_telemetry",
        action_name="get_telemetry",
        capability_id="get_telemetry",
        description="Query smart glasses battery and HUD state",
        supported_actions=("get_telemetry",),
    ),
)


# ============================================================================
# Factory Functions
# ============================================================================

def create_virtual_vision(
    device_id: str = "ATLAS_VISION_01",
    home_location: Optional[GeoLocation] = None,
) -> Tuple[DeviceIdentity, VirtualVisionAdapter]:
    """Factory creating fully-configured VirtualVision profile and adapter."""
    loc = home_location or GeoLocation(latitude=37.7749, longitude=-122.4194, altitude=2.5)
    identity = DeviceIdentity(
        device_id=device_id,
        device_type=DeviceType.STATIONARY_SENSOR,
        display_name="ATLAS Vision Environmental Sensor",
        home_location=loc,
        capabilities=_VISION_CAPABILITIES,
        is_simulation=True,
        connectivity_status=ConnectivityStatus.ONLINE,
        product_type=ProductType.VISION,
        product_role=ProductRole.OBSERVATION_SOURCE,
    )
    adapter = VirtualVisionAdapter(device_id=device_id, home_location=loc)
    return identity, adapter


def create_virtual_drone(
    device_id: str = "ATLAS_DRONE_01",
    home_location: Optional[GeoLocation] = None,
) -> Tuple[DeviceIdentity, VirtualDroneAdapter]:
    """Factory creating fully-configured VirtualDrone profile and adapter."""
    loc = home_location or GeoLocation(latitude=37.7749, longitude=-122.4194, altitude=0.0)
    identity = DeviceIdentity(
        device_id=device_id,
        device_type=DeviceType.DRONE_AERIAL,
        display_name="ATLAS Autonomous Drone",
        home_location=loc,
        capabilities=_DRONE_CAPABILITIES,
        is_simulation=True,
        connectivity_status=ConnectivityStatus.ONLINE,
        product_type=ProductType.DRONE,
        product_role=ProductRole.HYBRID,
    )
    adapter = VirtualDroneAdapter(device_id=device_id, home_location=loc)
    return identity, adapter


def create_virtual_rover(
    device_id: str = "ATLAS_ROVER_01",
    home_location: Optional[GeoLocation] = None,
) -> Tuple[DeviceIdentity, VirtualRoverAdapter]:
    """Factory creating fully-configured VirtualRover profile and adapter."""
    loc = home_location or GeoLocation(latitude=37.7750, longitude=-122.4195, altitude=0.0)
    identity = DeviceIdentity(
        device_id=device_id,
        device_type=DeviceType.ROBOT_GROUND,
        display_name="ATLAS Ground Rover",
        home_location=loc,
        capabilities=_ROVER_CAPABILITIES,
        is_simulation=True,
        connectivity_status=ConnectivityStatus.ONLINE,
        product_type=ProductType.ROVER,
        product_role=ProductRole.HYBRID,
    )
    adapter = VirtualRoverAdapter(device_id=device_id, home_location=loc)
    return identity, adapter


def create_virtual_glass(
    device_id: str = "ATLAS_GLASS_01",
    home_location: Optional[GeoLocation] = None,
) -> Tuple[DeviceIdentity, VirtualGlassAdapter]:
    """Factory creating fully-configured VirtualGlass profile and adapter."""
    loc = home_location or GeoLocation(latitude=37.7749, longitude=-122.4194, altitude=1.7)
    identity = DeviceIdentity(
        device_id=device_id,
        device_type=DeviceType.SMART_GLASSES,
        display_name="ATLAS Smart Glasses",
        home_location=loc,
        capabilities=_GLASS_CAPABILITIES,
        is_simulation=True,
        connectivity_status=ConnectivityStatus.ONLINE,
        product_type=ProductType.GLASS,
        product_role=ProductRole.HYBRID,
    )
    adapter = VirtualGlassAdapter(device_id=device_id, home_location=loc)
    return identity, adapter
