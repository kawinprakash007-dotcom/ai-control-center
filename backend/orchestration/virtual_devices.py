from dataclasses import dataclass, field
import time
from typing import Any, Dict, List, Optional, Tuple, Sequence

from core.interfaces.orchestration_interface import DeviceAdapterInterface
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
# Virtual Drone Adapter
# ============================================================================

class VirtualDroneAdapter(DeviceAdapterInterface):
    """
    Deterministic simulated flight controller and telemetry adapter for ATLAS Drones.
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

    def get_protocol_name(self) -> str:
        return "simulated_drone"

    def format_command(self, tool_call: ToolCall) -> Dict[str, Any]:
        return {
            "protocol": self.get_protocol_name(),
            "action": tool_call.action,
            "params": tool_call.parameters,
        }

    def execute_command(self, command: DeviceCommand) -> Result:
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
        )


# ============================================================================
# Virtual Rover Adapter
# ============================================================================

class VirtualRoverAdapter(DeviceAdapterInterface):
    """
    Deterministic simulated ground robotics rover adapter.
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

    def get_protocol_name(self) -> str:
        return "simulated_rover"

    def format_command(self, tool_call: ToolCall) -> Dict[str, Any]:
        return {
            "protocol": self.get_protocol_name(),
            "action": tool_call.action,
            "params": tool_call.parameters,
        }

    def execute_command(self, command: DeviceCommand) -> Result:
        action = command.action.lower()
        params = command.parameters
        observations: List[MultimodalObservation] = []

        if action in ("go_to_waypoint", "navigate"):
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
        )


# ============================================================================
# Virtual Glass Adapter
# ============================================================================

class VirtualGlassAdapter(DeviceAdapterInterface):
    """
    Deterministic simulated smart glasses (HUD & camera) adapter.
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
        self.battery_pct = 100.0
        self.connectivity = ConnectivityStatus.ONLINE

    @property
    def hud_message(self) -> str:
        return self.display_text

    def get_protocol_name(self) -> str:
        return "simulated_glass"

    def format_command(self, tool_call: ToolCall) -> Dict[str, Any]:
        return {
            "protocol": self.get_protocol_name(),
            "action": tool_call.action,
            "params": tool_call.parameters,
        }

    def execute_command(self, command: DeviceCommand) -> Result:
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
        )


# ============================================================================
# Virtual Device Factories
# ============================================================================

def create_virtual_drone(
    device_id: str = "ATLAS_DRONE_01",
    home_location: Optional[GeoLocation] = None,
) -> Tuple[DeviceIdentity, VirtualDroneAdapter]:
    """Factory creating fully-configured VirtualDrone profile and adapter."""
    loc = home_location or GeoLocation(latitude=37.7749, longitude=-122.4194, altitude=0.0)

    caps = (
        DeviceCapabilityDescriptor(
            capability_name="navigate",
            action_name="go_to_waypoint",
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
            parameters_schema={
                "properties": {
                    "target_altitude": {"type": "number", "minimum": 1.0, "maximum": 500.0},
                },
            },
        ),
        DeviceCapabilityDescriptor(
            capability_name="land",
            action_name="land",
        ),
        DeviceCapabilityDescriptor(
            capability_name="hover",
            action_name="hover",
        ),
        DeviceCapabilityDescriptor(
            capability_name="capture_image",
            action_name="capture_image",
            parameters_schema={
                "properties": {
                    "target": {"type": "string"},
                },
            },
        ),
        DeviceCapabilityDescriptor(
            capability_name="get_telemetry",
            action_name="get_telemetry",
        ),
    )

    identity = DeviceIdentity(
        device_id=device_id,
        device_type=DeviceType.DRONE_AERIAL,
        display_name="ATLAS Autonomous Drone",
        home_location=loc,
        capabilities=caps,
        is_simulation=True,
        connectivity_status=ConnectivityStatus.ONLINE,
    )
    adapter = VirtualDroneAdapter(device_id=device_id, home_location=loc)
    return identity, adapter


def create_virtual_rover(
    device_id: str = "ATLAS_ROVER_01",
    home_location: Optional[GeoLocation] = None,
) -> Tuple[DeviceIdentity, VirtualRoverAdapter]:
    """Factory creating fully-configured VirtualRover profile and adapter."""
    loc = home_location or GeoLocation(latitude=37.7750, longitude=-122.4195, altitude=0.0)

    caps = (
        DeviceCapabilityDescriptor(
            capability_name="navigate",
            action_name="go_to_waypoint",
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
        ),
        DeviceCapabilityDescriptor(
            capability_name="capture_image",
            action_name="capture_image",
            parameters_schema={
                "properties": {
                    "target": {"type": "string"},
                },
            },
        ),
        DeviceCapabilityDescriptor(
            capability_name="get_telemetry",
            action_name="get_telemetry",
        ),
    )

    identity = DeviceIdentity(
        device_id=device_id,
        device_type=DeviceType.ROBOT_GROUND,
        display_name="ATLAS Ground Rover",
        home_location=loc,
        capabilities=caps,
        is_simulation=True,
        connectivity_status=ConnectivityStatus.ONLINE,
    )
    adapter = VirtualRoverAdapter(device_id=device_id, home_location=loc)
    return identity, adapter


def create_virtual_glass(
    device_id: str = "ATLAS_GLASS_01",
    home_location: Optional[GeoLocation] = None,
) -> Tuple[DeviceIdentity, VirtualGlassAdapter]:
    """Factory creating fully-configured VirtualGlass profile and adapter."""
    loc = home_location or GeoLocation(latitude=37.7749, longitude=-122.4194, altitude=1.7)

    caps = (
        DeviceCapabilityDescriptor(
            capability_name="display_hud",
            action_name="display_hud",
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
            parameters_schema={
                "properties": {
                    "target": {"type": "string"},
                },
            },
        ),
        DeviceCapabilityDescriptor(
            capability_name="get_telemetry",
            action_name="get_telemetry",
        ),
    )

    identity = DeviceIdentity(
        device_id=device_id,
        device_type=DeviceType.SMART_GLASSES,
        display_name="ATLAS Smart Glasses",
        home_location=loc,
        capabilities=caps,
        is_simulation=True,
        connectivity_status=ConnectivityStatus.ONLINE,
    )
    adapter = VirtualGlassAdapter(device_id=device_id, home_location=loc)
    return identity, adapter
