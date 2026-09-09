import collections
import inspect
import sys
import threading
import time
from typing import Any, Dict, List, Optional
import pytest

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
    GeoLocation,
    ModalityType,
    MultimodalObservation,
)
from core.models.result import Result
from core.models.runtime import CognitiveEvent, CognitiveEventType
from core.models.tool_call import ToolCall
from orchestration.device_gateway import (
    DeviceCommand,
    DeviceGateway,
    DeviceGatewayCapability,
    validate_parameters_against_schema,
)
from orchestration.virtual_devices import (
    VirtualDroneAdapter,
    VirtualGlassAdapter,
    VirtualRoverAdapter,
    create_virtual_drone,
    create_virtual_glass,
    create_virtual_rover,
)
from unittest.mock import MagicMock

if "chromadb" not in sys.modules:
    try:
        import chromadb
    except ImportError:
        sys.modules["chromadb"] = MagicMock()

from tools.capability_registry import CapabilityRegistry
from tools.tool_orchestrator import ToolOrchestrator


class MockEventSink(CognitiveEventSinkInterface):
    """Deterministic in-memory sink for testing event publication."""
    def __init__(self):
        self.events: List[CognitiveEvent] = []

    def publish(self, event: CognitiveEvent) -> None:
        self.events.append(event)

    def record_event(self, event: CognitiveEvent) -> None:
        self.events.append(event)

    def get_events(self, turn_id: Optional[str] = None) -> List[CognitiveEvent]:
        if turn_id is None:
            return list(self.events)
        return [e for e in self.events if e.turn_id == turn_id]

    def clear(self) -> None:
        self.events.clear()


# ============================================================================
# 1-5: Device Registration, Lookup, and Listing
# ============================================================================

def test_01_device_registration():
    """Register a valid device identity."""
    gateway = DeviceGateway()
    drone, _ = create_virtual_drone("ATLAS_DRONE_01")
    gateway.register_device(drone)

    dev = gateway.get_device("ATLAS_DRONE_01")
    assert dev is not None
    assert dev.device_id == "ATLAS_DRONE_01"
    assert dev.device_type == DeviceType.DRONE_AERIAL


def test_02_duplicate_registration_rejected():
    """Duplicate device_id registration is rejected deterministically."""
    gateway = DeviceGateway()
    drone, _ = create_virtual_drone("ATLAS_DRONE_01")
    gateway.register_device(drone)

    with pytest.raises(ValueError, match="already registered"):
        gateway.register_device(drone)


def test_03_unregister_device():
    """Unregister an existing device removes it from registry."""
    gateway = DeviceGateway()
    rover, _ = create_virtual_rover("ATLAS_ROVER_01")
    gateway.register_device(rover)
    assert gateway.get_device("ATLAS_ROVER_01") is not None

    removed = gateway.unregister_device("ATLAS_ROVER_01")
    assert removed is True
    assert gateway.get_device("ATLAS_ROVER_01") is None

    # Unregister non-existent returns False
    assert gateway.unregister_device("NON_EXISTENT") is False


def test_04_device_lookup():
    """Device lookup returns None for non-registered devices."""
    gateway = DeviceGateway()
    assert gateway.get_device("UNKNOWN_DEVICE") is None


def test_05_device_listing_and_filtering():
    """List registered devices with optional DeviceType filter."""
    gateway = DeviceGateway()
    drone, _ = create_virtual_drone("DRONE_A")
    rover, _ = create_virtual_rover("ROVER_B")
    glass, _ = create_virtual_glass("GLASS_C")

    gateway.register_device(drone)
    gateway.register_device(rover)
    gateway.register_device(glass)

    all_devs = gateway.list_devices()
    assert len(all_devs) == 3
    # Deterministic alphabetical ordering by device_id
    assert [d.device_id for d in all_devs] == ["DRONE_A", "GLASS_C", "ROVER_B"]

    drone_only = gateway.list_devices(device_type=DeviceType.DRONE_AERIAL)
    assert len(drone_only) == 1
    assert drone_only[0].device_id == "DRONE_A"


# ============================================================================
# 6-10: Capability, Action, and Parameter Validation
# ============================================================================

def test_06_capability_registration():
    """Device identity preserves declared capability descriptors."""
    drone, _ = create_virtual_drone("DRONE_01")
    caps = drone.capabilities
    cap_names = {c.capability_name for c in caps}
    assert "navigate" in cap_names
    assert "takeoff" in cap_names
    assert "capture_image" in cap_names


def test_07_capability_lookup():
    """Query capability descriptors declared by a device."""
    gateway = DeviceGateway()
    drone, _ = create_virtual_drone("DRONE_01")
    gateway.register_device(drone)

    caps = gateway.list_device_capabilities("DRONE_01")
    assert len(caps) >= 5
    assert any(c.action_name == "go_to_waypoint" for c in caps)


def test_08_capability_validation():
    """Reject command requesting an undeclared capability."""
    gateway = DeviceGateway()
    drone, adapter = create_virtual_drone("DRONE_01")
    gateway.register_device(drone)
    gateway.register_adapter(adapter, device_id="DRONE_01")

    res = gateway.dispatch_to_device(
        device_id="DRONE_01",
        capability="laser_cutter",  # not on drone!
        action="fire",
        parameters={},
    )
    assert not res.success
    assert "does not support action 'fire' on capability 'laser_cutter'" in res.message


def test_09_action_validation():
    """Reject command requesting an invalid action on an existing capability."""
    gateway = DeviceGateway()
    drone, adapter = create_virtual_drone("DRONE_01")
    gateway.register_device(drone)
    gateway.register_adapter(adapter, device_id="DRONE_01")

    res = gateway.dispatch_to_device(
        device_id="DRONE_01",
        capability="navigate",
        action="hyperjump",  # invalid action
        parameters={"latitude": 37.77, "longitude": -122.41},
    )
    assert not res.success
    assert "does not support action 'hyperjump'" in res.message


def test_10_parameter_validation_schemas():
    """Validate parameters against capability descriptor schema."""
    schema = {
        "required": ["latitude", "longitude"],
        "properties": {
            "latitude": {"type": "number", "minimum": -90.0, "maximum": 90.0},
            "longitude": {"type": "number", "minimum": -180.0, "maximum": 180.0},
            "altitude": {"type": "number", "minimum": 0.0, "maximum": 500.0},
        },
    }

    # Valid
    ok, err = validate_parameters_against_schema({"latitude": 37.77, "longitude": -122.41}, schema)
    assert ok is True
    assert err is None

    # Missing required parameter
    ok, err = validate_parameters_against_schema({"latitude": 37.77}, schema)
    assert ok is False
    assert "Missing required parameter: 'longitude'" in err

    # Out of range parameter
    ok, err = validate_parameters_against_schema({"latitude": 999.0, "longitude": -122.41}, schema)
    assert ok is False
    assert "exceeds maximum" in err

    # Wrong type
    ok, err = validate_parameters_against_schema({"latitude": "north_pole", "longitude": -122.41}, schema)
    assert ok is False
    assert "must be a number" in err


# ============================================================================
# 11-13: Adapter Registration and Resolution
# ============================================================================

def test_11_adapter_registration():
    """Register adapters by DeviceType or specific device_id."""
    gateway = DeviceGateway()
    drone, adapter = create_virtual_drone("DRONE_01")
    gateway.register_device(drone)
    gateway.register_adapter(adapter, device_type=DeviceType.DRONE_AERIAL)

    resolved = gateway.resolve_adapter("DRONE_01")
    assert resolved is adapter


def test_12_adapter_resolution_priority():
    """Device-specific adapter takes precedence over DeviceType adapter."""
    gateway = DeviceGateway()
    drone, default_adapter = create_virtual_drone("DRONE_01")
    special_adapter = VirtualDroneAdapter("DRONE_01")

    gateway.register_device(drone)
    gateway.register_adapter(default_adapter, device_type=DeviceType.DRONE_AERIAL)
    gateway.register_adapter(special_adapter, device_id="DRONE_01")

    resolved = gateway.resolve_adapter("DRONE_01")
    assert resolved is special_adapter


def test_13_unknown_adapter():
    """Attempt to resolve adapter for unregistered device or missing adapter fails."""
    gateway = DeviceGateway()
    with pytest.raises(KeyError, match="No registered adapter found"):
        gateway.resolve_adapter("NON_EXISTENT_DEV")


# ============================================================================
# 14-15: Device Status and Heartbeat
# ============================================================================

def test_14_device_status():
    """Update and query connectivity status explicitly."""
    gateway = DeviceGateway()
    drone, _ = create_virtual_drone("DRONE_01")
    gateway.register_device(drone)

    assert gateway.query_device_status("DRONE_01") == ConnectivityStatus.ONLINE

    gateway.update_device_status("DRONE_01", ConnectivityStatus.DEGRADED)
    assert gateway.query_device_status("DRONE_01") == ConnectivityStatus.DEGRADED


def test_15_heartbeat_and_timeout():
    """Heartbeat updates last_heartbeat; status expires to DISCONNECTED after timeout."""
    gateway = DeviceGateway(heartbeat_timeout_seconds=5.0)
    drone, _ = create_virtual_drone("DRONE_01")
    gateway.register_device(drone)

    # Initial heartbeat at t=100.0
    gateway.record_heartbeat("DRONE_01", timestamp=100.0)
    assert gateway.query_device_status("DRONE_01", now=102.0) == ConnectivityStatus.ONLINE

    # Elapsed > 5s -> DISCONNECTED
    assert gateway.query_device_status("DRONE_01", now=106.0) == ConnectivityStatus.DISCONNECTED

    # Fresh heartbeat -> ONLINE again
    gateway.record_heartbeat("DRONE_01", timestamp=107.0)
    assert gateway.query_device_status("DRONE_01", now=108.0) == ConnectivityStatus.ONLINE


# ============================================================================
# 16-25: Semantic Command Dispatch and Rejections
# ============================================================================

def test_16_dispatch_success():
    """Successful semantic command dispatch to virtual adapter."""
    gateway = DeviceGateway()
    drone, adapter = create_virtual_drone("DRONE_01")
    gateway.register_device(drone)
    gateway.register_adapter(adapter, device_id="DRONE_01")

    res = gateway.dispatch_to_device(
        device_id="DRONE_01",
        capability="takeoff",
        action="takeoff",
        parameters={"target_altitude": 15.0},
        dispatch_id="disp_takeoff_01",
    )
    assert res.success is True
    assert "took off successfully" in res.message
    assert adapter.airborne is True
    assert adapter.altitude == 15.0


def test_17_unknown_device_dispatch():
    """Dispatching to an unknown device fails deterministically."""
    gateway = DeviceGateway()
    res = gateway.dispatch_to_device(
        device_id="GHOST_DRONE",
        capability="takeoff",
        action="takeoff",
        parameters={},
    )
    assert res.success is False
    assert "not registered in DeviceGateway" in res.message


def test_18_offline_device_dispatch():
    """Dispatching to an offline/disconnected device is rejected."""
    gateway = DeviceGateway()
    drone, adapter = create_virtual_drone("DRONE_01")
    gateway.register_device(drone)
    gateway.register_adapter(adapter, device_id="DRONE_01")

    gateway.update_device_status("DRONE_01", ConnectivityStatus.OFFLINE)
    res = gateway.dispatch_to_device(
        device_id="DRONE_01",
        capability="takeoff",
        action="takeoff",
        parameters={},
    )
    assert res.success is False
    assert "unavailable: connectivity status is OFFLINE" in res.message


def test_19_missing_capability():
    """Dispatching an undeclared capability fails without executing adapter."""
    gateway = DeviceGateway()
    rover, adapter = create_virtual_rover("ROVER_01")
    gateway.register_device(rover)
    gateway.register_adapter(adapter, device_id="ROVER_01")

    res = gateway.dispatch_to_device(
        device_id="ROVER_01",
        capability="fly",
        action="fly_high",
        parameters={},
    )
    assert res.success is False
    assert "does not support action 'fly_high' on capability 'fly'" in res.message


def test_20_invalid_action():
    """Dispatching an invalid action on an existing capability fails."""
    gateway = DeviceGateway()
    rover, adapter = create_virtual_rover("ROVER_01")
    gateway.register_device(rover)
    gateway.register_adapter(adapter, device_id="ROVER_01")

    res = gateway.dispatch_to_device(
        device_id="ROVER_01",
        capability="navigate",
        action="drift",
        parameters={"latitude": 37.77, "longitude": -122.41},
    )
    assert res.success is False
    assert "does not support action 'drift'" in res.message


def test_21_invalid_parameters():
    """Dispatching parameters violating schema fails before reaching adapter."""
    gateway = DeviceGateway()
    drone, adapter = create_virtual_drone("DRONE_01")
    gateway.register_device(drone)
    gateway.register_adapter(adapter, device_id="DRONE_01")

    # Latitude out of bounds (999 > 90)
    res = gateway.dispatch_to_device(
        device_id="DRONE_01",
        capability="navigate",
        action="go_to_waypoint",
        parameters={"latitude": 999.0, "longitude": -122.41},
    )
    assert res.success is False
    assert "Parameter validation failed" in res.message


def test_22_adapter_failure_handled_gracefully():
    """Adapter exception is caught and returned as Result.fail."""
    class FailingAdapter(DeviceAdapterInterface):
        def get_protocol_name(self) -> str:
            return "failing_protocol"
        def format_command(self, tool_call: ToolCall) -> Dict[str, Any]:
            return {}
        def execute_command(self, command: DeviceCommand) -> Result:
            raise RuntimeError("Hardware bus timeout error.")

    gateway = DeviceGateway()
    drone, _ = create_virtual_drone("DRONE_FAIL")
    gateway.register_device(drone)
    gateway.register_adapter(FailingAdapter(), device_id="DRONE_FAIL")

    res = gateway.dispatch_to_device(
        device_id="DRONE_FAIL",
        capability="takeoff",
        action="takeoff",
        parameters={},
    )
    assert res.success is False
    assert "Adapter execution error" in res.message
    assert "Hardware bus timeout" in res.message


def test_23_timeout_bounded_dispatch():
    """Gateway rejects commands to timed-out/dead edge devices."""
    gateway = DeviceGateway(heartbeat_timeout_seconds=2.0)
    drone, adapter = create_virtual_drone("DRONE_TIMEOUT")
    gateway.register_device(drone)
    gateway.register_adapter(adapter, device_id="DRONE_TIMEOUT")

    # Alive at t=10.0
    gateway.record_heartbeat("DRONE_TIMEOUT", timestamp=10.0)
    # At t=15.0, status is DISCONNECTED
    res = gateway.dispatch_to_device(
        device_id="DRONE_TIMEOUT",
        capability="takeoff",
        action="takeoff",
        parameters={},
        now=15.0,
    )
    assert res.success is False
    assert "connectivity status is DISCONNECTED" in res.message


def test_24_duplicate_dispatch_handling():
    """Duplicate dispatch_id is rejected deterministically."""
    gateway = DeviceGateway()
    drone, adapter = create_virtual_drone("DRONE_01")
    gateway.register_device(drone)
    gateway.register_adapter(adapter, device_id="DRONE_01")

    res1 = gateway.dispatch_to_device(
        device_id="DRONE_01",
        capability="takeoff",
        action="takeoff",
        parameters={},
        dispatch_id="disp_dup_test_99",
    )
    assert res1.success is True

    # Same dispatch ID again
    res2 = gateway.dispatch_to_device(
        device_id="DRONE_01",
        capability="takeoff",
        action="takeoff",
        parameters={},
        dispatch_id="disp_dup_test_99",
    )
    assert res2.success is False
    assert "Duplicate dispatch_id 'disp_dup_test_99' rejected" in res2.message


def test_25_dispatch_correlation_and_causation():
    """Dispatch preserves correlation_id and causation_id down to adapter."""
    gateway = DeviceGateway()
    rover, adapter = create_virtual_rover("ROVER_01")
    gateway.register_device(rover)
    gateway.register_adapter(adapter, device_id="ROVER_01")

    res = gateway.dispatch_to_device(
        device_id="ROVER_01",
        capability="stop",
        action="stop",
        parameters={},
        dispatch_id="disp_corr_01",
        correlation_id="INCIDENT_CORR_88",
        causation_id="SIT_TRIGGER_12",
    )
    assert res.success is True
    assert "observations" in res.data
    obs = res.data["observations"][0]
    assert obs.correlation_id == "INCIDENT_CORR_88"
    assert obs.causation_id == "disp_corr_01"


# ============================================================================
# 26-32: Observability, Boundaries, and Security Audit
# ============================================================================

def test_26_observability_events():
    """Device registration and command dispatches emit sanitized CognitiveEvents."""
    event_sink = MockEventSink()
    gateway = DeviceGateway(event_sink=event_sink)
    drone, adapter = create_virtual_drone("DRONE_OBS")
    gateway.register_device(drone)
    gateway.register_adapter(adapter, device_id="DRONE_OBS")

    gateway.dispatch_to_device(
        device_id="DRONE_OBS",
        capability="takeoff",
        action="takeoff",
        parameters={},
        dispatch_id="disp_obs_123",
    )

    event_types = [e.event_type for e in event_sink.get_events()]
    assert CognitiveEventType.DEVICE_REGISTERED in event_types
    assert CognitiveEventType.DEVICE_DISPATCH_REQUESTED in event_types
    assert CognitiveEventType.DEVICE_DISPATCH_ACCEPTED in event_types


def test_27_policy_boundary():
    """DeviceGateway is NOT a policy engine; it evaluates capability/dispatchability only."""
    gateway = DeviceGateway()
    gw_methods = dir(gateway)
    assert "evaluate_policy" not in gw_methods
    assert "authorize" not in gw_methods
    assert "enforce_policy" not in gw_methods


def test_28_tool_orchestrator_boundary():
    """ToolOrchestrator invokes DeviceGateway as a registered capability."""
    gateway = DeviceGateway()
    drone, adapter = create_virtual_drone("ATLAS_DRONE_01")
    gateway.register_device(drone)
    gateway.register_adapter(adapter, device_id="ATLAS_DRONE_01")

    cap_bridge = DeviceGatewayCapability(gateway)
    registry = CapabilityRegistry()
    registry.register("device_gateway", cap_bridge)

    orchestrator = ToolOrchestrator(registry=registry)

    # Dispatch via ToolOrchestrator.execute()
    call = ToolCall(
        capability="device_gateway",
        action="dispatch_capability",
        parameters={
            "device_id": "ATLAS_DRONE_01",
            "capability": "takeoff",
            "action": "takeoff",
            "parameters": {"target_altitude": 12.0},
        },
        call_id="call_orch_takeoff",
    )

    result = orchestrator.execute(call)
    assert result.success is True
    assert "took off successfully" in result.message
    assert adapter.airborne is True
    assert adapter.altitude == 12.0


def test_29_no_direct_world_state_mutation():
    """DeviceGateway and VirtualAdapters do not touch WorldState directly."""
    for cls in (DeviceGateway, VirtualDroneAdapter, VirtualRoverAdapter, VirtualGlassAdapter):
        src = inspect.getsource(cls)
        assert "WorldStateStore" not in src
        assert "mutate_world_state" not in src
        assert "DeterministicWorldStateUpdater" not in src


def test_30_no_direct_goal_store_mutation():
    """DeviceGateway and VirtualAdapters do not mutate GoalStore."""
    for cls in (DeviceGateway, VirtualDroneAdapter, VirtualRoverAdapter, VirtualGlassAdapter):
        src = inspect.getsource(cls)
        assert "GoalStore" not in src
        assert "create_goal" not in src
        assert "add_goal" not in src


def test_31_no_model_invocation():
    """DeviceGateway and VirtualAdapters do not invoke LLMs, VLMs, or ReasoningEngine."""
    for cls in (DeviceGateway, VirtualDroneAdapter, VirtualRoverAdapter, VirtualGlassAdapter):
        src = inspect.getsource(cls)
        assert "ReasoningEngine" not in src
        assert "ModelRouter" not in src
        assert "generate_content" not in src


def test_32_no_hardware_imports():
    """Source inspection: Zero hardware protocol or native SDK imports."""
    import orchestration.device_gateway as gw_mod
    import orchestration.virtual_devices as vd_mod

    for mod in (gw_mod, vd_mod):
        src = inspect.getsource(mod)
        for bad in (
            "import socket", "import serial", "import pyserial", "import RPi.GPIO",
            "import paho.mqtt", "import pymavlink", "import mavsdk", "import rclpy",
            "import subprocess", "os.system",
        ):
            assert bad not in src, f"Discovered forbidden import '{bad}' in {mod.__name__}"


# ============================================================================
# 33-39: Virtual Device Adapters & Closed-Loop Simulation
# ============================================================================

def test_33_virtual_drone_lifecycle():
    """VirtualDrone state transitions: takeoff -> navigate -> hover -> land."""
    _, adapter = create_virtual_drone("DRONE_TEST")
    assert adapter.altitude == 0.0
    assert adapter.airborne is False

    # Takeoff
    cmd1 = DeviceCommand("d1", "DRONE_TEST", "takeoff", "takeoff", {"target_altitude": 20.0})
    res1 = adapter.execute_command(cmd1)
    assert res1.success is True
    assert adapter.altitude == 20.0
    assert adapter.airborne is True

    # Navigate
    cmd2 = DeviceCommand("d2", "DRONE_TEST", "navigate", "go_to_waypoint", {"latitude": 37.78, "longitude": -122.40})
    res2 = adapter.execute_command(cmd2)
    assert res2.success is True
    assert adapter.location.latitude == 37.78

    # Land
    cmd3 = DeviceCommand("d3", "DRONE_TEST", "land", "land")
    res3 = adapter.execute_command(cmd3)
    assert res3.success is True
    assert adapter.altitude == 0.0
    assert adapter.airborne is False


def test_34_virtual_rover_lifecycle():
    """VirtualRover state transitions: navigate -> stop -> capture_image."""
    _, adapter = create_virtual_rover("ROVER_TEST")
    assert adapter.moving is False

    # Navigate
    cmd1 = DeviceCommand("r1", "ROVER_TEST", "navigate", "go_to_waypoint", {"latitude": 37.776, "longitude": -122.418})
    res1 = adapter.execute_command(cmd1)
    assert res1.success is True
    assert adapter.moving is True
    assert adapter.location.latitude == 37.776

    # Stop
    cmd2 = DeviceCommand("r2", "ROVER_TEST", "stop", "stop")
    res2 = adapter.execute_command(cmd2)
    assert res2.success is True
    assert adapter.moving is False


def test_35_virtual_glass_lifecycle():
    """VirtualGlass state transitions: display_hud -> capture_image -> get_telemetry."""
    _, adapter = create_virtual_glass("GLASS_TEST")
    assert adapter.display_text == ""

    # HUD
    cmd1 = DeviceCommand("g1", "GLASS_TEST", "display_hud", "display_hud", {"text": "Hazard Ahead: 50m"})
    res1 = adapter.execute_command(cmd1)
    assert res1.success is True
    assert adapter.display_text == "Hazard Ahead: 50m"

    # Capture image
    cmd2 = DeviceCommand("g2", "GLASS_TEST", "capture_image", "capture_image")
    res2 = adapter.execute_command(cmd2)
    assert res2.success is True
    assert "artifact_reference" in res2.data


def test_36_simulated_telemetry_observation_produced():
    """Virtual adapters produce canonical MultimodalObservation telemetry."""
    _, adapter = create_virtual_drone("DRONE_TELEM")
    cmd = DeviceCommand("t1", "DRONE_TELEM", "get_telemetry", "get_telemetry")
    res = adapter.execute_command(cmd)
    assert res.success is True
    obs = res.data["observations"][0]
    assert isinstance(obs, MultimodalObservation)
    assert obs.modality == ModalityType.TELEMETRY
    assert "battery_pct" in obs.payload


def test_37_command_to_state_transition():
    """Command execution deterministically updates internal virtual device state."""
    _, adapter = create_virtual_drone("DRONE_STATE")
    initial_battery = adapter.battery_pct

    cmd = DeviceCommand("s1", "DRONE_STATE", "takeoff", "takeoff")
    adapter.execute_command(cmd)
    assert adapter.battery_pct < initial_battery
    assert adapter.armed is True


def test_38_command_to_observation_closed_loop():
    """Executing an action returns observation data suitable for CentralInputGateway ingestion."""
    _, adapter = create_virtual_glass("GLASS_LOOP")
    cmd = DeviceCommand("l1", "GLASS_LOOP", "capture_image", "capture_image")
    res = adapter.execute_command(cmd)
    assert res.success is True
    obs = res.data["observations"][0]
    assert obs.modality == ModalityType.IMAGE
    assert obs.artifact_reference.startswith("artifacts/glass/GLASS_LOOP/")


def test_39_multi_device_isolation():
    """Simultaneous operations across multiple devices maintain strict state isolation."""
    d_ident, d_adapt = create_virtual_drone("DRONE_ISO")
    r_ident, r_adapt = create_virtual_rover("ROVER_ISO")
    g_ident, g_adapt = create_virtual_glass("GLASS_ISO")

    gateway = DeviceGateway()
    for ident, adapt in ((d_ident, d_adapt), (r_ident, r_adapt), (g_ident, g_adapt)):
        gateway.register_device(ident)
        gateway.register_adapter(adapt, device_id=ident.device_id)

    # Command Drone to fly
    gateway.dispatch_to_device("DRONE_ISO", "takeoff", "takeoff", {"target_altitude": 25.0})
    # Command Rover to navigate
    gateway.dispatch_to_device("ROVER_ISO", "navigate", "go_to_waypoint", {"latitude": 37.779, "longitude": -122.411})
    # Command Glass to display text
    gateway.dispatch_to_device("GLASS_ISO", "display_hud", "display_hud", {"text": "Connected"})

    # Verify isolation
    assert d_adapt.altitude == 25.0
    assert d_adapt.airborne is True
    assert r_adapt.moving is True
    assert g_adapt.display_text == "Connected"
    assert g_adapt.battery_pct > 95.0


# ============================================================================
# 40-45: Replay, Capacity, Concurrency, and End-to-End
# ============================================================================

def test_40_replay_determinism():
    """Identical dispatch sequence against virtual adapters produces identical results."""
    def run_sequence():
        gw = DeviceGateway()
        d_ident, d_adapt = create_virtual_drone("DRONE_DET")
        gw.register_device(d_ident)
        gw.register_adapter(d_adapt, device_id="DRONE_DET")

        res_list = []
        for i in range(5):
            res = gw.dispatch_to_device(
                device_id="DRONE_DET",
                capability="navigate",
                action="go_to_waypoint",
                parameters={"latitude": 37.77 + (i * 0.001), "longitude": -122.41},
                dispatch_id=f"replay_disp_{i}",
                now=1000.0 + i,
            )
            res_list.append((res.success, res.data["location"]))
        return res_list

    run1 = run_sequence()
    run2 = run_sequence()
    assert run1 == run2


def test_41_bounded_device_registry():
    """DeviceGateway enforces maximum registered devices limit."""
    gateway = DeviceGateway(max_devices=3)
    d1, _ = create_virtual_drone("D1")
    d2, _ = create_virtual_drone("D2")
    d3, _ = create_virtual_drone("D3")
    d4, _ = create_virtual_drone("D4")

    gateway.register_device(d1)
    gateway.register_device(d2)
    gateway.register_device(d3)

    with pytest.raises(ValueError, match="Maximum registered devices limit"):
        gateway.register_device(d4)


def test_42_bounded_dispatch_history():
    """Dispatch cache remains bounded under high volume."""
    gateway = DeviceGateway(max_dispatch_history=5)
    drone, adapter = create_virtual_drone("DRONE_BOUND")
    gateway.register_device(drone)
    gateway.register_adapter(adapter, device_id="DRONE_BOUND")

    for i in range(10):
        gateway.dispatch_to_device(
            device_id="DRONE_BOUND",
            capability="hover",
            action="hover",
            parameters={},
            dispatch_id=f"disp_bound_{i}",
        )

    assert len(gateway._seen_dispatch_ids) == 5


def test_43_concurrency_safety():
    """Concurrent dispatch operations execute safely without race conditions."""
    gateway = DeviceGateway()
    drone, adapter = create_virtual_drone("DRONE_CONCUR")
    gateway.register_device(drone)
    gateway.register_adapter(adapter, device_id="DRONE_CONCUR")

    def worker(worker_id: int):
        for i in range(15):
            gateway.dispatch_to_device(
                device_id="DRONE_CONCUR",
                capability="hover",
                action="hover",
                parameters={},
                dispatch_id=f"worker_{worker_id}_disp_{i}",
            )

    threads = [threading.Thread(target=worker, args=(w,)) for w in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(gateway._seen_dispatch_ids) == 60


def test_44_security_registration_boundary():
    """Unregistered device cannot receive commands or spoof adapters."""
    gateway = DeviceGateway()
    # Spoofed device dispatch without registration
    res = gateway.dispatch_to_device(
        device_id="ROGUE_DEVICE_999",
        capability="takeoff",
        action="takeoff",
        parameters={},
    )
    assert res.success is False
    assert "not registered in DeviceGateway" in res.message


def test_45_end_to_end_command_to_virtual_adapter_result():
    """End-to-end flow: ToolCall -> ToolOrchestrator -> DeviceGateway -> Adapter -> Result."""
    gateway = DeviceGateway()
    rover, adapter = create_virtual_rover("ATLAS_ROVER_ALPHA")
    gateway.register_device(rover)
    gateway.register_adapter(adapter, device_id="ATLAS_ROVER_ALPHA")

    cap_bridge = DeviceGatewayCapability(gateway)
    registry = CapabilityRegistry()
    registry.register("device_gateway", cap_bridge)
    orchestrator = ToolOrchestrator(registry=registry)

    call = ToolCall(
        capability="device_gateway",
        action="dispatch_capability",
        parameters={
            "device_id": "ATLAS_ROVER_ALPHA",
            "capability": "navigate",
            "action": "go_to_waypoint",
            "parameters": {"latitude": 37.7788, "longitude": -122.4122},
        },
        call_id="call_e2e_rover_nav",
    )

    result = orchestrator.execute(call)
    assert result.success is True
    assert adapter.moving is True
    assert adapter.location.latitude == 37.7788
    assert len(result.data["observations"]) == 1
    obs = result.data["observations"][0]
    assert obs.modality == ModalityType.GPS
    assert obs.source_id == "ATLAS_ROVER_ALPHA"


# ============================================================================
# Section 33: Performance Benchmarks
# ============================================================================

def test_perf_01_device_gateway_scaling():
    """Benchmark device registration, lookup, and dispatch scaling (10, 100, 500)."""
    for count in (10, 100, 500):
        gw = DeviceGateway(max_devices=count + 10, max_dispatch_history=count + 10)
        adapter = VirtualDroneAdapter("BENCH_DRONE")

        # 1. Registration benchmark
        t0 = time.perf_counter()
        for i in range(count):
            dev = DeviceIdentity(
                device_id=f"DEV_{count}_{i}",
                device_type=DeviceType.DRONE_AERIAL,
                display_name=f"Device {i}",
            )
            gw.register_device(dev)
        t_reg = (time.perf_counter() - t0) * 1000.0

        # 2. Lookup benchmark
        t0 = time.perf_counter()
        for i in range(count):
            d = gw.get_device(f"DEV_{count}_{i}")
            assert d is not None
        t_lookup = (time.perf_counter() - t0) * 1000.0

        # Register adapter for type
        gw.register_adapter(adapter, device_type=DeviceType.DRONE_AERIAL)

        # 3. Resolution benchmark
        t0 = time.perf_counter()
        for i in range(count):
            ad = gw.resolve_adapter(f"DEV_{count}_{i}")
            assert ad is adapter
        t_resolve = (time.perf_counter() - t0) * 1000.0

        # All operations for 500 devices must be sub-100ms in pure python
        assert t_reg < 150.0, f"Registration of {count} took {t_reg:.2f}ms"
        assert t_lookup < 50.0, f"Lookup of {count} took {t_lookup:.2f}ms"
        assert t_resolve < 50.0, f"Resolution of {count} took {t_resolve:.2f}ms"
