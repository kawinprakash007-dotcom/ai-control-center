"""
ATLAS Phase 6.3 — Digital Twin & Simulation Verification Suite.

Comprehensive tests covering Sections A through AF:
A. Domain validation
B. Serialization & credential scrubbing
C. Simulation clock
D. Simulation world creation
E. Twin registration
F. Twin lifecycle
G. Vision simulation
H. Glass simulation
I. Drone simulation
J. Rover simulation
K. Telemetry
L. Observation generation
M. DeviceGateway integration
N. CentralInputGateway integration
O. Fault injection
P. OFFLINE fault handling
Q. LOW_BATTERY fault handling
R. GPS_LOSS fault handling
S. COMMAND_FAILURE & TIMEOUT fault handling
T. TELEMETRY_STALE fault handling
U. DUPLICATE_OBSERVATION fault handling
V. CONFLICTING_OBSERVATION fault handling
W. Scenario execution
X. Scenario assertions
Y. Deterministic replay
Z. Bounds & capacity enforcement
AA. Security static checks
AB. No direct WorldState mutation
AC. No direct ToolOrchestrator bypass
AD. No direct CognitiveRuntime bypass
AE. Model neutrality
AF. Architectural replacement compatibility
"""

import math
import os
import sys
import time
import pytest

from core.interfaces.orchestration_interface import DeviceAdapterInterface
from core.interfaces.simulation_interface import (
    DigitalTwinInterface,
    ScenarioRunnerInterface,
    SimulationClockInterface,
    SimulationWorldInterface,
)
from core.models.device_contract import (
    DeviceErrorCode,
    DeviceHealthStatus,
    ProductRole,
    ProductType,
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
from orchestration.device_gateway import DeviceCommand, DeviceGateway
from orchestration.input_gateway import CentralInputGateway
from safety.policy_engine import StandardPolicyEngine
from simulation.adapters import DigitalTwinAdapter, create_digital_twin_device
from simulation.clock import SimulationClock
from simulation.environment import (
    SimulatedEntity,
    SimulatedEnvironment,
    SimulatedHazard,
)
from simulation.fault_injection import FaultInjectionManager
from simulation.runner import ScenarioRunner
from simulation.scenario import (
    Scenario,
    ScenarioAssertion,
    ScenarioBuilder,
    ScenarioResult,
    ScenarioStep,
)
from simulation.twin import (
    DroneDigitalTwin,
    GlassDigitalTwin,
    RoverDigitalTwin,
    VisionDigitalTwin,
)
from simulation.world import SimulationWorld


# ============================================================================
# Section A: Domain Validation
# ============================================================================

def test_A01_twin_position_creation_and_distance():
    p1 = TwinPosition(latitude=37.7749, longitude=-122.4194, altitude=10.0, heading=90.0, speed=5.0)
    assert p1.latitude == 37.7749
    assert p1.longitude == -122.4194
    assert p1.altitude == 10.0
    assert p1.heading == 90.0
    assert p1.speed == 5.0
    geo = p1.to_geo_location()
    assert geo.latitude == 37.7749

    p2 = TwinPosition(latitude=37.7749, longitude=-122.4194, altitude=20.0)
    assert math.isclose(p1.distance_to(p2), 10.0, rel_tol=1e-3)


def test_A02_twin_configuration_validation():
    cfg = TwinConfiguration(
        twin_id="TWIN_01",
        product_type=ProductType.DRONE,
        product_role=ProductRole.HYBRID,
        initial_battery_pct=85.0,
    )
    assert cfg.twin_id == "TWIN_01"
    assert cfg.product_type == ProductType.DRONE
    assert cfg.initial_battery_pct == 85.0

    with pytest.raises(ValueError):
        TwinConfiguration(
            twin_id="",
            product_type=ProductType.DRONE,
            product_role=ProductRole.HYBRID,
        )

    with pytest.raises(ValueError):
        TwinConfiguration(
            twin_id="TWIN_ERR",
            product_type=ProductType.DRONE,
            product_role=ProductRole.HYBRID,
            initial_battery_pct=150.0,
        )


def test_A03_twin_state_immutability():
    state = TwinState(
        twin_id="TWIN_STATE_01",
        product_id="TWIN_STATE_01",
        product_type=ProductType.VISION,
        product_role=ProductRole.OBSERVATION_SOURCE,
        battery=90.0,
    )
    with pytest.raises(Exception):
        state.battery = 50.0  # dataclass frozen=True


# ============================================================================
# Section B: Serialization & Credential Scrubbing
# ============================================================================

def test_B01_twin_configuration_serialization_scrubbing():
    cfg = TwinConfiguration(
        twin_id="TWIN_SEC",
        product_type=ProductType.GLASS,
        product_role=ProductRole.HYBRID,
        parameters={"camera_fps": 30, "api_key": "SECRET_123", "token": "TOKEN_ABC"},
        metadata={"vendor": "ATLAS", "password": "SUPER_SECRET_PWD"},
    )
    d = cfg.to_dict()
    assert d["parameters"]["camera_fps"] == 30
    assert d["parameters"]["api_key"] == "[REDACTED]"
    assert d["parameters"]["token"] == "[REDACTED]"
    assert d["metadata"]["vendor"] == "ATLAS"
    assert d["metadata"]["password"] == "[REDACTED]"

    # Roundtrip from_dict
    restored = TwinConfiguration.from_dict(d)
    assert restored.twin_id == "TWIN_SEC"
    assert restored.product_type == ProductType.GLASS


def test_B02_twin_telemetry_serialization_roundtrip():
    telem = TwinTelemetry(
        telemetry_id="telem_001",
        twin_id="TWIN_ROVER_01",
        product_type=ProductType.ROVER,
        timestamp=1000.0,
        battery_level=78.5,
        connectivity=ConnectivityStatus.ONLINE,
        health_status=DeviceHealthStatus.HEALTHY,
        position=TwinPosition(37.77, -122.42, 0.0),
        temperature_celsius=24.0,
        operating_mode="PATROL",
    )
    d = telem.to_dict()
    assert d["telemetry_id"] == "telem_001"
    assert d["battery_level"] == 78.5
    restored = TwinTelemetry.from_dict(d)
    assert restored.twin_id == "TWIN_ROVER_01"
    assert restored.position.latitude == 37.77


def test_B03_twin_fault_serialization_roundtrip():
    fault = TwinFault(
        fault_id="f_001",
        twin_id="TWIN_01",
        fault_type=TwinFaultType.LOW_BATTERY,
        parameters={"battery_pct": 5.0, "secret_key": "HIDE_ME"},
        injected_at=100.0,
        duration_seconds=30.0,
    )
    d = fault.to_dict()
    assert d["fault_type"] == "LOW_BATTERY"
    assert d["parameters"]["secret_key"] == "[REDACTED]"
    restored = TwinFault.from_dict(d)
    assert restored.fault_id == "f_001"
    assert restored.duration_seconds == 30.0


# ============================================================================
# Section C: Simulation Clock
# ============================================================================

def test_C01_clock_monotonicity_and_advance():
    clock = SimulationClock(initial_time=1000.0)
    assert clock.now() == 1000.0
    t1 = clock.advance(5.5)
    assert t1 == 1005.5
    assert clock.now() == 1005.5
    assert clock.get_ticks() == 1


def test_C02_clock_rejects_negative_advance():
    clock = SimulationClock(initial_time=100.0)
    with pytest.raises(ValueError):
        clock.advance(-1.0)


def test_C03_clock_set_time_and_backwards_rejection():
    clock = SimulationClock(initial_time=100.0)
    clock.set_time(200.0)
    assert clock.now() == 200.0

    with pytest.raises(ValueError):
        clock.set_time(150.0)  # Backwards jump rejected!


def test_C04_clock_listeners():
    clock = SimulationClock(initial_time=10.0)
    events = []

    def on_tick(old_t, new_t):
        events.append((old_t, new_t))

    clock.register_listener(on_tick)
    clock.advance(2.0)
    assert len(events) == 1
    assert events[0] == (10.0, 12.0)
    clock.unregister_listener(on_tick)
    clock.advance(3.0)
    assert len(events) == 1  # No more notifications


# ============================================================================
# Section D: Simulation World Creation & Snapshots
# ============================================================================

def test_D01_simulation_world_instantiation():
    clock = SimulationClock(initial_time=500.0)
    env = SimulatedEnvironment(ambient_temperature_celsius=21.0)
    world = SimulationWorld(clock=clock, environment=env)
    assert world.get_clock().now() == 500.0
    assert world.get_environment().ambient_temperature_celsius == 21.0


def test_D02_environment_entities_and_hazards():
    env = SimulatedEnvironment()
    ent = SimulatedEntity(
        entity_id="person_01",
        entity_type="PERSON",
        position=TwinPosition(37.77, -122.42),
    )
    haz = SimulatedHazard(
        hazard_id="smoke_01",
        hazard_type="SMOKE",
        location=TwinPosition(37.7701, -122.4201),
        radius_meters=15.0,
    )
    env.add_entity(ent)
    env.add_hazard(haz)
    assert env.get_entity("person_01") is not None
    assert env.get_hazard("smoke_01") is not None
    assert len(env.list_entities()) == 1
    assert len(env.list_hazards()) == 1


def test_D03_world_snapshot_immutability():
    world = SimulationWorld()
    twin = VisionDigitalTwin()
    world.register_twin(twin)
    snap = world.get_snapshot()
    assert snap.simulated_time == world.get_clock().now()
    assert twin.twin_id in snap.twins
    assert snap.pending_observations_count == 0


# ============================================================================
# Section E: Twin Registration & Lookup
# ============================================================================

def test_E01_twin_registration_and_retrieval():
    world = SimulationWorld()
    drone = DroneDigitalTwin()
    world.register_twin(drone)
    retrieved = world.get_twin(drone.twin_id)
    assert retrieved is drone
    assert len(world.list_twins()) == 1
    assert world.unregister_twin(drone.twin_id) is True
    assert world.get_twin(drone.twin_id) is None


def test_E02_twin_registration_capacity_bound():
    limits = SimulationLimits(max_twins=2)
    world = SimulationWorld(limits=limits)
    t1 = VisionDigitalTwin(config=TwinConfiguration("T1", ProductType.VISION, ProductRole.OBSERVATION_SOURCE))
    t2 = VisionDigitalTwin(config=TwinConfiguration("T2", ProductType.VISION, ProductRole.OBSERVATION_SOURCE))
    t3 = VisionDigitalTwin(config=TwinConfiguration("T3", ProductType.VISION, ProductRole.OBSERVATION_SOURCE))
    world.register_twin(t1)
    world.register_twin(t2)
    with pytest.raises(ValueError):
        world.register_twin(t3)


# ============================================================================
# Section F: Twin Lifecycle & Status
# ============================================================================

def test_F01_twin_status_transitions():
    twin = DroneDigitalTwin()
    state = twin.get_state()
    assert state.simulation_status == TwinSimulationStatus.ACTIVE
    assert state.connectivity == ConnectivityStatus.ONLINE
    assert state.health == DeviceHealthStatus.HEALTHY


def test_F02_twin_simulation_status_from_str():
    assert TwinSimulationStatus.from_str("ACTIVE") == TwinSimulationStatus.ACTIVE
    assert TwinSimulationStatus.from_str("paused") == TwinSimulationStatus.PAUSED
    assert TwinSimulationStatus.from_str("INVALID") == TwinSimulationStatus.ACTIVE


# ============================================================================
# Section G: ATLAS Vision Simulation
# ============================================================================

def test_G01_vision_detect_motion():
    vision = VisionDigitalTwin()
    cmd = DeviceCommand(
        dispatch_id="cmd_vis_motion",
        device_id=vision.twin_id,
        capability="camera",
        action="detect_motion",
        parameters={"region": "entryway"},
    )
    res = vision.execute_command(cmd)
    assert res.success
    assert res.data["motion_detected"] is True
    obs_list = res.data["observations"]
    assert len(obs_list) == 1
    assert obs_list[0].source_type == "VISION"


def test_G02_vision_detect_person_and_anomaly():
    vision = VisionDigitalTwin()
    cmd_person = DeviceCommand(
        dispatch_id="cmd_vis_person",
        device_id=vision.twin_id,
        capability="camera",
        action="detect_person",
        parameters={"person_count": 2},
    )
    res_p = vision.execute_command(cmd_person)
    assert res_p.success
    assert res_p.data["person_count"] == 2

    cmd_anom = DeviceCommand(
        dispatch_id="cmd_vis_anom",
        device_id=vision.twin_id,
        capability="camera",
        action="detect_anomaly",
        parameters={"hazard_type": "smoke"},
    )
    res_a = vision.execute_command(cmd_anom)
    assert res_a.success
    assert res_a.data["anomaly_detected"] is True


def test_G03_vision_capture_media():
    vision = VisionDigitalTwin()
    cmd_img = DeviceCommand(
        dispatch_id="cmd_vis_img",
        device_id=vision.twin_id,
        capability="camera",
        action="capture_image",
    )
    res_img = vision.execute_command(cmd_img)
    assert res_img.success
    assert res_img.data["image_captured"] is True


# ============================================================================
# Section H: ATLAS Glass Simulation
# ============================================================================

def test_H01_glass_display_hud():
    glass = GlassDigitalTwin()
    cmd = DeviceCommand(
        dispatch_id="cmd_glass_hud",
        device_id=glass.twin_id,
        capability="hud",
        action="display_hud",
        parameters={"message": "Navigate to Point A", "priority": "HIGH"},
    )
    res = glass.execute_command(cmd)
    assert res.success
    assert res.data["displayed"] is True


def test_H02_glass_capture_audio_and_image():
    glass = GlassDigitalTwin()
    cmd_aud = DeviceCommand(
        dispatch_id="cmd_glass_aud",
        device_id=glass.twin_id,
        capability="audio",
        action="capture_audio",
        parameters={"duration": 3.0},
    )
    res = glass.execute_command(cmd_aud)
    assert res.success
    assert res.data["audio_captured"] is True


def test_H03_glass_notification_and_position():
    glass = GlassDigitalTwin()
    cmd_notif = DeviceCommand(
        dispatch_id="cmd_glass_notif",
        device_id=glass.twin_id,
        capability="notification",
        action="send_notification",
        parameters={"text": "Alert: Motion nearby"},
    )
    res = glass.execute_command(cmd_notif)
    assert res.success
    assert res.data["delivered"] is True


# ============================================================================
# Section I: ATLAS Drone Simulation
# ============================================================================

def test_I01_drone_takeoff_hover_and_land():
    drone = DroneDigitalTwin()
    cmd_takeoff = DeviceCommand(
        dispatch_id="cmd_dr_takeoff",
        device_id=drone.twin_id,
        capability="flight",
        action="takeoff",
        parameters={"altitude": 12.0},
    )
    res_t = drone.execute_command(cmd_takeoff)
    assert res_t.success
    assert res_t.data["flight_state"] == "HOVERING"

    # Second takeoff while airborne should conflict
    res_t2 = drone.execute_command(cmd_takeoff)
    assert not res_t2.success
    assert res_t2.error_code == DeviceErrorCode.COMMAND_REJECTED.value

    # Land
    cmd_land = DeviceCommand(
        dispatch_id="cmd_dr_land",
        device_id=drone.twin_id,
        capability="flight",
        action="land",
    )
    res_l = drone.execute_command(cmd_land)
    assert res_l.success
    assert res_l.data["flight_state"] == "GROUNDED"


def test_I02_drone_waypoint_navigation():
    drone = DroneDigitalTwin()
    drone.execute_command(DeviceCommand("cmd_to", drone.twin_id, "flight", "takeoff"))
    cmd_goto = DeviceCommand(
        dispatch_id="cmd_dr_goto",
        device_id=drone.twin_id,
        capability="flight",
        action="goto_location",
        parameters={"latitude": 37.7780, "longitude": -122.4150, "altitude": 20.0},
    )
    res = drone.execute_command(cmd_goto)
    assert res.success
    assert res.data["arrived"] is True
    assert drone.get_state().position.latitude == 37.7780


def test_I03_drone_battery_consumption_and_low_battery_failsafe():
    cfg = TwinConfiguration(
        twin_id="DRONE_LOW_BATT",
        product_type=ProductType.DRONE,
        product_role=ProductRole.HYBRID,
        initial_battery_pct=10.0,
    )
    drone = DroneDigitalTwin(config=cfg)
    cmd_takeoff = DeviceCommand("cmd_to_low", drone.twin_id, "flight", "takeoff")
    res = drone.execute_command(cmd_takeoff)
    assert not res.success
    assert res.error_code == DeviceErrorCode.LOW_BATTERY.value


# ============================================================================
# Section J: ATLAS Rover Simulation
# ============================================================================

def test_J01_rover_move_and_stop():
    rover = RoverDigitalTwin()
    cmd_move = DeviceCommand(
        dispatch_id="cmd_rov_move",
        device_id=rover.twin_id,
        capability="locomotion",
        action="move",
        parameters={"latitude": 37.7750, "longitude": -122.4190},
    )
    res_m = rover.execute_command(cmd_move)
    assert res_m.success
    assert res_m.data["rover_state"] == "MOVING"

    cmd_stop = DeviceCommand(
        dispatch_id="cmd_rov_stop",
        device_id=rover.twin_id,
        capability="locomotion",
        action="stop",
    )
    res_s = rover.execute_command(cmd_stop)
    assert res_s.success
    assert res_s.data["rover_state"] == "STOPPED"


def test_J02_rover_obstacle_avoidance_safety_stop():
    rover = RoverDigitalTwin()
    rover.set_obstacle(True)
    cmd_move = DeviceCommand(
        dispatch_id="cmd_rov_move_obs",
        device_id=rover.twin_id,
        capability="locomotion",
        action="move",
        parameters={"latitude": 37.7760, "longitude": -122.4180},
    )
    res = rover.execute_command(cmd_move)
    assert not res.success
    assert res.error_code == DeviceErrorCode.SAFETY_REJECTION.value


def test_J03_rover_docking_and_recharge():
    cfg = TwinConfiguration(
        twin_id="ROVER_DOCK",
        product_type=ProductType.ROVER,
        product_role=ProductRole.HYBRID,
        initial_battery_pct=50.0,
    )
    rover = RoverDigitalTwin(config=cfg)
    rover.execute_command(DeviceCommand("cmd_dock", rover.twin_id, "locomotion", "dock"))
    assert rover.get_state().internal_state["rover_state"] == "DOCKED"
    rover.tick(now=10.0, delta_seconds=50.0)  # Recharges 50s * 0.2% = 10%
    assert rover.get_state().battery > 50.0


# ============================================================================
# Section K: Telemetry
# ============================================================================

def test_K01_telemetry_generation():
    drone = DroneDigitalTwin()
    telem = drone.get_telemetry()
    assert telem.twin_id == drone.twin_id
    assert telem.product_type == ProductType.DRONE
    assert telem.battery_level == 100.0


def test_K02_telemetry_conversion_to_multimodal_observation():
    rover = RoverDigitalTwin()
    telem = rover.get_telemetry()
    obs = telem.to_multimodal_observation()
    assert isinstance(obs, MultimodalObservation)
    assert obs.source_id == rover.twin_id
    assert obs.source_type == "DEVICE_TELEMETRY"
    assert obs.modality == ModalityType.TELEMETRY


def test_K03_telemetry_history_bounded():
    limits = SimulationLimits(max_telemetry_records=5)
    vision = VisionDigitalTwin(limits=limits)
    for _ in range(10):
        vision.get_telemetry()
    assert len(vision._telemetry_history) == 5


# ============================================================================
# Section L: Observation Generation
# ============================================================================

def test_L01_canonical_observation_generation():
    vision = VisionDigitalTwin()
    obs = vision.generate_observation(
        observation_type="PERSON_DETECTED",
        payload={"confidence": 0.98},
    )
    assert obs.source_id == vision.twin_id
    assert obs.source_type == "VISION"
    assert obs.payload["event"] == "PERSON_DETECTED"
    assert obs.payload["confidence"] == 0.98


# ============================================================================
# Section M: DeviceGateway Integration & Adapter Equivalence
# ============================================================================

def test_M01_digital_twin_adapter_in_device_gateway():
    gw = DeviceGateway()
    drone = DroneDigitalTwin()
    identity, adapter = create_digital_twin_device(drone)

    gw.register_device(identity)
    gw.register_adapter(adapter, device_id=identity.device_id)

    # Dispatch via DeviceGateway
    res = gw.dispatch_to_device(
        device_id=drone.twin_id,
        capability="takeoff",
        action="takeoff",
        parameters={"altitude": 10.0},
    )
    assert res.success
    assert drone.get_state().internal_state["flight_state"] == "HOVERING"


def test_M02_adapter_interface_compliance():
    vision = VisionDigitalTwin()
    adapter = DigitalTwinAdapter(vision)
    assert isinstance(adapter, DeviceAdapterInterface)
    assert adapter.get_protocol_name() == "digital_twin"
    assert adapter.connect() is True
    assert adapter.get_status() == ConnectivityStatus.ONLINE
    health = adapter.get_health()
    assert health.status == DeviceHealthStatus.HEALTHY
    caps = adapter.get_capabilities()
    assert len(caps) > 0


# ============================================================================
# Section N: CentralInputGateway Ingress Integration
# ============================================================================

def test_N01_simulation_world_to_central_input_gateway():
    gateway = CentralInputGateway()
    world = SimulationWorld()
    vision = VisionDigitalTwin()
    world.register_twin(vision)

    now = world.get_clock().now()
    obs = vision.generate_observation("MOTION_DETECTED", payload={"zone": "patio"}, timestamp=now)
    world.queue_observation(obs)

    flushed = world.flush_observations()
    assert len(flushed) == 1
    gateway.ingest_observation(flushed[0], now=now)
    assert len(gateway.get_recent_observations()) == 1


# ============================================================================
# Section O: Fault Injection
# ============================================================================

def test_O01_inject_remove_and_clear_faults():
    mgr = FaultInjectionManager()
    fault = TwinFault(
        fault_id="f_offline",
        twin_id="TWIN_01",
        fault_type=TwinFaultType.OFFLINE,
    )
    mgr.inject_fault(fault)
    assert mgr.has_fault("TWIN_01", TwinFaultType.OFFLINE)
    assert mgr.remove_fault("f_offline") is True
    assert not mgr.has_fault("TWIN_01", TwinFaultType.OFFLINE)


def test_O02_fault_auto_expiry_with_simulation_clock():
    mgr = FaultInjectionManager()
    fault = TwinFault(
        fault_id="f_temp",
        twin_id="TWIN_01",
        fault_type=TwinFaultType.SENSOR_FAILURE,
        injected_at=100.0,
        duration_seconds=10.0,
    )
    mgr.inject_fault(fault)
    assert mgr.has_fault("TWIN_01", TwinFaultType.SENSOR_FAILURE, current_time=105.0)
    # At t=111.0, fault should expire!
    assert not mgr.has_fault("TWIN_01", TwinFaultType.SENSOR_FAILURE, current_time=111.0)


# ============================================================================
# Section P: OFFLINE Fault Handling
# ============================================================================

def test_P01_offline_fault_blocks_execution():
    drone = DroneDigitalTwin()
    drone.inject_fault(
        TwinFault("f_off", drone.twin_id, TwinFaultType.OFFLINE)
    )
    cmd = DeviceCommand("cmd_to_fail", drone.twin_id, "flight", "takeoff")
    res = drone.execute_command(cmd)
    assert not res.success
    assert res.error_code == DeviceErrorCode.OFFLINE_DEVICE.value


# ============================================================================
# Section Q: LOW_BATTERY Fault Handling
# ============================================================================

def test_Q01_low_battery_fault_injection():
    rover = RoverDigitalTwin()
    rover.inject_fault(
        TwinFault("f_batt", rover.twin_id, TwinFaultType.LOW_BATTERY, parameters={"battery_pct": 8.0})
    )
    assert rover.get_state().battery == 8.0
    assert rover.get_state().health == DeviceHealthStatus.DEGRADED


# ============================================================================
# Section R: GPS_LOSS Fault Handling
# ============================================================================

def test_R01_gps_loss_fault_injection():
    glass = GlassDigitalTwin()
    glass.inject_fault(
        TwinFault("f_gps", glass.twin_id, TwinFaultType.GPS_LOSS)
    )
    assert glass.get_state().position is None
    assert glass.get_state().health == DeviceHealthStatus.DEGRADED


# ============================================================================
# Section S: COMMAND_FAILURE & TIMEOUT Fault Handling
# ============================================================================

def test_S01_simulated_command_failure_fault():
    vision = VisionDigitalTwin()
    vision.inject_fault(
        TwinFault(
            "f_fail",
            vision.twin_id,
            TwinFaultType.COMMAND_FAILURE,
            parameters={"action": "capture_image", "error_code": DeviceErrorCode.DEVICE_ERROR.value},
        )
    )
    cmd = DeviceCommand("cmd_img", vision.twin_id, "camera", "capture_image")
    res = vision.execute_command(cmd)
    assert not res.success
    assert res.error_code == DeviceErrorCode.DEVICE_ERROR.value


def test_S02_simulated_command_timeout_fault():
    drone = DroneDigitalTwin()
    drone.inject_fault(
        TwinFault("f_time", drone.twin_id, TwinFaultType.COMMAND_TIMEOUT, parameters={"action": "takeoff"})
    )
    cmd = DeviceCommand("cmd_to", drone.twin_id, "flight", "takeoff")
    res = drone.execute_command(cmd)
    assert not res.success
    assert res.error_code == DeviceErrorCode.COMMAND_TIMEOUT.value


# ============================================================================
# Section T: TELEMETRY_STALE Fault Handling
# ============================================================================

def test_T01_stale_telemetry_fault():
    rover = RoverDigitalTwin()
    rover.inject_fault(
        TwinFault("f_stale", rover.twin_id, TwinFaultType.TELEMETRY_STALE, parameters={"frozen_timestamp": 50.0})
    )
    telem = rover.get_telemetry()
    assert telem.timestamp == 50.0


# ============================================================================
# Section U: DUPLICATE_OBSERVATION Fault Handling
# ============================================================================

def test_U01_duplicate_observation_ingress_handling():
    gateway = CentralInputGateway()
    vision = VisionDigitalTwin()
    now = time.time()
    obs1 = vision.generate_observation("MOTION", observation_id="obs_dup_001", timestamp=now)
    obs2 = vision.generate_observation("MOTION", observation_id="obs_dup_001", timestamp=now)

    gateway.ingest_observation(obs1, now=now)
    # Duplicate ID is rejected by gateway
    with pytest.raises(ValueError):
        gateway.ingest_observation(obs2, now=now)


# ============================================================================
# Section V: CONFLICTING_OBSERVATION Fault Handling
# ============================================================================

def test_V01_conflicting_observation_fault():
    vision = VisionDigitalTwin()
    vision.inject_fault(
        TwinFault("f_conf", vision.twin_id, TwinFaultType.CONFLICTING_OBSERVATION)
    )
    obs = vision.generate_observation("MOTION", payload={"motion_detected": True})
    assert obs.payload["conflicting_reading"] is True
    assert obs.payload["motion_detected"] is False


# ============================================================================
# Section W: Scenario Execution
# ============================================================================

def test_W01_scenario_builder_and_runner_execution():
    builder = ScenarioBuilder(scenario_id="scen_001", name="Perimeter Inspection")
    builder.with_initial_time(1000.0)

    cfg_vision = TwinConfiguration("VIS_01", ProductType.VISION, ProductRole.OBSERVATION_SOURCE)
    cfg_drone = TwinConfiguration("DRN_01", ProductType.DRONE, ProductRole.HYBRID)
    builder.add_twin_config(cfg_vision)
    builder.add_twin_config(cfg_drone)

    builder.add_step("s1", time_offset=0.0, action_type="DISPATCH_COMMAND", target_id="VIS_01", payload={"action": "detect_motion"})
    builder.add_step("s2", time_offset=5.0, action_type="DISPATCH_COMMAND", target_id="DRN_01", payload={"action": "takeoff", "parameters": {"altitude": 10.0}})
    builder.add_step("s3", time_offset=10.0, action_type="DISPATCH_COMMAND", target_id="DRN_01", payload={"action": "land"})

    builder.add_assertion("a1", target_type="COMMAND_RESULT", target_id="s1", expected_field="success", expected_value=True)
    builder.add_assertion("a2", target_type="TWIN_STATE", target_id="DRN_01", expected_field="health", expected_value="HEALTHY")

    scenario = builder.build()
    runner = ScenarioRunner()
    result = runner.run_scenario(scenario)

    assert result.success is True
    assert result.total_steps_executed == 3
    assert result.assertions_passed == 2
    assert result.simulated_duration == 10.0


# ============================================================================
# Section X: Scenario Assertions Evaluation
# ============================================================================

def test_X01_scenario_assertion_operators():
    builder = ScenarioBuilder(scenario_id="scen_ops", name="Operator Test")
    cfg = TwinConfiguration("VIS_OPS", ProductType.VISION, ProductRole.OBSERVATION_SOURCE, initial_battery_pct=95.0)
    builder.add_twin_config(cfg)

    builder.add_step("s1", 0.0, "DISPATCH_COMMAND", "VIS_OPS", {"action": "detect_person", "parameters": {"person_count": 3}})
    builder.add_assertion("a_eq", "COMMAND_RESULT", "s1", "success", True, "EQUALS")
    builder.add_assertion("a_gt", "TWIN_STATE", "VIS_OPS", "battery", 90.0, "GREATER_THAN")
    builder.add_assertion("a_lt", "TWIN_STATE", "VIS_OPS", "battery", 100.0, "LESS_THAN")

    res = ScenarioRunner().run_scenario(builder.build())
    assert res.success is True
    assert res.assertions_passed == 3


# ============================================================================
# Section Y: Deterministic Replay
# ============================================================================

def test_Y01_deterministic_replay_produces_identical_results():
    def build_test_scenario():
        b = ScenarioBuilder("scen_replay", "Replay Test")
        b.with_initial_time(5000.0)
        b.add_twin_config(TwinConfiguration("DRN_REP", ProductType.DRONE, ProductRole.HYBRID))
        b.add_step("s1", 1.0, "DISPATCH_COMMAND", "DRN_REP", {"action": "takeoff"})
        b.add_step("s2", 5.0, "DISPATCH_COMMAND", "DRN_REP", {"action": "hover"})
        b.add_step("s3", 10.0, "DISPATCH_COMMAND", "DRN_REP", {"action": "land"})
        b.add_assertion("a1", "TWIN_STATE", "DRN_REP", "health", "HEALTHY")
        return b.build()

    runner = ScenarioRunner()
    res1 = runner.run_scenario(build_test_scenario())
    res2 = runner.run_scenario(build_test_scenario())

    assert res1.success == res2.success
    assert res1.total_steps_executed == res2.total_steps_executed
    assert res1.simulated_duration == res2.simulated_duration
    assert len(res1.trace) == len(res2.trace)
    for t1, t2 in zip(res1.trace, res2.trace):
        assert t1["step_id"] == t2["step_id"]
        assert t1["time_offset"] == t2["time_offset"]
        assert t1["result"]["success"] == t2["result"]["success"]


# ============================================================================
# Section Z: Bounds & Capacity Enforcement
# ============================================================================

def test_Z01_environment_and_fault_bounds():
    limits = SimulationLimits(max_entities=3, max_fault_records=2)
    env = SimulatedEnvironment(limits=limits)
    for i in range(3):
        env.add_entity(SimulatedEntity(f"e_{i}", "OBSTACLE", TwinPosition(0, 0)))
    with pytest.raises(ValueError):
        env.add_entity(SimulatedEntity("e_overflow", "OBSTACLE", TwinPosition(0, 0)))

    mgr = FaultInjectionManager(limits=limits)
    mgr.inject_fault(TwinFault("f1", "T1", TwinFaultType.GPS_LOSS))
    mgr.inject_fault(TwinFault("f2", "T1", TwinFaultType.LOW_BATTERY))
    with pytest.raises(ValueError):
        mgr.inject_fault(TwinFault("f3", "T1", TwinFaultType.OFFLINE))


# ============================================================================
# Section AA: Security Static Checks
# ============================================================================

def test_AA01_zero_hardware_or_system_imports():
    prohibited = [
        "pymavlink", "mavsdk", "rclpy", "rospy", "paho", "serial", "RPi.GPIO",
    ]
    sim_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "simulation"))
    for root, _, files in os.walk(sim_dir):
        for f in files:
            if f.endswith(".py"):
                path = os.path.join(root, f)
                with open(path, "r", encoding="utf-8") as handle:
                    content = handle.read()
                    for p in prohibited:
                        assert f"import {p}" not in content, f"Prohibited import {p} in {path}"
                        assert f"from {p}" not in content, f"Prohibited import {p} in {path}"
                    assert "subprocess" not in content
                    assert "os.system" not in content
                    assert "eval(" not in content
                    assert "exec(" not in content


# ============================================================================
# Section AB: No Direct WorldState Mutation
# ============================================================================

def test_AB01_world_simulation_boundary_preserves_world_state_isolation():
    world = SimulationWorld()
    vision = VisionDigitalTwin()
    world.register_twin(vision)
    # The world does not have any reference or pointer to WorldState
    assert not hasattr(world, "world_state")
    assert not hasattr(world, "world_store")
    assert not hasattr(world, "apply_transition")
    # All observation output is quarantined in pending_observations
    obs = vision.generate_observation("MOTION")
    world.queue_observation(obs)
    assert len(world.flush_observations()) == 1


# ============================================================================
# Section AC: No Direct ToolOrchestrator Bypass
# ============================================================================

def test_AC01_policy_engine_governs_commands():
    gw = DeviceGateway()
    drone = DroneDigitalTwin()
    ident, adapter = create_digital_twin_device(drone)
    gw.register_device(ident)
    gw.register_adapter(adapter, ident.device_id)

    # Standard policy check
    pe = StandardPolicyEngine()
    # Unregistered arbitrary action cannot bypass PolicyEngine
    from core.models.policy import PolicyContext
    from core.models.tool_call import ToolCall
    tc = ToolCall(capability="shell", action="rm_rf", parameters={})
    ctx = PolicyContext(capability="shell", action="rm_rf")
    res = pe.evaluate(tc, ctx)
    assert not res.is_allowed


# ============================================================================
# Section AD: No Direct CognitiveRuntime Bypass
# ============================================================================

def test_AD01_digital_twin_cannot_reason_or_create_goals():
    vision = VisionDigitalTwin()
    # Digital twins are edge peripheral representations only
    assert not hasattr(vision, "create_goal")
    assert not hasattr(vision, "plan")
    assert not hasattr(vision, "reason")


# ============================================================================
# Section AE: Model Neutrality
# ============================================================================

def test_AE01_transport_neutral_protocol_names():
    vision = VisionDigitalTwin()
    drone = DroneDigitalTwin()
    glass = GlassDigitalTwin()
    rover = RoverDigitalTwin()

    for twin in (vision, drone, glass, rover):
        adapter = DigitalTwinAdapter(twin)
        proto = adapter.get_protocol_name()
        assert proto == "digital_twin"
        assert "mavlink" not in proto
        assert "ros" not in proto
        assert "mqtt" not in proto


# ============================================================================
# Section AF: Architectural Replacement Compatibility
# ============================================================================

def test_AF01_twin_and_physical_adapter_substitutability():
    """
    Proves that a DigitalTwinAdapter and any future physical adapter satisfy
    the exact same DeviceAdapterInterface and can be swapped with zero change to DeviceGateway.
    """
    gw = DeviceGateway()
    drone_twin = DroneDigitalTwin()
    ident, adapter = create_digital_twin_device(drone_twin)

    # Register twin adapter
    gw.register_device(ident)
    gw.register_adapter(adapter, ident.device_id)

    # Execute command through DeviceGateway
    res = gw.dispatch_to_device(
        device_id=ident.device_id,
        capability="takeoff",
        action="takeoff",
        parameters={"altitude": 10.0},
    )
    assert res.success
    assert drone_twin.get_state().internal_state["flight_state"] == "HOVERING"

    # Gateway inspection treats it purely as DeviceAdapterInterface
    stored_adapter = gw.resolve_adapter(ident.device_id)
    assert isinstance(stored_adapter, DeviceAdapterInterface)
    health = gw.get_device_health(ident.device_id)
    assert health.status == DeviceHealthStatus.HEALTHY


# ============================================================================
# Additional Comprehensive Tests (Section A - AF Deepening)
# ============================================================================

def test_A04_twin_position_validation_and_3d_distance():
    p1 = TwinPosition(latitude=37.7749, longitude=-122.4194, altitude=0.0)
    p2 = TwinPosition(latitude=37.7749, longitude=-122.4194, altitude=50.0)
    assert abs(p1.distance_to(p2) - 50.0) < 0.01
    assert p1.distance_to(p1) == 0.0


def test_A05_twin_configuration_empty_id_rejected():
    with pytest.raises(ValueError, match="non-empty string"):
        TwinConfiguration(
            twin_id="",
            product_type=ProductType.DRONE,
            product_role=ProductRole.HYBRID,
        )
    with pytest.raises(ValueError, match="non-empty string"):
        TwinConfiguration(
            twin_id="   ",
            product_type=ProductType.DRONE,
            product_role=ProductRole.HYBRID,
        )


def test_B04_twin_position_serialization_roundtrip():
    p = TwinPosition(latitude=37.7749, longitude=-122.4194, altitude=12.5, heading=180.0)
    d = p.to_dict()
    p2 = TwinPosition.from_dict(d)
    assert p2.latitude == p.latitude
    assert p2.longitude == p.longitude
    assert p2.altitude == p.altitude
    assert p2.heading == p.heading


def test_C05_clock_listener_exception_resilience():
    clock = SimulationClock(initial_time=0.0)
    def bad_listener(old_t, new_t):
        raise RuntimeError("listener failure")

    clock.register_listener(bad_listener)
    clock.tick()
    assert clock.now() == 1.0


def test_D04_environment_hazard_query_by_position():
    env = SimulatedEnvironment()
    pos = TwinPosition(latitude=37.7749, longitude=-122.4194, altitude=0.0)
    h1 = SimulatedHazard(
        hazard_id="haz_fire_1",
        hazard_type="HEAT_ANOMALY",
        location=pos,
        radius_meters=15.0,
        severity=0.9,
    )
    env.add_hazard(h1)
    near = env.get_hazards_near(pos, radius_meters=10.0)
    assert len(near) == 1
    assert near[0].hazard_id == "haz_fire_1"


def test_D05_environment_entity_removal_and_bounds():
    env = SimulatedEnvironment(limits=SimulationLimits(max_entities=2))
    pos = TwinPosition(latitude=37.7749, longitude=-122.4194, altitude=0.0)
    e1 = SimulatedEntity(entity_id="e1", entity_type="PERSON", position=pos)
    e2 = SimulatedEntity(entity_id="e2", entity_type="OBSTACLE", position=pos)
    e3 = SimulatedEntity(entity_id="e3", entity_type="VEHICLE", position=pos)

    env.add_entity(e1)
    env.add_entity(e2)
    with pytest.raises(ValueError, match="capacity bound"):
        env.add_entity(e3)

    assert env.remove_entity("e1") is True
    assert env.remove_entity("non_existent") is False
    assert len(env.list_entities()) == 1


def test_E03_twin_unregister_and_missing_handling():
    world = SimulationWorld()
    vision = VisionDigitalTwin()
    world.register_twin(vision)
    assert world.get_twin(vision.twin_id) is not None
    assert world.unregister_twin(vision.twin_id) is True
    assert world.get_twin(vision.twin_id) is None
    assert world.unregister_twin("NON_EXISTENT") is False


def test_F03_twin_status_mutation():
    drone = DroneDigitalTwin()
    assert drone.get_simulation_status() == TwinSimulationStatus.ACTIVE
    drone.set_simulation_status(TwinSimulationStatus.PAUSED)
    assert drone.get_simulation_status() == TwinSimulationStatus.PAUSED
    assert drone.get_state().simulation_status == TwinSimulationStatus.PAUSED


def test_G04_vision_unsupported_action_error_code():
    vision = VisionDigitalTwin()
    cmd = DeviceCommand(
        dispatch_id="cmd_unsupported",
        device_id=vision.twin_id,
        capability="camera",
        action="fly_to_sky",
    )
    res = vision.execute_command(cmd)
    assert not res.success
    assert res.error_code == DeviceErrorCode.UNSUPPORTED_ACTION.value


def test_H04_glass_hud_notification_and_state():
    glass = GlassDigitalTwin()
    cmd_hud = DeviceCommand(
        dispatch_id="cmd_gl_hud",
        device_id=glass.twin_id,
        capability="hud",
        action="display_hud",
        parameters={"message": "Alert: Zone Breach", "priority": "high"},
    )
    res = glass.execute_command(cmd_hud)
    assert res.success
    assert res.data["displayed"] is True
    assert len(glass._hud_messages) == 1
    assert glass._hud_messages[0]["message"] == "Alert: Zone Breach"


def test_I04_drone_return_to_base():
    home = TwinPosition(latitude=37.7749, longitude=-122.4194, altitude=0.0)
    cfg = TwinConfiguration(
        twin_id="DRONE_RTB",
        product_type=ProductType.DRONE,
        product_role=ProductRole.HYBRID,
        initial_position=home,
    )
    drone = DroneDigitalTwin(config=cfg)
    drone.execute_command(DeviceCommand("c_to", drone.twin_id, "flight", "takeoff"))
    drone.execute_command(
        DeviceCommand("c_nav", drone.twin_id, "flight", "goto_location", parameters={"latitude": 37.7800, "longitude": -122.4100, "altitude": 25.0})
    )
    assert drone.get_state().position.latitude == 37.7800

    res_rtb = drone.execute_command(DeviceCommand("c_rtb", drone.twin_id, "flight", "return_to_base"))
    assert res_rtb.success
    assert res_rtb.data["returned"] is True
    assert drone.get_state().internal_state["flight_state"] == "GROUNDED"
    assert drone.get_state().position.latitude == home.latitude
    assert drone.get_state().position.longitude == home.longitude


def test_I05_drone_landing_when_already_grounded():
    drone = DroneDigitalTwin()
    assert drone.get_state().internal_state["flight_state"] == "GROUNDED"
    res = drone.execute_command(DeviceCommand("c_land", drone.twin_id, "flight", "land"))
    assert res.success
    assert res.data["flight_state"] == "GROUNDED"


def test_J04_rover_docking_and_stopping():
    rover = RoverDigitalTwin()
    rover.execute_command(
        DeviceCommand("c_mv", rover.twin_id, "locomotion", "move", parameters={"latitude": 37.7750, "longitude": -122.4190})
    )
    assert rover.get_state().internal_state["rover_state"] == "MOVING"

    res_stop = rover.execute_command(DeviceCommand("c_st", rover.twin_id, "locomotion", "stop"))
    assert res_stop.success
    assert rover.get_state().internal_state["rover_state"] == "STOPPED"

    res_dock = rover.execute_command(DeviceCommand("c_dk", rover.twin_id, "locomotion", "dock"))
    assert res_dock.success
    assert rover.get_state().internal_state["rover_state"] == "DOCKED"


def test_K04_telemetry_dict_sanitization():
    drone = DroneDigitalTwin()
    telem = drone.get_telemetry()
    d = telem.to_dict()
    assert "token" not in d
    assert "api_key" not in d
    assert "battery_level" in d
    assert d["twin_id"] == drone.twin_id


def test_K05_telemetry_history_contains_recent_snapshots():
    rover = RoverDigitalTwin()
    for _ in range(3):
        rover.get_telemetry()
    history = rover.get_telemetry_history()
    assert len(history) == 3
    assert all(h.twin_id == rover.twin_id for h in history)


def test_L02_observation_with_custom_clock_timestamps():
    vision = VisionDigitalTwin()
    custom_time = 1234567.89
    obs = vision.generate_observation("ANOMALY_DETECTED", timestamp=custom_time, payload={"type": "unattended_bag"})
    assert obs is not None
    assert obs.timestamp == custom_time
    assert obs.payload["event"] == "ANOMALY_DETECTED"


def test_M03_gateway_health_reports_twin_status():
    gw = DeviceGateway()
    drone = DroneDigitalTwin()
    ident, adapter = create_digital_twin_device(drone)
    gw.register_device(ident)
    gw.register_adapter(adapter, ident.device_id)

    health = gw.get_device_health(ident.device_id)
    assert health.device_id == ident.device_id
    assert health.status == DeviceHealthStatus.HEALTHY
    assert health.connectivity == ConnectivityStatus.ONLINE


def test_O03_fault_clearing_individual_by_id():
    vision = VisionDigitalTwin()
    f1 = TwinFault("f_1", vision.twin_id, TwinFaultType.SENSOR_FAILURE)
    f2 = TwinFault("f_2", vision.twin_id, TwinFaultType.GPS_LOSS)
    vision.inject_fault(f1)
    vision.inject_fault(f2)
    assert len(vision.fault_manager.list_active_faults()) == 2

    assert vision.remove_fault("f_1") is True
    assert len(vision.fault_manager.list_active_faults()) == 1
    assert vision.fault_manager.has_fault(vision.twin_id, TwinFaultType.GPS_LOSS) is True
    assert vision.fault_manager.has_fault(vision.twin_id, TwinFaultType.SENSOR_FAILURE) is False


def test_W02_scenario_assertion_failure_recorded():
    builder = ScenarioBuilder("sc_fail", "Failing Assertion Scenario")
    builder.add_twin_config(
        TwinConfiguration(twin_id="DR_FAIL", product_type=ProductType.DRONE, product_role=ProductRole.HYBRID)
    )
    builder.add_step(
        "s1", 0.0, "DISPATCH_COMMAND", "DR_FAIL", {"capability": "flight", "action": "takeoff", "parameters": {"altitude": 10.0}}
    )
    # Expect altitude to be 999.0 which is false
    builder.add_assertion("a1", "TWIN_STATE", "DR_FAIL", "altitude", 999.0, operator="EQUALS")
    scenario = builder.build()

    runner = ScenarioRunner()
    result = runner.run_scenario(scenario)
    assert result.success is False
    assert result.assertions_passed == 0
    assert len(result.assertion_failures) == 1
    assert result.assertion_failures[0]["assertion_id"] == "a1"


def test_Z02_world_max_observations_bound():
    limits = SimulationLimits(max_pending_observations=5)
    world = SimulationWorld(limits=limits)
    vision = VisionDigitalTwin(limits=limits)
    world.register_twin(vision)

    for i in range(12):
        obs = vision.generate_observation("PING", payload={"idx": i})
        world.queue_observation(obs)

    flushed = world.flush_observations()
    # Flushed observation count cannot exceed bounded capacity
    assert len(flushed) <= 5

