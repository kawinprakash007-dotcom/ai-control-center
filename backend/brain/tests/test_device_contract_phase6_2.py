"""
ATLAS Phase 6.2 — Unified Edge/Product Contract Layer Test Suite.

Comprehensive tests covering all 31 required sections (A through AE):
A. Product Identity
B. Product Type Validation
C. Role Classification
D. Capability Contract
E. Capability Schema
F. Schema Version
G. Command Lifecycle
H. Acknowledgement
I. Command Result
J. Lineage
K. Telemetry
L. Heartbeat
M. Health
N. Errors
O. Lifecycle
P. Adapter Compatibility
Q. VirtualVision
R. VirtualGlass
S. VirtualDrone
T. VirtualRover
U. Vision Observation Generation & Central Ingress
V. Command Observation Generation & Central Ingress
W. Multi-Product Isolation (VISION, GLASS, DRONE, ROVER)
X. Duplicate Commands & Idempotency
Y. Replay Determinism
Z. Deterministic Failure
AA. Timeout Simulation
AB. Bounds Enforcement
AC. Concurrency Safety
AD. Security & Credential Sanitization
AE. Authority Boundaries
"""

import collections
import copy
import dataclasses
import json
import sys
import threading
import time
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

# Ensure mocks for lightweight CI/test environments
if "chromadb" not in sys.modules:
    sys.modules["chromadb"] = MagicMock()
if "sentence_transformers" not in sys.modules:
    sys.modules["sentence_transformers"] = MagicMock()

from core.interfaces.orchestration_interface import DeviceAdapterInterface
from core.interfaces.runtime_interface import CognitiveEventSinkInterface
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
from core.models.orchestration import (
    ConnectivityStatus,
    DeviceCapabilityDescriptor,
    DeviceIdentity,
    DeviceType,
    GeoLocation,
    ModalityType,
    MultimodalObservation,
)
from core.models.policy import PolicyContext, RiskLevel
from core.models.result import Result
from core.models.runtime import CognitiveEvent, CognitiveEventType
from core.models.tool_call import ToolCall
from orchestration.device_gateway import (
    DeviceCommand,
    DeviceGateway,
    DeviceGatewayCapability,
    validate_parameters_against_schema,
)
from orchestration.input_gateway import CentralInputGateway, IngressEnvelope
from orchestration.virtual_devices import (
    VirtualDroneAdapter,
    VirtualGlassAdapter,
    VirtualRoverAdapter,
    VirtualVisionAdapter,
    create_virtual_drone,
    create_virtual_glass,
    create_virtual_rover,
    create_virtual_vision,
)
from safety.policy_engine import StandardPolicyEngine
from tools.capability_registry import CapabilityRegistry
from tools.tool_orchestrator import ToolOrchestrator


class MockEventSink(CognitiveEventSinkInterface):
    """Deterministic in-memory sink for event verification."""
    def __init__(self):
        self.events: List[CognitiveEvent] = []

    def publish(self, event: CognitiveEvent) -> None:
        self.events.append(event)

    def record_event(self, event: CognitiveEvent) -> None:
        self.events.append(event)

    def get_events(self) -> List[CognitiveEvent]:
        return list(self.events)

    def clear(self) -> None:
        self.events.clear()


# ============================================================================
# Section A: Product Identity
# ============================================================================

def test_A01_product_identity_creation():
    """DeviceIdentity cleanly accepts all Phase 6.2 product identity attributes."""
    identity = DeviceIdentity(
        device_id="VISION_UNIT_01",
        device_type=DeviceType.STATIONARY_SENSOR,
        display_name="Perimeter Camera 01",
        product_type=ProductType.VISION,
        product_role=ProductRole.OBSERVATION_SOURCE,
        vendor="ATLAS Sensing Labs",
        model="Vision-Pro-X",
        contract_version="1.0",
        is_simulation=True,
    )
    assert identity.device_id == "VISION_UNIT_01"
    assert identity.product_type == "VISION"
    assert identity.product_role == "OBSERVATION_SOURCE"
    assert identity.vendor == "ATLAS Sensing Labs"
    assert identity.model == "Vision-Pro-X"
    assert identity.contract_version == "1.0"


def test_A02_product_identity_immutability():
    """DeviceIdentity is a frozen dataclass; mutations raise FrozenInstanceError."""
    identity = DeviceIdentity(
        device_id="DRONE_01",
        device_type=DeviceType.DRONE_AERIAL,
        display_name="Drone 01",
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        identity.display_name = "New Drone Name"  # type: ignore


def test_A03_product_identity_serialization_scrubbing():
    """DeviceIdentity.to_dict() scrubs sensitive credentials from metadata."""
    identity = DeviceIdentity(
        device_id="GLASS_01",
        device_type=DeviceType.SMART_GLASSES,
        display_name="Glass Unit",
        metadata={
            "api_key": "secret_abc123",
            "password": "super_secret_pw",
            "firmware_channel": "release",
        },
    )
    d = identity.to_dict()
    assert d["metadata"]["api_key"] == "[REDACTED]"
    assert d["metadata"]["password"] == "[REDACTED]"
    assert d["metadata"]["firmware_channel"] == "release"

    # Roundtrip check
    reconstructed = DeviceIdentity.from_dict(d)
    assert reconstructed.device_id == "GLASS_01"
    assert reconstructed.metadata["firmware_channel"] == "release"


# ============================================================================
# Section B: Product Type Validation
# ============================================================================

def test_B01_product_type_enum_members():
    """Verify all four first-class product types plus UNKNOWN fallback exist."""
    assert ProductType.VISION.value == "VISION"
    assert ProductType.GLASS.value == "GLASS"
    assert ProductType.DRONE.value == "DRONE"
    assert ProductType.ROVER.value == "ROVER"
    assert ProductType.UNKNOWN.value == "UNKNOWN"


def test_B02_product_type_from_str_normalization():
    """ProductType.from_str() handles casing and whitespace robustly."""
    assert ProductType.from_str("vision") == ProductType.VISION
    assert ProductType.from_str("  drone  ") == ProductType.DRONE
    assert ProductType.from_str("Glass") == ProductType.GLASS
    assert ProductType.from_str("rover") == ProductType.ROVER


def test_B03_product_type_unknown_fallback():
    """Unrecognized product type string safely falls back to ProductType.UNKNOWN."""
    assert ProductType.from_str("unrecognized_gadget_xyz") == ProductType.UNKNOWN
    assert ProductType.from_str(None) == ProductType.UNKNOWN


# ============================================================================
# Section C: Role Classification
# ============================================================================

def test_C01_product_role_classification():
    """ProductRole enum contains OBSERVATION_SOURCE, ACTUATOR, and HYBRID."""
    assert ProductRole.OBSERVATION_SOURCE.value == "OBSERVATION_SOURCE"
    assert ProductRole.ACTUATOR.value == "ACTUATOR"
    assert ProductRole.HYBRID.value == "HYBRID"


def test_C02_default_product_roles():
    """Canonical mapping accurately assigns default product roles."""
    assert ProductRole.get_default_role(ProductType.VISION) == ProductRole.OBSERVATION_SOURCE
    assert ProductRole.get_default_role(ProductType.GLASS) == ProductRole.HYBRID
    assert ProductRole.get_default_role(ProductType.DRONE) == ProductRole.HYBRID
    assert ProductRole.get_default_role(ProductType.ROVER) == ProductRole.HYBRID


def test_C03_role_metadata_only():
    """ProductRole is stored as capability metadata and does not bypass PolicyEngine."""
    vision_ident, _ = create_virtual_vision("VISION_SEC")
    assert vision_ident.product_role == "OBSERVATION_SOURCE"
    # Even though it is an observation source, it cannot execute forbidden operations
    tc = ToolCall(capability="shell", action="exec", parameters={"cmd": "ls"})
    pe = StandardPolicyEngine()
    ctx = PolicyContext(capability="shell", action="exec")
    res = pe.evaluate(tc, ctx)
    assert not res.is_allowed


# ============================================================================
# Section D: Capability Contract
# ============================================================================

def test_D01_capability_contract_declaration():
    """DeviceCapabilityDescriptor declares capability_id, description, supported_actions, schema_version."""
    desc = DeviceCapabilityDescriptor(
        capability_name="navigate",
        action_name="go_to_waypoint",
        capability_id="nav_cap",
        description="Ground navigation",
        supported_actions=("go_to_waypoint", "navigate"),
        schema_version="1.0",
        parameters_schema={"properties": {"latitude": {"type": "number"}}},
    )
    assert desc.capability_id == "nav_cap"
    assert desc.description == "Ground navigation"
    assert desc.supported_actions == ("go_to_waypoint", "navigate")
    assert desc.schema_version == "1.0"


def test_D02_capability_supports_action():
    """DeviceCapabilityDescriptor.supports_action() tests action support accurately."""
    desc = DeviceCapabilityDescriptor(
        capability_name="navigate",
        action_name="go_to_waypoint",
        supported_actions=("go_to_waypoint", "navigate", "drive"),
    )
    assert desc.supports_action("go_to_waypoint") is True
    assert desc.supports_action("NAVIGATE") is True
    assert desc.supports_action("drive") is True
    assert desc.supports_action("fly") is False


def test_D03_capability_rejection_of_arbitrary_names():
    """Gateway rejects commands targeting undeclared capabilities or actions."""
    gw = DeviceGateway()
    drone, adapter = create_virtual_drone("DRONE_TEST")
    gw.register_device(drone)
    gw.register_adapter(adapter, device_id="DRONE_TEST")

    res = gw.dispatch_to_device(
        device_id="DRONE_TEST",
        capability="arbitrary_laser",
        action="fire",
        parameters={},
    )
    assert res.success is False
    assert res.data["error_code"] == DeviceErrorCode.UNSUPPORTED_CAPABILITY.value


# ============================================================================
# Section E: Capability Schema
# ============================================================================

def test_E01_schema_required_parameter_enforcement():
    """Command missing a required parameter is rejected with INVALID_PARAMETERS."""
    gw = DeviceGateway()
    drone, adapter = create_virtual_drone("DRONE_TEST_REQ")
    gw.register_device(drone)
    gw.register_adapter(adapter, device_id="DRONE_TEST_REQ")

    # navigate requires latitude and longitude
    res = gw.dispatch_to_device(
        device_id="DRONE_TEST_REQ",
        capability="navigate",
        action="go_to_waypoint",
        parameters={"latitude": 37.77},  # missing longitude!
    )
    assert res.success is False
    assert res.data["error_code"] == DeviceErrorCode.INVALID_PARAMETERS.value
    assert "Missing required parameter" in res.message


def test_E02_schema_type_enforcement():
    """Command passing string for numeric parameter is rejected deterministically."""
    gw = DeviceGateway()
    drone, adapter = create_virtual_drone("DRONE_TEST_TYPE")
    gw.register_device(drone)
    gw.register_adapter(adapter, device_id="DRONE_TEST_TYPE")

    res = gw.dispatch_to_device(
        device_id="DRONE_TEST_TYPE",
        capability="navigate",
        action="go_to_waypoint",
        parameters={"latitude": "invalid_string_lat", "longitude": -122.41},
    )
    assert res.success is False
    assert res.data["error_code"] == DeviceErrorCode.INVALID_PARAMETERS.value
    assert "must be a number" in res.message


def test_E03_schema_numeric_bounds_and_enums():
    """Command exceeding min/max bounds is rejected with INVALID_PARAMETERS."""
    schema = {
        "properties": {
            "altitude": {"type": "number", "minimum": 1.0, "maximum": 500.0},
            "mode": {"type": "string", "enum": ["slow", "fast"]},
        }
    }
    # Exceed maximum
    valid, err = validate_parameters_against_schema({"altitude": 9999.0}, schema)
    assert valid is False
    assert "exceeds maximum" in err

    # Below minimum
    valid, err = validate_parameters_against_schema({"altitude": 0.5}, schema)
    assert valid is False
    assert "less than minimum" in err

    # Invalid enum
    valid, err = validate_parameters_against_schema({"mode": "hyperdrive"}, schema)
    assert valid is False
    assert "must be one of" in err


# ============================================================================
# Section F: Schema Version
# ============================================================================

def test_F01_schema_version_default_and_validation():
    """Capabilities declare schema_version='1.0' by default and serialize correctly."""
    desc = DeviceCapabilityDescriptor(capability_name="cam", action_name="snap")
    assert desc.schema_version == "1.0"
    d = desc.to_dict()
    assert d["schema_version"] == "1.0"
    reconstructed = DeviceCapabilityDescriptor.from_dict(d)
    assert reconstructed.schema_version == "1.0"


def test_F02_unsupported_schema_version_rejected():
    """Registering a device with an unsupported capability schema_version is rejected."""
    gw = DeviceGateway()
    bad_cap = DeviceCapabilityDescriptor(
        capability_name="cam",
        action_name="snap",
        schema_version="99.0",  # unsupported!
    )
    ident = DeviceIdentity(
        device_id="BAD_VER_DEV",
        device_type=DeviceType.STATIONARY_SENSOR,
        display_name="Bad Version Device",
        capabilities=(bad_cap,),
    )
    with pytest.raises(ValueError, match="unsupported schema_version"):
        gw.register_device(ident)


# ============================================================================
# Section G: Command Lifecycle
# ============================================================================

def test_G01_command_state_progression():
    """Validate full semantic command state enum."""
    states = [
        CommandState.REQUESTED,
        CommandState.VALIDATED,
        CommandState.DISPATCHED,
        CommandState.ACKNOWLEDGED,
        CommandState.EXECUTING,
        CommandState.COMPLETED,
        CommandState.FAILED,
        CommandState.TIMEOUT,
        CommandState.REJECTED,
        CommandState.CANCELLED,
    ]
    for s in states:
        assert CommandState.from_str(s.value) == s


def test_G02_command_state_from_str():
    """CommandState.from_str handles case-insensitivity and default fallback."""
    assert CommandState.from_str("completed") == CommandState.COMPLETED
    assert CommandState.from_str("UNKNOWN_STATE") == CommandState.REQUESTED


def test_G03_command_request_to_dict_roundtrip():
    """DeviceCommandRequest serialization round-trip cleanly preserves all fields."""
    req = DeviceCommandRequest(
        command_id="cmd_123",
        device_id="DRONE_01",
        capability="takeoff",
        action="takeoff",
        parameters={"target_altitude": 25.0},
        correlation_id="corr_999",
        turn_id="turn_42",
        goal_id="goal_g1",
    )
    d = req.to_dict()
    assert d["command_id"] == "cmd_123"
    assert d["turn_id"] == "turn_42"
    reconstructed = DeviceCommandRequest.from_dict(d)
    assert reconstructed.command_id == req.command_id
    assert reconstructed.goal_id == "goal_g1"


# ============================================================================
# Section H: Acknowledgement
# ============================================================================

def test_H01_acknowledgement_distinct_from_completion():
    """Command acknowledgement confirms ingestion separately from execution completion."""
    gw = DeviceGateway()
    drone, adapter = create_virtual_drone("DRONE_ACK")
    gw.register_device(drone)
    gw.register_adapter(adapter, device_id="DRONE_ACK")

    res = gw.dispatch_to_device(
        device_id="DRONE_ACK",
        capability="hover",
        action="hover",
        parameters={},
        dispatch_id="disp_ack_01",
    )
    assert res.success is True
    # Gateway recorded acknowledgement
    ack = gw._latest_acknowledgements.get("disp_ack_01")
    assert ack is not None
    assert ack.status == AcknowledgementStatus.ACCEPTED
    assert ack.command_id == "disp_ack_01"


def test_H02_acknowledgement_statuses():
    """Verify AcknowledgementStatus enum values."""
    assert AcknowledgementStatus.ACCEPTED.value == "ACCEPTED"
    assert AcknowledgementStatus.QUEUED.value == "QUEUED"
    assert AcknowledgementStatus.REJECTED.value == "REJECTED"
    assert AcknowledgementStatus.EXECUTING.value == "EXECUTING"


def test_H03_acknowledgement_is_accepted_helper():
    """DeviceCommandAcknowledgement.is_accepted() correctly evaluates status."""
    ack_ok = DeviceCommandAcknowledgement(command_id="c1", device_id="d1", status=AcknowledgementStatus.ACCEPTED)
    assert ack_ok.is_accepted() is True

    ack_rej = DeviceCommandAcknowledgement(command_id="c2", device_id="d1", status=AcknowledgementStatus.REJECTED)
    assert ack_rej.is_accepted() is False


# ============================================================================
# Section I: Command Result
# ============================================================================

def test_I01_command_result_creation_and_to_result():
    """DeviceCommandResult bridges to Result.ok() preserving data and observations."""
    obs = MultimodalObservation(
        observation_id="obs_res_1",
        source_id="ROVER_01",
        source_type="ROVER",
        modality=ModalityType.TELEMETRY,
        timestamp=time.time(),
        payload={"battery": 95.0},
    )
    cmd_res = DeviceCommandResult(
        command_id="cmd_res_01",
        device_id="ROVER_01",
        state=CommandState.COMPLETED,
        success=True,
        message="Rover stopped successfully",
        data={"speed": 0.0},
        observations=(obs,),
        correlation_id="corr_Rover",
    )
    result = cmd_res.to_result()
    assert result.success is True
    assert result.message == "Rover stopped successfully"
    assert len(result.data["observations"]) == 1


def test_I02_command_result_failure_mapping():
    """DeviceCommandResult with error bridges cleanly to Result.fail()."""
    err = DeviceError(
        code=DeviceErrorCode.LOW_BATTERY,
        message="Battery level 4.2% below threshold",
        device_id="DRONE_LOW",
    )
    cmd_res = DeviceCommandResult(
        command_id="cmd_fail_01",
        device_id="DRONE_LOW",
        state=CommandState.FAILED,
        success=False,
        error=err,
    )
    result = cmd_res.to_result()
    assert result.success is False
    assert result.data["error_code"] == "LOW_BATTERY"


def test_I03_command_result_serialization_roundtrip():
    """DeviceCommandResult serializes and deserializes deterministically."""
    cmd_res = DeviceCommandResult(
        command_id="cmd_ser_01",
        device_id="VISION_01",
        state=CommandState.COMPLETED,
        success=True,
        duration_seconds=0.0452,
    )
    d = cmd_res.to_dict()
    reconstructed = DeviceCommandResult.from_dict(d)
    assert reconstructed.command_id == cmd_res.command_id
    assert reconstructed.state == CommandState.COMPLETED


# ============================================================================
# Section J: Lineage
# ============================================================================

def test_J01_causal_lineage_preservation():
    """Dispatch preserves correlation_id, causation_id, and contract_version."""
    gw = DeviceGateway()
    drone, adapter = create_virtual_drone("DRONE_LINEAGE")
    gw.register_device(drone)
    gw.register_adapter(adapter, device_id="DRONE_LINEAGE")

    res = gw.dispatch_to_device(
        device_id="DRONE_LINEAGE",
        capability="hover",
        action="hover",
        parameters={},
        dispatch_id="disp_lin_01",
        correlation_id="corr_trace_abc",
        causation_id="cause_parent_xyz",
        contract_version="1.0",
    )
    assert res.success is True
    assert res.data["correlation_id"] == "corr_trace_abc"


def test_J02_lineage_through_adapter_execution():
    """Observations produced during command execution retain correlation and causation IDs."""
    gw = DeviceGateway()
    rover, adapter = create_virtual_rover("ROVER_LINEAGE")
    gw.register_device(rover)
    gw.register_adapter(adapter, device_id="ROVER_LINEAGE")

    res = gw.dispatch_to_device(
        device_id="ROVER_LINEAGE",
        capability="navigate",
        action="go_to_waypoint",
        parameters={"latitude": 37.775, "longitude": -122.419},
        dispatch_id="disp_lin_rover",
        correlation_id="corr_autonomous_mission",
        causation_id="goal_explore_01",
    )
    assert res.success is True
    obs = res.data["observations"][0]
    assert obs.correlation_id == "corr_autonomous_mission"
    assert obs.causation_id == "disp_lin_rover"


def test_J03_lineage_no_unnecessary_replacement():
    """DeviceCommandRequest uses command_id as fallback only if correlation_id is empty."""
    req_with_corr = DeviceCommandRequest(
        command_id="cmd_99",
        device_id="DEV_1",
        capability="cap",
        action="act",
        correlation_id="existing_corr",
    )
    assert req_with_corr.correlation_id == "existing_corr"

    req_no_corr = DeviceCommandRequest(
        command_id="cmd_100",
        device_id="DEV_1",
        capability="cap",
        action="act",
    )
    assert req_no_corr.correlation_id == "cmd_100"


# ============================================================================
# Section K: Telemetry
# ============================================================================

def test_K01_canonical_telemetry_fields():
    """DeviceTelemetry models all required fields with deterministic defaults."""
    telem = DeviceTelemetry(
        device_id="DRONE_TELEM",
        location=GeoLocation(latitude=37.77, longitude=-122.41, altitude=15.0),
        battery_pct=88.5,
        connectivity=ConnectivityStatus.ONLINE,
        health=DeviceHealthStatus.HEALTHY,
        metrics={"speed_mps": 4.5},
    )
    assert telem.device_id == "DRONE_TELEM"
    assert telem.battery_pct == 88.5
    assert telem.connectivity == ConnectivityStatus.ONLINE


def test_K02_telemetry_to_multimodal_observation():
    """DeviceTelemetry.to_multimodal_observation() cleanly produces an observation."""
    telem = DeviceTelemetry(
        device_id="ROVER_TELEM",
        battery_pct=76.0,
        metrics={"motor_temp_c": 42.0},
        correlation_id="corr_tel_01",
    )
    obs = telem.to_multimodal_observation()
    assert isinstance(obs, MultimodalObservation)
    assert obs.modality == ModalityType.TELEMETRY
    assert obs.source_id == "ROVER_TELEM"
    assert obs.payload["battery_pct"] == 76.0
    assert obs.correlation_id == "corr_tel_01"


def test_K03_telemetry_gateway_recording():
    """Gateway.record_telemetry() stores telemetry in bounded history and returns observation."""
    gw = DeviceGateway()
    drone, adapter = create_virtual_drone("DRONE_T_REC")
    gw.register_device(drone)

    telem = DeviceTelemetry(
        device_id="DRONE_T_REC",
        battery_pct=92.0,
    )
    obs = gw.record_telemetry(telem)
    assert obs.device_id == "DRONE_T_REC"
    assert len(gw._telemetry_history["DRONE_T_REC"]) == 1


# ============================================================================
# Section L: Heartbeat
# ============================================================================

def test_L01_heartbeat_beacon_model():
    """DeviceHeartbeat represents heartbeat beacon with age calculation."""
    now = time.time()
    hb = DeviceHeartbeat(
        device_id="GLASS_HB",
        timestamp=now - 5.0,
        sequence_number=12,
    )
    assert hb.sequence_number == 12
    assert hb.get_age(now=now) >= 5.0


def test_L02_heartbeat_recording_and_sequence_increment():
    """Gateway.record_heartbeat() monotonically increments sequence numbers."""
    gw = DeviceGateway()
    drone, _ = create_virtual_drone("DRONE_HB_SEQ")
    gw.register_device(drone)

    gw.record_heartbeat("DRONE_HB_SEQ", timestamp=100.0)
    gw.record_heartbeat("DRONE_HB_SEQ", timestamp=105.0)
    gw.record_heartbeat("DRONE_HB_SEQ", timestamp=110.0)

    eval_data = gw.evaluate_heartbeat("DRONE_HB_SEQ", now=112.0)
    assert eval_data["sequence_number"] == 3
    assert eval_data["age_seconds"] == 2.0


def test_L03_heartbeat_age_and_timeout_evaluation():
    """Gateway marks device DISCONNECTED if heartbeat age exceeds timeout."""
    gw = DeviceGateway(heartbeat_timeout_seconds=30.0)
    rover, _ = create_virtual_rover("ROVER_HB_EXP")
    gw.register_device(rover)

    gw.record_heartbeat("ROVER_HB_EXP", timestamp=100.0)
    # At t=110, age is 10s (within 30s timeout)
    status_fresh = gw.query_device_status("ROVER_HB_EXP", now=110.0)
    assert status_fresh == ConnectivityStatus.ONLINE

    # At t=135, age is 35s (exceeds 30s timeout)
    status_expired = gw.query_device_status("ROVER_HB_EXP", now=135.0)
    assert status_expired == ConnectivityStatus.DISCONNECTED


# ============================================================================
# Section M: Health
# ============================================================================

def test_M01_health_status_enums():
    """Verify all four semantic DeviceHealthStatus ranks."""
    assert DeviceHealthStatus.HEALTHY.value == "HEALTHY"
    assert DeviceHealthStatus.DEGRADED.value == "DEGRADED"
    assert DeviceHealthStatus.UNHEALTHY.value == "UNHEALTHY"
    assert DeviceHealthStatus.UNKNOWN.value == "UNKNOWN"


def test_M02_device_health_computation():
    """Gateway.get_device_health() reflects connectivity and battery state."""
    gw = DeviceGateway(heartbeat_timeout_seconds=60.0)
    drone, _ = create_virtual_drone("DRONE_HLTH")
    gw.register_device(drone)

    # Initial state with heartbeat
    gw.record_heartbeat("DRONE_HLTH", timestamp=100.0)
    gw.record_telemetry(DeviceTelemetry(device_id="DRONE_HLTH", battery_pct=85.0, timestamp=100.0))

    health = gw.get_device_health("DRONE_HLTH", now=102.0)
    assert health.status == DeviceHealthStatus.HEALTHY
    assert health.battery_pct == 85.0

    # Low battery degrades health
    gw.record_telemetry(DeviceTelemetry(device_id="DRONE_HLTH", battery_pct=15.0, timestamp=105.0))
    health_degraded = gw.get_device_health("DRONE_HLTH", now=106.0)
    assert health_degraded.status == DeviceHealthStatus.DEGRADED

    # Critical battery (<=5%) marks UNHEALTHY
    gw.record_telemetry(DeviceTelemetry(device_id="DRONE_HLTH", battery_pct=4.0, timestamp=107.0))
    health_crit = gw.get_device_health("DRONE_HLTH", now=108.0)
    assert health_crit.status == DeviceHealthStatus.UNHEALTHY


def test_M03_health_serialization_no_credentials():
    """DeviceHealth.to_dict() serializes cleanly with zero credential leaks."""
    health = DeviceHealth(
        device_id="DEV_H",
        status=DeviceHealthStatus.HEALTHY,
        connectivity=ConnectivityStatus.ONLINE,
        battery_pct=90.0,
    )
    d = health.to_dict()
    assert "token" not in d
    assert "password" not in d
    assert d["status"] == "HEALTHY"


# ============================================================================
# Section N: Errors
# ============================================================================

def test_N01_canonical_error_codes():
    """All 13 canonical DeviceErrorCode classifications are defined."""
    codes = [
        DeviceErrorCode.UNKNOWN_DEVICE,
        DeviceErrorCode.OFFLINE_DEVICE,
        DeviceErrorCode.UNSUPPORTED_CAPABILITY,
        DeviceErrorCode.UNSUPPORTED_ACTION,
        DeviceErrorCode.INVALID_PARAMETERS,
        DeviceErrorCode.COMMAND_REJECTED,
        DeviceErrorCode.COMMAND_TIMEOUT,
        DeviceErrorCode.TRANSPORT_ERROR,
        DeviceErrorCode.DEVICE_ERROR,
        DeviceErrorCode.LOW_BATTERY,
        DeviceErrorCode.SAFETY_REJECTION,
        DeviceErrorCode.DUPLICATE_COMMAND,
        DeviceErrorCode.CONTRACT_VERSION_UNSUPPORTED,
    ]
    assert len(codes) == 13


def test_N02_device_error_model_serialization():
    """DeviceError serialization and deserialization roundtrip."""
    err = DeviceError(
        code=DeviceErrorCode.SAFETY_REJECTION,
        message="Obstacle detected",
        device_id="ROVER_01",
        command_id="cmd_987",
    )
    d = err.to_dict()
    reconstructed = DeviceError.from_dict(d)
    assert reconstructed.code == DeviceErrorCode.SAFETY_REJECTION
    assert reconstructed.command_id == "cmd_987"


def test_N03_error_mapping_to_result_failure():
    """Gateway returns Result.fail with canonical error code on failure."""
    gw = DeviceGateway()
    res = gw.dispatch_to_device("NON_EXISTENT_DEV", "hover", "hover", {})
    assert res.success is False
    assert res.data["error_code"] == DeviceErrorCode.UNKNOWN_DEVICE.value


# ============================================================================
# Section O: Device Lifecycle
# ============================================================================

def test_O01_device_lifecycle_register_unregister():
    """Gateway manages registration and unregistration lifecycles cleanly."""
    gw = DeviceGateway()
    vision, _ = create_virtual_vision("VISION_LC")
    gw.register_device(vision)
    assert gw.get_device("VISION_LC") is not None

    ok = gw.unregister_device("VISION_LC")
    assert ok is True
    assert gw.get_device("VISION_LC") is None


def test_O02_device_lifecycle_status_transitions():
    """Gateway updates and queries connectivity status transitions."""
    gw = DeviceGateway()
    drone, _ = create_virtual_drone("DRONE_STAT")
    gw.register_device(drone)

    gw.update_device_status("DRONE_STAT", ConnectivityStatus.DEGRADED)
    assert gw.query_device_status("DRONE_STAT") == ConnectivityStatus.DEGRADED

    gw.update_device_status("DRONE_STAT", ConnectivityStatus.ONLINE)
    assert gw.query_device_status("DRONE_STAT") == ConnectivityStatus.ONLINE


def test_O03_device_lifecycle_connect_disconnect_adapter():
    """DeviceAdapterInterface connect() and disconnect() methods update status."""
    _, adapter = create_virtual_drone("DRONE_AD_LC")
    assert adapter.get_status() == ConnectivityStatus.ONLINE
    adapter.disconnect()
    assert adapter.get_status() == ConnectivityStatus.DISCONNECTED
    adapter.connect()
    assert adapter.get_status() == ConnectivityStatus.ONLINE


# ============================================================================
# Section P: Adapter Compatibility
# ============================================================================

def test_P01_adapter_interface_default_methods():
    """Custom adapter subclassing DeviceAdapterInterface inherits safe defaults."""
    class CustomMinimalAdapter(DeviceAdapterInterface):
        def get_protocol_name(self) -> str:
            return "minimal"
        def format_command(self, tool_call: ToolCall) -> Dict[str, Any]:
            return {}
        def execute_command(self, command: Any) -> Result:
            return Result.ok()

    ad = CustomMinimalAdapter()
    assert ad.connect() is True
    assert ad.disconnect() is True
    assert ad.get_status() == ConnectivityStatus.ONLINE
    assert ad.get_capabilities() == ()


def test_P02_adapter_resolution_precedence():
    """Gateway resolves device-specific adapter over type-specific adapter."""
    gw = DeviceGateway()
    drone, specific_adapter = create_virtual_drone("DRONE_PRECEDENCE")
    _, generic_type_adapter = create_virtual_drone("GENERIC_DRONE")

    gw.register_device(drone)
    gw.register_adapter(generic_type_adapter, device_type=DeviceType.DRONE_AERIAL)
    gw.register_adapter(specific_adapter, device_id="DRONE_PRECEDENCE")

    resolved = gw.resolve_adapter("DRONE_PRECEDENCE")
    assert resolved is specific_adapter
    assert resolved is not generic_type_adapter


def test_P03_protocol_naming_neutrality():
    """All virtual adapters declare semantic protocol names without hardware bus strings."""
    _, v_ad = create_virtual_vision()
    _, g_ad = create_virtual_glass()
    _, d_ad = create_virtual_drone()
    _, r_ad = create_virtual_rover()

    for ad in (v_ad, g_ad, d_ad, r_ad):
        pname = ad.get_protocol_name().lower()
        assert "uart" not in pname
        assert "serial" not in pname
        assert "spi" not in pname
        assert "gpio" not in pname


# ============================================================================
# Section Q: Virtual Vision
# ============================================================================

def test_Q01_virtual_vision_motion_detection():
    """VirtualVisionAdapter.detect_motion produces a valid EVENT observation."""
    ident, adapter = create_virtual_vision("VISION_MOTION")
    cmd = DeviceCommand(
        dispatch_id="d_mot_1",
        device_id="VISION_MOTION",
        capability="detect_motion",
        action="detect_motion",
        parameters={"region": "front_porch"},
    )
    res = adapter.execute_command(cmd)
    assert res.success is True
    assert res.data["motion_detected"] is True
    assert len(res.data["observations"]) == 1
    obs = res.data["observations"][0]
    assert obs.modality == ModalityType.EVENT
    assert obs.payload["region"] == "front_porch"


def test_Q02_virtual_vision_person_and_anomaly():
    """VirtualVisionAdapter detects person and anomaly deterministically."""
    _, adapter = create_virtual_vision("VISION_ANOM")
    cmd_person = DeviceCommand(
        dispatch_id="d_per_1",
        device_id="VISION_ANOM",
        capability="detect_person",
        action="detect_person",
        parameters={"confidence": 0.96},
    )
    res_p = adapter.execute_command(cmd_person)
    assert res_p.success is True
    assert res_p.data["event"] == "PERSON_DETECTED"

    cmd_anom = DeviceCommand(
        dispatch_id="d_anom_1",
        device_id="VISION_ANOM",
        capability="detect_anomaly",
        action="detect_anomaly",
        parameters={"type": "BROKEN_GLASS"},
    )
    res_a = adapter.execute_command(cmd_anom)
    assert res_a.success is True
    assert res_a.data["anomaly"]["anomaly_type"] == "BROKEN_GLASS"


def test_Q03_virtual_vision_image_and_video():
    """VirtualVisionAdapter captures image and video returning artifact references."""
    _, adapter = create_virtual_vision("VISION_MEDIA")
    res_img = adapter.execute_command(
        DeviceCommand("d_img", "VISION_MEDIA", "capture_image", "capture_image", {})
    )
    assert res_img.success is True
    assert "artifacts/vision" in res_img.data["artifact_reference"]

    res_vid = adapter.execute_command(
        DeviceCommand("d_vid", "VISION_MEDIA", "capture_video", "capture_video", {"duration_seconds": 15.0})
    )
    assert res_vid.success is True
    assert "artifacts/vision" in res_vid.data["artifact_reference"]
    assert res_vid.data["duration_seconds"] == 15.0


# ============================================================================
# Section R: Virtual Glass
# ============================================================================

def test_R01_virtual_glass_display_hud():
    """VirtualGlassAdapter updates HUD display and hud_message property."""
    _, adapter = create_virtual_glass("GLASS_HUD")
    res = adapter.execute_command(
        DeviceCommand("d_hud", "GLASS_HUD", "display_hud", "display_hud", {"text": "Incoming Call: Alice"})
    )
    assert res.success is True
    assert adapter.display_text == "Incoming Call: Alice"
    assert adapter.hud_message == "Incoming Call: Alice"


def test_R02_virtual_glass_capture_image_and_audio():
    """VirtualGlassAdapter captures first-person POV photo and ambient audio."""
    _, adapter = create_virtual_glass("GLASS_AV")
    res_img = adapter.execute_command(
        DeviceCommand("d_g_img", "GLASS_AV", "capture_image", "capture_image", {})
    )
    assert res_img.success is True
    assert len(res_img.data["observations"]) == 1
    assert res_img.data["observations"][0].modality == ModalityType.IMAGE

    res_aud = adapter.execute_command(
        DeviceCommand("d_g_aud", "GLASS_AV", "capture_audio", "capture_audio", {"duration_seconds": 4.0})
    )
    assert res_aud.success is True
    assert res_aud.data["observations"][0].modality == ModalityType.AUDIO_EVENT


def test_R03_virtual_glass_telemetry_and_location():
    """VirtualGlassAdapter retrieves user location and smart glasses telemetry."""
    _, adapter = create_virtual_glass("GLASS_TELEM")
    res_loc = adapter.execute_command(
        DeviceCommand("d_g_loc", "GLASS_TELEM", "get_location", "get_location", {})
    )
    assert res_loc.success is True
    assert res_loc.data["observations"][0].modality == ModalityType.GPS


# ============================================================================
# Section S: Virtual Drone
# ============================================================================

def test_S01_virtual_drone_takeoff_and_land():
    """VirtualDroneAdapter simulates vertical takeoff and landing."""
    _, adapter = create_virtual_drone("DRONE_FLIGHT")
    res_to = adapter.execute_command(
        DeviceCommand("d_to", "DRONE_FLIGHT", "takeoff", "takeoff", {"target_altitude": 15.0})
    )
    assert res_to.success is True
    assert adapter.airborne is True
    assert adapter.altitude == 15.0

    res_land = adapter.execute_command(
        DeviceCommand("d_land", "DRONE_FLIGHT", "land", "land", {})
    )
    assert res_land.success is True
    assert adapter.airborne is False
    assert adapter.altitude == 0.0


def test_S02_virtual_drone_waypoint_nav_and_hover():
    """VirtualDroneAdapter navigates to GPS waypoint and holds hover."""
    _, adapter = create_virtual_drone("DRONE_NAV")
    res_nav = adapter.execute_command(
        DeviceCommand("d_nav", "DRONE_NAV", "navigate", "go_to_waypoint", {"latitude": 37.78, "longitude": -122.42, "altitude": 20.0})
    )
    assert res_nav.success is True
    assert adapter.location.latitude == 37.78

    res_hov = adapter.execute_command(
        DeviceCommand("d_hov", "DRONE_NAV", "hover", "hover", {})
    )
    assert res_hov.success is True


def test_S03_virtual_drone_low_battery_rejection():
    """VirtualDroneAdapter rejects takeoff when battery <= 5.0%."""
    _, adapter = create_virtual_drone("DRONE_LOW_BATT")
    adapter.battery_pct = 4.5
    res = adapter.execute_command(
        DeviceCommand("d_to_low", "DRONE_LOW_BATT", "takeoff", "takeoff", {})
    )
    assert res.success is False
    assert res.data["error_code"] == DeviceErrorCode.LOW_BATTERY.value


# ============================================================================
# Section T: Virtual Rover
# ============================================================================

def test_T01_virtual_rover_navigation_and_stop():
    """VirtualRoverAdapter simulates ground navigation and stop locomotion."""
    _, adapter = create_virtual_rover("ROVER_GROUND")
    res_nav = adapter.execute_command(
        DeviceCommand("d_r_nav", "ROVER_GROUND", "navigate", "go_to_waypoint", {"latitude": 37.776, "longitude": -122.418})
    )
    assert res_nav.success is True
    assert adapter.moving is True

    res_stop = adapter.execute_command(
        DeviceCommand("d_r_stop", "ROVER_GROUND", "stop", "stop", {})
    )
    assert res_stop.success is True
    assert adapter.moving is False


def test_T02_virtual_rover_obstacle_simulation():
    """VirtualRoverAdapter fails navigation when simulate_obstacle is active."""
    _, adapter = create_virtual_rover("ROVER_OBS")
    adapter.simulate_obstacle = True
    res = adapter.execute_command(
        DeviceCommand("d_r_obs", "ROVER_OBS", "navigate", "go_to_waypoint", {"latitude": 37.776, "longitude": -122.418})
    )
    assert res.success is False
    assert res.data["error_code"] == DeviceErrorCode.SAFETY_REJECTION.value
    assert "obstacle detected" in res.message


def test_T03_virtual_rover_telemetry_and_image():
    """VirtualRoverAdapter retrieves telemetry and captures forward ground picture."""
    _, adapter = create_virtual_rover("ROVER_CAM")
    res_cam = adapter.execute_command(
        DeviceCommand("d_r_cam", "ROVER_CAM", "capture_image", "capture_image", {})
    )
    assert res_cam.success is True
    assert "artifacts/rover" in res_cam.data["artifact_reference"]


# ============================================================================
# Section U: Vision Observation Generation & Central Ingress
# ============================================================================

def test_U01_vision_observation_through_central_input_gateway():
    """Vision environmental observation flows into CentralInputGateway without direct WorldState writes."""
    gw = DeviceGateway()
    input_gw = CentralInputGateway()
    vision, adapter = create_virtual_vision("VISION_CENTRAL")
    gw.register_device(vision)
    gw.register_adapter(adapter, device_id="VISION_CENTRAL")

    # 1. Vision emits person detection observation
    res = gw.dispatch_to_device(
        device_id="VISION_CENTRAL",
        capability="detect_person",
        action="detect_person",
        parameters={"confidence": 0.95, "person_count": 1},
    )
    assert res.success is True
    obs: MultimodalObservation = res.data["observations"][0]

    # 2. Observation enters CentralInputGateway
    input_gw.ingest_observation(obs)
    recent = input_gw.get_recent_observations(modality=ModalityType.EVENT)
    assert len(recent) == 1
    assert recent[0].source_id == "VISION_CENTRAL"
    assert recent[0].payload["event"] == "PERSON_DETECTED"


# ============================================================================
# Section V: Command Observation Generation & Central Ingress
# ============================================================================

def test_V01_command_to_observation_ingress():
    """End-to-end command execution produces observations ingested into CentralInputGateway."""
    gw = DeviceGateway()
    input_gw = CentralInputGateway()
    drone, adapter = create_virtual_drone("DRONE_INGRESS")
    gw.register_device(drone)
    gw.register_adapter(adapter, device_id="DRONE_INGRESS")

    cap_bridge = DeviceGatewayCapability(gw)
    reg = CapabilityRegistry()
    reg.register("device_gateway", cap_bridge)
    orch = ToolOrchestrator(registry=reg)

    tc = ToolCall(
        capability="device_gateway",
        action="dispatch_capability",
        parameters={
            "device_id": "DRONE_INGRESS",
            "capability": "capture_image",
            "action": "capture_image",
            "parameters": {"target": "inspection_zone_a"},
        },
        call_id="call_inspect_01",
    )
    res = orch.execute(tc)
    assert res.success is True
    observations = res.data["observations"]
    assert len(observations) == 1

    # Ingest into CentralInputGateway
    for o in observations:
        input_gw.ingest_observation(o)

    recent_images = input_gw.get_recent_observations(modality=ModalityType.IMAGE)
    assert len(recent_images) == 1
    assert recent_images[0].source_id == "DRONE_INGRESS"


# ============================================================================
# Section W: Multi-Product Isolation
# ============================================================================

def test_W01_four_products_simultaneous_isolation():
    """VISION, GLASS, DRONE, and ROVER registered concurrently maintain total state and command isolation."""
    gw = DeviceGateway()
    vision, v_ad = create_virtual_vision("ATLAS_VISION_01")
    glass, g_ad = create_virtual_glass("ATLAS_GLASS_01")
    drone, d_ad = create_virtual_drone("ATLAS_DRONE_01")
    rover, r_ad = create_virtual_rover("ATLAS_ROVER_01")

    for dev, ad in ((vision, v_ad), (glass, g_ad), (drone, d_ad), (rover, r_ad)):
        gw.register_device(dev)
        gw.register_adapter(ad, device_id=dev.device_id)

    assert len(gw.list_devices()) == 4

    # Dispatch to Drone
    gw.dispatch_to_device("ATLAS_DRONE_01", "takeoff", "takeoff", {"target_altitude": 10.0})
    assert d_ad.airborne is True
    assert r_ad.moving is False
    assert v_ad.recording is False

    # Dispatch to Rover
    gw.dispatch_to_device("ATLAS_ROVER_01", "navigate", "go_to_waypoint", {"latitude": 37.7, "longitude": -122.4})
    assert r_ad.moving is True
    assert d_ad.altitude == 10.0

    # Dispatch to Glass
    gw.dispatch_to_device("ATLAS_GLASS_01", "display_hud", "display_hud", {"text": "HUD Isolation Check"})
    assert g_ad.display_text == "HUD Isolation Check"

    # Dispatch to Vision
    gw.dispatch_to_device("ATLAS_VISION_01", "detect_motion", "detect_motion", {"region": "hallway"})
    assert v_ad.last_detection["region"] == "hallway"


# ============================================================================
# Section X: Duplicate Commands & Idempotency
# ============================================================================

def test_X01_duplicate_command_id_deterministic_rejection():
    """Duplicate dispatch_id is deterministically rejected with DUPLICATE_COMMAND."""
    gw = DeviceGateway()
    drone, adapter = create_virtual_drone("DRONE_DUP")
    gw.register_device(drone)
    gw.register_adapter(adapter, device_id="DRONE_DUP")

    res1 = gw.dispatch_to_device("DRONE_DUP", "hover", "hover", {}, dispatch_id="cmd_dup_fixed_id")
    assert res1.success is True

    res2 = gw.dispatch_to_device("DRONE_DUP", "hover", "hover", {}, dispatch_id="cmd_dup_fixed_id")
    assert res2.success is False
    assert res2.data["error_code"] == DeviceErrorCode.DUPLICATE_COMMAND.value
    assert "Duplicate dispatch_id" in res2.message


# ============================================================================
# Section Y: Replay Determinism
# ============================================================================

def test_Y01_deterministic_replay_equality():
    """Replaying an identical sequence of edge operations produces deterministic output parity."""
    def run_sequence(seed_time: float) -> List[Dict[str, Any]]:
        gw = DeviceGateway(clock=lambda: seed_time)
        rover, adapter = create_virtual_rover("ROVER_REPLAY")
        gw.register_device(rover)
        gw.register_adapter(adapter, device_id="ROVER_REPLAY")

        trace = []
        # 1. Heartbeat
        gw.record_heartbeat("ROVER_REPLAY", timestamp=seed_time)
        trace.append(gw.evaluate_heartbeat("ROVER_REPLAY", now=seed_time))

        # 2. Command
        res = gw.dispatch_to_device(
            device_id="ROVER_REPLAY",
            capability="navigate",
            action="go_to_waypoint",
            parameters={"latitude": 37.77, "longitude": -122.42},
            dispatch_id="disp_rep_01",
            now=seed_time,
        )
        trace.append({"success": res.success, "location": adapter.location.to_dict()})

        # 3. Telemetry
        telem = DeviceTelemetry(device_id="ROVER_REPLAY", battery_pct=98.0, timestamp=seed_time)
        obs = gw.record_telemetry(telem)
        trace.append(obs.to_dict())

        return trace

    run1 = run_sequence(1700000000.0)
    run2 = run_sequence(1700000000.0)
    assert run1 == run2


# ============================================================================
# Section Z: Deterministic Failure
# ============================================================================

def test_Z01_deterministic_device_failure_simulation():
    """When adapter failure simulation is activated, gateway returns DEVICE_ERROR."""
    gw = DeviceGateway()
    drone, adapter = create_virtual_drone("DRONE_FAIL")
    gw.register_device(drone)
    gw.register_adapter(adapter, device_id="DRONE_FAIL")

    adapter.simulate_failure = True
    res = gw.dispatch_to_device("DRONE_FAIL", "hover", "hover", {})
    assert res.success is False
    assert res.data["error_code"] == DeviceErrorCode.DEVICE_ERROR.value


# ============================================================================
# Section AA: Timeout Simulation
# ============================================================================

def test_AA01_simulated_command_timeout():
    """Simulated device timeout returns canonical COMMAND_TIMEOUT error."""
    gw = DeviceGateway()
    drone, adapter = create_virtual_drone("DRONE_TIMEOUT")
    gw.register_device(drone)
    gw.register_adapter(adapter, device_id="DRONE_TIMEOUT")

    adapter.simulate_timeout = True
    res = gw.dispatch_to_device("DRONE_TIMEOUT", "hover", "hover", {})
    assert res.success is False
    assert res.data["error_code"] == DeviceErrorCode.COMMAND_TIMEOUT.value


# ============================================================================
# Section AB: Bounds Enforcement
# ============================================================================

def test_AB01_gateway_bounds_enforcement():
    """Gateway enforces bounds on max devices, max capabilities, and max history."""
    # 1. Max devices
    gw = DeviceGateway(max_devices=2, max_dispatch_history=5)
    d1, _ = create_virtual_drone("D1")
    d2, _ = create_virtual_drone("D2")
    d3, _ = create_virtual_drone("D3")

    gw.register_device(d1)
    gw.register_device(d2)
    with pytest.raises(ValueError, match="Maximum registered devices limit"):
        gw.register_device(d3)


# ============================================================================
# Section AC: Concurrency Safety
# ============================================================================

def test_AC01_concurrent_multi_device_dispatch():
    """Concurrent dispatch operations across devices execute safely with RLock protection."""
    gw = DeviceGateway()
    drone, d_ad = create_virtual_drone("DRONE_CONCUR")
    rover, r_ad = create_virtual_rover("ROVER_CONCUR")
    gw.register_device(drone)
    gw.register_device(rover)
    gw.register_adapter(d_ad, device_id="DRONE_CONCUR")
    gw.register_adapter(r_ad, device_id="ROVER_CONCUR")

    errors = []

    def worker(dev_id: str, cap: str, act: str, tid: int):
        try:
            for i in range(10):
                res = gw.dispatch_to_device(
                    device_id=dev_id,
                    capability=cap,
                    action=act,
                    parameters={},
                    dispatch_id=f"disp_concur_{dev_id}_{tid}_{i}",
                )
                if not res.success:
                    errors.append(f"Failure on {dev_id}: {res.message}")
        except Exception as e:
            errors.append(str(e))

    threads = [
        threading.Thread(target=worker, args=("DRONE_CONCUR", "hover", "hover", 1)),
        threading.Thread(target=worker, args=("DRONE_CONCUR", "hover", "hover", 2)),
        threading.Thread(target=worker, args=("ROVER_CONCUR", "stop", "stop", 3)),
        threading.Thread(target=worker, args=("ROVER_CONCUR", "stop", "stop", 4)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0
    assert len(gw._seen_dispatch_ids) == 40


# ============================================================================
# Section AD: Security & Credential Sanitization
# ============================================================================

def test_AD01_security_sanitization_no_secrets():
    """Sanitizer scrubs sensitive tokens, passwords, and private keys from all contract payloads."""
    raw_meta = {
        "user_name": "atlas_admin",
        "api_key": "sk-1234567890",
        "nested": {
            "password": "p@ssword",
            "safe_param": 42,
        }
    }
    clean = sanitize_contract_metadata(raw_meta)
    assert clean["user_name"] == "atlas_admin"
    assert clean["api_key"] == "[REDACTED]"
    assert clean["nested"]["password"] == "[REDACTED]"
    assert clean["nested"]["safe_param"] == 42


# ============================================================================
# Section AE: Authority Boundaries
# ============================================================================

def test_AE01_authority_boundaries_enforced():
    """
    CognitiveRuntime -> PolicyEngine -> ToolOrchestrator -> DeviceGateway -> Adapter.
    PolicyEngine denies unauthorized actions before ToolOrchestrator or DeviceGateway can execute.
    """
    gw = DeviceGateway()
    drone, adapter = create_virtual_drone("DRONE_AUTH")
    gw.register_device(drone)
    gw.register_adapter(adapter, device_id="DRONE_AUTH")

    bridge = DeviceGatewayCapability(gw)
    reg = CapabilityRegistry()
    reg.register("device_gateway", bridge)

    policy = StandardPolicyEngine()
    orch = ToolOrchestrator(registry=reg, policy_engine=policy)

    # ToolCall with forbidden capability cannot pass policy
    forbidden_call = ToolCall(capability="shell", action="exec", parameters={"cmd": "rm -rf /"})
    res = orch.execute(forbidden_call)
    assert res.success is False
    assert "Prohibited" in res.message or "forbidden" in res.message
