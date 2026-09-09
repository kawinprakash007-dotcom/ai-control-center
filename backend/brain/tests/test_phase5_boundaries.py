import inspect
import sys
import pytest

from core.models.orchestration import (
    ConnectivityStatus,
    DeviceCapabilityDescriptor,
    DeviceIdentity,
    DeviceType,
    GeoLocation,
    ModalityType,
    MultimodalObservation,
    Situation,
    SituationCategory,
    SituationEvidence,
    SituationSeverity,
    SituationStatus,
)
from core.models.world_state import WorldState, WorldCondition, WorldEntity
import core.models.orchestration as orchestration_module
import core.interfaces.orchestration_interface as orchestration_interfaces


def test_situation_is_not_world_state():
    """
    CRITICAL ARCHITECTURAL BOUNDARY:
    A Situation is an intermediate interpretation of observations, NOT a WorldState.
    """
    assert Situation is not WorldState
    assert not issubclass(Situation, WorldState)

    sit = Situation(
        situation_id="sit_001",
        category=SituationCategory.ENVIRONMENTAL,
        title="High Ambient Heat",
        description="Temperature exceeding 40C",
        severity=SituationSeverity.HIGH,
        confidence=0.9,
        status=SituationStatus.ACTIVE,
        involved_entities=("hvac_zone_1",),
        supporting_evidence=(),
    )

    assert not isinstance(sit, WorldState)
    # Situation does not expose WorldState entity mapping / versioning APIs
    assert not hasattr(sit, "entities")
    assert not hasattr(sit, "conditions")
    assert not hasattr(sit, "relationships")
    assert not hasattr(sit, "version")


def test_situation_cannot_directly_mutate_world_state():
    """
    CRITICAL ARCHITECTURAL BOUNDARY:
    Situations cannot directly mutate WorldState.
    WorldState updates MUST pass through the explicit WorldStateUpdater boundary.
    """
    sit = Situation(
        situation_id="sit_002",
        category=SituationCategory.SYSTEM_HEALTH,
        title="Degraded Comm",
        description="Packet loss detected",
        severity=SituationSeverity.MEDIUM,
        confidence=0.8,
        status=SituationStatus.ACTIVE,
        involved_entities=("comms_unit",),
        supporting_evidence=(),
    )

    for method_name in dir(sit):
        assert "apply_transition" not in method_name
        assert "update_condition" not in method_name
        assert "mutate" not in method_name
        assert "commit" not in method_name


def test_situation_does_not_contain_executable_tool_call_semantics():
    """
    CRITICAL ARCHITECTURAL BOUNDARY:
    A Situation contains semantic and contextual fields only; NO executable ToolCall or action dispatch.
    """
    sit_fields = set(Situation.__dataclass_fields__.keys())
    # Ensure no executable or tool dispatch fields exist
    forbidden_fields = {
        "tool_call",
        "tool_name",
        "action_parameters",
        "execute",
        "handler",
        "dispatch",
        "callback",
        "coroutine",
    }
    assert not (sit_fields & forbidden_fields)


def test_device_capability_descriptor_does_not_execute_actions():
    """
    CRITICAL ARCHITECTURAL BOUNDARY:
    DeviceCapabilityDescriptor ONLY declares metadata and schemas.
    It has no execute(), invoke(), or call() methods.
    Execution authority remains strictly with ToolOrchestrator.
    """
    cap = DeviceCapabilityDescriptor(
        capability_name="turn_on_light",
        action_name="gpio_write_high",
    )
    assert not hasattr(cap, "execute")
    assert not hasattr(cap, "invoke")
    assert not hasattr(cap, "run")
    assert not hasattr(cap, "call")
    assert not hasattr(cap, "trigger")


def test_device_identity_does_not_encode_transport_details():
    """
    CRITICAL ARCHITECTURAL BOUNDARY:
    DeviceIdentity is hardware-independent and transport-agnostic.
    DO NOT embed serial protocol, UART configuration, GPIO, MAVLink, ROS2, or ESP32 registers.
    """
    dev_fields = set(DeviceIdentity.__dataclass_fields__.keys())
    forbidden_transport_fields = {
        "baud_rate",
        "baudrate",
        "serial_port",
        "uart_port",
        "gpio_pin",
        "i2c_address",
        "spi_bus",
        "mavlink_system_id",
        "mavlink_component_id",
        "ros2_node",
        "ros_topic",
        "esp32_register",
        "ip_address",
        "mac_address",
        "socket",
    }
    assert not (dev_fields & forbidden_transport_fields), f"Found transport fields in DeviceIdentity: {dev_fields & forbidden_transport_fields}"


def test_new_models_do_not_import_hardware_or_network_libraries():
    """
    SECURITY & ARCHITECTURE AUDIT:
    Confirm that the new model and interface modules do NOT import any
    hardware, subprocess, or network socket libraries.
    """
    forbidden_imports = {
        "socket",
        "urllib.request",
        "requests",
        "httpx",
        "aiohttp",
        "serial",
        "pyserial",
        "RPi.GPIO",
        "smbus",
        "paho.mqtt",
        "pymavlink",
        "rclpy",
        "subprocess",
        "os.system",
    }

    orchestration_src = inspect.getsource(orchestration_module)
    interface_src = inspect.getsource(orchestration_interfaces)

    for forbidden in forbidden_imports:
        assert f"import {forbidden}" not in orchestration_src, f"Discovered forbidden import '{forbidden}' in orchestration models"
        assert f"from {forbidden}" not in orchestration_src, f"Discovered forbidden import '{forbidden}' in orchestration models"
        assert f"import {forbidden}" not in interface_src, f"Discovered forbidden import '{forbidden}' in orchestration interfaces"
        assert f"from {forbidden}" not in interface_src, f"Discovered forbidden import '{forbidden}' in orchestration interfaces"


def test_no_code_execution_or_eval_in_models():
    """
    SECURITY AUDIT:
    Ensure zero eval, exec, compile, or pickle usage in new models.
    """
    orchestration_src = inspect.getsource(orchestration_module)
    interface_src = inspect.getsource(orchestration_interfaces)

    for code_str in [orchestration_src, interface_src]:
        assert "eval(" not in code_str
        assert "exec(" not in code_str
        assert "pickle" not in code_str
        assert "__import__" not in code_str
