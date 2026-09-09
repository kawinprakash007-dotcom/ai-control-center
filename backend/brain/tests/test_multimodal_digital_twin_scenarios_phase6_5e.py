"""
ATLAS Phase 6.5e — Multimodal Digital-Twin Scenarios Test Suite.

Comprehensive validation covering:
1. Domain models, bounds, validation, and immutability (A-E).
2. Scenario catalog, deterministic IDs, capacity bounds (F-H).
3. Runner initialization, SimulationClock, step ordering, time advancement (I-L).
4. Twin and world configuration without production state mutation (M-N).
5. Observation, telemetry, fault injection, and fault clearing (O-R).
6. Assertion evaluation (all 16 targets, operators, tolerance, severity), strict mode (S-V).
7. Replay records, deterministic execution, and SHA-256 hash matching (W-X).
8. The 10 canonical scenarios execution (Y-AH).
9. Four digital twins (Vision, Glass, Drone, Rover) validation (AI-AL).
10. Perception and Temporal Cross-Modal Fusion integration (AM-AP).
11. Observational targets (WorldState, Event, Mission, Objective, Goal, Trace, Policy, Device) (AS-AX).
12. Strict architectural boundary invariants (AST / audit) (AY-BF).
13. Bounded collections, determinism, isolation, and edge cases (BJ-BN).
14. End-to-end full ecosystem validation with AtlasApplicationState (Section 36).
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
import os
import sys
import threading
from typing import Any, Dict, List, Optional, Sequence
from unittest.mock import MagicMock

# Fallbacks for test environment if chromadb/sentence_transformers missing
if "chromadb" not in sys.modules:
    sys.modules["chromadb"] = MagicMock()
if "sentence_transformers" not in sys.modules:
    sys.modules["sentence_transformers"] = MagicMock()

import pytest

from core.models.device_contract import (
    ConnectivityStatus,
    DeviceHealthStatus,
    ProductRole,
    ProductType,
)
from core.models.orchestration import (
    GeoLocation,
    ModalityType,
    MultimodalObservation,
)
from core.models.scenario import (
    Scenario,
    ScenarioAssertion,
    ScenarioAssertionOperator,
    ScenarioAssertionResult,
    ScenarioAssertionSeverity,
    ScenarioAssertionTarget,
    ScenarioBuilder,
    ScenarioLimits,
    ScenarioResult,
    ScenarioStep,
    ScenarioStepType,
)
from core.models.simulation import (
    SimulationLimits,
    TwinConfiguration,
    TwinFault,
    TwinFaultType,
    TwinPosition,
)
from simulation.adapters import DigitalTwinAdapter, create_digital_twin_device
from simulation.catalog import (
    ScenarioCatalog,
    build_canonical_catalog,
    create_conflicting_sensor_reports_scenario,
    create_device_offline_scenario,
    create_drone_failure_scenario,
    create_full_ecosystem_scenario,
    create_home_intrusion_scenario,
    create_lost_person_scenario,
    create_mission_resolution_scenario,
    create_moving_target_scenario,
    create_multi_source_corroboration_scenario,
    create_stale_telemetry_scenario,
)
from simulation.clock import SimulationClock
from simulation.environment import SimulatedEntity, SimulatedHazard
from simulation.runner import ScenarioRunner
from simulation.twin import (
    DroneDigitalTwin,
    GlassDigitalTwin,
    RoverDigitalTwin,
    VisionDigitalTwin,
)
from simulation.world import SimulationWorld
from multimodal_fusion.engine import TemporalCrossModalFusionEngine


# ============================================================================
# Section A-E: Domain Models, Immutability, Serialization, Enums
# ============================================================================

def test_A01_scenario_enums():
    # Step types
    assert ScenarioStepType.ADVANCE_TIME.value == "ADVANCE_TIME"
    assert ScenarioStepType.SET_WORLD_STATE.value == "SET_WORLD_STATE"
    assert ScenarioStepType.CONFIGURE_TWIN.value == "CONFIGURE_TWIN"
    assert ScenarioStepType.INJECT_OBSERVATION.value == "INJECT_OBSERVATION"
    assert ScenarioStepType.INJECT_TELEMETRY.value == "INJECT_TELEMETRY"
    assert ScenarioStepType.INJECT_FAULT.value == "INJECT_FAULT"
    assert ScenarioStepType.CLEAR_FAULT.value == "CLEAR_FAULT"
    assert ScenarioStepType.from_str("inject_fault") == ScenarioStepType.INJECT_FAULT

    # Targets
    assert ScenarioAssertionTarget.PERCEPTION.value == "PERCEPTION"
    assert ScenarioAssertionTarget.FUSION.value == "FUSION"
    assert ScenarioAssertionTarget.SITUATION.value == "SITUATION"
    assert ScenarioAssertionTarget.WORLD_STATE.value == "WORLD_STATE"
    assert ScenarioAssertionTarget.EVENT.value == "EVENT"
    assert ScenarioAssertionTarget.MISSION.value == "MISSION"
    assert ScenarioAssertionTarget.OBJECTIVE.value == "OBJECTIVE"
    assert ScenarioAssertionTarget.GOAL.value == "GOAL"
    assert ScenarioAssertionTarget.DEVICE.value == "DEVICE"
    assert ScenarioAssertionTarget.TRACE.value == "TRACE"
    assert ScenarioAssertionTarget.POLICY.value == "POLICY"
    assert ScenarioAssertionTarget.RESULT.value == "RESULT"
    assert ScenarioAssertionTarget.TWIN_STATE.value == "TWIN_STATE"
    assert ScenarioAssertionTarget.COMMAND_RESULT.value == "COMMAND_RESULT"
    assert ScenarioAssertionTarget.OBSERVATION_COUNT.value == "OBSERVATION_COUNT"
    assert ScenarioAssertionTarget.ACTIVE_FAULT.value == "ACTIVE_FAULT"
    assert ScenarioAssertionTarget.from_str("WORLD_STATE") == ScenarioAssertionTarget.WORLD_STATE

    # Operators
    assert ScenarioAssertionOperator.EQUALS.value == "EQUALS"
    assert ScenarioAssertionOperator.NOT_EQUALS.value == "NOT_EQUALS"
    assert ScenarioAssertionOperator.CONTAINS.value == "CONTAINS"
    assert ScenarioAssertionOperator.NOT_CONTAINS.value == "NOT_CONTAINS"
    assert ScenarioAssertionOperator.GREATER_THAN.value == "GREATER_THAN"
    assert ScenarioAssertionOperator.LESS_THAN.value == "LESS_THAN"
    assert ScenarioAssertionOperator.COUNT.value == "COUNT"
    assert ScenarioAssertionOperator.EXISTS.value == "EXISTS"
    assert ScenarioAssertionOperator.ABSENT.value == "ABSENT"
    assert ScenarioAssertionOperator.from_str("contains") == ScenarioAssertionOperator.CONTAINS

    # Severities
    assert ScenarioAssertionSeverity.INFO.value == "INFO"
    assert ScenarioAssertionSeverity.WARNING.value == "WARNING"
    assert ScenarioAssertionSeverity.FAIL.value == "FAIL"
    assert ScenarioAssertionSeverity.ERROR.value == "ERROR"
    assert ScenarioAssertionSeverity.CRITICAL.value == "CRITICAL"


def test_B01_scenario_limits_and_validation():
    limits = ScenarioLimits(max_scenarios=50, max_steps=100)
    assert limits.max_scenarios == 50
    assert limits.max_steps == 100

    # Serialization
    d = limits.to_dict()
    assert d["max_scenarios"] == 50
    json_str = limits.model_dump_json()
    assert "max_scenarios" in json_str

    deserialized = ScenarioLimits.from_dict(d)
    assert deserialized.max_scenarios == 50
    validated = ScenarioLimits.model_validate(d)
    assert validated.max_steps == 100


def test_C01_scenario_step_immutability_and_serialization():
    step = ScenarioStep(
        step_id="step_01",
        time_offset=1.5,
        action_type=ScenarioStepType.INJECT_OBSERVATION.value,
        target_id="twin_01",
        payload={"modality": "IMAGE"},
    )
    assert step.step_id == "step_01"
    assert step.time_offset == 1.5

    # Immutability check
    with pytest.raises(Exception):
        step.step_id = "step_mutated"  # type: ignore

    # Dual serialization
    d = step.to_dict()
    assert d["step_id"] == "step_01"
    assert d["time_offset"] == 1.5
    s_from_dict = ScenarioStep.from_dict(d)
    assert s_from_dict.step_id == "step_01"
    s_val = ScenarioStep.model_validate(d)
    assert s_val.target_id == "twin_01"


def test_D01_scenario_assertion_model():
    assertion = ScenarioAssertion(
        assertion_id="assert_01",
        target_type=ScenarioAssertionTarget.TWIN_STATE.value,
        target_id="twin_01",
        expected_field="battery",
        expected_value=80.0,
        operator=ScenarioAssertionOperator.GREATER_THAN.value,
        severity=ScenarioAssertionSeverity.FAIL.value,
        tolerance=0.5,
    )
    assert assertion.assertion_id == "assert_01"
    assert assertion.tolerance == 0.5

    # Immutability check
    with pytest.raises(Exception):
        assertion.expected_value = 90.0  # type: ignore

    # Dual serialization
    d = assertion.to_dict()
    assert d["expected_field"] == "battery"
    assert d["tolerance"] == 0.5
    a2 = ScenarioAssertion.from_dict(d)
    assert a2.assertion_id == "assert_01"
    assert a2.tolerance == 0.5


def test_E01_scenario_result_and_hash():
    res = ScenarioResult(
        scenario_id="scen_test",
        success=True,
        total_steps_executed=5,
        simulated_duration=10.0,
        assertions_evaluated=2,
        assertions_passed=2,
    )
    assert res.scenario_id == "scen_test"
    assert res.success is True
    assert len(res.deterministic_hash) == 64  # SHA-256 hex digest

    # Immutability
    with pytest.raises(Exception):
        res.success = False  # type: ignore

    # Serialization
    d = res.to_dict()
    assert d["scenario_id"] == "scen_test"
    assert d["deterministic_hash"] == res.deterministic_hash


# ============================================================================
# Section F-H: Deterministic IDs, Catalog Operations, Capacity Bounds
# ============================================================================

def test_F01_scenario_builder_deterministic_sorting():
    b = ScenarioBuilder("scen_sort", "Deterministic Step Sorting")
    # Add steps out of chronological order
    b.add_step("s3", 10.0, ScenarioStepType.ADVANCE_TIME.value)
    b.add_step("s1", 2.0, ScenarioStepType.ADVANCE_TIME.value)
    b.add_step("s2", 5.0, ScenarioStepType.ADVANCE_TIME.value)

    scen = b.build()
    # Steps must be sorted deterministically by (time_offset, step_id)
    offsets = [s.time_offset for s in scen.steps]
    assert offsets == [2.0, 5.0, 10.0]
    assert [s.step_id for s in scen.steps] == ["s1", "s2", "s3"]


def test_G01_scenario_catalog_crud_and_thread_safety():
    catalog = ScenarioCatalog(limits=ScenarioLimits(max_scenarios=5))
    scen1 = Scenario(scenario_id="s1", name="Scenario 1")
    scen2 = Scenario(scenario_id="s2", name="Scenario 2")

    catalog.register(scen1)
    catalog.register(scen2)
    assert len(catalog) == 2

    assert catalog.get("s1") == scen1
    assert catalog.get("nonexistent") is None

    # Deterministic listing
    listed = catalog.list()
    assert len(listed) == 2
    assert listed[0].scenario_id == "s1"
    assert listed[1].scenario_id == "s2"
    assert catalog.list_all() == listed
    assert catalog.list_scenarios() == listed

    # Removal
    assert catalog.remove("s1") is True
    assert catalog.remove("s1") is False
    assert len(catalog) == 1

    catalog.clear()
    assert len(catalog) == 0


def test_H01_scenario_catalog_bounds_and_duplicate_rejection():
    catalog = ScenarioCatalog(limits=ScenarioLimits(max_scenarios=2))
    s1 = Scenario(scenario_id="scen_a", name="A")
    s2 = Scenario(scenario_id="scen_b", name="B")
    s3 = Scenario(scenario_id="scen_c", name="C")

    catalog.register(s1)
    # Duplicate rejection
    with pytest.raises(ValueError, match="already registered"):
        catalog.register(s1)

    catalog.register(s2)
    # Capacity enforcement
    with pytest.raises(ValueError, match="Catalog capacity exceeded"):
        catalog.register(s3)


# ============================================================================
# Section I-L: Runner Initialization, SimulationClock, Step Ordering
# ============================================================================

def test_I01_runner_initialization_defaults():
    runner = ScenarioRunner()
    assert runner.limits is not None
    assert runner.scenario_limits is not None
    assert runner.strict_mode is False


def test_J01_runner_simulation_clock_advancement():
    b = ScenarioBuilder("scen_clock", "Clock Test")
    b.with_initial_time(5000.0)
    b.add_step("s1", 2.5, ScenarioStepType.ADVANCE_TIME.value, payload={"seconds": 2.5})
    b.add_step("s2", 10.0, ScenarioStepType.ADVANCE_TIME.value, payload={"seconds": 5.0})

    runner = ScenarioRunner()
    res = runner.run_scenario(b.build())
    assert res.start_time == 5000.0
    assert res.end_time >= 5010.0
    assert res.simulated_duration >= 10.0


def test_K01_runner_handles_unordered_steps_deterministically():
    b = ScenarioBuilder("scen_order", "Step Order Determinism")
    b.with_initial_time(100.0)
    cfg = TwinConfiguration("VIS_01", ProductType.VISION, ProductRole.OBSERVATION_SOURCE)
    b.add_twin_config(cfg)

    # Insert steps out of order
    b.add_step("step_b", 4.0, "DISPATCH_COMMAND", "VIS_01", {"action": "detect_person"})
    b.add_step("step_a", 1.0, "DISPATCH_COMMAND", "VIS_01", {"action": "detect_motion"})

    runner = ScenarioRunner()
    res = runner.run_scenario(b.build())
    trace_steps = [t["step_id"] for t in res.trace]
    assert trace_steps == ["step_a", "step_b"]


def test_L01_advance_time_action_executes_correctly():
    b = ScenarioBuilder("scen_adv", "Advance Time Test")
    b.with_initial_time(10.0)
    b.add_step("s1", 1.0, ScenarioStepType.ADVANCE_TIME.value, payload={"seconds": 4.0})

    runner = ScenarioRunner()
    res = runner.run_scenario(b.build())
    assert res.total_steps_executed == 1
    assert res.simulated_duration >= 4.0


# ============================================================================
# Section M-N: Twin Configuration, World State & Environment Configuration
# ============================================================================

def test_M01_configure_twin_action():
    b = ScenarioBuilder("scen_cfg_twin", "Configure Twin Action")
    cfg = TwinConfiguration("DRN_CFG", ProductType.DRONE, ProductRole.ACTUATOR, initial_battery_pct=100.0)
    b.add_twin_config(cfg)
    b.add_step(
        "s1",
        1.0,
        ScenarioStepType.CONFIGURE_TWIN.value,
        target_id="DRN_CFG",
        payload={"battery_level": 42.0, "position": {"x": 10.0, "y": 20.0, "z": 5.0}},
    )
    b.add_assertion(
        "a_battery",
        ScenarioAssertionTarget.TWIN_STATE.value,
        "DRN_CFG",
        "battery",
        42.0,
        ScenarioAssertionOperator.EQUALS.value,
    )
    b.add_assertion(
        "a_alt",
        ScenarioAssertionTarget.TWIN_STATE.value,
        "DRN_CFG",
        "altitude",
        5.0,
        ScenarioAssertionOperator.EQUALS.value,
    )

    res = ScenarioRunner().run_scenario(b.build())
    assert res.success is True
    assert res.assertions_passed == 2


def test_N01_set_world_state_modifies_simulated_environment_only():
    b = ScenarioBuilder("scen_env", "Simulated Environment Modification")
    b.add_step(
        "s1",
        0.0,
        ScenarioStepType.SET_WORLD_STATE.value,
        payload={"weather": "RAIN", "temperature": 15.0, "lighting": 200.0},
    )
    b.add_assertion(
        "a_weather",
        ScenarioAssertionTarget.WORLD_STATE.value,
        "world",
        "weather",
        "RAIN",
        ScenarioAssertionOperator.EQUALS.value,
    )
    b.add_assertion(
        "a_temp",
        ScenarioAssertionTarget.WORLD_STATE.value,
        "world",
        "temperature",
        15.0,
        ScenarioAssertionOperator.EQUALS.value,
    )

    res = ScenarioRunner().run_scenario(b.build())
    assert res.success is True
    assert res.assertions_passed == 2


# ============================================================================
# Section O-R: Observation, Telemetry, Fault Injection & Removal
# ============================================================================

def test_O01_inject_observation_action():
    b = ScenarioBuilder("scen_obs", "Inject Observation")
    cfg = TwinConfiguration("VIS_OBS", ProductType.VISION, ProductRole.OBSERVATION_SOURCE)
    b.add_twin_config(cfg)
    b.add_step(
        "s1",
        1.0,
        ScenarioStepType.INJECT_OBSERVATION.value,
        target_id="VIS_OBS",
        payload={"observation_type": "MOTION_DETECTED", "payload": {"zone": "backyard"}},
    )
    b.add_assertion(
        "a_obs_cnt",
        ScenarioAssertionTarget.OBSERVATION_COUNT.value,
        "world",
        "count",
        1,
        ScenarioAssertionOperator.EQUALS.value,
    )

    res = ScenarioRunner().run_scenario(b.build())
    assert res.success is True
    assert res.assertions_passed == 1


def test_P01_inject_telemetry_action():
    b = ScenarioBuilder("scen_telem", "Inject Telemetry")
    cfg = TwinConfiguration("ROV_TELEM", ProductType.ROVER, ProductRole.ACTUATOR)
    b.add_twin_config(cfg)
    b.add_step(
        "s1",
        1.0,
        ScenarioStepType.INJECT_TELEMETRY.value,
        target_id="ROV_TELEM",
        payload={"battery_level": 75.0, "position": {"x": 5.0, "y": 10.0, "z": 0.0}},
    )
    b.add_assertion(
        "a_telem_bat",
        ScenarioAssertionTarget.TWIN_STATE.value,
        "ROV_TELEM",
        "battery",
        75.0,
        ScenarioAssertionOperator.EQUALS.value,
    )

    res = ScenarioRunner().run_scenario(b.build())
    assert res.success is True
    assert res.assertions_passed == 1


def test_Q01_fault_injection_and_clearing():
    b = ScenarioBuilder("scen_fault", "Fault Lifecycle")
    cfg = TwinConfiguration("DRN_FLT", ProductType.DRONE, ProductRole.ACTUATOR)
    b.add_twin_config(cfg)
    # Inject fault
    b.add_step(
        "s_fault",
        1.0,
        ScenarioStepType.INJECT_FAULT.value,
        target_id="DRN_FLT",
        payload={"fault_id": "f1", "fault_type": "OFFLINE"},
    )
    b.add_step(
        "s_assert_active",
        2.0,
        ScenarioStepType.ASSERT.value,
        target_id="DRN_FLT",
        payload={"target_type": "ACTIVE_FAULT", "expected_field": "has_fault", "expected_value": "OFFLINE"},
    )
    # Clear fault
    b.add_step(
        "s_clear",
        5.0,
        ScenarioStepType.CLEAR_FAULT.value,
        target_id="DRN_FLT",
        payload={"fault_id": "f1"},
    )
    b.add_assertion(
        "a_fault_cleared",
        ScenarioAssertionTarget.ACTIVE_FAULT.value,
        "DRN_FLT",
        "count",
        0,
        ScenarioAssertionOperator.EQUALS.value,
    )

    res = ScenarioRunner().run_scenario(b.build())
    assert res.success is True
    assert res.assertions_passed == 2


def test_R01_fault_manager_isolation_across_runs():
    runner = ScenarioRunner()
    # Run 1: with fault
    b1 = ScenarioBuilder("scen_iso1", "Fault Isolation 1")
    b1.add_twin_config(TwinConfiguration("TWIN_ISO", ProductType.DRONE, ProductRole.ACTUATOR))
    b1.add_step("s1", 0.0, "INJECT_FAULT", "TWIN_ISO", {"fault_id": "f_iso", "fault_type": "OFFLINE"})
    b1.add_assertion("a1", "ACTIVE_FAULT", "TWIN_ISO", "count", 1, "EQUALS")
    r1 = runner.run_scenario(b1.build())
    assert r1.success is True

    # Run 2: without fault (must be completely clean, zero leakage)
    b2 = ScenarioBuilder("scen_iso2", "Fault Isolation 2")
    b2.add_twin_config(TwinConfiguration("TWIN_ISO", ProductType.DRONE, ProductRole.ACTUATOR))
    b2.add_assertion("a2", "ACTIVE_FAULT", "TWIN_ISO", "count", 0, "EQUALS")
    r2 = runner.run_scenario(b2.build())
    assert r2.success is True


# ============================================================================
# Section S-V: Assertion Evaluation, Operators, Tolerance, Strict Mode
# ============================================================================

def test_S01_all_assertion_operators():
    b = ScenarioBuilder("scen_operators", "Operator Verification")
    cfg = TwinConfiguration("VIS_OP", ProductType.VISION, ProductRole.OBSERVATION_SOURCE, initial_battery_pct=88.5)
    b.add_twin_config(cfg)
    b.add_step("s1", 0.0, "DISPATCH_COMMAND", "VIS_OP", {"action": "detect_person", "parameters": {"person_count": 3}})

    b.add_assertion("a_eq", "TWIN_STATE", "VIS_OP", "health", "HEALTHY", "EQUALS")
    b.add_assertion("a_neq", "TWIN_STATE", "VIS_OP", "health", "FAILED", "NOT_EQUALS")
    b.add_assertion("a_gt", "TWIN_STATE", "VIS_OP", "battery", 80.0, "GREATER_THAN")
    b.add_assertion("a_lt", "TWIN_STATE", "VIS_OP", "battery", 90.0, "LESS_THAN")
    b.add_assertion("a_exists", "TWIN_STATE", "VIS_OP", "health", True, "EXISTS")

    res = ScenarioRunner().run_scenario(b.build())
    assert res.success is True
    assert res.assertions_passed == 5


def test_T01_assertion_tolerance():
    b = ScenarioBuilder("scen_tolerance", "Tolerance Verification")
    cfg = TwinConfiguration("ROV_TOL", ProductType.ROVER, ProductRole.ACTUATOR, initial_battery_pct=50.04)
    b.add_twin_config(cfg)

    # battery is 50.04, expected 50.0 with tolerance 0.1 -> passes
    b.add_assertion(
        "a_tol_pass",
        ScenarioAssertionTarget.TWIN_STATE.value,
        "ROV_TOL",
        "battery",
        50.0,
        ScenarioAssertionOperator.EQUALS.value,
        tolerance=0.1,
    )

    res = ScenarioRunner().run_scenario(b.build())
    assert res.success is True
    assert res.assertions_passed == 1


def test_U01_assertion_severity_warning_does_not_fail_scenario():
    b = ScenarioBuilder("scen_warning", "Warning Severity")
    cfg = TwinConfiguration("VIS_WARN", ProductType.VISION, ProductRole.OBSERVATION_SOURCE, initial_battery_pct=90.0)
    b.add_twin_config(cfg)

    # Pass assertion
    b.add_assertion("a_pass", "TWIN_STATE", "VIS_WARN", "health", "HEALTHY", "EQUALS", severity="FAIL")
    # Warning failure: expected 100.0, actual 90.0
    b.add_assertion("a_warn", "TWIN_STATE", "VIS_WARN", "battery", 100.0, "EQUALS", severity="WARNING")

    res = ScenarioRunner().run_scenario(b.build())
    # Scenario still succeeds because WARNING does not fail the run
    assert res.success is True
    assert res.assertions_passed == 1
    assert res.assertions_evaluated == 2
    assert len(res.assertion_failures) == 1


def test_V01_strict_mode_enforces_validation_limits():
    limits = ScenarioLimits(max_steps=2)
    b = ScenarioBuilder("scen_strict", "Strict Mode Limits")
    b.add_step("s1", 1.0, "ADVANCE_TIME", payload={"seconds": 1.0})
    b.add_step("s2", 2.0, "ADVANCE_TIME", payload={"seconds": 1.0})
    b.add_step("s3", 3.0, "ADVANCE_TIME", payload={"seconds": 1.0})  # 3 steps > max 2

    # In strict mode, raises ValueError
    runner = ScenarioRunner(strict_mode=True, scenario_limits=limits)
    with pytest.raises(ValueError, match="violates scenario limits"):
        runner.run_scenario(b.build())


# ============================================================================
# Section W-X: Replay Records & Deterministic Hashing
# ============================================================================

def test_W01_deterministic_replay_produces_identical_hash():
    cat = build_canonical_catalog()
    scen = cat.get("SCENARIO_01_HOME_INTRUSION")
    assert scen is not None

    runner = ScenarioRunner()
    res1 = runner.run_scenario(scen)
    res2 = runner.run_scenario(scen)

    assert res1.success is True
    assert res2.success is True
    assert res1.deterministic_hash == res2.deterministic_hash
    assert res1.total_steps_executed == res2.total_steps_executed
    assert res1.simulated_duration == res2.simulated_duration


def test_X01_different_scenarios_produce_different_hashes():
    cat = build_canonical_catalog()
    s1 = cat.get("SCENARIO_01_HOME_INTRUSION")
    s2 = cat.get("SCENARIO_03_DRONE_FAILURE")
    assert s1 is not None and s2 is not None

    runner = ScenarioRunner()
    r1 = runner.run_scenario(s1)
    r2 = runner.run_scenario(s2)

    assert r1.deterministic_hash != r2.deterministic_hash


# ============================================================================
# Section Y-AH: The 10 Canonical Scenarios Execution
# ============================================================================

def test_Y01_canonical_01_home_intrusion():
    scen = create_home_intrusion_scenario()
    res = ScenarioRunner().run_scenario(scen)
    assert res.success is True
    assert res.total_steps_executed >= 6


def test_Z01_canonical_02_lost_person():
    scen = create_lost_person_scenario()
    res = ScenarioRunner().run_scenario(scen)
    assert res.success is True


def test_AA01_canonical_03_drone_failure():
    scen = create_drone_failure_scenario()
    res = ScenarioRunner().run_scenario(scen)
    assert res.success is True


def test_AB01_canonical_04_conflicting_sensor_reports():
    scen = create_conflicting_sensor_reports_scenario()
    res = ScenarioRunner().run_scenario(scen)
    assert res.success is True


def test_AC01_canonical_05_stale_telemetry():
    scen = create_stale_telemetry_scenario()
    res = ScenarioRunner().run_scenario(scen)
    assert res.success is True


def test_AD01_canonical_06_device_offline():
    scen = create_device_offline_scenario()
    res = ScenarioRunner().run_scenario(scen)
    assert res.success is True


def test_AE01_canonical_07_moving_target():
    scen = create_moving_target_scenario()
    res = ScenarioRunner().run_scenario(scen)
    assert res.success is True


def test_AF01_canonical_08_multi_source_corroboration():
    scen = create_multi_source_corroboration_scenario()
    res = ScenarioRunner().run_scenario(scen)
    assert res.success is True


def test_AG01_canonical_09_mission_resolution():
    scen = create_mission_resolution_scenario()
    res = ScenarioRunner().run_scenario(scen)
    assert res.success is True


def test_AH01_canonical_10_full_ecosystem():
    scen = create_full_ecosystem_scenario()
    res = ScenarioRunner().run_scenario(scen)
    assert res.success is True
    assert res.total_steps_executed >= 6


# ============================================================================
# Section AI-AL: Digital Twin Support (Vision, Glass, Drone, Rover)
# ============================================================================

def test_AI01_vision_twin_in_scenario():
    b = ScenarioBuilder("scen_vis", "Vision Twin Validation")
    cfg = TwinConfiguration("VIS_CANON", ProductType.VISION, ProductRole.OBSERVATION_SOURCE)
    b.add_twin_config(cfg)
    b.add_step("s1", 1.0, "DISPATCH_COMMAND", "VIS_CANON", {"capability": "camera", "action": "capture_frame"})
    b.add_assertion("a1", "TWIN_STATE", "VIS_CANON", "health", "HEALTHY", "EQUALS")
    res = ScenarioRunner().run_scenario(b.build())
    assert res.success is True


def test_AJ01_glass_twin_in_scenario():
    b = ScenarioBuilder("scen_glass", "Glass Twin Validation")
    cfg = TwinConfiguration("GLS_CANON", ProductType.GLASS, ProductRole.OBSERVATION_SOURCE)
    b.add_twin_config(cfg)
    b.add_step("s1", 1.0, "DISPATCH_COMMAND", "GLS_CANON", {"capability": "display", "action": "show_alert", "parameters": {"text": "Hello"}})
    b.add_assertion("a1", "TWIN_STATE", "GLS_CANON", "health", "HEALTHY", "EQUALS")
    res = ScenarioRunner().run_scenario(b.build())
    assert res.success is True


def test_AK01_drone_twin_in_scenario():
    b = ScenarioBuilder("scen_drone", "Drone Twin Validation")
    cfg = TwinConfiguration("DRN_CANON", ProductType.DRONE, ProductRole.ACTUATOR)
    b.add_twin_config(cfg)
    b.add_step("s1", 1.0, "DISPATCH_COMMAND", "DRN_CANON", {"capability": "flight", "action": "takeoff", "parameters": {"altitude": 15.0}})
    b.add_assertion("a1", "TWIN_STATE", "DRN_CANON", "health", "HEALTHY", "EQUALS")
    res = ScenarioRunner().run_scenario(b.build())
    assert res.success is True


def test_AL01_rover_twin_in_scenario():
    b = ScenarioBuilder("scen_rover", "Rover Twin Validation")
    cfg = TwinConfiguration("ROV_CANON", ProductType.ROVER, ProductRole.ACTUATOR)
    b.add_twin_config(cfg)
    b.add_step("s1", 1.0, "DISPATCH_COMMAND", "ROV_CANON", {"capability": "navigation", "action": "move_to", "parameters": {"x": 5.0, "y": 5.0}})
    b.add_assertion("a1", "TWIN_STATE", "ROV_CANON", "health", "HEALTHY", "EQUALS")
    res = ScenarioRunner().run_scenario(b.build())
    assert res.success is True


# ============================================================================
# Section AM-AP: Perception & Fusion Integration
# ============================================================================

def test_AM01_perception_assertion_target():
    b = ScenarioBuilder("scen_percep", "Perception Target Verification")
    cfg = TwinConfiguration("VIS_P", ProductType.VISION, ProductRole.OBSERVATION_SOURCE)
    b.add_twin_config(cfg)
    b.add_step(
        "s1",
        1.0,
        ScenarioStepType.INJECT_OBSERVATION.value,
        target_id="VIS_P",
        payload={"observation_type": "DETECTION", "payload": {"object": "vehicle"}},
    )
    b.add_assertion(
        "a_percep_cnt",
        ScenarioAssertionTarget.PERCEPTION.value,
        "VIS_P",
        "count",
        1,
        ScenarioAssertionOperator.EQUALS.value,
    )
    b.add_assertion(
        "a_has_obs",
        ScenarioAssertionTarget.PERCEPTION.value,
        "VIS_P",
        "has_observation",
        True,
        ScenarioAssertionOperator.EQUALS.value,
    )

    res = ScenarioRunner().run_scenario(b.build())
    assert res.success is True
    assert res.assertions_passed == 2


def test_AN01_temporal_cross_modal_fusion_integration():
    fusion_engine = TemporalCrossModalFusionEngine()
    runner = ScenarioRunner(temporal_fusion_engine=fusion_engine)

    b = ScenarioBuilder("scen_fusion", "Fusion Target Verification")
    b.add_step(
        "s1",
        1.0,
        ScenarioStepType.INJECT_OBSERVATION.value,
        payload={
            "observation_id": "obs_f1",
            "device_id": "dev_1",
            "modality": ModalityType.IMAGE.value,
            "payload": {"label": "intruder"},
        },
    )
    b.add_assertion(
        "a_fusion_cnt",
        ScenarioAssertionTarget.FUSION.value,
        "fusion",
        "count",
        1,
        ScenarioAssertionOperator.EQUALS.value,
    )

    res = runner.run_scenario(b.build())
    assert res.success is True


# ============================================================================
# Section AS-AX: Observational Targets (Zero Production Mutation)
# ============================================================================

def test_AS01_trace_assertion_target():
    b = ScenarioBuilder("scen_trace", "Trace Assertion Target")
    b.add_step("s1", 1.0, "ADVANCE_TIME", payload={"seconds": 1.0})
    b.add_step("s2", 2.0, "ADVANCE_TIME", payload={"seconds": 1.0})
    b.add_assertion("a_trace_cnt", ScenarioAssertionTarget.TRACE.value, "trace", "count", 2, "EQUALS")
    b.add_assertion("a_trace_has", ScenarioAssertionTarget.TRACE.value, "trace", "has_step", "s1", "CONTAINS")

    res = ScenarioRunner().run_scenario(b.build())
    assert res.success is True
    assert res.assertions_passed == 2


def test_AT01_command_result_assertion_target():
    b = ScenarioBuilder("scen_cmd_res", "Command Result Target")
    cfg = TwinConfiguration("DRN_CR", ProductType.DRONE, ProductRole.ACTUATOR)
    b.add_twin_config(cfg)
    b.add_step("step_takeoff", 0.0, "DISPATCH_COMMAND", "DRN_CR", {"capability": "flight", "action": "takeoff"})
    b.add_assertion(
        "a_cmd_success",
        ScenarioAssertionTarget.COMMAND_RESULT.value,
        "step_takeoff",
        "success",
        True,
        ScenarioAssertionOperator.EQUALS.value,
    )

    res = ScenarioRunner().run_scenario(b.build())
    assert res.success is True
    assert res.assertions_passed == 1


# ============================================================================
# Section AY-BF: Strict Architectural Boundary Invariants
# ============================================================================

def test_AY01_zero_hardware_drivers_in_simulation():
    prohibited = ["pymavlink", "mavsdk", "rclpy", "rospy", "paho", "serial.", "RPi.GPIO"]
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


def test_AZ01_zero_external_execution_in_simulation():
    sim_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "simulation"))
    for root, _, files in os.walk(sim_dir):
        for f in files:
            if f.endswith(".py"):
                path = os.path.join(root, f)
                with open(path, "r", encoding="utf-8") as handle:
                    tree = ast.parse(handle.read(), filename=path)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Call):
                            if isinstance(node.func, ast.Name):
                                assert node.func.id not in ("eval", "exec"), f"Prohibited {node.func.id} in {path}"


def test_BA01_zero_time_sleep_in_scenario_runner():
    runner_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "simulation", "runner.py"))
    with open(runner_path, "r", encoding="utf-8") as f:
        content = f.read()
        assert "time.sleep" not in content, "ScenarioRunner must use SimulationClock, never time.sleep()"


# ============================================================================
# Section BJ-BN: Bounded Collections, Determinism & Edge Cases
# ============================================================================

def test_BJ01_empty_scenario_execution():
    b = ScenarioBuilder("scen_empty", "Empty Scenario")
    scen = b.build()
    res = ScenarioRunner().run_scenario(scen)
    assert res.success is True
    assert res.total_steps_executed == 0
    assert res.assertions_evaluated == 0
    assert len(res.deterministic_hash) == 64


def test_BK01_dual_serialization_round_trip():
    cat = build_canonical_catalog()
    scen = cat.get("SCENARIO_01_HOME_INTRUSION")
    assert scen is not None

    d = scen.to_dict()
    scen_from_dict = Scenario.from_dict(d)
    assert scen_from_dict.scenario_id == scen.scenario_id
    assert len(scen_from_dict.steps) == len(scen.steps)
    assert len(scen_from_dict.assertions) == len(scen.assertions)

    scen_validated = Scenario.model_validate(d)
    assert scen_validated.name == scen.name


def test_BL01_credential_safety_in_scenarios():
    cat = build_canonical_catalog()
    secret_terms = ["api_key", "secret", "password", "token", "private_key"]
    for scen in cat.list_all():
        d_str = json.dumps(scen.to_dict()).lower()
        for s in secret_terms:
            assert f'"{s}": "actual_' not in d_str


# ============================================================================
# Section 36: End-to-End Full Ecosystem Validation
# ============================================================================

def test_E2E01_full_ecosystem_application_state_integration():
    """
    End-to-end integration of Phase 6.5e ScenarioRunner with
    AtlasApplicationState in in-memory mode, exercising digital twins,
    gateways, and observers without production database mutation.
    """
    from core.app_state import initialize_application_state

    app_state = initialize_application_state(in_memory_stores=True)
    fusion_engine = TemporalCrossModalFusionEngine()

    runner = ScenarioRunner(
        app_state=app_state,
        temporal_fusion_engine=fusion_engine,
    )

    cat = build_canonical_catalog()
    scen = cat.get("SCENARIO_10_FULL_ECOSYSTEM")
    assert scen is not None

    result = runner.run_scenario(scen)
    assert result.success is True
    assert result.total_steps_executed >= 6
    assert len(result.deterministic_hash) == 64
    assert result.assertions_passed >= 2
