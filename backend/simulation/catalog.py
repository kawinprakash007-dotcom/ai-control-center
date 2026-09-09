"""
ATLAS Phase 6.5e — Scenario Catalog & Canonical Library.

Provides:
- Thread-safe, bounded ScenarioCatalog registry
- 10 canonical deterministic multi-product simulation scenarios
- Deterministic iteration and replay compatibility
"""

from __future__ import annotations

import collections
import threading
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from core.models.device_contract import ProductRole, ProductType
from core.models.orchestration import GeoLocation, ModalityType
from core.models.scenario import (
    Scenario,
    ScenarioAssertion,
    ScenarioAssertionOperator,
    ScenarioAssertionSeverity,
    ScenarioAssertionTarget,
    ScenarioBuilder,
    ScenarioLimits,
    ScenarioStep,
    ScenarioStepType,
)
from core.models.simulation import (
    TwinConfiguration,
    TwinFaultType,
    TwinPosition,
)


class ScenarioCatalog:
    """
    Thread-safe, bounded catalog for registering and retrieving simulation scenarios.
    Enforces deterministic iteration and unique scenario identifiers.
    """

    def __init__(self, limits: Optional[ScenarioLimits] = None):
        self.limits = limits or ScenarioLimits()
        self._lock = threading.RLock()
        self._scenarios: Dict[str, Scenario] = collections.OrderedDict()

    def register(self, scenario: Scenario) -> None:
        """
        Register a scenario in the catalog.
        Rejects duplicate IDs and enforces capacity bounds.
        """
        with self._lock:
            if not scenario or not isinstance(scenario, Scenario):
                raise TypeError("Expected a valid Scenario instance.")
            sid = scenario.scenario_id
            if sid in self._scenarios:
                raise ValueError(f"Scenario with ID '{sid}' is already registered in catalog.")
            if len(self._scenarios) >= self.limits.max_scenarios:
                raise ValueError(
                    f"Catalog capacity exceeded (max {self.limits.max_scenarios} scenarios)."
                )
            self._scenarios[sid] = scenario

    def get(self, scenario_id: str) -> Optional[Scenario]:
        """Lookup a scenario by its unique identifier."""
        with self._lock:
            return self._scenarios.get(scenario_id)

    def list(self) -> Sequence[Scenario]:
        """
        List all registered scenarios sorted deterministically by scenario_id.
        """
        with self._lock:
            return tuple(sorted(self._scenarios.values(), key=lambda s: s.scenario_id))

    def list_scenarios(self) -> Sequence[Scenario]:
        """Alias for list()."""
        return self.list()

    def list_all(self) -> Sequence[Scenario]:
        """Alias for list()."""
        return self.list()

    def remove(self, scenario_id: str) -> bool:
        """Remove a scenario by ID. Returns True if removed, False if not found."""
        with self._lock:
            if scenario_id in self._scenarios:
                del self._scenarios[scenario_id]
                return True
            return False

    def clear(self) -> None:
        """Clear all scenarios from the catalog."""
        with self._lock:
            self._scenarios.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._scenarios)

    def __contains__(self, scenario_id: str) -> bool:
        with self._lock:
            return scenario_id in self._scenarios

    def __iter__(self) -> Iterator[Scenario]:
        with self._lock:
            return iter(tuple(sorted(self._scenarios.values(), key=lambda s: s.scenario_id)))


# ============================================================================
# Standard Digital Twin Configurations Helper
# ============================================================================

def make_standard_twin_configs() -> Dict[str, TwinConfiguration]:
    """Create baseline digital twin configurations for all 4 ATLAS edge products."""
    loc_base = TwinPosition(latitude=37.7749, longitude=-122.4194, altitude=10.0)
    return {
        "twin_vision_01": TwinConfiguration(
            twin_id="twin_vision_01",
            product_type=ProductType.VISION,
            product_role=ProductRole.OBSERVATION_SOURCE,
            initial_position=loc_base,
            initial_battery_pct=100.0,
            capabilities=("capture_frame", "detect_objects", "motion_detect"),
        ),
        "twin_glass_01": TwinConfiguration(
            twin_id="twin_glass_01",
            product_type=ProductType.GLASS,
            product_role=ProductRole.OBSERVATION_SOURCE,
            initial_position=TwinPosition(latitude=37.7748, longitude=-122.4195, altitude=1.7),
            initial_battery_pct=95.0,
            capabilities=("glance", "capture_speech", "display_hud"),
        ),
        "twin_drone_01": TwinConfiguration(
            twin_id="twin_drone_01",
            product_type=ProductType.DRONE,
            product_role=ProductRole.ACTUATOR,
            initial_position=TwinPosition(latitude=37.7750, longitude=-122.4190, altitude=30.0),
            initial_battery_pct=90.0,
            capabilities=("takeoff", "land", "goto_waypoint", "survey_aerial"),
        ),
        "twin_rover_01": TwinConfiguration(
            twin_id="twin_rover_01",
            product_type=ProductType.ROVER,
            product_role=ProductRole.ACTUATOR,
            initial_position=TwinPosition(latitude=37.7747, longitude=-122.4193, altitude=0.5),
            initial_battery_pct=85.0,
            capabilities=("navigate", "stop", "scan_lidar", "radar_ping"),
        ),
    }


# ============================================================================
# 10 Canonical Scenarios
# ============================================================================

def create_home_intrusion_scenario() -> Scenario:
    """
    Scenario 1: HOME_INTRUSION
    Multi-product edge coordination:
    T+0: Vision operational.
    T+5: Vision detects person at entrance.
    T+7: Drone spatial fix near entrance.
    T+8: Rover telemetry records gate perimeter movement.
    T+10: Glass wearer GPS is away from entrance.
    T+12: Advance time and evaluate cross-modal correlation.
    """
    twins = make_standard_twin_configs()
    b = ScenarioBuilder("SCENARIO_01_HOME_INTRUSION", "Home Intrusion Multi-Product Detection")
    b.with_initial_time(1000000.0)
    for cfg in twins.values():
        b.add_twin_config(cfg)

    # Steps
    b.add_step("step_01", 0.0, ScenarioStepType.ADVANCE_TIME, payload={"seconds": 5.0})
    b.add_step(
        "step_02",
        5.0,
        ScenarioStepType.INJECT_OBSERVATION,
        target_id="twin_vision_01",
        payload={
            "observation_id": "obs_vis_person",
            "modality": ModalityType.IMAGE.value,
            "payload": {"label": "person_intruder", "attributes": {"confidence": 0.95, "status": "detected"}},
        },
    )
    b.add_step(
        "step_03",
        7.0,
        ScenarioStepType.INJECT_OBSERVATION,
        target_id="twin_drone_01",
        payload={
            "observation_id": "obs_drone_gps",
            "modality": ModalityType.GPS.value,
            "payload": {"label": "entrance_perimeter", "attributes": {"altitude": 25.0}},
            "location": {"latitude": 37.7750, "longitude": -122.4192, "altitude": 25.0},
        },
    )
    b.add_step(
        "step_04",
        8.0,
        ScenarioStepType.INJECT_TELEMETRY,
        target_id="twin_rover_01",
        payload={"metrics": {"gate_movement": 1.0, "radar_distance": 4.5}},
    )
    b.add_step(
        "step_05",
        10.0,
        ScenarioStepType.INJECT_OBSERVATION,
        target_id="twin_glass_01",
        payload={
            "observation_id": "obs_glass_wearer",
            "modality": ModalityType.GPS.value,
            "payload": {"label": "wearer_away", "attributes": {"status": "away_from_entrance"}},
            "location": {"latitude": 37.7740, "longitude": -122.4180, "altitude": 2.0},
        },
    )
    b.add_step("step_06", 12.0, ScenarioStepType.ADVANCE_TIME, payload={"seconds": 2.0})

    # Assertions
    b.add_assertion(
        "assert_obs_count",
        ScenarioAssertionTarget.OBSERVATION_COUNT,
        "world",
        "count",
        3,
        ScenarioAssertionOperator.GREATER_THAN,
        ScenarioAssertionSeverity.FAIL,
    )
    b.add_assertion(
        "assert_vision_health",
        ScenarioAssertionTarget.TWIN_STATE,
        "twin_vision_01",
        "health",
        "HEALTHY",
        ScenarioAssertionOperator.EQUALS,
        ScenarioAssertionSeverity.FAIL,
    )
    return b.build()


def create_lost_person_scenario() -> Scenario:
    """
    Scenario 2: LOST_PERSON
    Multi-product search & correlation across Glass, Vision, Drone, and Rover.
    """
    twins = make_standard_twin_configs()
    b = ScenarioBuilder("SCENARIO_02_LOST_PERSON", "Lost Person Multi-Product Search")
    b.with_initial_time(1000000.0)
    for cfg in twins.values():
        b.add_twin_config(cfg)

    b.add_step(
        "step_01",
        1.0,
        ScenarioStepType.INJECT_OBSERVATION,
        target_id="twin_glass_01",
        payload={
            "observation_id": "obs_glass_lkl",
            "modality": ModalityType.GPS.value,
            "payload": {"label": "last_known_location", "attributes": {"entity_id": "person_lost"}},
            "location": {"latitude": 37.7760, "longitude": -122.4200, "altitude": 5.0},
        },
    )
    b.add_step(
        "step_02",
        3.0,
        ScenarioStepType.INJECT_OBSERVATION,
        target_id="twin_drone_01",
        payload={
            "observation_id": "obs_drone_sighting",
            "modality": ModalityType.IMAGE.value,
            "payload": {"label": "person_sighting", "attributes": {"entity_id": "person_lost"}},
            "location": {"latitude": 37.7761, "longitude": -122.4201, "altitude": 20.0},
        },
    )
    b.add_step(
        "step_02b",
        4.0,
        ScenarioStepType.INJECT_OBSERVATION,
        target_id="twin_rover_01",
        payload={
            "observation_id": "obs_rover_thermal",
            "modality": ModalityType.IMAGE.value,
            "payload": {"label": "thermal_signature", "attributes": {"entity_id": "person_lost"}},
            "location": {"latitude": 37.7762, "longitude": -122.4202, "altitude": 0.5},
        },
    )
    b.add_step("step_03", 5.0, ScenarioStepType.ADVANCE_TIME, payload={"seconds": 5.0})

    b.add_assertion(
        "assert_lost_person_evidence",
        ScenarioAssertionTarget.OBSERVATION_COUNT,
        "world",
        "count",
        2,
        ScenarioAssertionOperator.GREATER_THAN,
        ScenarioAssertionSeverity.FAIL,
    )
    return b.build()


def create_drone_failure_scenario() -> Scenario:
    """
    Scenario 3: DRONE_FAILURE
    Injects command failure into Drone and observes failure state / alternative readiness.
    """
    twins = make_standard_twin_configs()
    b = ScenarioBuilder("SCENARIO_03_DRONE_FAILURE", "Drone Failure & Alternate Readiness")
    b.with_initial_time(1000000.0)
    for cfg in twins.values():
        b.add_twin_config(cfg)

    b.add_step(
        "step_01",
        2.0,
        ScenarioStepType.INJECT_FAULT,
        target_id="twin_drone_01",
        payload={"fault_id": "fault_drone_cmd", "fault_type": TwinFaultType.COMMAND_FAILURE.value},
    )
    b.add_step(
        "step_02",
        3.0,
        ScenarioStepType.DISPATCH_COMMAND,
        target_id="twin_drone_01",
        payload={"capability": "takeoff", "action": "takeoff", "parameters": {"altitude": 10.0}},
    )
    b.add_step("step_03", 5.0, ScenarioStepType.ADVANCE_TIME, payload={"seconds": 2.0})

    b.add_assertion(
        "assert_drone_has_fault",
        ScenarioAssertionTarget.ACTIVE_FAULT,
        "twin_drone_01",
        "has_fault",
        TwinFaultType.COMMAND_FAILURE.value,
        ScenarioAssertionOperator.EQUALS,
        ScenarioAssertionSeverity.FAIL,
    )
    b.add_assertion(
        "assert_cmd_failed",
        ScenarioAssertionTarget.COMMAND_RESULT,
        "step_02",
        "success",
        False,
        ScenarioAssertionOperator.EQUALS,
        ScenarioAssertionSeverity.FAIL,
    )
    b.add_assertion(
        "assert_rover_healthy",
        ScenarioAssertionTarget.TWIN_STATE,
        "twin_rover_01",
        "health",
        "HEALTHY",
        ScenarioAssertionOperator.EQUALS,
        ScenarioAssertionSeverity.FAIL,
    )
    return b.build()


def create_conflicting_sensor_reports_scenario() -> Scenario:
    """
    Scenario 4: CONFLICTING_SENSOR_REPORTS
    Vision reports path clear; Rover reports obstacle detected at coincident time/location.
    """
    twins = make_standard_twin_configs()
    b = ScenarioBuilder("SCENARIO_04_CONFLICTING_SENSOR_REPORTS", "Conflicting Sensor Observations")
    b.with_initial_time(1000000.0)
    for cfg in twins.values():
        b.add_twin_config(cfg)

    loc = {"latitude": 37.7749, "longitude": -122.4194}
    b.add_step(
        "step_01",
        1.0,
        ScenarioStepType.INJECT_OBSERVATION,
        target_id="twin_vision_01",
        payload={
            "observation_id": "obs_vis_clear",
            "modality": ModalityType.IMAGE.value,
            "payload": {"label": "path_clear", "attributes": {"status": "clear", "speed": 10.0}},
            "location": loc,
        },
    )
    b.add_step(
        "step_02",
        1.02,
        ScenarioStepType.INJECT_OBSERVATION,
        target_id="twin_rover_01",
        payload={
            "observation_id": "obs_rover_blocked",
            "modality": ModalityType.TELEMETRY.value,
            "payload": {"label": "obstacle_detected", "attributes": {"status": "blocked", "speed": 0.0}},
            "location": loc,
        },
    )
    b.add_step("step_03", 2.0, ScenarioStepType.ADVANCE_TIME, payload={"seconds": 1.0})

    b.add_assertion(
        "assert_two_observations_ingested",
        ScenarioAssertionTarget.OBSERVATION_COUNT,
        "world",
        "count",
        2,
        ScenarioAssertionOperator.EQUALS,
        ScenarioAssertionSeverity.FAIL,
    )
    return b.build()


def create_stale_telemetry_scenario() -> Scenario:
    """
    Scenario 5: STALE_TELEMETRY
    Injects telemetry with historical timestamp to verify staleness handling.
    """
    twins = make_standard_twin_configs()
    b = ScenarioBuilder("SCENARIO_05_STALE_TELEMETRY", "Stale Telemetry Handling")
    b.with_initial_time(1000000.0)
    for cfg in twins.values():
        b.add_twin_config(cfg)

    # Ingest telemetry with timestamp 400s in the past relative to sim clock
    b.add_step(
        "step_01",
        10.0,
        ScenarioStepType.INJECT_OBSERVATION,
        target_id="twin_drone_01",
        payload={
            "observation_id": "obs_drone_stale_telem",
            "modality": ModalityType.TELEMETRY.value,
            "timestamp": 1000000.0 - 400.0,
            "payload": {"label": "battery_telemetry", "attributes": {"battery": 80.0}},
        },
    )
    b.add_step("step_02", 11.0, ScenarioStepType.ADVANCE_TIME, payload={"seconds": 1.0})

    b.add_assertion(
        "assert_stale_observation_present",
        ScenarioAssertionTarget.OBSERVATION_COUNT,
        "world",
        "count",
        1,
        ScenarioAssertionOperator.EQUALS,
        ScenarioAssertionSeverity.FAIL,
    )
    return b.build()


def create_device_offline_scenario() -> Scenario:
    """
    Scenario 6: DEVICE_OFFLINE
    Takes Drone offline and supplies Vision, Rover, and Glass observations.
    """
    twins = make_standard_twin_configs()
    b = ScenarioBuilder("SCENARIO_06_DEVICE_OFFLINE", "Device Offline Fault & Alternate Observation")
    b.with_initial_time(1000000.0)
    for cfg in twins.values():
        b.add_twin_config(cfg)

    b.add_step(
        "step_01",
        1.0,
        ScenarioStepType.INJECT_FAULT,
        target_id="twin_drone_01",
        payload={"fault_id": "fault_drone_off", "fault_type": TwinFaultType.OFFLINE.value},
    )
    b.add_step(
        "step_02",
        2.0,
        ScenarioStepType.INJECT_OBSERVATION,
        target_id="twin_vision_01",
        payload={
            "observation_id": "obs_vis_backup",
            "modality": ModalityType.IMAGE.value,
            "payload": {"label": "area_surveillance", "attributes": {"status": "clear"}},
        },
    )
    b.add_step("step_03", 3.0, ScenarioStepType.ADVANCE_TIME, payload={"seconds": 1.0})

    b.add_assertion(
        "assert_drone_offline",
        ScenarioAssertionTarget.ACTIVE_FAULT,
        "twin_drone_01",
        "has_fault",
        TwinFaultType.OFFLINE.value,
        ScenarioAssertionOperator.EQUALS,
        ScenarioAssertionSeverity.FAIL,
    )
    b.add_assertion(
        "assert_vision_online",
        ScenarioAssertionTarget.TWIN_STATE,
        "twin_vision_01",
        "connectivity",
        "ONLINE",
        ScenarioAssertionOperator.EQUALS,
        ScenarioAssertionSeverity.FAIL,
    )
    return b.build()


def create_moving_target_scenario() -> Scenario:
    """
    Scenario 7: MOVING_TARGET
    Generates target observations across t=0 (A), t=5 (B), t=10 (C).
    """
    twins = make_standard_twin_configs()
    b = ScenarioBuilder("SCENARIO_07_MOVING_TARGET", "Moving Target Kinematic Tracking")
    b.with_initial_time(1000000.0)
    for cfg in twins.values():
        b.add_twin_config(cfg)

    # Point A
    b.add_step(
        "step_01",
        0.0,
        ScenarioStepType.INJECT_OBSERVATION,
        target_id="twin_drone_01",
        payload={
            "observation_id": "obs_target_a",
            "modality": ModalityType.GPS.value,
            "payload": {"label": "tracked_target", "attributes": {"target_id": "veh_01", "waypoint": "A"}},
            "location": {"latitude": 37.7749, "longitude": -122.4194},
        },
    )
    # Point B
    b.add_step(
        "step_02",
        5.0,
        ScenarioStepType.INJECT_OBSERVATION,
        target_id="twin_drone_01",
        payload={
            "observation_id": "obs_target_b",
            "modality": ModalityType.GPS.value,
            "payload": {"label": "tracked_target", "attributes": {"target_id": "veh_01", "waypoint": "B"}},
            "location": {"latitude": 37.7759, "longitude": -122.4194},
        },
    )
    # Point C
    b.add_step(
        "step_03",
        10.0,
        ScenarioStepType.INJECT_OBSERVATION,
        target_id="twin_drone_01",
        payload={
            "observation_id": "obs_target_c",
            "modality": ModalityType.GPS.value,
            "payload": {"label": "tracked_target", "attributes": {"target_id": "veh_01", "waypoint": "C"}},
            "location": {"latitude": 37.7769, "longitude": -122.4194},
        },
    )
    b.add_step("step_04", 12.0, ScenarioStepType.ADVANCE_TIME, payload={"seconds": 2.0})

    b.add_assertion(
        "assert_three_waypoint_observations",
        ScenarioAssertionTarget.OBSERVATION_COUNT,
        "world",
        "count",
        3,
        ScenarioAssertionOperator.EQUALS,
        ScenarioAssertionSeverity.FAIL,
    )
    return b.build()


def create_multi_source_corroboration_scenario() -> Scenario:
    """
    Scenario 8: MULTI_SOURCE_CORROBORATION
    Coordinated multi-product observations: Vision (person), Drone (person), Rover (movement), Glass (location).
    """
    twins = make_standard_twin_configs()
    b = ScenarioBuilder("SCENARIO_08_MULTI_SOURCE_CORROBORATION", "Multi-Source Product Corroboration")
    b.with_initial_time(1000000.0)
    for cfg in twins.values():
        b.add_twin_config(cfg)

    loc = {"latitude": 37.7749, "longitude": -122.4194}
    b.add_step(
        "step_01",
        1.0,
        ScenarioStepType.INJECT_OBSERVATION,
        target_id="twin_vision_01",
        payload={
            "observation_id": "obs_vis_person",
            "modality": ModalityType.IMAGE.value,
            "payload": {"label": "person", "attributes": {"confidence": 0.95}},
            "location": loc,
        },
    )
    b.add_step(
        "step_02",
        1.05,
        ScenarioStepType.INJECT_OBSERVATION,
        target_id="twin_drone_01",
        payload={
            "observation_id": "obs_drone_person",
            "modality": ModalityType.IMAGE.value,
            "payload": {"label": "person", "attributes": {"confidence": 0.90}},
            "location": loc,
        },
    )
    b.add_step(
        "step_03",
        1.1,
        ScenarioStepType.INJECT_OBSERVATION,
        target_id="twin_rover_01",
        payload={
            "observation_id": "obs_rover_radar",
            "modality": ModalityType.TELEMETRY.value,
            "payload": {"label": "movement", "attributes": {"radar_distance": 2.5}},
            "location": loc,
        },
    )
    b.add_step(
        "step_04",
        1.15,
        ScenarioStepType.INJECT_OBSERVATION,
        target_id="twin_glass_01",
        payload={
            "observation_id": "obs_glass_gps",
            "modality": ModalityType.GPS.value,
            "payload": {"label": "wearer", "attributes": {"speed": 0.0}},
            "location": loc,
        },
    )
    b.add_step("step_05", 2.0, ScenarioStepType.ADVANCE_TIME, payload={"seconds": 1.0})

    b.add_assertion(
        "assert_four_observations_corroborated",
        ScenarioAssertionTarget.OBSERVATION_COUNT,
        "world",
        "count",
        4,
        ScenarioAssertionOperator.EQUALS,
        ScenarioAssertionSeverity.FAIL,
    )
    return b.build()


def create_mission_resolution_scenario() -> Scenario:
    """
    Scenario 9: MISSION_RESOLUTION
    Provides multi-source evidence supporting a tactical patrol/investigation mission.
    """
    twins = make_standard_twin_configs()
    b = ScenarioBuilder("SCENARIO_09_MISSION_RESOLUTION", "Tactical Mission Evidence Resolution")
    b.with_initial_time(1000000.0)
    for cfg in twins.values():
        b.add_twin_config(cfg)

    b.add_step(
        "step_01",
        1.0,
        ScenarioStepType.INJECT_OBSERVATION,
        target_id="twin_rover_01",
        payload={
            "observation_id": "obs_patrol_route",
            "modality": ModalityType.TELEMETRY.value,
            "payload": {"label": "patrol_checkpoint", "attributes": {"checkpoint_id": "CP_01", "status": "verified"}},
        },
    )
    b.add_step(
        "step_02",
        3.0,
        ScenarioStepType.INJECT_OBSERVATION,
        target_id="twin_drone_01",
        payload={
            "observation_id": "obs_aerial_survey",
            "modality": ModalityType.GPS.value,
            "payload": {"label": "survey_complete", "attributes": {"coverage": 1.0}},
        },
    )
    b.add_step("step_03", 5.0, ScenarioStepType.ADVANCE_TIME, payload={"seconds": 2.0})

    b.add_assertion(
        "assert_mission_observations_ingested",
        ScenarioAssertionTarget.OBSERVATION_COUNT,
        "world",
        "count",
        2,
        ScenarioAssertionOperator.EQUALS,
        ScenarioAssertionSeverity.FAIL,
    )
    b.add_assertion(
        "assert_rover_operational",
        ScenarioAssertionTarget.TWIN_STATE,
        "twin_rover_01",
        "health",
        "HEALTHY",
        ScenarioAssertionOperator.EQUALS,
        ScenarioAssertionSeverity.FAIL,
    )
    return b.build()


def create_full_ecosystem_scenario() -> Scenario:
    """
    Scenario 10: FULL_ECOSYSTEM_SCENARIO
    Complete multi-modal scenario exercising:
    Vision + Glass + Drone + Rover digital twins,
    Visual perception, spatial perception, telemetry perception,
    temporal fusion, cross-modal fusion, SituationFusion, and DeviceGateway.
    """
    twins = make_standard_twin_configs()
    b = ScenarioBuilder("SCENARIO_10_FULL_ECOSYSTEM", "Full ATLAS Multimodal Ecosystem Scenario")
    b.with_initial_time(1000000.0)
    for cfg in twins.values():
        b.add_twin_config(cfg)

    # 1. Setup initial environment entity
    b.add_step(
        "step_01",
        0.0,
        ScenarioStepType.SPAWN_ENTITY,
        payload={"entity_id": "suspect_vehicle", "entity_type": "VEHICLE", "position": {"latitude": 37.7749, "longitude": -122.4194}},
    )
    # 2. Vision detects vehicle
    b.add_step(
        "step_02",
        2.0,
        ScenarioStepType.INJECT_OBSERVATION,
        target_id="twin_vision_01",
        payload={
            "observation_id": "obs_eco_vis",
            "modality": ModalityType.IMAGE.value,
            "payload": {"label": "vehicle", "attributes": {"entity_id": "suspect_vehicle", "speed": 15.0}},
            "location": {"latitude": 37.7749, "longitude": -122.4194},
        },
    )
    # 3. Drone aerial GPS fix
    b.add_step(
        "step_03",
        2.1,
        ScenarioStepType.INJECT_OBSERVATION,
        target_id="twin_drone_01",
        payload={
            "observation_id": "obs_eco_drone",
            "modality": ModalityType.GPS.value,
            "payload": {"label": "vehicle", "attributes": {"entity_id": "suspect_vehicle"}},
            "location": {"latitude": 37.77492, "longitude": -122.41941, "altitude": 35.0},
        },
    )
    # 4. Rover radar telemetry
    b.add_step(
        "step_04",
        2.2,
        ScenarioStepType.INJECT_OBSERVATION,
        target_id="twin_rover_01",
        payload={
            "observation_id": "obs_eco_rover",
            "modality": ModalityType.TELEMETRY.value,
            "payload": {"label": "vehicle_radar", "attributes": {"radar_range": 12.0, "speed": 15.0}},
            "location": {"latitude": 37.7749, "longitude": -122.4194},
        },
    )
    # 5. Glass speech callout
    b.add_step(
        "step_05",
        2.3,
        ScenarioStepType.INJECT_OBSERVATION,
        target_id="twin_glass_01",
        payload={
            "observation_id": "obs_eco_glass",
            "modality": ModalityType.AUDIO_EVENT.value,
            "payload": {"label": "speech_transcript", "attributes": {"text": "Vehicle approaching north perimeter"}},
            "location": {"latitude": 37.7748, "longitude": -122.4195},
        },
    )
    # 6. Advance simulation time
    b.add_step("step_06", 5.0, ScenarioStepType.ADVANCE_TIME, payload={"seconds": 3.0})

    # Assertions
    b.add_assertion(
        "assert_eco_observations_ingested",
        ScenarioAssertionTarget.OBSERVATION_COUNT,
        "world",
        "count",
        4,
        ScenarioAssertionOperator.GREATER_THAN,
        ScenarioAssertionSeverity.FAIL,
    )
    b.add_assertion(
        "assert_all_four_twins_healthy",
        ScenarioAssertionTarget.TWIN_STATE,
        "twin_drone_01",
        "health",
        "HEALTHY",
        ScenarioAssertionOperator.EQUALS,
        ScenarioAssertionSeverity.FAIL,
    )
    return b.build()


def build_canonical_catalog(limits: Optional[ScenarioLimits] = None) -> ScenarioCatalog:
    """Construct and populate a ScenarioCatalog with all 10 canonical scenarios."""
    cat = ScenarioCatalog(limits=limits)
    cat.register(create_home_intrusion_scenario())
    cat.register(create_lost_person_scenario())
    cat.register(create_drone_failure_scenario())
    cat.register(create_conflicting_sensor_reports_scenario())
    cat.register(create_stale_telemetry_scenario())
    cat.register(create_device_offline_scenario())
    cat.register(create_moving_target_scenario())
    cat.register(create_multi_source_corroboration_scenario())
    cat.register(create_mission_resolution_scenario())
    cat.register(create_full_ecosystem_scenario())
    return cat
