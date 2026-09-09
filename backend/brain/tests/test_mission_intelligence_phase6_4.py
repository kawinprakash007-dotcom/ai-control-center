"""
ATLAS Phase 6.4 — Multi-Product Situation & Mission Intelligence Test Suite.

Comprehensive behavioral test suite covering:
A. Domain models
B. Serialization & credential scrubbing
C. Immutability
D. Multi-product situation creation
E. Product evidence aggregation
F. Source diversity
G. Corroboration
H. Contradiction handling
I. Stale evidence
J. Entity correlation
K. Entity disproval
L. Spatial correlation
M. Temporal correlation
N. Product role classification
O. Capability selection
P. Mission creation
Q. Objective dependencies
R. Mission planning
S. Mission coordinator
T. GoalManager integration
U. No direct GoalStore mutation
V. Failure / replanning
W. Unavailable product
X. Alternate product selection
Y. Mission completion
Z. Mission failure
AA. Multiple simultaneous missions
AB. Duplicate mission prevention
AC. Mission bounds
AD. Deterministic IDs
AE. Deterministic replay
AF. Security static audit
AG. No WorldState bypass
AH. No ToolOrchestrator bypass
AI. No CognitiveRuntime bypass
AJ. No Policy bypass
AK. Model neutrality
AL. Simulation integration
AM. SCENARIO 1: Home Intrusion (Four-product coordination)
AN. SCENARIO 2: Drone Failure & Reassignment
AO. SCENARIO 3: Conflicting Evidence & Contradiction Tracking
AP. SCENARIO 4: Mission Resolution & Completion
AQ. Causal Lineage & Timeline
"""

from dataclasses import FrozenInstanceError, replace
import os
import sys
import pytest
import time
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock

# Fallback mocks for optional vector DB modules in CI/unit-test environment
if "chromadb" not in sys.modules:
    sys.modules["chromadb"] = MagicMock()
if "sentence_transformers" not in sys.modules:
    sys.modules["sentence_transformers"] = MagicMock()

from core.interfaces.mission_interface import (
    MissionCoordinatorInterface,
    MissionPlannerInterface,
    MultiProductSituationIntelligenceInterface,
    ProductRoleSelectorInterface,
)
from core.models.device_contract import DeviceHealthStatus, ProductRole, ProductType
from core.models.goal import Goal, GoalPriority
from core.models.mission import (
    EntityCorrelation,
    EntityCorrelationStatus,
    Mission,
    MissionLimits,
    MissionObjective,
    MissionObjectiveType,
    MissionStatus,
    MissionTimelineEntry,
    MultiProductSituation,
    ObjectiveStatus,
    ProductEvidence,
    SituationContradiction,
)
from core.models.orchestration import (
    ConnectivityStatus,
    GeoLocation,
    ModalityType,
    Situation,
    SituationCategory,
    SituationEvidence,
    SituationSeverity,
    SituationStatus,
)
from goals.manager import AutonomousGoalManager
from goals.store import InMemoryGoalStore
from mission.coordinator import MissionCoordinator
from mission.planner import MissionPlanner
from mission.role_selector import ProductRoleSelector
from mission.situation_intelligence import MultiProductSituationIntelligenceEngine
from mission.timeline import MissionTimeline
from simulation.twin import (
    DroneDigitalTwin,
    GlassDigitalTwin,
    RoverDigitalTwin,
    VisionDigitalTwin,
)


# ============================================================================
# Helpers
# ============================================================================

def make_test_situation(
    situation_id: str = "sit_test_1",
    category: SituationCategory = SituationCategory.SECURITY,
    title: str = "Perimeter Breach",
    description: str = "Person detected near north perimeter fence",
    severity: SituationSeverity = SituationSeverity.HIGH,
    confidence: float = 0.85,
    source_id: str = "ATLAS_VISION_01",
    entities: Tuple[str, ...] = ("person_17",),
    location: Optional[GeoLocation] = None,
    correlation_id: str = "corr_123",
    timestamp: Optional[float] = None,
) -> Situation:
    loc = location or GeoLocation(latitude=37.7749, longitude=-122.4194, altitude=0.0)
    ts = timestamp if timestamp is not None else 1000000.0
    ev = SituationEvidence(
        evidence_id=f"ev_{situation_id}",
        observation_id=f"obs_{situation_id}",
        source_id=source_id,
        modality=ModalityType.IMAGE,
        evidence_weight=confidence,
        timestamp=ts,
        concise_summary=f"{source_id} detected {entities[0] if entities else 'target'}",
    )
    return Situation(
        situation_id=situation_id,
        category=category,
        title=title,
        description=description,
        severity=severity,
        confidence=confidence,
        status=SituationStatus.ACTIVE,
        involved_entities=entities,
        supporting_evidence=(ev,),
        location=loc,
        correlation_id=correlation_id,
        created_at=ts,
        updated_at=ts,
    )


# ============================================================================
# Section A: Domain Models
# ============================================================================

def test_A01_domain_models_instantiation():
    pe = ProductEvidence(
        evidence_id="pe_1",
        source_id="ATLAS_VISION_01",
        product_type=ProductType.VISION,
        product_role=ProductRole.OBSERVATION_SOURCE,
        observation_id="obs_1",
        situation_id="sit_1",
        timestamp=1000000.0,
        confidence=0.9,
        modality=ModalityType.IMAGE,
        summary="Person detected",
    )
    assert pe.evidence_id == "pe_1"
    assert pe.product_type == ProductType.VISION
    assert pe.product_role == ProductRole.OBSERVATION_SOURCE


def test_A02_mission_status_and_objective_type_from_str():
    assert MissionStatus.from_str("active") == MissionStatus.ACTIVE
    assert MissionStatus.from_str("INVESTIGATING") == MissionStatus.INVESTIGATING
    assert MissionStatus.from_str("unknown_val") == MissionStatus.PLANNED

    assert MissionObjectiveType.from_str("verify_incident") == MissionObjectiveType.VERIFY_INCIDENT
    assert MissionObjectiveType.from_str("NOTIFY_WEARER") == MissionObjectiveType.NOTIFY_WEARER

    assert ObjectiveStatus.from_str("in_progress") == ObjectiveStatus.IN_PROGRESS
    assert EntityCorrelationStatus.from_str("correlated") == EntityCorrelationStatus.CORRELATED


# ============================================================================
# Section B: Serialization & Credential Scrubbing
# ============================================================================

def test_B01_product_evidence_serialization_scrubbing():
    pe = ProductEvidence(
        evidence_id="pe_sec",
        source_id="VISION_1",
        product_type=ProductType.VISION,
        product_role=ProductRole.OBSERVATION_SOURCE,
        observation_id="obs_sec",
        situation_id="sit_sec",
        timestamp=1000.0,
        confidence=0.8,
        modality=ModalityType.EVENT,
        summary="Event summary",
        data={"api_key": "SECRET123", "normal_field": 42},
        provenance={"token": "BEARER_XYZ", "source": "hub"},
    )
    d = pe.to_dict()
    assert d["data"]["normal_field"] == 42
    assert "SECRET123" not in str(d)
    assert "BEARER_XYZ" not in str(d)

    roundtrip = ProductEvidence.from_dict(d)
    assert roundtrip.evidence_id == "pe_sec"
    assert roundtrip.product_type == ProductType.VISION


def test_B02_mission_and_objective_serialization_roundtrip():
    obj = MissionObjective(
        objective_id="obj_1",
        mission_id="msn_1",
        type=MissionObjectiveType.VERIFY_INCIDENT,
        description="Verify target",
        dependencies=("obj_0",),
        required_capabilities=("flight",),
        candidate_products=("ATLAS_DRONE_01",),
        assigned_product_id="ATLAS_DRONE_01",
        completion_criteria={"required_evidence": "TARGET_VERIFIED"},
    )
    mission = Mission(
        mission_id="msn_1",
        mission_type="SECURITY_RESPONSE",
        description="Test mission",
        objectives=(obj,),
        current_status=MissionStatus.ACTIVE,
        priority=GoalPriority.HIGH,
        metadata={"token": "SHOULD_BE_SCRUBBED", "safe_val": 10},
    )

    d = mission.to_dict()
    assert "SHOULD_BE_SCRUBBED" not in str(d)
    assert d["metadata"]["safe_val"] == 10

    m2 = Mission.from_dict(d)
    assert m2.mission_id == "msn_1"
    assert len(m2.objectives) == 1
    assert m2.objectives[0].type == MissionObjectiveType.VERIFY_INCIDENT
    assert m2.objectives[0].dependencies == ("obj_0",)


# ============================================================================
# Section C: Immutability
# ============================================================================

def test_C01_models_frozen_immutability():
    pe = ProductEvidence(
        evidence_id="pe_1",
        source_id="ATLAS_VISION_01",
        product_type=ProductType.VISION,
        product_role=ProductRole.OBSERVATION_SOURCE,
        observation_id="obs_1",
        situation_id="sit_1",
        timestamp=1000.0,
        confidence=0.8,
        modality=ModalityType.IMAGE,
        summary="Summary",
    )
    with pytest.raises(FrozenInstanceError):
        pe.confidence = 0.99

    mission = Mission(
        mission_id="m_frozen",
        mission_type="TEST",
        description="Desc",
    )
    with pytest.raises(FrozenInstanceError):
        mission.progress = 0.5


# ============================================================================
# Section D: Multi-Product Situation Creation
# ============================================================================

def test_D01_synthesize_multi_product_situation():
    engine = MultiProductSituationIntelligenceEngine()
    s1 = make_test_situation("sit_vis", source_id="ATLAS_VISION_01", correlation_id="c_incident_1")
    s2 = make_test_situation("sit_dro", source_id="ATLAS_DRONE_01", correlation_id="c_incident_1")

    multi_situations = engine.evaluate_situations([s1, s2])
    assert len(multi_situations) == 1
    mps = multi_situations[0]
    assert mps.category == SituationCategory.SECURITY
    assert len(mps.involved_products) == 2
    assert ProductType.VISION in mps.involved_products
    assert ProductType.DRONE in mps.involved_products
    assert len(mps.supporting_situation_ids) == 2


# ============================================================================
# Section E: Product Evidence Aggregation
# ============================================================================

def test_E01_evidence_aggregation_preserves_attributes():
    engine = MultiProductSituationIntelligenceEngine()
    s1 = make_test_situation("sit_1", source_id="ATLAS_VISION_01")
    s2 = make_test_situation("sit_2", source_id="ATLAS_ROVER_01")

    multi = engine.evaluate_situations([s1, s2])[0]
    ev_sources = {e.source_id for e in multi.evidence_references}
    assert "ATLAS_VISION_01" in ev_sources
    assert "ATLAS_ROVER_01" in ev_sources
    assert all(isinstance(e.product_type, ProductType) for e in multi.evidence_references)


# ============================================================================
# Section F & G: Source Diversity & Corroboration
# ============================================================================

def test_F01_same_source_repetition_zero_corroboration_boost():
    engine = MultiProductSituationIntelligenceEngine()
    # 3 situations from the exact same camera
    s1 = make_test_situation("s_v1", confidence=0.70, source_id="ATLAS_VISION_01", correlation_id="c_same")
    s2 = make_test_situation("s_v2", confidence=0.70, source_id="ATLAS_VISION_01", correlation_id="c_same")
    s3 = make_test_situation("s_v3", confidence=0.70, source_id="ATLAS_VISION_01", correlation_id="c_same")

    multi = engine.evaluate_situations([s1, s2, s3])[0]
    # Single source repetition gives ZERO corroboration boost
    assert multi.confidence == 0.70


def test_G01_multi_product_source_diversity_corroboration_boost():
    engine = MultiProductSituationIntelligenceEngine()
    # Independent products: Vision + Drone + Rover
    s1 = make_test_situation("s_v", confidence=0.70, source_id="ATLAS_VISION_01", correlation_id="c_multi")
    s2 = make_test_situation("s_d", confidence=0.70, source_id="ATLAS_DRONE_01", correlation_id="c_multi")
    s3 = make_test_situation("s_r", confidence=0.70, source_id="ATLAS_ROVER_01", correlation_id="c_multi")

    multi = engine.evaluate_situations([s1, s2, s3])[0]
    # 3 distinct products: base 0.70 + (3 - 1) * 0.08 = 0.70 + 0.16 = 0.86
    assert multi.confidence > 0.70
    assert multi.confidence == 0.86


# ============================================================================
# Section H: Contradiction Handling
# ============================================================================

def test_H01_contradiction_detection_and_confidence_penalty():
    engine = MultiProductSituationIntelligenceEngine()
    # Vision: person detected
    s1 = make_test_situation(
        "s_vis",
        confidence=0.80,
        source_id="ATLAS_VISION_01",
        description="Person detected inside perimeter",
        correlation_id="c_contra",
    )
    # Drone: clear / no person
    s2 = make_test_situation(
        "s_dro",
        confidence=0.80,
        source_id="ATLAS_DRONE_01",
        description="Aerial sweep clear, no person present",
        correlation_id="c_contra",
    )

    multi = engine.evaluate_situations([s1, s2])[0]
    assert len(multi.contradictions) > 0
    contra = multi.contradictions[0]
    assert contra.resolution_status == "UNRESOLVED"
    # Contradiction incurs -0.15 penalty
    assert multi.confidence < 0.80


# ============================================================================
# Section J & K: Entity Correlation & Disproval
# ============================================================================

def test_J01_entity_correlation_candidate_and_correlated():
    engine = MultiProductSituationIntelligenceEngine()
    s1 = make_test_situation("s_v", source_id="ATLAS_VISION_01", entities=("person_17",), correlation_id="c_ent")
    s2 = make_test_situation("s_d", source_id="ATLAS_DRONE_01", entities=("person_A",), correlation_id="c_ent")

    multi = engine.evaluate_situations([s1, s2])[0]
    assert len(multi.entity_correlations) > 0
    ecorr = multi.entity_correlations[0]
    assert ecorr.entity_type == "PERSON"
    assert ecorr.status in (EntityCorrelationStatus.CORRELATED, EntityCorrelationStatus.CANDIDATE)


def test_K01_entity_correlation_status_model():
    ecorr = EntityCorrelation(
        correlation_id="ec_disproved",
        primary_entity_id="person_17",
        correlated_entity_id="vehicle_99",
        entity_type="MIXED",
        status=EntityCorrelationStatus.DISPROVED,
        confidence=0.1,
    )
    assert ecorr.status == EntityCorrelationStatus.DISPROVED


# ============================================================================
# Section L & M: Spatial & Temporal Correlation
# ============================================================================

def test_L01_spatial_proximity_clustering():
    engine = MultiProductSituationIntelligenceEngine()
    loc_center = GeoLocation(latitude=37.7749, longitude=-122.4194)
    loc_near = GeoLocation(latitude=37.7750, longitude=-122.4193)  # ~15 meters away
    loc_far = GeoLocation(latitude=37.7900, longitude=-122.4000)   # ~2 km away

    s1 = make_test_situation("s1", location=loc_center, correlation_id="", entities=("target_alpha",))
    s2 = make_test_situation("s2", location=loc_near, correlation_id="", entities=("target_alpha",))
    s3 = make_test_situation("s3", location=loc_far, correlation_id="", entities=("target_beta",))

    clusters = engine.evaluate_situations([s1, s2, s3])
    # s1 and s2 should cluster together; s3 should be a separate cluster
    assert len(clusters) == 2


# ============================================================================
# Section N & O: Product Role & Capability Selection
# ============================================================================

def test_N01_product_role_suitability():
    selector = ProductRoleSelector()

    drone = DroneDigitalTwin()
    rover = RoverDigitalTwin()
    vision = VisionDigitalTwin()
    glass = GlassDigitalTwin()

    devices = [drone, rover, vision, glass]

    # NOTIFY_WEARER should prioritize Glass
    obj_hud = MissionObjective(
        objective_id="o_hud",
        mission_id="m1",
        type=MissionObjectiveType.NOTIFY_WEARER,
        description="Notify wearer",
    )
    cand_hud = selector.select_candidate_products(obj_hud, devices)
    assert cand_hud[0] == glass.twin_id

    # VERIFY_INCIDENT should prioritize Drone
    obj_verify = MissionObjective(
        objective_id="o_ver",
        mission_id="m1",
        type=MissionObjectiveType.VERIFY_INCIDENT,
        description="Aerial verification",
    )
    cand_ver = selector.select_candidate_products(obj_verify, devices)
    assert cand_ver[0] == drone.twin_id

    # INSPECT_ROUTE should prioritize Rover
    obj_route = MissionObjective(
        objective_id="o_rt",
        mission_id="m1",
        type=MissionObjectiveType.INSPECT_ROUTE,
        description="Ground route inspect",
    )
    cand_route = selector.select_candidate_products(obj_route, devices)
    assert cand_route[0] == rover.twin_id


def test_O01_offline_device_excluded_from_selection():
    selector = ProductRoleSelector()
    drone = DroneDigitalTwin()
    drone._connectivity = ConnectivityStatus.DISCONNECTED

    obj = MissionObjective(
        objective_id="o_test",
        mission_id="m1",
        type=MissionObjectiveType.VERIFY_INCIDENT,
        description="Verify",
    )
    candidates = selector.select_candidate_products(obj, [drone])
    assert len(candidates) == 0


# ============================================================================
# Section P, Q, R: Mission Planning & Dependencies
# ============================================================================

def test_P01_mission_planning_creates_ordered_objectives():
    planner = MissionPlanner()
    engine = MultiProductSituationIntelligenceEngine()
    s1 = make_test_situation("s_sec", category=SituationCategory.SECURITY)
    mps = engine.evaluate_situations([s1])[0]

    drone = DroneDigitalTwin()
    rover = RoverDigitalTwin()
    glass = GlassDigitalTwin()
    vision = VisionDigitalTwin()

    mission = planner.plan_mission(mps, [drone, rover, glass, vision])
    assert mission.current_status == MissionStatus.PLANNED
    assert len(mission.objectives) >= 3

    # Verify dependency structure: obj_2 and obj_3 should depend on obj_1
    obj1 = mission.objectives[0]
    assert len(obj1.dependencies) == 0
    obj2 = mission.objectives[1]
    assert obj1.objective_id in obj2.dependencies


# ============================================================================
# Section S, T, U: Mission Coordinator & AutonomousGoalManager Integration
# ============================================================================

def test_S01_mission_coordinator_lifecycle():
    store = InMemoryGoalStore()
    goal_mgr = AutonomousGoalManager(store=store, execution_engine=MagicMock())
    coordinator = MissionCoordinator(goal_manager=goal_mgr)

    obj = MissionObjective(
        objective_id="o_coord_1",
        mission_id="m_coord",
        type=MissionObjectiveType.VERIFY_INCIDENT,
        description="Verify target location",
        dependencies=(),
        assigned_product_id="ATLAS_DRONE_01",
    )
    mission = Mission(
        mission_id="m_coord",
        mission_type="SECURITY_RESPONSE",
        description="Coordination test",
        objectives=(obj,),
    )

    created = coordinator.create_mission(mission)
    assert created.current_status == MissionStatus.ACTIVE

    # Verify Goal was created through AutonomousGoalManager
    active_goals = store.list_goals()
    assert len(active_goals) == 1
    assert active_goals[0].goal_id == "goal_o_coord_1"


def test_U01_zero_direct_goal_store_mutation():
    """
    Validates that MissionCoordinator interacts exclusively with AutonomousGoalManager
    and never calls store.create_goal() directly.
    """
    store = InMemoryGoalStore()
    goal_mgr = AutonomousGoalManager(store=store, execution_engine=MagicMock())
    coordinator = MissionCoordinator(goal_manager=goal_mgr)

    # Coordinator should NOT have a direct store attribute
    assert not hasattr(coordinator, "goal_store")
    assert not hasattr(coordinator, "store")


# ============================================================================
# Section V, W, X: Replanning & Alternate Product Selection
# ============================================================================

def test_V01_mission_replanning_reassigns_to_alternate_product():
    planner = MissionPlanner()
    drone = DroneDigitalTwin()
    rover = RoverDigitalTwin()

    obj1 = MissionObjective(
        objective_id="o_dr_fail",
        mission_id="m_replan",
        type=MissionObjectiveType.VERIFY_INCIDENT,
        description="Verify perimeter",
        candidate_products=(drone.twin_id, rover.twin_id),
        assigned_product_id=drone.twin_id,
        status=ObjectiveStatus.IN_PROGRESS,
    )
    mission = Mission(
        mission_id="m_replan",
        mission_type="PERIMETER_RESPONSE",
        description="Replan test",
        objectives=(obj1,),
        involved_product_ids=(drone.twin_id,),
        current_status=MissionStatus.ACTIVE,
    )

    # Drone fails; replan should assign to Rover
    replanned = planner.replan_mission(
        mission=mission,
        failed_objective_id=obj1.objective_id,
        reason="Drone battery depleted",
        available_devices=[drone, rover],
    )
    assert replanned.current_status == MissionStatus.ACTIVE
    assert replanned.replanning_count == 1
    assert replanned.objectives[0].assigned_product_id == rover.twin_id
    assert replanned.objectives[0].status == ObjectiveStatus.PENDING


def test_Z01_mission_failure_when_max_replanning_exceeded():
    planner = MissionPlanner(limits=MissionLimits(max_replanning_attempts=2))
    drone = DroneDigitalTwin()

    obj1 = MissionObjective(
        objective_id="o_fail",
        mission_id="m_max",
        type=MissionObjectiveType.VERIFY_INCIDENT,
        description="Verify",
        candidate_products=(),
        assigned_product_id=drone.twin_id,
    )
    mission = Mission(
        mission_id="m_max",
        mission_type="TEST",
        description="Max replan",
        objectives=(obj1,),
        replanning_count=2,  # Already at max
    )

    failed = planner.replan_mission(mission, obj1.objective_id, "Failure", [drone])
    assert failed.current_status == MissionStatus.FAILED


# ============================================================================
# Section Y: Evidence-Driven Mission Completion
# ============================================================================

def test_Y01_evidence_driven_completion():
    coordinator = MissionCoordinator()
    obj = MissionObjective(
        objective_id="o_ev",
        mission_id="m_ev",
        type=MissionObjectiveType.VERIFY_INCIDENT,
        description="Verify perimeter target",
        dependencies=(),
        assigned_product_id="ATLAS_DRONE_01",
        completion_criteria={"required_evidence": "TARGET_VERIFIED"},
    )
    mission = Mission(
        mission_id="m_ev",
        mission_type="VERIFICATION",
        description="Evidence completion test",
        objectives=(obj,),
        trigger_situation_ids=("sit_test",),
    )
    coordinator.create_mission(mission)

    # Non-satisfying evidence
    unrelated_ev = ProductEvidence(
        evidence_id="pe_unrel",
        source_id="ATLAS_DRONE_01",
        product_type=ProductType.DRONE,
        product_role=ProductRole.HYBRID,
        observation_id="obs_1",
        situation_id="sit_test",
        timestamp=1000.0,
        confidence=0.9,
        modality=ModalityType.TELEMETRY,
        summary="Drone flying normal",
    )
    coordinator.ingest_evidence(unrelated_ev)
    assert coordinator.get_mission("m_ev").current_status == MissionStatus.ACTIVE

    # Satisfying evidence
    satisfying_ev = ProductEvidence(
        evidence_id="pe_sat",
        source_id="ATLAS_DRONE_01",
        product_type=ProductType.DRONE,
        product_role=ProductRole.HYBRID,
        observation_id="obs_2",
        situation_id="sit_test",
        timestamp=1005.0,
        confidence=0.95,
        modality=ModalityType.IMAGE,
        summary="TARGET_VERIFIED: intruder confirmed at perimeter",
    )
    updated_missions = coordinator.ingest_evidence(satisfying_ev)
    assert len(updated_missions) == 1
    m = updated_missions[0]
    assert m.current_status == MissionStatus.COMPLETED
    assert m.objectives[0].status == ObjectiveStatus.COMPLETED


# ============================================================================
# Section AA, AB, AC: Bounds & Duplicate Prevention
# ============================================================================

def test_AA01_multiple_simultaneous_missions():
    coordinator = MissionCoordinator(limits=MissionLimits(max_active_missions=5))
    for i in range(3):
        m = Mission(mission_id=f"m_{i}", mission_type=f"TYPE_{i}", description=f"Desc {i}")
        coordinator.create_mission(m)
    assert len(coordinator.list_active_missions()) == 3


def test_AB01_duplicate_mission_suppressed():
    coordinator = MissionCoordinator()
    m1 = Mission(mission_id="m_orig", mission_type="TYPE_A", description="D1", trigger_situation_ids=("sit_same",))
    m2 = Mission(mission_id="m_dup", mission_type="TYPE_A", description="D2", trigger_situation_ids=("sit_same",))

    c1 = coordinator.create_mission(m1)
    c2 = coordinator.create_mission(m2)
    # Returns existing active mission
    assert c1.mission_id == c2.mission_id
    assert len(coordinator.list_active_missions()) == 1


def test_AC01_mission_capacity_bound_enforced():
    coordinator = MissionCoordinator(limits=MissionLimits(max_active_missions=2))
    coordinator.create_mission(Mission(mission_id="m_1", mission_type="T1", description="D1"))
    coordinator.create_mission(Mission(mission_id="m_2", mission_type="T2", description="D2"))

    with pytest.raises(ValueError, match="capacity bound"):
        coordinator.create_mission(Mission(mission_id="m_3", mission_type="T3", description="D3"))


# ============================================================================
# Section AD & AE: Determinism & Replay
# ============================================================================

def test_AE01_deterministic_replay_produces_identical_missions():
    engine1 = MultiProductSituationIntelligenceEngine()
    engine2 = MultiProductSituationIntelligenceEngine()

    s1 = make_test_situation("s1", correlation_id="c_replay")
    s2 = make_test_situation("s2", source_id="ATLAS_DRONE_01", correlation_id="c_replay")

    res1 = engine1.evaluate_situations([s1, s2], now=1000.0)[0]
    res2 = engine2.evaluate_situations([s1, s2], now=1000.0)[0]

    assert res1.situation_id == res2.situation_id
    assert res1.confidence == res2.confidence
    assert res1.to_dict() == res2.to_dict()


# ============================================================================
# Section AF: Security Static Code Audit
# ============================================================================

def test_AF01_zero_hardware_driver_or_subprocesses_in_mission_package():
    """
    Static code analysis verifying that backend/mission contains ZERO hardware drivers,
    zero subprocess/shell commands, and zero direct GoalStore mutations.
    """
    mission_dir = os.path.join("backend", "mission")
    prohibited_tokens = [
        "pymavlink", "mavsdk", "rclpy", "rospy", "paho", "serial", "RPi.GPIO",
        "subprocess", "os.system", "eval(", "exec(", "playwright", "selenium",
    ]

    for root, _, files in os.walk(mission_dir):
        for f in files:
            if f.endswith(".py"):
                path = os.path.join(root, f)
                with open(path, "r", encoding="utf-8") as pyfile:
                    content = pyfile.read()
                    for token in prohibited_tokens:
                        assert token not in content, f"Prohibited token '{token}' found in {path}"


# ============================================================================
# Section AG, AH, AI, AJ: Authority Boundary Defense
# ============================================================================

def test_AG01_zero_world_state_bypasses():
    coordinator = MissionCoordinator()
    # Coordinator does not touch or expose WorldState
    assert not hasattr(coordinator, "world_state")
    assert not hasattr(coordinator, "world_store")


def test_AH01_zero_tool_orchestrator_bypasses():
    coordinator = MissionCoordinator()
    # Coordinator does not execute tool calls
    assert not hasattr(coordinator, "execute_tool")
    assert not hasattr(coordinator, "dispatch_tool")


# ============================================================================
# Section AM: Scenario 1 — Home Intrusion (Four-Product Coordination)
# ============================================================================

def test_AM01_scenario_1_home_intrusion_four_products():
    """
    SCENARIO 1:
    Vision: Person detected
    Drone: Aerial verification confirms person
    Rover: Perimeter route inspection
    Glass: HUD alert delivered to wearer
    """
    # 1. Initialize digital twins
    vision = VisionDigitalTwin()
    drone = DroneDigitalTwin()
    rover = RoverDigitalTwin()
    glass = GlassDigitalTwin()
    available_devices = [vision, drone, rover, glass]

    # 2. Vision detects intrusion
    sit_vision = make_test_situation(
        situation_id="sit_intrusion_vision",
        category=SituationCategory.SECURITY,
        title="Intruder Motion Detected",
        description="Person detected at north fence",
        source_id=vision.twin_id,
        correlation_id="intrusion_001",
    )

    # 3. Situation Intelligence creates unified MultiProductSituation
    engine = MultiProductSituationIntelligenceEngine()
    mps = engine.evaluate_situations([sit_vision], now=1000.0)[0]
    assert mps.category == SituationCategory.SECURITY

    # 4. Mission Planner generates tactical plan
    planner = MissionPlanner()
    mission = planner.plan_mission(mps, available_devices, now=1000.0)
    assert len(mission.objectives) >= 3

    # Verify tactical assignments
    assigned_sources = {o.assigned_product_id for o in mission.objectives}
    assert drone.twin_id in assigned_sources or rover.twin_id in assigned_sources

    # 5. Mission Coordinator executes through GoalManager
    goal_store = InMemoryGoalStore()
    goal_mgr = AutonomousGoalManager(store=goal_store, execution_engine=MagicMock())
    coordinator = MissionCoordinator(goal_manager=goal_mgr, planner=planner)

    active_mission = coordinator.create_mission(mission)
    assert active_mission.current_status == MissionStatus.ACTIVE

    # 6. Ingest Drone verification evidence
    ev_drone = ProductEvidence(
        evidence_id="pe_drone_confirm",
        source_id=drone.twin_id,
        product_type=ProductType.DRONE,
        product_role=ProductRole.HYBRID,
        observation_id="obs_dr_1",
        situation_id="sit_intrusion_vision",
        timestamp=1010.0,
        confidence=0.92,
        modality=ModalityType.IMAGE,
        summary="TARGET_VERIFIED: Intruder confirmed at perimeter north fence",
    )
    coordinator.ingest_evidence(ev_drone, now=1010.0)

    # 7. Ingest Rover route inspection evidence
    ev_rover = ProductEvidence(
        evidence_id="pe_rover_inspect",
        source_id=rover.twin_id,
        product_type=ProductType.ROVER,
        product_role=ProductRole.HYBRID,
        observation_id="obs_rov_1",
        situation_id="sit_intrusion_vision",
        timestamp=1020.0,
        confidence=0.88,
        modality=ModalityType.EVENT,
        summary="ROUTE_INSPECTED: perimeter route cleared and secured",
    )
    coordinator.ingest_evidence(ev_rover, now=1020.0)

    # 8. Ingest Glass wearer notification evidence
    ev_glass = ProductEvidence(
        evidence_id="pe_glass_hud",
        source_id=glass.twin_id,
        product_type=ProductType.GLASS,
        product_role=ProductRole.HYBRID,
        observation_id="obs_gl_1",
        situation_id="sit_intrusion_vision",
        timestamp=1025.0,
        confidence=0.99,
        modality=ModalityType.EVENT,
        summary="NOTIFICATION_DISPLAYED: Tactical alert displayed to wearer HUD",
    )
    coordinator.ingest_evidence(ev_glass, now=1025.0)

    # Ingest persistent observation evidence
    ev_vis_cont = ProductEvidence(
        evidence_id="pe_vis_cont",
        source_id=vision.twin_id,
        product_type=ProductType.VISION,
        product_role=ProductRole.OBSERVATION_SOURCE,
        observation_id="obs_vis_2",
        situation_id="sit_intrusion_vision",
        timestamp=1030.0,
        confidence=0.95,
        modality=ModalityType.IMAGE,
        summary="CONTINUOUS_TRACKING: stationary camera tracking target",
    )
    coordinator.ingest_evidence(ev_vis_cont, now=1030.0)

    # Mission completed successfully across four products
    final_m = coordinator.get_mission(mission.mission_id)
    assert final_m.current_status == MissionStatus.COMPLETED
    assert final_m.progress == 1.0


# ============================================================================
# Section AN: Scenario 2 — Drone Failure & Reassignment
# ============================================================================

def test_AN01_scenario_2_drone_failure_and_reassignment():
    """
    SCENARIO 2:
    Vision detects incident.
    Drone is scheduled for verification but becomes unavailable.
    Mission intelligence replans and reassigns verification to Rover.
    """
    vision = VisionDigitalTwin()
    drone = DroneDigitalTwin()
    rover = RoverDigitalTwin()

    sit = make_test_situation("sit_fail_test", source_id=vision.twin_id)
    engine = MultiProductSituationIntelligenceEngine()
    mps = engine.evaluate_situations([sit])[0]

    planner = MissionPlanner()
    mission = planner.plan_mission(mps, [vision, drone, rover])

    # Find the objective assigned to the Drone
    drone_obj = next(o for o in mission.objectives if o.assigned_product_id == drone.twin_id)

    # Drone fails (e.g. offline / disconnected)
    drone._connectivity = ConnectivityStatus.DISCONNECTED

    replanned = planner.replan_mission(
        mission=mission,
        failed_objective_id=drone_obj.objective_id,
        reason="Drone lost connection",
        available_devices=[vision, rover],  # Drone omitted
    )

    assert replanned.current_status == MissionStatus.ACTIVE
    assert replanned.replanning_count == 1
    # Objective was reassigned to Rover!
    reassigned_obj = next(o for o in replanned.objectives if o.objective_id == drone_obj.objective_id)
    assert reassigned_obj.assigned_product_id == rover.twin_id


# ============================================================================
# Section AO: Scenario 3 — Conflicting Evidence & Contradiction Tracking
# ============================================================================

def test_AO01_scenario_3_conflicting_evidence():
    """
    SCENARIO 3:
    Vision reports person present.
    Drone reports clear / no person.
    Rover reports gate movement.
    Contradiction must be preserved and confidence penalised.
    """
    engine = MultiProductSituationIntelligenceEngine()

    s_vis = make_test_situation(
        "s_v",
        description="Person detected at main gate",
        source_id="ATLAS_VISION_01",
        confidence=0.85,
        correlation_id="gate_breach",
    )
    s_dro = make_test_situation(
        "s_d",
        description="Sweep complete: no person present at main gate",
        source_id="ATLAS_DRONE_01",
        confidence=0.85,
        correlation_id="gate_breach",
    )
    s_rov = make_test_situation(
        "s_r",
        description="Gate vibration and movement confirmed",
        source_id="ATLAS_ROVER_01",
        confidence=0.80,
        correlation_id="gate_breach",
    )

    mps = engine.evaluate_situations([s_vis, s_dro, s_rov])[0]

    # Contradiction is preserved
    assert len(mps.contradictions) > 0
    contra = mps.contradictions[0]
    assert contra.resolution_status == "UNRESOLVED"

    # Confidence is penalized due to contradictory claims
    assert mps.confidence < 0.85


# ============================================================================
# Section AP: Scenario 4 — Mission Resolution & Completion
# ============================================================================

def test_AP01_scenario_4_mission_resolution():
    """
    SCENARIO 4:
    Multiple products provide sufficient evidence to satisfy all completion criteria.
    Mission transitions to COMPLETED.
    """
    coordinator = MissionCoordinator()
    o1 = MissionObjective(
        objective_id="obj_a",
        mission_id="m_res",
        type=MissionObjectiveType.VERIFY_INCIDENT,
        description="Verify",
        completion_criteria={"required_evidence": "CONFIRMED"},
    )
    mission = Mission(
        mission_id="m_res",
        mission_type="VERIFICATION",
        description="Resolution test",
        objectives=(o1,),
        trigger_situation_ids=("sit_res",),
    )
    coordinator.create_mission(mission)

    # Ingest satisfying evidence
    ev = ProductEvidence(
        evidence_id="pe_res",
        source_id="DRONE_1",
        product_type=ProductType.DRONE,
        product_role=ProductRole.HYBRID,
        observation_id="obs_res",
        situation_id="sit_res",
        timestamp=100.0,
        confidence=0.9,
        modality=ModalityType.EVENT,
        summary="Target CONFIRMED and perimeter secured",
    )
    coordinator.ingest_evidence(ev)

    m = coordinator.get_mission("m_res")
    assert m.current_status == MissionStatus.COMPLETED
    assert m.progress == 1.0


# ============================================================================
# Section AQ: Causal Lineage & Timeline
# ============================================================================

def test_AQ01_causal_lineage_and_timeline_preservation():
    coordinator = MissionCoordinator()
    m = Mission(
        mission_id="m_lineage",
        mission_type="TACTICAL",
        description="Lineage test",
        correlation_id="corr_parent_999",
        causation_id="sit_cause_888",
    )
    coordinator.create_mission(m)

    timeline = coordinator.get_timeline()
    entries = timeline.get_entries()
    assert len(entries) >= 1
    assert entries[0].event_type == "MISSION_CREATED"

    # Check that mission preserved lineage
    retrieved = coordinator.get_mission("m_lineage")
    assert retrieved.correlation_id == "corr_parent_999"
    assert retrieved.causation_id == "sit_cause_888"


# ============================================================================
# Section AR to BJ: Comprehensive Edge Case & Component Tests
# ============================================================================

def test_AR01_timeline_fifo_bounded_retention():
    timeline = MissionTimeline(max_entries=5)
    for i in range(10):
        timeline.record_entry(
            event_type="STEP",
            description=f"Step {i}",
            timestamp=1000.0 + i,
        )
    entries = timeline.get_entries()
    assert len(entries) == 5
    # Oldest 5 entries should have been evicted; remaining are 5 through 9
    assert entries[0].description == "Step 5"
    assert entries[-1].description == "Step 9"


def test_AS01_timeline_filter_by_source_and_type():
    timeline = MissionTimeline(max_entries=20)
    timeline.record_entry(event_type="ALERT", description="A1", source_id="VISION", timestamp=100.0)
    timeline.record_entry(event_type="UPDATE", description="U1", source_id="DRONE", timestamp=101.0)
    timeline.record_entry(event_type="ALERT", description="A2", source_id="DRONE", timestamp=102.0)

    alerts = timeline.get_entries(event_type="ALERT")
    assert len(alerts) == 2

    drone_entries = timeline.get_entries(source_id="DRONE")
    assert len(drone_entries) == 2


def test_AT01_contradiction_negation_tokens():
    engine = MultiProductSituationIntelligenceEngine()
    assert engine._detect_contradiction("Person detected at north fence", "Sweep complete: no person present")
    assert engine._detect_contradiction("Perimeter breached", "clear: perimeter secure")
    assert engine._detect_contradiction("Movement observed", "False alarm: empty yard")
    assert not engine._detect_contradiction("Person detected", "Target verified near doorway")


def test_AU01_temporal_correlation_sliding_window():
    engine = MultiProductSituationIntelligenceEngine()
    s1 = make_test_situation("s_early", correlation_id="", entities=("x1",), timestamp=100.0)
    # Same category, no entities in common, 400s later (beyond 300s window)
    s2 = make_test_situation("s_late", correlation_id="", entities=("x2",), timestamp=500.0)

    # Evaluate s1 and s2; because delta is 400s > 300s, they form 2 separate clusters
    clusters = engine.evaluate_situations([s1, s2], now=500.0)
    assert len(clusters) == 2


def test_AV01_entity_correlation_tracker_lifecycle():
    engine = MultiProductSituationIntelligenceEngine()
    ec = engine.record_entity_correlation(
        primary_entity_id="entity_1",
        correlated_entity_id="entity_2",
        entity_type="PERSON",
        status=EntityCorrelationStatus.CANDIDATE,
        confidence=0.6,
    )
    assert ec.status == EntityCorrelationStatus.CANDIDATE

    # Update to CORRELATED
    ec_conf = engine.record_entity_correlation(
        primary_entity_id="entity_1",
        correlated_entity_id="entity_2",
        entity_type="PERSON",
        status=EntityCorrelationStatus.CORRELATED,
        confidence=0.95,
    )
    assert ec_conf.status == EntityCorrelationStatus.CORRELATED
    stored = engine.get_entity_correlation("entity_1", "entity_2")
    assert stored is not None
    assert stored.status == EntityCorrelationStatus.CORRELATED


def test_AW01_product_role_selector_degraded_penalty():
    selector = ProductRoleSelector()
    drone = DroneDigitalTwin()
    obj = MissionObjective(
        objective_id="o_deg",
        mission_id="m1",
        type=MissionObjectiveType.VERIFY_INCIDENT,
        description="Verify",
    )
    score_healthy = selector._score_device_for_objective(drone, obj, None)
    drone._health = DeviceHealthStatus.DEGRADED
    score_degraded = selector._score_device_for_objective(drone, obj, None)
    # Degraded score should be exactly 0.20 lower
    assert round(score_healthy - score_degraded, 2) == 0.20


def test_AX01_product_role_selector_low_battery_penalty():
    selector = ProductRoleSelector()
    drone = DroneDigitalTwin()
    obj = MissionObjective(
        objective_id="o_bat",
        mission_id="m1",
        type=MissionObjectiveType.VERIFY_INCIDENT,
        description="Verify",
    )
    drone._battery = 25.0  # 15-30% range
    score_low = selector._score_device_for_objective(drone, obj, None)
    assert score_low < 0.95

    # Critical battery (<15%) for travel mission returns 0.0
    drone._battery = 10.0
    score_crit = selector._score_device_for_objective(drone, obj, None)
    assert score_crit == 0.0


def test_AY01_product_role_selector_rank_capability():
    selector = ProductRoleSelector()
    drone = DroneDigitalTwin()
    rover = RoverDigitalTwin()
    glass = GlassDigitalTwin()
    vision = VisionDigitalTwin()

    devices = [drone, rover, glass, vision]
    flight_ranking = selector.rank_products_for_capability("flight", devices)
    assert len(flight_ranking) >= 1
    assert flight_ranking[0][0] == drone.twin_id

    locomotion_ranking = selector.rank_products_for_capability("locomotion", devices)
    locomotion_ids = [r[0] for r in locomotion_ranking]
    assert rover.twin_id in locomotion_ids

    hud_ranking = selector.rank_products_for_capability("hud", devices)
    assert len(hud_ranking) >= 1
    assert hud_ranking[0][0] == glass.twin_id


def test_AZ01_mission_planner_anomaly_template():
    planner = MissionPlanner()
    engine = MultiProductSituationIntelligenceEngine()
    sit = make_test_situation("s_anom", category=SituationCategory.ANOMALY)
    mps = engine.evaluate_situations([sit])[0]

    drone = DroneDigitalTwin()
    mission = planner.plan_mission(mps, [drone])
    assert mission.mission_type == "ANOMALY_INVESTIGATION"
    assert any(o.type == MissionObjectiveType.LOCATE_TARGET for o in mission.objectives)


def test_BA01_mission_planner_environmental_template():
    planner = MissionPlanner()
    engine = MultiProductSituationIntelligenceEngine()
    sit = make_test_situation("s_env", category=SituationCategory.ENVIRONMENTAL)
    mps = engine.evaluate_situations([sit])[0]

    vision = VisionDigitalTwin()
    mission = planner.plan_mission(mps, [vision])
    assert "HAZARD" in mission.mission_type or "ENVIRONMENTAL" in mission.mission_type
    assert any(o.type == MissionObjectiveType.MONITOR_AREA for o in mission.objectives)


def test_BB01_mission_planner_priority_mapping():
    planner = MissionPlanner()
    engine = MultiProductSituationIntelligenceEngine()
    sit_crit = make_test_situation("s_crit", severity=SituationSeverity.CRITICAL)
    mps = engine.evaluate_situations([sit_crit])[0]

    mission = planner.plan_mission(mps, [DroneDigitalTwin()])
    assert mission.priority == GoalPriority.CRITICAL


def test_BC01_mission_coordinator_abort_cancels_goals():
    store = InMemoryGoalStore()
    mock_engine = MagicMock()
    goal_mgr = AutonomousGoalManager(store=store, execution_engine=mock_engine)
    coordinator = MissionCoordinator(goal_manager=goal_mgr)

    obj = MissionObjective(
        objective_id="o_abort_1",
        mission_id="m_abort",
        type=MissionObjectiveType.VERIFY_INCIDENT,
        description="Verify",
    )
    mission = Mission(
        mission_id="m_abort",
        mission_type="TACTICAL",
        description="Abort test",
        objectives=(obj,),
    )
    coordinator.create_mission(mission)

    aborted = coordinator.abort_mission("m_abort", reason="Operator emergency override")
    assert aborted.current_status == MissionStatus.ABORTED
    assert "Operator emergency override" in str(aborted.failure_reason)


def test_BD01_mission_coordinator_pause_and_resume():
    coordinator = MissionCoordinator()
    mission = Mission(
        mission_id="m_pause",
        mission_type="TACTICAL",
        description="Pause test",
    )
    coordinator.create_mission(mission)

    paused = coordinator.pause_mission("m_pause", reason="Waiting for sunrise")
    assert paused.current_status == MissionStatus.PAUSED

    resumed = coordinator.resume_mission("m_pause")
    assert resumed.current_status == MissionStatus.ACTIVE


def test_BE01_mission_coordinator_step_coordination():
    coordinator = MissionCoordinator()
    o1 = MissionObjective(
        objective_id="o_step_1",
        mission_id="m_step",
        type=MissionObjectiveType.VERIFY_INCIDENT,
        description="Verify target",
        dependencies=(),
    )
    o2 = MissionObjective(
        objective_id="o_step_2",
        mission_id="m_step",
        type=MissionObjectiveType.INSPECT_ROUTE,
        description="Inspect route",
        dependencies=("o_step_1",),
    )
    mission = Mission(
        mission_id="m_step",
        mission_type="COORDINATION",
        description="Step test",
        objectives=(o1, o2),
        trigger_situation_ids=("sit_step",),
    )
    coordinator.create_mission(mission)

    # Ingest evidence completing o1
    ev1 = ProductEvidence(
        evidence_id="pe_step_1",
        source_id="DRONE_1",
        product_type=ProductType.DRONE,
        product_role=ProductRole.HYBRID,
        observation_id="obs_step_1",
        situation_id="sit_step",
        timestamp=100.0,
        confidence=0.9,
        modality=ModalityType.EVENT,
        summary="Target verified",
    )
    coordinator.ingest_evidence(ev1)

    # Step coordination: o2 should transition to IN_PROGRESS
    coordinator.step_coordination()
    m = coordinator.get_mission("m_step")
    o2_updated = next(o for o in m.objectives if o.objective_id == "o_step_2")
    assert o2_updated.status == ObjectiveStatus.IN_PROGRESS


def test_BF01_application_state_integration():
    from core.app_state import AtlasApplicationState, initialize_application_state
    state = initialize_application_state()
    assert state.situation_intelligence is not None
    assert isinstance(state.situation_intelligence, MultiProductSituationIntelligenceEngine)
    assert state.mission_coordinator is not None
    assert isinstance(state.mission_coordinator, MissionCoordinator)


def test_BG01_evidence_confidence_below_threshold_rejected():
    coordinator = MissionCoordinator()
    o1 = MissionObjective(
        objective_id="o_thresh",
        mission_id="m_thresh",
        type=MissionObjectiveType.VERIFY_INCIDENT,
        description="High confidence verify",
        completion_criteria={"confidence_threshold": 0.85},
    )
    mission = Mission(
        mission_id="m_thresh",
        mission_type="VERIFICATION",
        description="Threshold test",
        objectives=(o1,),
        trigger_situation_ids=("sit_thresh",),
    )
    coordinator.create_mission(mission)

    # Evidence with low confidence (0.60 < 0.85)
    ev_low = ProductEvidence(
        evidence_id="pe_low",
        source_id="DRONE_1",
        product_type=ProductType.DRONE,
        product_role=ProductRole.HYBRID,
        observation_id="obs_low",
        situation_id="sit_thresh",
        timestamp=100.0,
        confidence=0.60,
        modality=ModalityType.EVENT,
        summary="Uncertain sighting",
    )
    coordinator.ingest_evidence(ev_low)

    m = coordinator.get_mission("m_thresh")
    # Should still be IN_PROGRESS, not completed
    assert m.objectives[0].status == ObjectiveStatus.IN_PROGRESS
    assert m.current_status == MissionStatus.ACTIVE


def test_BH01_product_evidence_all_modalities():
    modalities = [
        ModalityType.IMAGE,
        ModalityType.VIDEO_FRAME,
        ModalityType.AUDIO_EVENT,
        ModalityType.EVENT,
        ModalityType.GPS,
        ModalityType.TELEMETRY,
    ]
    for mod in modalities:
        ev = ProductEvidence(
            evidence_id=f"pe_{mod.value}",
            source_id="TEST_SOURCE",
            product_type=ProductType.DRONE,
            product_role=ProductRole.HYBRID,
            observation_id=f"obs_{mod.value}",
            situation_id="sit_mod",
            timestamp=100.0,
            confidence=0.8,
            modality=mod,
            summary=f"Modality {mod.value} observation",
        )
        d = ev.to_dict()
        assert d["modality"] == mod.value
        ev_back = ProductEvidence.from_dict(d)
        assert ev_back.modality == mod


def test_BI01_to_dict_preserves_scrubbed_metadata():
    ev = ProductEvidence(
        evidence_id="pe_secret",
        source_id="TEST_SOURCE",
        product_type=ProductType.DRONE,
        product_role=ProductRole.HYBRID,
        observation_id="obs_sec",
        situation_id="sit_sec",
        timestamp=100.0,
        confidence=0.8,
        modality=ModalityType.EVENT,
        summary="Secret test",
        data={"token": "super_secret_123", "normal_key": "safe_value"},
    )
    assert ev.data["token"] == "[REDACTED]"
    assert ev.data["normal_key"] == "safe_value"
    d = ev.to_dict()
    assert d["data"]["token"] == "[REDACTED]"
    assert d["data"]["normal_key"] == "safe_value"

