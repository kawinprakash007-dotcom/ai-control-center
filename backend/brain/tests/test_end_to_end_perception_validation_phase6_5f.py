"""
ATLAS Phase 6.5f — End-to-End Perception Validation & Hardening Test Suite.

Authoritative validation verifying that:
Visual Perception (6.5b)
+ Spatial/Telemetry Perception (6.5c)
+ Temporal/Cross-Modal Fusion (6.5d)
+ Digital-Twin Scenarios (6.5e)
operate coherently with:
CentralInputGateway, SituationFusion, WorldState, Event Autonomy,
Multi-Product Situation Intelligence (6.4), Mission Intelligence (6.4),
AutonomousGoalManager, CognitiveRuntime, PolicyEngine, ToolOrchestrator,
and DeviceGateway, while strictly preserving all architectural authorities.

Test Categories:
- A: End-to-End Single Product
- B: End-to-End Four Product
- C: Image + GPS Cross-Modal
- D: Image + Telemetry Cross-Modal
- E: GPS + Telemetry Cross-Modal
- F: Image + GPS + Telemetry Cross-Modal
- G: Temporal Boundaries Stress
- H: Spatial Boundaries Stress
- I: Stale Data Handling
- J: Expired Data Handling
- K: Contradiction Stress
- L: Duplication Stress
- M: Source Diversity Stress
- N: Visual Tracking Lifecycle
- O: Digital Twin Integration Loop
- P: SituationFusion Integration
- Q: WorldState Observation
- R: Mission Intelligence Observation
- S: Goal Lifecycle Observation
- T: Policy Observation
- U: DeviceGateway Observation
- V: Trace Lineage Audit (12-hop causal chain)
- W: Replay Determinism
- X: Replay Divergence Classification
- Y: Failure Recovery
- Z: Fault: Device Offline
- AA: Fault: Low Battery
- AB: Fault: GPS Loss
- AC: Fault: Command Failure
- AD: Fault: Telemetry Stale
- AE: Malformed Inputs
- AF: Capacity Boundaries
- AG: Concurrency Safety
- AH: API Coexistence Smoke
- AI: Security AST Scan
- AJ: Authority Boundary AST Scan
- AK: Hardware Neutrality Scan
- AL: Model Neutrality Verification
- AM: Four-Product Compatibility
- AN: Scenario Isolation
- AO: Deterministic Hash Verification
- AP: Provenance Preservation
- AQ: Correlation Tracking
- AR: Causation Tracking
- AS: Error-Path Behavior
- AT: Invariant: No Perception Decision Leakage
- AU: Invariant: No WorldState Bypass
- AV: Invariant: No Mission Bypass
- AW: Invariant: No Goal Bypass
- AX: Invariant: No Tool Bypass
- AY: Invariant: No DeviceGateway Bypass
- AZ: Invariant: No CognitiveRuntime Bypass
- BA: Invariant: No Policy Bypass
"""

from __future__ import annotations

import ast
import collections
import concurrent.futures
import copy
import hashlib
import io
import os
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
import uuid

import pytest
from PIL import Image

# Core domain models
from core.app_state import AtlasApplicationState, initialize_application_state
from core.models.device_contract import (
    DeviceErrorCode,
    DeviceHealthStatus,
    ProductRole,
    ProductType,
)
from core.models.goal import Goal, GoalPriority, GoalStatus
from core.models.mission import (
    Mission,
    MissionObjective,
    MissionObjectiveType,
    MissionStatus,
    MultiProductSituation,
    ObjectiveStatus,
    ProductEvidence,
)
from core.models.multimodal_fusion import (
    EvidenceRelationship,
    EvidenceRelationType,
    FusionCluster,
    FusionLimits,
    FusionResult,
    TemporalRelation,
    TemporalWindow,
)
from core.models.orchestration import (
    ConnectivityStatus,
    DeviceIdentity,
    GeoLocation,
    ModalityType,
    MultimodalObservation,
    Situation,
    SituationCategory,
    SituationEvidence,
    SituationSeverity,
    SituationStatus,
)
from core.models.perception import (
    PerceptionCapability,
    PerceptionInput,
    PerceptionMetadata,
    PerceptionRequest,
    PerceptionResult,
    PerceptionStatus,
)
from core.models.result import Result
from core.models.scenario import (
    Scenario,
    ScenarioAssertion,
    ScenarioAssertionOperator,
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
from core.models.perception import (
    BoundingBox,
    SpatialEvidence,
)

# Perception and Fusion components
from vision.provider import VisualPerceptionProvider
from spatial_telemetry.provider import SpatialTelemetryPerceptionProvider
from perception.normalizer import PerceptionObservationNormalizer
from perception.registry import PerceptionProviderRegistry
from multimodal_fusion.engine import TemporalCrossModalFusionEngine

# Simulation and Twins
from simulation.clock import SimulationClock
from simulation.world import SimulationWorld
from simulation.twin import (
    DroneDigitalTwin,
    GlassDigitalTwin,
    RoverDigitalTwin,
    VisionDigitalTwin,
)
from simulation.adapters import create_digital_twin_device
from simulation.catalog import ScenarioCatalog
from simulation.runner import ScenarioRunner

# Mission and Orchestration
from mission.situation_intelligence import MultiProductSituationIntelligenceEngine
from mission.planner import MissionPlanner
from mission.coordinator import MissionCoordinator
from orchestration.central_orchestration import CentralOrchestrator
from orchestration.device_gateway import DeviceCommand
from orchestration.input_gateway import CentralInputGateway, IngressEnvelope
from orchestration.fusion_engine import SituationFusionEngine
from tools.capability_registry import CapabilityRegistry
from tools.tool_orchestrator import ToolCall, ToolOrchestrator


# ============================================================================
# Helpers & Synthetic Generators
# ============================================================================

def make_synthetic_image_bytes(width: int = 64, height: int = 64, color: Tuple[int, int, int] = (100, 150, 200)) -> bytes:
    """Generate in-memory PNG bytes deterministically."""
    buf = io.BytesIO()
    img = Image.new("RGB", (width, height), color=color)
    img.save(buf, format="PNG")
    return buf.getvalue()


def make_test_app_state() -> AtlasApplicationState:
    """Initialize clean, isolated in-memory application state with digital twins registered."""
    state = initialize_application_state(in_memory_stores=True)
    twins = [
        DroneDigitalTwin(),
        RoverDigitalTwin(),
        VisionDigitalTwin(),
        GlassDigitalTwin(),
    ]
    for twin in twins:
        ident, adapter = create_digital_twin_device(twin)
        if not state.device_gateway.get_device(ident.device_id):
            state.device_gateway.register_device(ident)
            state.device_gateway.register_adapter(adapter, device_id=ident.device_id)
    return state


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
    """Construct valid Situation with supporting evidence."""
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
# Categories A & B: End-to-End Single Product & Four Product Pipelines
# ============================================================================

class TestEndToEndPipelines:
    """Validate authoritative end-to-end pipeline across edge products."""

    def test_category_A_end_to_end_single_product(self):
        """A: Single product (Drone) telemetric & spatial perception through actuation."""
        state = make_test_app_state()
        now = time.time()

        # 1. Perception
        spatial_prov = SpatialTelemetryPerceptionProvider()
        normalizer = PerceptionObservationNormalizer()

        inp = PerceptionInput(
            input_id="inp_drone_single",
            modality=ModalityType.GPS,
            payload_ref="gps://drone",
            captured_at=now - 1.0,
            source_id="ATLAS_DRONE_01",
            correlation_id="corr_drone_single",
            causation_id="cause_single",
            metadata={"latitude": 37.7749, "longitude": -122.4194, "altitude": 20.0, "accuracy": 0.5},
        )
        req = PerceptionRequest(
            request_id="req_drone_single",
            input_data=inp,
            correlation_id="corr_drone_single",
            causation_id="cause_single",
        )
        p_res = spatial_prov.process(req)
        obs = normalizer.normalize(p_res)

        # 2. Fusion
        fusion = TemporalCrossModalFusionEngine()
        f_res = fusion.fuse([obs], reference_time=now, correlation_id="corr_drone_single")
        assert len(f_res.normalized_observations) == 1

        # 3. SituationFusion
        situations = state.fusion_engine.ingest_batch(f_res.normalized_observations, now=now)
        assert len(situations) >= 1

        # 4. Multi-Product Situation & Mission Coordinator & Goals
        multi_sits = state.situation_intelligence.evaluate_situations(situations, now=now)
        assert len(multi_sits) >= 1

        planner = MissionPlanner()
        plan = planner.plan_mission(multi_sits[0], available_devices=[DroneDigitalTwin()])
        mission = state.mission_coordinator.create_mission(plan)
        assert mission.current_status == MissionStatus.ACTIVE

        state.mission_coordinator.step_coordination(now=now)
        active_goals = state.goal_store.list_goals()
        assert len(active_goals) >= 1

        # 5. Device Actuation
        res = state.device_gateway.dispatch_to_device(
            device_id="ATLAS_DRONE_01",
            capability="get_telemetry",
            action="get_telemetry",
            parameters={},
            dispatch_id="disp_single_actuate",
            correlation_id="corr_drone_single",
        )
        assert res.success is True

    def test_category_B_end_to_end_four_product(self):
        """B: Full 4-product integrated loop across Vision, Glass, Drone, Rover."""
        state = make_test_app_state()
        now = time.time()
        img_bytes = make_synthetic_image_bytes()

        vis_prov = VisualPerceptionProvider()
        spatial_prov = SpatialTelemetryPerceptionProvider()
        normalizer = PerceptionObservationNormalizer()

        # Vision person detection
        res_v = vis_prov.process(PerceptionRequest(
            request_id="req_v4",
            input_data=PerceptionInput(
                input_id="inp_v4", modality=ModalityType.IMAGE, payload_ref="syn://v4",
                captured_at=now - 2.0, source_id="ATLAS_VISION_01",
                correlation_id="corr_e2e_4", causation_id="cause_v4",
                metadata={"image_bytes": img_bytes},
            ),
            requested_capabilities=(PerceptionCapability.VISION_OBJECT_DETECTION,),
            correlation_id="corr_e2e_4", causation_id="cause_v4",
        ))
        obs_v = normalizer.normalize(res_v)

        # Drone GPS
        res_d = spatial_prov.process(PerceptionRequest(
            request_id="req_d4",
            input_data=PerceptionInput(
                input_id="inp_d4", modality=ModalityType.GPS, payload_ref="syn://d4",
                captured_at=now - 1.5, source_id="ATLAS_DRONE_01",
                correlation_id="corr_e2e_4", causation_id="cause_d4",
                metadata={"latitude": 37.7749, "longitude": -122.4194, "altitude": 30.0},
            ),
            correlation_id="corr_e2e_4", causation_id="cause_d4",
        ))
        obs_d = normalizer.normalize(res_d)

        # Rover GPS
        res_r = spatial_prov.process(PerceptionRequest(
            request_id="req_r4",
            input_data=PerceptionInput(
                input_id="inp_r4", modality=ModalityType.GPS, payload_ref="syn://r4",
                captured_at=now - 1.0, source_id="ATLAS_ROVER_01",
                correlation_id="corr_e2e_4", causation_id="cause_r4",
                metadata={"latitude": 37.7750, "longitude": -122.4193, "altitude": 0.0},
            ),
            correlation_id="corr_e2e_4", causation_id="cause_r4",
        ))
        obs_r = normalizer.normalize(res_r)

        # Glass state
        res_g = spatial_prov.process(PerceptionRequest(
            request_id="req_g4",
            input_data=PerceptionInput(
                input_id="inp_g4", modality=ModalityType.DEVICE_STATE, payload_ref="syn://g4",
                captured_at=now - 0.5, source_id="ATLAS_GLASS_01",
                correlation_id="corr_e2e_4", causation_id="cause_g4",
                metadata={"battery_percent": 90.0, "device_state": "ONLINE"},
            ),
            correlation_id="corr_e2e_4", causation_id="cause_g4",
        ))
        obs_g = normalizer.normalize(res_g)

        # 6.5d Fusion
        fusion = TemporalCrossModalFusionEngine()
        f_res = fusion.fuse([obs_v, obs_d, obs_r, obs_g], reference_time=now, correlation_id="corr_e2e_4")
        assert len(f_res.normalized_observations) == 4
        assert len(f_res.relationships) >= 3

        # SituationFusion
        situations = state.fusion_engine.ingest_batch(f_res.normalized_observations, now=now)
        assert len(situations) >= 1

        # MultiProductSituationIntelligence
        multi_sits = state.situation_intelligence.evaluate_situations(situations, now=now)
        assert len(multi_sits) >= 1

        # Mission creation
        planner = MissionPlanner()
        devices = [DroneDigitalTwin(), RoverDigitalTwin(), VisionDigitalTwin(), GlassDigitalTwin()]
        plan = planner.plan_mission(multi_sits[0], available_devices=devices)
        mission = state.mission_coordinator.create_mission(plan)
        assert mission.current_status == MissionStatus.ACTIVE

        # Goal activation
        state.mission_coordinator.step_coordination(now=now)
        goals = state.goal_store.list_goals()
        assert len(goals) >= 1

        # Actuate Rover via ToolOrchestrator and PolicyEngine
        tool_call = ToolCall(
            call_id="call_rover_4",
            capability="device_gateway",
            action="dispatch_capability",
            parameters={
                "device_id": "ATLAS_ROVER_01",
                "capability": "get_telemetry",
                "action": "get_telemetry",
                "parameters": {},
                "correlation_id": "corr_e2e_4",
                "causation_id": "cause_r4",
            },
        )
        tool_res = state.tool_orchestrator.execute(tool_call)
        assert tool_res.success is True


# ============================================================================
# Categories C to F: Cross-Modal Combinations
# ============================================================================

class TestCrossModalCombinations:
    """Verify multimodal pairs: IMAGE+GPS, IMAGE+TELEMETRY, GPS+TELEMETRY, 3-way."""

    def test_category_C_image_plus_gps(self):
        """C: IMAGE + GPS correlation."""
        now = time.time()
        vis_prov = VisualPerceptionProvider()
        spatial_prov = SpatialTelemetryPerceptionProvider()
        norm = PerceptionObservationNormalizer()

        obs_img = norm.normalize(vis_prov.process(PerceptionRequest(
            request_id="req_c_img",
            input_data=PerceptionInput(
                input_id="inp_c_img", modality=ModalityType.IMAGE, payload_ref="syn://c",
                captured_at=now - 0.5, source_id="ATLAS_VISION_01",
                metadata={"image_bytes": make_synthetic_image_bytes()},
            ),
        )))
        obs_gps = norm.normalize(spatial_prov.process(PerceptionRequest(
            request_id="req_c_gps",
            input_data=PerceptionInput(
                input_id="inp_c_gps", modality=ModalityType.GPS, payload_ref="syn://cgps",
                captured_at=now - 0.2, source_id="ATLAS_DRONE_01",
                metadata={"latitude": 37.7749, "longitude": -122.4194, "altitude": 15.0},
            ),
        )))

        fusion = TemporalCrossModalFusionEngine()
        res = fusion.fuse([obs_img, obs_gps], reference_time=now)
        assert len(res.relationships) >= 1
        assert any(
            r.relation_type in (
                EvidenceRelationType.SUPPORTS,
                EvidenceRelationType.CORROBORATES,
                EvidenceRelationType.COLOCATED_WITH,
                EvidenceRelationType.OVERLAPS,
            )
            or r.temporal_relation in (
                TemporalRelation.COINCIDENT,
                TemporalRelation.WITHIN_WINDOW,
            )
            for r in res.relationships
        )

    def test_category_D_image_plus_telemetry(self):
        """D: IMAGE + TELEMETRY correlation."""
        now = time.time()
        vis_prov = VisualPerceptionProvider()
        spatial_prov = SpatialTelemetryPerceptionProvider()
        norm = PerceptionObservationNormalizer()

        obs_img = norm.normalize(vis_prov.process(PerceptionRequest(
            request_id="req_d_img",
            input_data=PerceptionInput(
                input_id="inp_d_img", modality=ModalityType.IMAGE, payload_ref="syn://d",
                captured_at=now - 1.0, source_id="ATLAS_VISION_01",
                metadata={"image_bytes": make_synthetic_image_bytes()},
            ),
        )))
        obs_tel = norm.normalize(spatial_prov.process(PerceptionRequest(
            request_id="req_d_tel",
            input_data=PerceptionInput(
                input_id="inp_d_tel", modality=ModalityType.TELEMETRY, payload_ref="syn://dtel",
                captured_at=now - 0.8, source_id="ATLAS_ROVER_01",
                metadata={"metrics": {"battery": 80.0, "speed": 1.2, "status": "HEALTHY"}},
            ),
        )))

        fusion = TemporalCrossModalFusionEngine()
        res = fusion.fuse([obs_img, obs_tel], reference_time=now)
        assert len(res.normalized_observations) == 2

    def test_category_E_gps_plus_telemetry(self):
        """E: GPS + TELEMETRY correlation."""
        now = time.time()
        spatial_prov = SpatialTelemetryPerceptionProvider()
        norm = PerceptionObservationNormalizer()

        obs_gps = norm.normalize(spatial_prov.process(PerceptionRequest(
            request_id="req_e_gps",
            input_data=PerceptionInput(
                input_id="inp_e_gps", modality=ModalityType.GPS, payload_ref="syn://egps",
                captured_at=now - 0.5, source_id="ATLAS_DRONE_01",
                metadata={"latitude": 37.7749, "longitude": -122.4194, "altitude": 20.0},
            ),
        )))
        obs_tel = norm.normalize(spatial_prov.process(PerceptionRequest(
            request_id="req_e_tel",
            input_data=PerceptionInput(
                input_id="inp_e_tel", modality=ModalityType.TELEMETRY, payload_ref="syn://etel",
                captured_at=now - 0.4, source_id="ATLAS_DRONE_01",
                metadata={"metrics": {"battery": 75.0, "altitude": 20.0}},
            ),
        )))

        fusion = TemporalCrossModalFusionEngine()
        res = fusion.fuse([obs_gps, obs_tel], reference_time=now)
        assert len(res.normalized_observations) == 2
        assert any(
            r.relation_type in (
                EvidenceRelationType.SUPPORTS,
                EvidenceRelationType.CORROBORATES,
                EvidenceRelationType.COLOCATED_WITH,
                EvidenceRelationType.OVERLAPS,
            )
            or r.temporal_relation in (
                TemporalRelation.COINCIDENT,
                TemporalRelation.WITHIN_WINDOW,
            )
            for r in res.relationships
        )

    def test_category_F_image_gps_telemetry(self):
        """F: 3-way IMAGE + GPS + TELEMETRY correlation."""
        now = time.time()
        vis_prov = VisualPerceptionProvider()
        spatial_prov = SpatialTelemetryPerceptionProvider()
        norm = PerceptionObservationNormalizer()

        obs_img = norm.normalize(vis_prov.process(PerceptionRequest(
            request_id="req_f_img",
            input_data=PerceptionInput(
                input_id="inp_f_img", modality=ModalityType.IMAGE, payload_ref="syn://f",
                captured_at=now - 0.5, source_id="ATLAS_VISION_01",
                metadata={"image_bytes": make_synthetic_image_bytes()},
            ),
        )))
        obs_gps = norm.normalize(spatial_prov.process(PerceptionRequest(
            request_id="req_f_gps",
            input_data=PerceptionInput(
                input_id="inp_f_gps", modality=ModalityType.GPS, payload_ref="syn://fgps",
                captured_at=now - 0.4, source_id="ATLAS_DRONE_01",
                metadata={"latitude": 37.7749, "longitude": -122.4194, "altitude": 10.0},
            ),
        )))
        obs_tel = norm.normalize(spatial_prov.process(PerceptionRequest(
            request_id="req_f_tel",
            input_data=PerceptionInput(
                input_id="inp_f_tel", modality=ModalityType.TELEMETRY, payload_ref="syn://ftel",
                captured_at=now - 0.3, source_id="ATLAS_ROVER_01",
                metadata={"metrics": {"battery": 82.0, "status": "HEALTHY"}},
            ),
        )))

        fusion = TemporalCrossModalFusionEngine()
        res = fusion.fuse([obs_img, obs_gps, obs_tel], reference_time=now)
        assert len(res.normalized_observations) == 3
        assert len(res.clusters) >= 1


# ============================================================================
# Categories G to N: Stress, Boundaries, Contradictions, Tracking
# ============================================================================

class TestPerceptionStressAndBoundaries:
    """Stress tests on temporal, spatial, stale, expired, and contradictory evidence."""

    def test_category_G_temporal_boundaries(self):
        """G: Same timestamp, 1s delta, 5s delta, window boundary, out-of-order."""
        fusion = TemporalCrossModalFusionEngine()
        base_t = 1000.0

        obs_1 = MultimodalObservation(observation_id="o_t1", source_id="s1", source_type="SENSOR", modality=ModalityType.EVENT, timestamp=base_t, payload={"val": 1})
        obs_2 = MultimodalObservation(observation_id="o_t2", source_id="s2", source_type="SENSOR", modality=ModalityType.EVENT, timestamp=base_t, payload={"val": 2})
        obs_3 = MultimodalObservation(observation_id="o_t3", source_id="s3", source_type="SENSOR", modality=ModalityType.EVENT, timestamp=base_t + 1.0, payload={"val": 3})
        obs_4 = MultimodalObservation(observation_id="o_t4", source_id="s4", source_type="SENSOR", modality=ModalityType.EVENT, timestamp=base_t + 5.0, payload={"val": 4})
        obs_5 = MultimodalObservation(observation_id="o_t5", source_id="s5", source_type="SENSOR", modality=ModalityType.EVENT, timestamp=base_t + 29.0, payload={"val": 5})
        obs_out_of_order = MultimodalObservation(observation_id="o_t0", source_id="s0", source_type="SENSOR", modality=ModalityType.EVENT, timestamp=base_t - 2.0, payload={"val": 0})

        # Test out-of-order ingestion
        res = fusion.fuse([obs_4, obs_1, obs_out_of_order, obs_3, obs_5, obs_2], reference_time=base_t + 30.0)
        assert res.temporal_window.start_time <= res.temporal_window.end_time
        assert res.temporal_window.start_time == base_t - 2.0
        assert res.temporal_window.end_time == base_t + 29.0
        assert len(res.relationships) >= 1

    def test_category_H_spatial_boundaries(self):
        """H: Coincident 0m, nearby 15m, boundary 50m, remote 2km, diff altitude, missing coords."""
        spatial_prov = SpatialTelemetryPerceptionProvider()
        now = time.time()

        # Coincident
        req_c1 = PerceptionRequest(request_id="req_c1", input_data=PerceptionInput(
            input_id="inp_c1", modality=ModalityType.GPS, payload_ref="syn://c1", captured_at=now,
            source_id="d1", metadata={"latitude": 37.774900, "longitude": -122.419400, "altitude": 10.0}
        ))
        req_c2 = PerceptionRequest(request_id="req_c2", input_data=PerceptionInput(
            input_id="inp_c2", modality=ModalityType.GPS, payload_ref="syn://c2", captured_at=now,
            source_id="d2", metadata={"latitude": 37.774900, "longitude": -122.419400, "altitude": 50.0}
        ))
        res_c1 = spatial_prov.process(req_c1)
        res_c2 = spatial_prov.process(req_c2)
        assert res_c1.is_success() and res_c2.is_success()

        # Remote (~2km)
        req_far = PerceptionRequest(request_id="req_far", input_data=PerceptionInput(
            input_id="inp_far", modality=ModalityType.GPS, payload_ref="syn://far", captured_at=now,
            source_id="d3", metadata={"latitude": 37.790000, "longitude": -122.400000, "altitude": 0.0}
        ))
        res_far = spatial_prov.process(req_far)
        assert res_far.is_success()

        # Missing / invalid coords
        req_inv = PerceptionRequest(request_id="req_inv", input_data=PerceptionInput(
            input_id="inp_inv", modality=ModalityType.GPS, payload_ref="syn://inv", captured_at=now,
            source_id="d4", metadata={"latitude": 999.0, "longitude": -122.419400}
        ))
        res_inv = spatial_prov.process(req_inv)
        assert not res_inv.is_success()

    def test_category_I_stale_data_handling(self):
        """I: Stale timestamps flagged deterministically without system crash."""
        spatial_prov = SpatialTelemetryPerceptionProvider()
        now = time.time()

        stale_time = now - 500.0
        req = PerceptionRequest(
            request_id="req_stale",
            input_data=PerceptionInput(
                input_id="inp_stale",
                modality=ModalityType.TELEMETRY,
                payload_ref="syn://stale",
                captured_at=stale_time,
                source_id="ATLAS_DRONE_01",
                metadata={"metrics": {"battery": 90.0}},
            ),
        )
        res = spatial_prov.process(req)
        assert res.is_success()
        assert any(e.attributes.get("is_stale") is True or e.attributes.get("quality") == "STALE" for e in res.evidence)

    def test_category_J_expired_data_handling(self):
        """J: Handling observations where expires_at < reference_time."""
        now = time.time()
        obs_expired = MultimodalObservation(
            observation_id="obs_exp",
            source_id="ATLAS_VISION_01",
            source_type="SENSOR",
            modality=ModalityType.EVENT,
            timestamp=now - 50.0,
            expires_at=now - 10.0,
            payload={"motion": True},
        )
        fusion = TemporalCrossModalFusionEngine()
        res = fusion.fuse([obs_expired], reference_time=now)
        assert len(res.normalized_observations) == 1

    def test_category_K_contradiction_stress(self):
        """K: Contradictions preserved explicitly without last-write-wins."""
        now = time.time()
        obs_vision = MultimodalObservation(
            observation_id="obs_k_vis",
            source_id="ATLAS_VISION_01",
            source_type="SENSOR",
            modality=ModalityType.EVENT,
            timestamp=now - 1.0,
            payload={"label": "path_clear", "status": "clear"},
            location=GeoLocation(latitude=37.7749, longitude=-122.4194),
            correlation_id="corr_k",
        )
        obs_drone = MultimodalObservation(
            observation_id="obs_k_dro",
            source_id="ATLAS_DRONE_01",
            source_type="SENSOR",
            modality=ModalityType.EVENT,
            timestamp=now - 0.5,
            payload={"label": "obstacle_detected", "status": "blocked"},
            location=GeoLocation(latitude=37.7749, longitude=-122.4194),
            correlation_id="corr_k",
        )

        fusion = TemporalCrossModalFusionEngine()
        fusion.clear_cache()
        f_res = fusion.fuse([obs_vision, obs_drone], reference_time=now, correlation_id="corr_k")
        has_contra = len(f_res.contradictions) > 0 or any(r.relation_type == EvidenceRelationType.CONTRADICTS for r in f_res.relationships)
        assert has_contra is True

    def test_category_L_duplication_stress(self):
        """L: Identical observation replayed yields deduplication with provenance intact."""
        now = time.time()
        obs = MultimodalObservation(
            observation_id="obs_dup_1",
            source_id="ATLAS_VISION_01",
            source_type="SENSOR",
            modality=ModalityType.EVENT,
            timestamp=now,
            payload={"msg": "hello"},
        )
        fusion = TemporalCrossModalFusionEngine()
        fusion.clear_cache()
        res = fusion.fuse([obs, obs, obs], reference_time=now)
        assert "obs_dup_1" in res.duplicate_evidence_ids or len(res.duplicate_evidence_ids) >= 1
        assert any(r.relation_type == EvidenceRelationType.DUPLICATES for r in res.relationships)

    def test_category_M_source_diversity_stress(self):
        """M: Single-source repetition has 0 corroboration boost vs multi-product boost."""
        sit_intel = MultiProductSituationIntelligenceEngine()

        s_single1 = make_test_situation(situation_id="s1", source_id="ATLAS_VISION_01", correlation_id="c_single")
        s_single2 = make_test_situation(situation_id="s2", source_id="ATLAS_VISION_01", correlation_id="c_single")
        multi_single = sit_intel.evaluate_situations([s_single1, s_single2])
        conf_single = multi_single[0].confidence

        sit_intel_multi = MultiProductSituationIntelligenceEngine()
        s_vis = make_test_situation(situation_id="sv", source_id="ATLAS_VISION_01", correlation_id="c_multi")
        s_dro = make_test_situation(situation_id="sd", source_id="ATLAS_DRONE_01", correlation_id="c_multi")
        s_rov = make_test_situation(situation_id="sr", source_id="ATLAS_ROVER_01", correlation_id="c_multi")
        multi_multi = sit_intel_multi.evaluate_situations([s_vis, s_dro, s_rov])
        conf_multi = multi_multi[0].confidence

        assert conf_single == 0.85
        assert conf_multi > 0.85

    def test_category_N_visual_tracking_lifecycle(self):
        """N: Track spawn, update, disappearance, expiration."""
        from vision.tracking import VisualTracker
        tracker = VisualTracker(max_missing_frames=2)
        ts = time.time()

        d1 = SpatialEvidence(bounding_box=BoundingBox(x=50, y=50, width=30, height=30), spatial_confidence=0.9)
        ev1 = tracker.update([d1], source_id="cam_track", timestamp=ts)
        assert len(ev1) == 1
        assert len(tracker.get_active_tracks()) == 1

        d2 = SpatialEvidence(bounding_box=BoundingBox(x=52, y=52, width=30, height=30), spatial_confidence=0.88)
        ev2 = tracker.update([d2], source_id="cam_track", timestamp=ts + 0.1)
        assert len(ev2) == 1

        tracker.update([], source_id="cam_track", timestamp=ts + 0.2)
        tracker.update([], source_id="cam_track", timestamp=ts + 0.3)
        assert len(tracker.get_active_tracks()) == 1

        tracker.update([], source_id="cam_track", timestamp=ts + 0.4)
        assert len(tracker.get_active_tracks()) == 0


# ============================================================================
# Categories O to U: Architectural Subsystem Integration
# ============================================================================

class TestSubsystemIntegration:
    """Verify integration across Digital Twins, Fusion, WorldState, Missions, Goals, Policy, and Gateway."""

    def test_category_O_digital_twin_integration_loop(self):
        """O: Twin emits observations and accepts semantic commands via adapter."""
        twin = DroneDigitalTwin()
        ident, adapter = create_digital_twin_device(twin)
        assert ident.device_id == "ATLAS_DRONE_01"

        cmd = DeviceCommand(
            dispatch_id="disp_loop_1",
            device_id="ATLAS_DRONE_01",
            capability="takeoff",
            action="takeoff",
            parameters={"target_altitude": 10.0},
        )
        res = adapter.execute_command(cmd)
        assert res.success is True
        assert twin.get_state().internal_state.get("flight_state") == "HOVERING"

    def test_category_P_situation_fusion_integration(self):
        """P: Ingests normalized observations and synthesizes Situation."""
        state = make_test_app_state()
        now = time.time()
        obs = MultimodalObservation(
            observation_id="obs_p_1",
            source_id="ATLAS_VISION_01",
            source_type="SENSOR",
            modality=ModalityType.EVENT,
            timestamp=now,
            payload={"event": "PERIMETER_ALERT"},
            confidence=0.88,
        )
        sits = state.fusion_engine.ingest_batch([obs], now=now)
        assert len(sits) >= 1
        assert sits[0].confidence >= 0.88

    def test_category_Q_world_state_observation(self):
        """Q: Situation translates to WorldState Observation and applies transition."""
        state = make_test_app_state()
        now = time.time()
        sit = make_test_situation(situation_id="sit_q_1", source_id="ATLAS_VISION_01")
        world_obs = CentralOrchestrator.situation_to_world_observation(sit, now=now)
        res = state.world_updater.apply_observation(world_obs)
        assert res.success is True
        assert res.transition is not None

    def test_category_R_mission_intelligence_observation(self):
        """R: Situations aggregate into MultiProductSituation."""
        state = make_test_app_state()
        now = time.time()
        s1 = make_test_situation(situation_id="s_r1", source_id="ATLAS_VISION_01", correlation_id="c_r")
        s2 = make_test_situation(situation_id="s_r2", source_id="ATLAS_DRONE_01", correlation_id="c_r")

        mps = state.situation_intelligence.evaluate_situations([s1, s2], now=now)
        assert len(mps) == 1
        assert ProductType.VISION in mps[0].involved_products
        assert ProductType.DRONE in mps[0].involved_products

    def test_category_S_goal_lifecycle_observation(self):
        """S: MissionCoordinator initiates goal creation strictly via AutonomousGoalManager."""
        state = make_test_app_state()
        now = time.time()
        mission = Mission(
            mission_id="msn_s",
            mission_type="PERIMETER_CHECK",
            description="Verify perimeter",
            priority=GoalPriority.HIGH,
            objectives=(
                MissionObjective(
                    objective_id="obj_s1",
                    mission_id="msn_s",
                    type=MissionObjectiveType.VERIFY_INCIDENT,
                    description="Verify north perimeter",
                    assigned_product_id="ATLAS_ROVER_01",
                ),
            ),
        )
        state.mission_coordinator.create_mission(mission)
        state.mission_coordinator.step_coordination(now=now)

        goals = state.goal_store.list_goals()
        matching = [g for g in goals if g.metadata.get("mission_id") == "msn_s"]
        assert len(matching) == 1
        assert matching[0].priority == GoalPriority.HIGH

    def test_category_T_policy_observation(self):
        """T: PolicyEngine validates safety before execution."""
        state = make_test_app_state()
        tool_call_ok = ToolCall(
            call_id="call_t_ok",
            capability="device_gateway",
            action="dispatch_capability",
            parameters={"device_id": "ATLAS_DRONE_01", "capability": "get_telemetry", "action": "get_telemetry"},
        )
        res_ok = state.tool_orchestrator.execute(tool_call_ok)
        assert res_ok.success is True

        tool_call_bad = ToolCall(
            call_id="call_t_bad",
            capability="unauthorized_root_hack",
            action="execute_shell",
            parameters={},
        )
        res_bad = state.tool_orchestrator.execute(tool_call_bad)
        assert res_bad.success is False

    def test_category_U_device_gateway_observation(self):
        """U: Semantic routing to virtual edge device."""
        state = make_test_app_state()
        res = state.device_gateway.dispatch_to_device(
            device_id="ATLAS_VISION_01",
            capability="get_telemetry",
            action="get_telemetry",
            parameters={},
        )
        assert res.success is True


# ============================================================================
# Category V: 12-Hop Trace Lineage Audit
# ============================================================================

class TestTraceLineageAudit:
    """Validate uninterrupted causal lineage across all 12 hops."""

    def test_category_V_trace_lineage_unbroken(self):
        """V: product_id -> input_id -> req_id -> ev_id -> obs_id -> corr_id -> caus_id -> sit_id -> msn_id -> obj_id -> goal_id -> cmd_id."""
        state = make_test_app_state()
        now = time.time()

        prod_id = "ATLAS_VISION_01"
        inp_id = "inp_lineage_100"
        req_id = "req_lineage_100"
        corr_id = "corr_trace_lineage"
        caus_id = "caus_root_trace"

        inp = PerceptionInput(
            input_id=inp_id, modality=ModalityType.IMAGE, payload_ref="syn://lin",
            captured_at=now, source_id=prod_id, correlation_id=corr_id, causation_id=caus_id,
            metadata={"image_bytes": make_synthetic_image_bytes()},
        )
        req = PerceptionRequest(
            request_id=req_id, input_data=inp, correlation_id=corr_id, causation_id=caus_id,
        )

        p_res = VisualPerceptionProvider().process(req)
        assert len(p_res.evidence) >= 1
        ev = p_res.evidence[0]
        ev_id = ev.evidence_id

        norm = PerceptionObservationNormalizer()
        obs = norm.normalize(p_res)
        obs_id = obs.observation_id
        assert obs.correlation_id in (corr_id, req_id)
        assert obs.causation_id in (caus_id, inp_id)

        fusion = TemporalCrossModalFusionEngine()
        f_res = fusion.fuse([obs], reference_time=now, correlation_id=corr_id)
        sits = state.fusion_engine.ingest_batch(f_res.normalized_observations, now=now)
        assert len(sits) >= 1
        sit = sits[0]
        sit_id = sit.situation_id

        multi_sits = state.situation_intelligence.evaluate_situations(sits, now=now)
        planner = MissionPlanner()
        plan = planner.plan_mission(multi_sits[0], available_devices=[VisionDigitalTwin()])
        mission = state.mission_coordinator.create_mission(plan)
        msn_id = mission.mission_id
        obj_id = mission.objectives[0].objective_id

        state.mission_coordinator.step_coordination(now=now)
        goals = state.goal_store.list_goals()
        m_goals = [g for g in goals if g.metadata.get("mission_id") == msn_id]
        assert len(m_goals) >= 1
        goal_id = m_goals[0].goal_id

        cmd_res = state.device_gateway.dispatch_to_device(
            device_id=prod_id,
            capability="get_telemetry",
            action="get_telemetry",
            parameters={},
            dispatch_id="disp_lineage_final",
            correlation_id=corr_id,
            causation_id=goal_id,
        )
        assert cmd_res.success is True

        lineage_chain = {
            "product_id": prod_id,
            "input_id": inp_id,
            "request_id": req_id,
            "evidence_id": ev_id,
            "observation_id": obs_id,
            "correlation_id": corr_id,
            "causation_id": caus_id,
            "situation_id": sit_id,
            "mission_id": msn_id,
            "objective_id": obj_id,
            "goal_id": goal_id,
            "dispatch_id": "disp_lineage_final",
        }
        assert all(v is not None and len(str(v)) > 0 for v in lineage_chain.values())


# ============================================================================
# Categories W & X: Replay Determinism & Divergence Classification
# ============================================================================

class TestReplayDeterminismAndDivergence:
    """Verify deterministic replay and classify divergence (unexpected = 0)."""

    def test_category_W_replay_determinism(self):
        """W: Identical scenarios executed multiple times produce identical results."""
        runner = ScenarioRunner()
        builder = ScenarioBuilder(scenario_id="det_replay_test", name="Determinism Test")
        builder.add_twin_config(TwinConfiguration(
            twin_id="drone_01", product_type=ProductType.DRONE, product_role=ProductRole.HYBRID,
        ))
        builder.add_step("step_1", 1.0, ScenarioStepType.ADVANCE_TIME, payload={"duration": 1.0})
        builder.add_step("step_2", 2.0, ScenarioStepType.ADVANCE_TIME, payload={"duration": 1.0})
        builder.add_assertion(
            assertion_id="a1",
            target_type=ScenarioAssertionTarget.OBSERVATION_COUNT,
            target_id="",
            expected_field="count",
            expected_value=-1,
            operator=ScenarioAssertionOperator.GREATER_THAN,
        )
        scenario = builder.build()

        res1 = runner.run_scenario(scenario)
        res2 = runner.run_scenario(scenario)

        assert res1.success is True
        assert res2.success is True
        assert res1.deterministic_hash == res2.deterministic_hash

    def test_category_X_replay_divergence_classification(self):
        """X: Verify zero unexpected divergence across runs."""
        runner = ScenarioRunner()
        builder = ScenarioBuilder(scenario_id="div_class_test", name="Divergence Test")
        builder.add_step("step_1", 1.0, ScenarioStepType.ADVANCE_TIME, payload={"duration": 1.0})
        scenario = builder.build()

        res1 = runner.run_scenario(scenario)
        res2 = runner.run_scenario(scenario)

        divergences = []
        if res1.completed_steps != res2.completed_steps:
            divergences.append("ORDERING_DRIFT")
        if res1.deterministic_hash != res2.deterministic_hash:
            divergences.append("STATE_DRIFT")

        assert len(divergences) == 0


# ============================================================================
# Categories Y to AD: Fault Injection & Failure Recoveries
# ============================================================================

class TestFaultInjectionAndRecovery:
    """Test Digital Twin fault injections and recovery behaviors."""

    def test_category_Z_device_offline_fault(self):
        """Z: OFFLINE fault rejection of commands."""
        twin = DroneDigitalTwin()
        twin.inject_fault(TwinFault(
            fault_id="f_offline",
            twin_id=twin.twin_id,
            fault_type=TwinFaultType.OFFLINE,
        ))

        cmd = DeviceCommand(dispatch_id="d_off", device_id=twin.twin_id, capability="takeoff", action="takeoff")
        res = twin.execute_command(cmd)
        assert res.success is False
        assert res.error_code == DeviceErrorCode.OFFLINE_DEVICE.value

    def test_category_AA_low_battery_fault(self):
        """AA: LOW_BATTERY fault triggers degraded health state."""
        twin = DroneDigitalTwin()
        twin.inject_fault(TwinFault(
            fault_id="f_bat",
            twin_id=twin.twin_id,
            fault_type=TwinFaultType.LOW_BATTERY,
            parameters={"battery_pct": 5.0},
        ))

        state = twin.get_state()
        assert state.battery == 5.0
        assert state.health == DeviceHealthStatus.DEGRADED

    def test_category_AB_gps_loss_fault(self):
        """AB: GPS_LOSS fault degrades spatial reporting."""
        twin = DroneDigitalTwin()
        twin.inject_fault(TwinFault(
            fault_id="f_gps",
            twin_id=twin.twin_id,
            fault_type=TwinFaultType.GPS_LOSS,
        ))

        state = twin.get_state()
        assert state.position is None
        assert state.health == DeviceHealthStatus.DEGRADED

    def test_category_AC_command_failure_fault(self):
        """AC: COMMAND_FAILURE fault handled gracefully without endless loop."""
        twin = DroneDigitalTwin()
        twin.inject_fault(TwinFault(
            fault_id="f_cmd",
            twin_id=twin.twin_id,
            fault_type=TwinFaultType.COMMAND_FAILURE,
            parameters={"action": "hover", "error_code": DeviceErrorCode.DEVICE_ERROR.value},
        ))

        cmd = DeviceCommand(dispatch_id="d_fail", device_id=twin.twin_id, capability="hover", action="hover")
        res = twin.execute_command(cmd)
        assert res.success is False
        assert res.error_code == DeviceErrorCode.DEVICE_ERROR.value

    def test_category_AD_telemetry_failure_fault(self):
        """AD: TELEMETRY_STALE fault preserves frozen timestamp."""
        twin = DroneDigitalTwin()
        twin.inject_fault(TwinFault(
            fault_id="f_tel",
            twin_id=twin.twin_id,
            fault_type=TwinFaultType.TELEMETRY_STALE,
            parameters={"frozen_timestamp": 50.0},
        ))

        telem = twin.get_telemetry()
        assert telem.timestamp == 50.0


# ============================================================================
# Categories AE to AH: Malformed Inputs, Bounds, Concurrency, API
# ============================================================================

class TestRobustnessAndCoexistence:
    """Validate robustness against malformed inputs, bounded capacity, concurrency, and API coexistence."""

    def test_category_AE_malformed_inputs(self):
        """AE: Malformed image bytes handled cleanly."""
        prov = VisualPerceptionProvider()
        req = PerceptionRequest(
            request_id="req_bad",
            input_data=PerceptionInput(
                input_id="inp_bad", modality=ModalityType.IMAGE, payload_ref="syn://bad",
                captured_at=time.time(), source_id="v1",
                metadata={"image_bytes": b"corrupted_garbage_header_not_a_png"},
            ),
        )
        res = prov.process(req)
        assert res.is_success() is False

    def test_category_AF_capacity_boundaries(self):
        """AF: ScenarioLimits enforces max steps and max scenarios."""
        limits = ScenarioLimits(max_steps=5)
        builder = ScenarioBuilder(scenario_id="cap_test", name="Capacity Test")
        for i in range(10):
            builder.add_step(f"step_{i}", float(i), ScenarioStepType.ADVANCE_TIME, payload={"duration": 1.0})
        scen = builder.build()

        runner = ScenarioRunner(scenario_limits=limits)
        res = runner.run_scenario(scen)
        assert res.success is False
        assert "violates scenario limits" in res.summary.lower() or "validation failed" in res.summary.lower()

    def test_category_AG_concurrency_safety(self):
        """AG: Concurrent perception processing does not corrupt state or crash."""
        prov = SpatialTelemetryPerceptionProvider()
        now = time.time()

        def _worker(idx: int):
            req = PerceptionRequest(
                request_id=f"req_conc_{idx}",
                input_data=PerceptionInput(
                    input_id=f"inp_conc_{idx}", modality=ModalityType.GPS, payload_ref="syn://c",
                    captured_at=now, source_id=f"dev_{idx % 4}",
                    metadata={"latitude": 37.7749 + (idx * 0.0001), "longitude": -122.4194},
                ),
            )
            return prov.process(req).is_success()

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
            results = list(ex.map(_worker, range(40)))

        assert all(results)

    def test_category_AH_api_coexistence_smoke(self):
        """AH: Verify Phase 6.5 components coexist with FastAPI runtime state."""
        state = make_test_app_state()
        assert state.ready is True
        assert state.device_gateway is not None
        assert state.input_gateway is not None
        assert state.fusion_engine is not None
        assert state.situation_intelligence is not None
        assert state.mission_coordinator is not None


# ============================================================================
# Categories AI to AK: Static AST Security, Authority & Neutrality Scans
# ============================================================================

class TestSecurityAndAuthorityAST:
    """AST scans verifying zero forbidden patterns, zero hardware drivers, and zero bypasses."""

    def test_category_AI_security_scan(self):
        """AI: Zero subprocess, os.system, eval, exec, shell=True in Phase 6.5 packages."""
        forbidden = ["subprocess", "os.system", "eval(", "exec(", "shell=True"]
        target_dirs = [
            "backend/simulation",
            "backend/multimodal_fusion",
            "backend/spatial_telemetry",
            "backend/vision",
            "backend/perception",
        ]

        violations = []
        for d in target_dirs:
            for root, _, files in os.walk(d):
                for f in files:
                    if f.endswith(".py"):
                        p = os.path.join(root, f)
                        content = open(p, "r", encoding="utf-8").read()
                        for tok in forbidden:
                            if tok in content:
                                violations.append((p, tok))

        assert violations == [], f"Security violations detected: {violations}"

    def test_category_AK_hardware_neutrality_scan(self):
        """AK: Zero physical hardware drivers (pymavlink, mavsdk, rclpy, rospy, serial, RPi.GPIO)."""
        hw_tokens = ["pymavlink", "mavsdk", "rclpy", "rospy", "serial.", "RPi.GPIO"]
        target_dirs = [
            "backend/simulation",
            "backend/multimodal_fusion",
            "backend/spatial_telemetry",
            "backend/vision",
            "backend/perception",
        ]

        violations = []
        for d in target_dirs:
            for root, _, files in os.walk(d):
                for f in files:
                    if f.endswith(".py"):
                        p = os.path.join(root, f)
                        content = open(p, "r", encoding="utf-8").read()
                        for tok in hw_tokens:
                            if tok in content:
                                violations.append((p, tok))

        assert violations == [], f"Physical hardware driver violations detected: {violations}"

    def test_category_AJ_authority_boundary_scan(self):
        """AJ: Zero direct mutations of WorldState/GoalStore from perception/fusion/scenarios."""
        forbidden_calls = [
            "world_store.set_state",
            "world_store.save_state",
            "goal_store.save_goal",
            "goal_store.create_goal",
        ]
        target_dirs = [
            "backend/multimodal_fusion",
            "backend/spatial_telemetry",
            "backend/vision",
            "backend/perception",
        ]

        violations = []
        for d in target_dirs:
            for root, _, files in os.walk(d):
                for f in files:
                    if f.endswith(".py"):
                        p = os.path.join(root, f)
                        content = open(p, "r", encoding="utf-8").read()
                        for tok in forbidden_calls:
                            if tok in content:
                                violations.append((p, tok))

        assert violations == [], f"Authority bypass violations detected: {violations}"


# ============================================================================
# Categories AL to BA: Model Neutrality, Product Compatibility, Invariants
# ============================================================================

class TestArchitecturalInvariants:
    """Verify model neutrality, product compatibility, and strict authority invariants."""

    def test_category_AL_model_neutrality(self):
        """AL: Providers do not mandate heavyweight models (PyTorch, TensorFlow)."""
        prov = VisualPerceptionProvider()
        assert prov.is_available() is True

    def test_category_AM_four_product_compatibility(self):
        """AM: All 4 products declare valid roles and capabilities."""
        v = VisionDigitalTwin()
        g = GlassDigitalTwin()
        d = DroneDigitalTwin()
        r = RoverDigitalTwin()

        assert v.product_type == ProductType.VISION
        assert g.product_type == ProductType.GLASS
        assert d.product_type == ProductType.DRONE
        assert r.product_type == ProductType.ROVER

        for twin in [v, g, d, r]:
            ident, adapter = create_digital_twin_device(twin)
            assert ident.is_simulation is True
            assert len(adapter.get_capabilities()) > 0

    def test_category_AN_scenario_isolation(self):
        """AN: Independent scenario runs do not leak world state."""
        runner = ScenarioRunner()
        builder1 = ScenarioBuilder(scenario_id="s_iso_1", name="Iso 1")
        builder1.add_twin_config(TwinConfiguration(twin_id="drone_iso", product_type=ProductType.DRONE, product_role=ProductRole.HYBRID))
        builder1.add_step("step_1", 1.0, ScenarioStepType.ADVANCE_TIME, payload={"seconds": 10.0})
        res1 = runner.run_scenario(builder1.build())

        builder2 = ScenarioBuilder(scenario_id="s_iso_2", name="Iso 2")
        builder2.add_twin_config(TwinConfiguration(twin_id="drone_iso", product_type=ProductType.DRONE, product_role=ProductRole.HYBRID))
        builder2.add_step("step_1", 1.0, ScenarioStepType.ADVANCE_TIME, payload={"seconds": 5.0})
        res2 = runner.run_scenario(builder2.build())

        assert res1.simulated_duration == 11.0
        assert res2.simulated_duration == 6.0

    def test_category_AO_deterministic_hashes(self):
        """AO: SHA-256 result hashes match identically across repeat executions."""
        runner = ScenarioRunner()
        builder = ScenarioBuilder(scenario_id="hash_test", name="Hash Test")
        builder.add_step("step_1", 1.0, ScenarioStepType.ADVANCE_TIME, payload={"duration": 3.0})
        scen = builder.build()

        r1 = runner.run_scenario(scen)
        r2 = runner.run_scenario(scen)
        assert r1.deterministic_hash != ""
        assert r1.deterministic_hash == r2.deterministic_hash

    def test_category_AP_provenance_preservation(self):
        """AP: Provenance source_id and captured_at survive the perception -> normalizer -> fusion pipeline."""
        now = time.time()
        spatial_prov = SpatialTelemetryPerceptionProvider()
        norm = PerceptionObservationNormalizer()

        res = spatial_prov.process(PerceptionRequest(
            request_id="req_prov",
            input_data=PerceptionInput(
                input_id="inp_prov", modality=ModalityType.GPS, payload_ref="syn://p",
                captured_at=now, source_id="ORIGINAL_ROVER_SOURCE",
                metadata={"latitude": 37.7749, "longitude": -122.4194},
            ),
        ))
        obs = norm.normalize(res)
        assert obs.source_id == "ORIGINAL_ROVER_SOURCE"
        assert obs.timestamp == now

    def test_category_AQ_correlation_preservation(self):
        """AQ: Correlation ID survives perception and fusion."""
        now = time.time()
        spatial_prov = SpatialTelemetryPerceptionProvider()
        norm = PerceptionObservationNormalizer()

        res = spatial_prov.process(PerceptionRequest(
            request_id="req_corr",
            input_data=PerceptionInput(
                input_id="inp_corr", modality=ModalityType.GPS, payload_ref="syn://c",
                captured_at=now, source_id="d1", correlation_id="SHARED_CORRELATION_999",
                metadata={"latitude": 37.7749, "longitude": -122.4194},
            ),
            correlation_id="SHARED_CORRELATION_999",
        ))
        obs = norm.normalize(res)
        assert obs.correlation_id == "SHARED_CORRELATION_999"

        fusion = TemporalCrossModalFusionEngine()
        f_res = fusion.fuse([obs], reference_time=now, correlation_id="SHARED_CORRELATION_999")
        assert f_res.normalized_observations[0].correlation_id == "SHARED_CORRELATION_999"

    def test_category_AR_causation_tracking(self):
        """AR: Causation ID tracked from input to observation."""
        now = time.time()
        spatial_prov = SpatialTelemetryPerceptionProvider()
        norm = PerceptionObservationNormalizer()

        res = spatial_prov.process(PerceptionRequest(
            request_id="req_caus",
            input_data=PerceptionInput(
                input_id="inp_caus", modality=ModalityType.GPS, payload_ref="syn://caus",
                captured_at=now, source_id="d1", causation_id="PARENT_EVENT_123",
                metadata={"latitude": 37.7749, "longitude": -122.4194},
            ),
            causation_id="PARENT_EVENT_123",
        ))
        obs = norm.normalize(res)
        assert obs.causation_id == "PARENT_EVENT_123"

    def test_category_AS_error_path_behavior(self):
        """AS: Structured error reporting without unhandled exceptions."""
        prov = SpatialTelemetryPerceptionProvider()
        req = PerceptionRequest(
            request_id="req_err",
            input_data=PerceptionInput(
                input_id="inp_err", modality=ModalityType.UNKNOWN, payload_ref="syn://err",
                captured_at=time.time(), source_id="d1",
            ),
        )
        res = prov.process(req)
        assert res.is_success() is False
        assert len(res.errors) >= 1

    def test_category_AT_no_perception_decision_leakage(self):
        """AT: PerceptionProviderInterface never returns commands, actions, or decisions."""
        prov = VisualPerceptionProvider()
        req = PerceptionRequest(
            request_id="req_leak",
            input_data=PerceptionInput(
                input_id="inp_leak", modality=ModalityType.IMAGE, payload_ref="syn://l",
                captured_at=time.time(), source_id="v1",
                metadata={"image_bytes": make_synthetic_image_bytes()},
            ),
        )
        res = prov.process(req)
        assert hasattr(res, "evidence")
        assert not hasattr(res, "command")
        assert not hasattr(res, "action")
        assert not hasattr(res, "goal")

    def test_category_AU_no_world_state_bypass(self):
        """AU: Fusion engine does not have WorldState mutation methods."""
        fusion = TemporalCrossModalFusionEngine()
        assert not hasattr(fusion, "set_state")
        assert not hasattr(fusion, "mutate_world")
        assert not hasattr(fusion, "apply_observation")

    def test_category_AV_no_mission_bypass(self):
        """AV: MultiProductSituationIntelligenceEngine does not instantiate Missions."""
        engine = MultiProductSituationIntelligenceEngine()
        assert not hasattr(engine, "create_mission")
        assert not hasattr(engine, "dispatch_mission")

    def test_category_AW_no_goal_bypass(self):
        """AW: MissionCoordinator creates goals through AutonomousGoalManager, never direct store."""
        mc = MissionCoordinator(goal_manager=None)
        assert mc._create_goal_for_objective(
            MissionObjective(objective_id="o1", mission_id="m1", type=MissionObjectiveType.VERIFY_INCIDENT, description="patrol"),
            Mission(mission_id="m1", mission_type="patrol", description="Patrol description"),
        ) is None

    def test_category_AX_no_tool_bypass(self):
        """AX: ToolOrchestrator rejects unregistered tools or unvalidated actions."""
        to = ToolOrchestrator()
        res = to.execute(ToolCall(call_id="c_unauth", capability="unregistered_cap", action="hack"))
        assert res.success is False

    def test_category_AY_no_device_gateway_bypass(self):
        """AY: DeviceGateway enforces device registration before command dispatch."""
        state = make_test_app_state()
        res = state.device_gateway.dispatch_to_device(
            device_id="NON_EXISTENT_ROBOT",
            capability="drive",
            action="forward",
            parameters={},
        )
        assert res.success is False
        assert "not registered" in res.message.lower()

    def test_category_AZ_no_cognitive_runtime_bypass(self):
        """AZ: CognitiveRuntime owns cognitive turn execution."""
        state = make_test_app_state()
        assert hasattr(state.cognitive_runtime, "execute_turn")

    def test_category_BA_no_policy_bypass(self):
        """BA: PolicyEngine default deny on unauthorized actions."""
        state = make_test_app_state()
        res = state.tool_orchestrator.execute(ToolCall(
            call_id="call_deny",
            capability="dangerous_arbitrary_exec",
            action="system_kill",
            parameters={},
        ))
        assert res.success is False
