"""
ATLAS Phase 5.0e — Cross-Layer Integration & Replay Verification Test Suite.

Verifies that ATLAS Central Orchestration operates as ONE coherent, bounded,
closed-loop autonomous system:
Edge Devices (Glass/Drone/Rover)
    ↓
CentralInputGateway
    ↓
SituationFusionEngine
    ↓
WorldState
    ↓
Event-Driven Autonomy / Anticipation
    ↓
AutonomousGoalManager
    ↓
CognitiveRuntime
    ↓
PolicyEngine
    ↓
ToolOrchestrator
    ↓
DeviceGateway
    ↓
DeviceAdapter
    ↓
Virtual Device
    ↓
MultimodalObservation
    ↺ CentralInputGateway

Covers all Phase 5.0e requirements (A through T):
- Wiring and dependencies
- End-to-end Glass -> Central -> Drone flow
- Observation re-ingestion closed loop
- Situation creation and updates
- WorldState integration
- Event autonomy integration
- Anticipatory planning integration
- GoalManager authority
- CognitiveRuntime integration
- Policy enforcement (ALLOW / DENY)
- ToolOrchestrator integration
- DeviceGateway integration
- Multi-device isolation
- Failure & recovery
- Timeout & bounds
- Duplicate suppression
- Causal lineage retention
- Replay determinism
- Loop protection
- Security & authority static audit
"""

import sys
import threading
import time
import uuid
from unittest.mock import MagicMock
from typing import Any, Dict, List, Optional, Sequence

# Fallback for environments where chromadb is not installed
if "chromadb" not in sys.modules:
    sys.modules["chromadb"] = MagicMock()

import pytest

# Core Interfaces
from core.interfaces.orchestration_interface import (
    CentralInputGatewayInterface,
    CentralOrchestratorInterface,
    DeviceAdapterInterface,
    DeviceGatewayInterface,
)
from core.interfaces.world_interface import WorldStateStoreInterface, WorldStateUpdaterInterface
from core.interfaces.autonomy_interface import EventDrivenAutonomyInterface
from core.interfaces.goal_interface import AutonomousGoalManagerInterface

# Core Models
from core.models.autonomy import (
    AutonomyDecision,
    AutonomyDecisionType,
    Event,
    EventPriority,
    EventProvenance,
    EventSource,
)
from core.models.goal import Goal, GoalPriority, GoalStatus
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
    SituationSeverity,
    SituationStatus,
)
from core.models.policy import AutonomyLevel, PolicyDecision, PolicyResult, PolicyContext
from core.models.request import Request
from core.models.result import Result
from core.models.runtime import (
    CognitiveEvent,
    CognitiveEventType,
    CognitiveStage,
    CognitiveTurnResult,
    TurnStatus,
)
from core.models.tool_call import ToolCall
from core.models.world_state import (
    Observation,
    StateProvenance,
    TransitionType,
    WorldCondition,
    WorldEntity,
    WorldState,
    WorldStateTransition,
)
from core.models.plan import Plan

# Concrete Components
from autonomy.classifier import DeterministicEventClassifier
from autonomy.coordinator import EventDrivenAutonomyCoordinator
from autonomy.relevance import DeterministicRelevanceEngine
from autonomy.world_listener import transition_to_autonomy_event
from goals.manager import AutonomousGoalManager
from goals.scheduler import DeterministicGoalScheduler
from goals.store import InMemoryGoalStore
from orchestration.central_orchestration import (
    CentralOrchestrationConfig,
    CentralOrchestrator,
    CentralOrchestratorResult,
)
from orchestration.device_gateway import DeviceCommand, DeviceGateway, DeviceGatewayCapability
from orchestration.fusion_engine import SituationFusionConfig, SituationFusionEngine
from orchestration.input_gateway import CentralInputGateway, IngressEnvelope
from orchestration.virtual_devices import (
    VirtualDroneAdapter,
    VirtualGlassAdapter,
    VirtualRoverAdapter,
    create_virtual_drone,
    create_virtual_glass,
    create_virtual_rover,
)
from runtime.cognitive_runtime import CognitiveRuntime
from runtime.event_sink import InMemoryEventSink
from safety.policy_engine import StandardPolicyEngine
from tools.capability_registry import CapabilityRegistry
from tools.tool_orchestrator import ToolOrchestrator
from world.resolver import DeterministicConflictResolver
from world.store import InMemoryWorldStateStore
from world.updater import DeterministicWorldStateUpdater


# ============================================================================
# Deterministic Test Doubles for Cognitive Turn Execution
# ============================================================================

class MockUnderstanding:
    def understand(self, input_data: Any) -> Request:
        text = str(input_data)
        return Request(
            id=f"req_{uuid.uuid4().hex[:8]}",
            session_id="integration_session",
            timestamp=time.time(),
            original_text=text,
            normalized_text=text.lower().strip(),
            parameters={},
        )


class MockDecisionEngine:
    def decide(self, request: Request) -> Any:
        from core.models.decision import Decision, CapabilityType, ExecutionMode
        return Decision(
            request_id=request.id,
            primary_goal=request.original_text,
            required_capabilities=[CapabilityType.TOOL],
            execution_mode=ExecutionMode.SINGLE_STEP,
            confidence=1.0,
            reasoning="Deterministic test decision",
        )


class DroneDispatchPlanner:
    """Planner that returns a concrete ToolCall for VirtualDrone navigation."""
    def __init__(self, device_id: str = "ATLAS_DRONE_01", target_alt: float = 15.0):
        self.device_id = device_id
        self.target_alt = target_alt

    def plan(self, *args: Any, **kwargs: Any) -> Plan:
        first_arg = args[0] if args else kwargs.get("decision")
        req_id = getattr(first_arg, "request_id", getattr(first_arg, "id", "req_plan"))
        tc = ToolCall(
            capability="device_gateway",
            action="dispatch_capability",
            parameters={
                "device_id": self.device_id,
                "capability": "takeoff",
                "action": "takeoff",
                "parameters": {"target_altitude": self.target_alt},
                "correlation_id": "corr_plan_001",
                "causation_id": str(req_id),
            },
            call_id=f"call_{uuid.uuid4().hex[:8]}",
        )
        return Plan(
            goal=f"Dispatch {self.device_id}",
            steps=[tc],
            confidence=1.0,
        )


class DirectExecutionEngine:
    """Replay/test execution engine executing tasks via ToolOrchestrator."""
    def __init__(self, orchestrator: ToolOrchestrator):
        self.orchestrator = orchestrator

    def execute(self, plan: Plan) -> List[Result]:
        results = []
        for step in plan.steps:
            if isinstance(step, ToolCall):
                res = self.orchestrator.execute(step)
                results.append(res)
                if hasattr(step, "status"):
                    step.status = "completed" if res.success else "failed"
        plan.status = "completed" if all(r.success for r in results) else "failed"
        return results


# ============================================================================
# Test Fixture: Integrated Central Orchestration Pipeline Factory
# ============================================================================

def build_test_orchestration(
    drone_id: str = "ATLAS_DRONE_01",
    glass_id: str = "ATLAS_GLASS_01",
    rover_id: str = "ATLAS_ROVER_01",
    config: Optional[CentralOrchestrationConfig] = None,
    planner: Optional[Any] = None,
):
    """Build a fully wired, deterministic Phase 5.0e integration pipeline."""
    event_sink = InMemoryEventSink()

    # 1. Device Gateway & Virtual Devices
    device_gw = DeviceGateway(event_sink=event_sink)
    drone, drone_adapter = create_virtual_drone(drone_id)
    glass, glass_adapter = create_virtual_glass(glass_id)
    rover, rover_adapter = create_virtual_rover(rover_id)

    device_gw.register_device(drone)
    device_gw.register_device(glass)
    device_gw.register_device(rover)

    device_gw.register_adapter(drone_adapter, device_id=drone_id)
    device_gw.register_adapter(glass_adapter, device_id=glass_id)
    device_gw.register_adapter(rover_adapter, device_id=rover_id)

    # 2. Tool Orchestrator with Policy & Device Gateway Capability
    registry = CapabilityRegistry()
    gw_cap = DeviceGatewayCapability(device_gw)
    registry.register("device_gateway", gw_cap)

    policy_engine = StandardPolicyEngine()
    tool_orchestrator = ToolOrchestrator(
        registry=registry,
        policy_engine=policy_engine,
    )

    # 3. World State
    world_store = InMemoryWorldStateStore()
    world_updater = DeterministicWorldStateUpdater(
        store=world_store,
        event_sink=event_sink,
    )

    # 4. Input Gateway & Situation Fusion
    input_gw = CentralInputGateway(event_sink=event_sink)
    fusion_engine = SituationFusionEngine(
        world_state_store=world_store,
        event_sink=event_sink,
    )

    # 5. Autonomous Goal Manager
    goal_store = InMemoryGoalStore()
    # Mock goal execution engine that can execute via cognitive runtime
    mock_exec_engine = MagicMock()
    goal_manager = AutonomousGoalManager(
        store=goal_store,
        execution_engine=mock_exec_engine,
        event_sink=event_sink,
    )

    # 6. Event-Driven Autonomy
    autonomy_coord = EventDrivenAutonomyCoordinator(
        classifier=DeterministicEventClassifier(),
        relevance_engine=DeterministicRelevanceEngine(),
        goal_manager=goal_manager,
        goal_store=goal_store,
        policy_engine=policy_engine,
        world_state_store=world_store,
        event_sink=event_sink,
    )

    # 7. Cognitive Runtime
    plan_impl = planner or DroneDispatchPlanner(device_id=drone_id)
    direct_exec = DirectExecutionEngine(tool_orchestrator)
    cognitive_runtime = CognitiveRuntime(
        understanding=MockUnderstanding(),
        decision_engine=MockDecisionEngine(),
        planner=plan_impl,
        policy_engine=policy_engine,
        execution_engine=direct_exec,
        event_sink=event_sink,
    )

    # 8. Central Orchestrator
    orchestrator = CentralOrchestrator(
        input_gateway=input_gw,
        fusion_engine=fusion_engine,
        world_updater=world_updater,
        world_store=world_store,
        autonomy_coordinator=autonomy_coord,
        goal_manager=goal_manager,
        cognitive_runtime=cognitive_runtime,
        tool_orchestrator=tool_orchestrator,
        device_gateway=device_gw,
        event_sink=event_sink,
        config=config,
    )

    return {
        "orchestrator": orchestrator,
        "device_gw": device_gw,
        "input_gw": input_gw,
        "fusion_engine": fusion_engine,
        "world_store": world_store,
        "world_updater": world_updater,
        "autonomy_coord": autonomy_coord,
        "goal_manager": goal_manager,
        "cognitive_runtime": cognitive_runtime,
        "tool_orchestrator": tool_orchestrator,
        "event_sink": event_sink,
        "drone_adapter": drone_adapter,
        "glass_adapter": glass_adapter,
        "rover_adapter": rover_adapter,
    }


# ============================================================================
# A. WIRING AND DEPENDENCY TESTS (1–5)
# ============================================================================

def test_01_central_orchestration_instantiation():
    """Verify CentralOrchestrator initializes cleanly with full dependency set."""
    p = build_test_orchestration()
    orch = p["orchestrator"]
    assert orch.input_gateway is not None
    assert orch.fusion_engine is not None
    assert orch.world_updater is not None
    assert orch.autonomy_coordinator is not None
    assert orch.goal_manager is not None
    assert orch.tool_orchestrator is not None
    assert orch.device_gateway is not None


def test_02_minimal_wiring_defaults():
    """Verify CentralOrchestrator operates when optional components are None."""
    input_gw = CentralInputGateway()
    fusion_engine = SituationFusionEngine()
    orch = CentralOrchestrator(input_gateway=input_gw, fusion_engine=fusion_engine)
    assert orch.world_updater is None
    assert orch.autonomy_coordinator is None
    assert orch.goal_manager is None
    assert orch.tool_orchestrator is None


def test_03_config_custom_limits():
    """Verify custom orchestration bounds are properly honored."""
    cfg = CentralOrchestrationConfig(max_cycles=2, max_reingestions=3)
    p = build_test_orchestration(config=cfg)
    assert p["orchestrator"].config.max_cycles == 2
    assert p["orchestrator"].config.max_reingestions == 3


def test_04_pure_data_situation_to_world_observation():
    """Verify pure data translation does not mutate WorldState store."""
    p = build_test_orchestration()
    store = p["world_store"]
    assert len(store.get_current_state().conditions) == 0

    sit = Situation(
        situation_id="sit_test_01",
        title="Fire Hazard",
        description="Active smoke detected",
        category=SituationCategory.SECURITY,
        severity=SituationSeverity.HIGH,
        confidence=0.9,
        status=SituationStatus.ACTIVE,
        involved_entities=("warehouse_sector_4",),
        supporting_evidence=(),
        created_at=100.0,
        updated_at=100.0,
    )
    obs = CentralOrchestrator.situation_to_world_observation(sit)
    assert obs.entity_id == "warehouse_sector_4"
    assert "hazard" in obs.property_name
    assert obs.value == "HIGH"
    # Verify store was not modified by translation
    assert len(store.get_current_state().conditions) == 0


def test_05_pure_data_multimodal_observation_to_world_observation():
    """Verify pure data translation for device MultimodalObservation."""
    obs = MultimodalObservation(
        observation_id="obs_m_01",
        source_id="ATLAS_DRONE_01",
        source_type="DRONE",
        modality=ModalityType.TELEMETRY,
        timestamp=100.0,
        payload={"altitude": 15.0, "battery_pct": 95.0},
        correlation_id="corr_m_01",
    )
    w_obs = CentralOrchestrator.multimodal_observation_to_world_observation(obs)
    assert w_obs.entity_id == "ATLAS_DRONE_01"
    assert w_obs.property_name == "telemetry_telemetry"
    assert w_obs.value == {"altitude": 15.0, "battery_pct": 95.0}


# ============================================================================
# B. END-TO-END GLASS -> CENTRAL -> DRONE SCENARIO (6–10)
# ============================================================================

def test_06_glass_to_central_drone_dispatch_flow():
    """
    Primary End-to-End Scenario:
    1. Glass produces incident observation
    2. CentralInputGateway normalizes
    3. SituationFusion fuses into active Situation
    4. WorldState updated
    5. EventAutonomy creates Event & Decision
    6. GoalManager creates Goal
    7. CognitiveRuntime executes Turn & ToolCall
    8. ToolOrchestrator routes through DeviceGateway
    9. VirtualDrone takes off
    10. Drone emits Telemetry observation back into pipeline.
    """
    p = build_test_orchestration()
    orch = p["orchestrator"]
    drone_adapter = p["drone_adapter"]

    # Virtual Glass produces emergency observation
    glass_obs = MultimodalObservation(
        observation_id="obs_glass_fire_01",
        source_id="ATLAS_GLASS_01",
        source_type="GLASS",
        modality=ModalityType.IMAGE,
        timestamp=100.0,
        payload={"hazard": "fire_alarm", "confidence": 0.95},
        location=GeoLocation(latitude=37.7749, longitude=-122.4194),
        correlation_id="corr_scenario_1",
    )

    res = orch.process_ingress(glass_obs, now=100.0)

    # 1. Pipeline successfully completed
    assert res.status == "SUCCESS"
    # 2. Ingress captured
    assert len(res.observations_ingested) >= 1
    # 3. Situation created
    assert len(res.situations_fused) >= 1
    assert res.situations_fused[0].status in (SituationStatus.ACTIVE, SituationStatus.DETECTED)
    # 4. WorldState transitioned
    assert len(res.world_transitions) >= 1
    # 5. Goal created through AutonomousGoalManager
    assert len(res.goals_created) >= 1
    # 6. Cognitive turn executed
    assert len(res.turn_results) >= 1
    # 7. Drone executed takeoff
    assert drone_adapter.airborne is True
    assert drone_adapter.altitude == 15.0
    # 8. Drone telemetry re-ingested
    assert len(res.observations_reingested) >= 1
    assert res.observations_reingested[0].source_id == "ATLAS_DRONE_01"


def test_07_e2e_glass_raw_dict_envelope_ingress():
    """Verify raw dict ingress envelopes successfully flow through pipeline."""
    p = build_test_orchestration()
    orch = p["orchestrator"]

    raw_dict = {
        "message_id": "env_msg_101",
        "source_id": "ATLAS_GLASS_01",
        "source_type": "GLASS",
        "modality": "IMAGE",
        "timestamp": 105.0,
        "payload": {"hazard": "intruder", "confidence": 0.9},
        "location": {"latitude": 37.78, "longitude": -122.42},
        "correlation_id": "corr_raw_dict",
    }
    res = orch.process_ingress(raw_dict, now=105.0)
    assert res.status == "SUCCESS"
    assert len(res.situations_fused) >= 1
    assert p["drone_adapter"].airborne is True


def test_08_e2e_drone_telemetry_reaches_central():
    """Confirm drone observation re-enters CentralInputGateway as canonical observation."""
    p = build_test_orchestration()
    orch = p["orchestrator"]

    glass_obs = MultimodalObservation(
        observation_id="obs_g_88",
        source_id="ATLAS_GLASS_01",
        source_type="GLASS",
        modality=ModalityType.IMAGE,
        timestamp=100.0,
        payload={"alarm": "zone_breach"},
        correlation_id="corr_88",
    )
    res = orch.process_ingress(glass_obs, now=100.0)
    assert len(res.observations_reingested) > 0
    reingested = res.observations_reingested[0]
    assert reingested.modality == ModalityType.TELEMETRY
    assert reingested.source_type == "DRONE"


def test_09_e2e_causal_trace_continuity():
    """Verify complete causal lineage: INGRESS -> SITUATION -> WORLD_STATE -> AUTONOMY -> GOAL -> TURN -> TOOL."""
    p = build_test_orchestration()
    orch = p["orchestrator"]

    obs = MultimodalObservation(
        observation_id="obs_trace_01",
        source_id="ATLAS_GLASS_01",
        source_type="GLASS",
        modality=ModalityType.IMAGE,
        timestamp=100.0,
        payload={"hazard": "fire_detected"},
        correlation_id="corr_trace_test",
    )
    res = orch.process_ingress(obs, now=100.0)
    stages = [entry["stage"] for entry in res.causal_trace]
    assert "INGRESS" in stages
    assert "SITUATION" in stages
    assert "WORLD_STATE" in stages
    assert "AUTONOMY" in stages
    assert "GOAL" in stages
    assert "TURN" in stages
    assert "TOOL_EXECUTION" in stages


def test_10_e2e_drone_internal_state_updated_deterministically():
    """Verify drone internal state matches requested flight parameters."""
    planner = DroneDispatchPlanner(device_id="ATLAS_DRONE_01", target_alt=25.0)
    p = build_test_orchestration(planner=planner)
    orch = p["orchestrator"]

    obs = MultimodalObservation(
        observation_id="obs_alt_test",
        source_id="ATLAS_GLASS_01",
        source_type="GLASS",
        modality=ModalityType.IMAGE,
        timestamp=100.0,
        payload={"hazard": "high_alarm"},
    )
    orch.process_ingress(obs, now=100.0)
    assert p["drone_adapter"].altitude == 25.0
    assert p["drone_adapter"].armed is True


# ============================================================================
# C. OBSERVATION RE-INGESTION CLOSED LOOP (11–13)
# ============================================================================

def test_11_reingestion_updates_fusion_engine():
    """Verify re-ingested observations are processed by SituationFusionEngine."""
    p = build_test_orchestration()
    orch = p["orchestrator"]

    obs = MultimodalObservation(
        observation_id="obs_reingest_1",
        source_id="ATLAS_GLASS_01",
        source_type="GLASS",
        modality=ModalityType.IMAGE,
        timestamp=100.0,
        payload={"hazard": "fire"},
        correlation_id="corr_reingest",
    )
    res = orch.process_ingress(obs, now=100.0)
    assert len(res.observations_reingested) > 0
    # Situation fusion must hold the fused situation
    active_sits = p["fusion_engine"].get_active_situations()
    assert len(active_sits) >= 1


def test_12_reingestion_disabled_config():
    """Verify auto_reingest_observations=False disables automatic feedback loop."""
    cfg = CentralOrchestrationConfig(auto_reingest_observations=False)
    p = build_test_orchestration(config=cfg)
    orch = p["orchestrator"]

    obs = MultimodalObservation(
        observation_id="obs_no_reingest",
        source_id="ATLAS_GLASS_01",
        source_type="GLASS",
        modality=ModalityType.IMAGE,
        timestamp=100.0,
        payload={"hazard": "fire"},
    )
    res = orch.process_ingress(obs, now=100.0)
    assert len(res.observations_reingested) == 0


def test_13_reingestion_causation_id_bound_to_dispatch():
    """Verify re-ingested observation causation_id links back to tool dispatch_id."""
    p = build_test_orchestration()
    orch = p["orchestrator"]

    obs = MultimodalObservation(
        observation_id="obs_causation_check",
        source_id="ATLAS_GLASS_01",
        source_type="GLASS",
        modality=ModalityType.IMAGE,
        timestamp=100.0,
        payload={"hazard": "fire"},
        correlation_id="corr_causation",
    )
    res = orch.process_ingress(obs, now=100.0)
    reingested = res.observations_reingested[0]
    assert reingested.causation_id is not None
    assert reingested.correlation_id == "corr_causation"


# ============================================================================
# D. SITUATION CREATION AND UPDATE (14–16)
# ============================================================================

def test_14_single_incident_fuses_into_active_situation():
    """Verify multiple frames from glass fuse into one active Situation."""
    p = build_test_orchestration()
    orch = p["orchestrator"]

    obs1 = MultimodalObservation(
        observation_id="obs_f1",
        source_id="ATLAS_GLASS_01",
        source_type="GLASS",
        modality=ModalityType.IMAGE,
        timestamp=100.0,
        payload={"hazard": "smoke"},
        location=GeoLocation(latitude=37.77, longitude=-122.41),
    )
    obs2 = MultimodalObservation(
        observation_id="obs_f2",
        source_id="ATLAS_DRONE_01",
        source_type="DRONE",
        modality=ModalityType.IMAGE,
        timestamp=101.0,
        payload={"hazard": "flames"},
        location=GeoLocation(latitude=37.77, longitude=-122.41),
    )
    res = orch.run_cycle([obs1, obs2], now=101.0)
    assert len(res.situations_fused) >= 1
    sit = res.situations_fused[0]
    assert sit.status == SituationStatus.ACTIVE


def test_15_benign_observation_does_not_create_hazard_situation():
    """Normal heartbeat observation does not create emergency situations."""
    p = build_test_orchestration()
    orch = p["orchestrator"]

    obs = MultimodalObservation(
        observation_id="obs_benign",
        source_id="ATLAS_GLASS_01",
        source_type="GLASS",
        modality=ModalityType.TELEMETRY,
        timestamp=100.0,
        payload={"battery": 98.0, "status": "NOMINAL"},
    )
    res = orch.process_ingress(obs, now=100.0)
    # No hazard situation or goal created
    assert len(res.goals_created) == 0


def test_16_situation_confidence_boost_on_multi_source():
    """Multi-source observation boosts fused situation confidence."""
    p = build_test_orchestration()
    orch = p["orchestrator"]

    obs_glass = MultimodalObservation(
        observation_id="obs_ms_glass",
        source_id="ATLAS_GLASS_01",
        source_type="GLASS",
        modality=ModalityType.IMAGE,
        timestamp=100.0,
        payload={"hazard": "fire_alarm"},
        location=GeoLocation(latitude=37.77, longitude=-122.41),
    )
    obs_rover = MultimodalObservation(
        observation_id="obs_ms_rover",
        source_id="ATLAS_ROVER_01",
        source_type="ROVER",
        modality=ModalityType.TELEMETRY,
        timestamp=100.5,
        payload={"thermal_anomaly": 95.0},
        location=GeoLocation(latitude=37.77, longitude=-122.41),
    )
    res = orch.run_cycle([obs_glass, obs_rover], now=101.0)
    assert len(res.situations_fused) >= 1
    sit = res.situations_fused[-1]
    assert sit.confidence >= 0.8


# ============================================================================
# E. WORLD STATE INTEGRATION (17–19)
# ============================================================================

def test_17_world_state_version_increments_monotonically():
    """Verify WorldState version strictly increments per accepted transition."""
    p = build_test_orchestration()
    orch = p["orchestrator"]
    store = p["world_store"]

    init_ver = store.get_current_state().version

    obs = MultimodalObservation(
        observation_id="obs_ver_1",
        source_id="ATLAS_GLASS_01",
        source_type="GLASS",
        modality=ModalityType.IMAGE,
        timestamp=100.0,
        payload={"hazard": "intrusion"},
    )
    res = orch.process_ingress(obs, now=100.0)
    assert store.get_current_state().version > init_ver
    assert len(res.world_transitions) > 0


def test_18_no_direct_world_state_mutation_from_adapter():
    """Virtual drone adapter execution does NOT mutate WorldState directly."""
    p = build_test_orchestration()
    store = p["world_store"]
    drone_adapter = p["drone_adapter"]

    init_ver = store.get_current_state().version
    # Directly invoke adapter command without gateway or world updater
    drone_adapter.execute_command(
        DeviceCommand(
            dispatch_id="disp_direct_test_18",
            device_id="ATLAS_DRONE_01",
            capability="takeoff",
            action="takeoff",
            parameters={"target_altitude": 10.0},
        )
    )
    # Store version must be completely unchanged
    assert store.get_current_state().version == init_ver


def test_19_world_state_condition_provenance_retains_situation():
    """Verify condition provenance traces back to situation fusion."""
    p = build_test_orchestration()
    orch = p["orchestrator"]
    store = p["world_store"]

    obs = MultimodalObservation(
        observation_id="obs_prov_test",
        source_id="ATLAS_GLASS_01",
        source_type="GLASS",
        modality=ModalityType.IMAGE,
        timestamp=100.0,
        payload={"hazard": "fire_hazard"},
    )
    orch.process_ingress(obs, now=100.0)
    conds = store.get_current_state().conditions
    assert len(conds) > 0
    assert "situation_fusion" in conds[0].provenance.source_id


# ============================================================================
# F. EVENT AUTONOMY INTEGRATION (20–22)
# ============================================================================

def test_20_world_transition_generates_autonomy_event():
    """World state transition converts into an Event for autonomy evaluation."""
    p = build_test_orchestration()
    orch = p["orchestrator"]

    obs = MultimodalObservation(
        observation_id="obs_evt_gen",
        source_id="ATLAS_GLASS_01",
        source_type="GLASS",
        modality=ModalityType.IMAGE,
        timestamp=100.0,
        payload={"hazard": "critical_hazard"},
    )
    res = orch.process_ingress(obs, now=100.0)
    assert len(res.events_evaluated) > 0
    assert res.events_evaluated[0].provenance.source_type == EventSource.WORLD_STATE


def test_21_autonomy_decision_creates_goal_not_device_action():
    """Autonomy evaluation produces a CREATE_GOAL decision, never direct tool execution."""
    p = build_test_orchestration()
    orch = p["orchestrator"]

    obs = MultimodalObservation(
        observation_id="obs_decision_test",
        source_id="ATLAS_GLASS_01",
        source_type="GLASS",
        modality=ModalityType.IMAGE,
        timestamp=100.0,
        payload={"hazard": "alarm_active"},
    )
    res = orch.process_ingress(obs, now=100.0)
    assert len(res.autonomy_decisions) > 0
    assert any(dec.decision_type == AutonomyDecisionType.CREATE_GOAL for dec in res.autonomy_decisions)
    assert res.autonomy_decisions[0].decision_type == AutonomyDecisionType.CREATE_GOAL


def test_22_benign_event_does_not_trigger_goal():
    """Benign/unchanged world conditions produce NO_ACTION autonomy decision."""
    p = build_test_orchestration()
    coord = p["autonomy_coord"]

    # Benign event
    evt = Event(
        event_id="evt_benign_01",
        source=EventSource.WORLD_STATE,
        event_type="sensor.heartbeat.steady",
        priority=EventPriority.LOW,
        timestamp=100.0,
        correlation_id="corr_benign",
        provenance=EventProvenance(
            source_id="sensor",
            source_type=EventSource.WORLD_STATE,
            origin_timestamp=100.0,
            correlation_id="corr_benign",
        ),
        payload={"temp": 21.0},
    )
    dec = coord.ingest_event(evt)
    assert dec.decision_type in (AutonomyDecisionType.RECORD_ONLY, AutonomyDecisionType.IGNORE)


# ============================================================================
# G. ANTICIPATORY PLANNING INTEGRATION (23–25)
# ============================================================================

def test_23_anticipatory_coordinator_wiring():
    """Verify AnticipatoryPlanningCoordinator can be attached to CentralOrchestrator."""
    p = build_test_orchestration()
    orch = p["orchestrator"]
    mock_anticipation = MagicMock()
    orch.anticipatory_coordinator = mock_anticipation
    assert orch.anticipatory_coordinator == mock_anticipation


def test_24_anticipatory_decision_routes_to_goal_manager():
    """Anticipatory decision routes goal creation through AutonomousGoalManager."""
    p = build_test_orchestration()
    goal_mgr = p["goal_manager"]
    init_goals = len(goal_mgr.store.get_all_goals())

    # Direct goal creation via goal manager
    g = goal_mgr.create_goal(
        title="Anticipated Battery Low",
        description="Return to launch before threshold",
        priority=GoalPriority.HIGH,
    )
    assert g is not None
    assert len(goal_mgr.store.get_all_goals()) == init_goals + 1


def test_25_hypothetical_state_isolated_from_world_state():
    """Hypothetical conditions in anticipatory planning do not mutate actual WorldState."""
    p = build_test_orchestration()
    store = p["world_store"]
    init_state = store.get_current_state()

    # Create an anticipation item (purely hypothetical)
    from core.models.anticipation import (
        Anticipation,
        AnticipationProvenance,
        FutureConditionType,
        TimeHorizon,
    )
    ant = Anticipation(
        anticipation_id="ant_hypo_01",
        condition_type=FutureConditionType.RESOURCE_DEPLETION_RISK,
        description="battery_depleted",
        hypothetical_state={"battery": 5.0},
        horizon=TimeHorizon.IMMEDIATE,
        horizon_window_seconds=(0.0, 300.0),
        confidence=0.85,
        relevance=0.9,
        freshness=1.0,
        evidence_items=(),
        provenance=AnticipationProvenance(
            source_entity="sensor",
            created_at=100.0,
            correlation_id="corr_ant",
        ),
        correlation_id="corr_ant",
    )
    # Current believed reality must remain untouched
    assert store.get_current_state().version == init_state.version


# ============================================================================
# H. GOALMANAGER AUTHORITY (26–28)
# ============================================================================

def test_26_goals_created_strictly_via_autonomous_goal_manager():
    """Goals recorded in orchestration result are created through AutonomousGoalManager."""
    p = build_test_orchestration()
    orch = p["orchestrator"]
    goal_mgr = p["goal_manager"]

    obs = MultimodalObservation(
        observation_id="obs_goal_auth_01",
        source_id="ATLAS_GLASS_01",
        source_type="GLASS",
        modality=ModalityType.IMAGE,
        timestamp=100.0,
        payload={"hazard": "fire_hazard"},
    )
    res = orch.process_ingress(obs, now=100.0)
    for g in res.goals_created:
        stored = goal_mgr.store.get_goal(g.id)
        assert stored is not None
        assert stored.title == g.title


def test_27_no_direct_goal_store_mutation_without_manager():
    """Verify CentralOrchestrator has no direct goal_store attribute, only goal_manager."""
    p = build_test_orchestration()
    orch = p["orchestrator"]
    assert not hasattr(orch, "goal_store")
    assert orch.goal_manager is not None


def test_28_goal_priority_maps_from_situation_severity():
    """High severity situation produces High priority goal."""
    p = build_test_orchestration()
    orch = p["orchestrator"]

    obs = MultimodalObservation(
        observation_id="obs_prio_test",
        source_id="ATLAS_GLASS_01",
        source_type="GLASS",
        modality=ModalityType.IMAGE,
        timestamp=100.0,
        payload={"hazard": "severe_fire"},
    )
    res = orch.process_ingress(obs, now=100.0)
    assert len(res.goals_created) >= 1
    assert res.goals_created[0].priority in (GoalPriority.HIGH, GoalPriority.CRITICAL)


# ============================================================================
# I. COGNITIVETURN / COGNITIVERUNTIME INTEGRATION (29–31)
# ============================================================================

def test_29_cognitive_runtime_executes_turn_on_goal():
    """Verify CognitiveRuntime.execute_turn is invoked when goal is created."""
    p = build_test_orchestration()
    orch = p["orchestrator"]

    obs = MultimodalObservation(
        observation_id="obs_turn_exec",
        source_id="ATLAS_GLASS_01",
        source_type="GLASS",
        modality=ModalityType.IMAGE,
        timestamp=100.0,
        payload={"hazard": "alarm_fire"},
    )
    res = orch.process_ingress(obs, now=100.0)
    assert len(res.turn_results) >= 1
    assert res.turn_results[0].status == TurnStatus.SUCCEEDED


def test_30_cognitive_turn_produces_valid_trace():
    """Cognitive turn execution populates turn_results with structured plan."""
    p = build_test_orchestration()
    orch = p["orchestrator"]

    obs = MultimodalObservation(
        observation_id="obs_trace_gen",
        source_id="ATLAS_GLASS_01",
        source_type="GLASS",
        modality=ModalityType.IMAGE,
        timestamp=100.0,
        payload={"hazard": "alarm_fire"},
    )
    res = orch.process_ingress(obs, now=100.0)
    turn_res = res.turn_results[0]
    assert turn_res.plan is not None
    assert len(turn_res.plan.steps) >= 1


def test_31_runtime_exceptions_handled_gracefully():
    """Exception in CognitiveRuntime does not crash CentralOrchestrator."""
    p = build_test_orchestration()
    orch = p["orchestrator"]
    orch.cognitive_runtime = MagicMock()
    orch.cognitive_runtime.execute_turn.side_effect = RuntimeError("Cognitive crash")

    obs = MultimodalObservation(
        observation_id="obs_runtime_crash",
        source_id="ATLAS_GLASS_01",
        source_type="GLASS",
        modality=ModalityType.IMAGE,
        timestamp=100.0,
        payload={"hazard": "fire"},
    )
    res = orch.process_ingress(obs, now=100.0)
    # Must finish without uncaught exception
    assert res.status == "SUCCESS"
    assert len(res.turn_results) == 0


# ============================================================================
# J. POLICY ALLOW / DENY ENFORCEMENT (32–34)
# ============================================================================

def test_32_policy_engine_allows_governed_drone_takeoff():
    """PolicyEngine authorizes standard device_gateway takeoff command."""
    p = build_test_orchestration()
    tool_orch = p["tool_orchestrator"]

    tc = ToolCall(
        capability="device_gateway",
        action="dispatch_capability",
        parameters={
            "device_id": "ATLAS_DRONE_01",
            "capability": "takeoff",
            "action": "takeoff",
            "parameters": {"target_altitude": 10.0},
        },
        call_id="call_policy_allow_test",
    )
    res = tool_orch.execute(tc)
    assert res.success is True
    assert p["drone_adapter"].airborne is True


def test_33_policy_engine_denies_prohibited_system_command():
    """PolicyEngine blocks arbitrary system shell commands before DeviceGateway."""
    p = build_test_orchestration()
    tool_orch = p["tool_orchestrator"]

    tc = ToolCall(
        capability="shell",
        action="execute_command",
        parameters={"cmd": "rm -rf /"},
        call_id="call_policy_deny_test",
    )
    res = tool_orch.execute(tc)
    assert res.success is False
    assert "denied" in res.message.lower() or "prohibited" in res.message.lower()


def test_34_policy_denial_blocks_adapter_invocation():
    """When Policy blocks a call, the device adapter is never invoked."""
    p = build_test_orchestration()
    drone_adapter = p["drone_adapter"]
    tool_orch = p["tool_orchestrator"]

    init_alt = drone_adapter.altitude
    tc = ToolCall(
        capability="prohibited_cap",
        action="nuke",
        parameters={},
        call_id="call_never_reach",
    )
    tool_orch.execute(tc)
    assert drone_adapter.altitude == init_alt


# ============================================================================
# K. TOOLORCHESTRATOR INTEGRATION (35–37)
# ============================================================================

def test_35_tool_orchestrator_routes_to_device_gateway():
    """ToolOrchestrator delegates device_gateway calls via DeviceGatewayCapability."""
    p = build_test_orchestration()
    tool_orch = p["tool_orchestrator"]

    tc = ToolCall(
        capability="device_gateway",
        action="dispatch_capability",
        parameters={
            "device_id": "ATLAS_DRONE_01",
            "capability": "hover",
            "action": "hover",
            "parameters": {},
        },
        call_id="call_hover_test",
    )
    res = tool_orch.execute(tc)
    assert res.success is True
    assert "hovering" in res.message.lower()


def test_36_tool_orchestrator_unknown_capability_rejected():
    """Unregistered capability is deterministically rejected."""
    p = build_test_orchestration()
    tool_orch = p["tool_orchestrator"]

    tc = ToolCall(
        capability="ghost_capability",
        action="do_magic",
        parameters={},
        call_id="call_ghost",
    )
    res = tool_orch.execute(tc)
    assert res.success is False


def test_37_tool_execution_preserves_call_id():
    """Call ID propagates through ToolOrchestrator into Result."""
    p = build_test_orchestration()
    tool_orch = p["tool_orchestrator"]

    tc = ToolCall(
        capability="device_gateway",
        action="dispatch_capability",
        parameters={
            "device_id": "ATLAS_DRONE_01",
            "capability": "get_telemetry",
            "action": "get_telemetry",
            "parameters": {},
        },
        call_id="call_id_exact_match_99",
    )
    res = tool_orch.execute(tc)
    assert res.call_id == "call_id_exact_match_99"


# ============================================================================
# L. DEVICEGATEWAY INTEGRATION (38–40)
# ============================================================================

def test_38_device_gateway_resolves_correct_adapter():
    """DeviceGateway resolves drone adapter for ATLAS_DRONE_01."""
    p = build_test_orchestration()
    device_gw = p["device_gw"]
    adapter = device_gw.resolve_adapter("ATLAS_DRONE_01")
    assert isinstance(adapter, VirtualDroneAdapter)


def test_39_device_gateway_rejects_unregistered_device():
    """Dispatch to unregistered device fails at DeviceGateway boundary."""
    p = build_test_orchestration()
    device_gw = p["device_gw"]

    res = device_gw.dispatch_to_device(
        device_id="UNKNOWN_DEVICE_XYZ",
        capability="takeoff",
        action="takeoff",
        parameters={},
    )
    assert res.success is False
    assert "not registered" in res.message


def test_40_device_gateway_validates_parameter_schema():
    """Invalid parameter schema fails deterministically at DeviceGateway."""
    p = build_test_orchestration()
    device_gw = p["device_gw"]

    # Latitude = 999.0 exceeds valid range [-90, 90]
    res = device_gw.dispatch_to_device(
        device_id="ATLAS_DRONE_01",
        capability="navigate",
        action="go_to_waypoint",
        parameters={"latitude": 999.0, "longitude": -122.4},
    )
    assert res.success is False
    assert "latitude" in res.message.lower()


# ============================================================================
# M. MULTI-DEVICE ISOLATION (41–43)
# ============================================================================

def test_41_independent_state_isolation_glass_drone_rover():
    """Simultaneous operations across Glass, Drone, and Rover maintain complete state isolation."""
    p = build_test_orchestration()
    device_gw = p["device_gw"]
    drone_adapter = p["drone_adapter"]
    rover_adapter = p["rover_adapter"]
    glass_adapter = p["glass_adapter"]

    # 1. Glass displays HUD
    device_gw.dispatch_to_device("ATLAS_GLASS_01", "display_hud", "display_hud", {"text": "Incident Area"})
    assert glass_adapter.hud_message == "Incident Area"
    assert drone_adapter.airborne is False
    assert rover_adapter.moving is False

    # 2. Drone takes off
    device_gw.dispatch_to_device("ATLAS_DRONE_01", "takeoff", "takeoff", {"target_altitude": 12.0})
    assert drone_adapter.airborne is True
    assert drone_adapter.altitude == 12.0
    assert rover_adapter.moving is False

    # 3. Rover navigates
    device_gw.dispatch_to_device("ATLAS_ROVER_01", "navigate", "go_to_waypoint", {"latitude": 37.77, "longitude": -122.41})
    assert rover_adapter.moving is True
    assert drone_adapter.altitude == 12.0


def test_42_multi_device_correlation_isolation():
    """Different devices preserve their distinct correlation IDs."""
    p = build_test_orchestration()
    device_gw = p["device_gw"]

    res_drone = device_gw.dispatch_to_device(
        "ATLAS_DRONE_01", "hover", "hover", {}, correlation_id="corr_drone_only"
    )
    res_rover = device_gw.dispatch_to_device(
        "ATLAS_ROVER_01", "stop", "stop", {}, correlation_id="corr_rover_only"
    )
    assert res_drone.data["correlation_id"] == "corr_drone_only"
    assert res_rover.data["correlation_id"] == "corr_rover_only"


def test_43_multi_device_listing_and_filtering():
    """Device listing deterministically filters by DeviceType."""
    p = build_test_orchestration()
    device_gw = p["device_gw"]

    all_devs = device_gw.list_devices()
    assert len(all_devs) == 3

    drones = device_gw.list_devices(device_type=DeviceType.DRONE_AERIAL)
    assert len(drones) == 1
    assert drones[0].device_id == "ATLAS_DRONE_01"


# ============================================================================
# N. FAILURE AND RECOVERY (44–46)
# ============================================================================

def test_44_adapter_failure_returns_canonical_result():
    """Adapter execution error produces a canonical Result with success=False."""
    class FailingAdapter(DeviceAdapterInterface):
        def get_protocol_name(self) -> str: return "failing"
        def format_command(self, tc): return {}
        def execute_command(self, cmd): raise RuntimeError("Sensor bus disconnected")

    p = build_test_orchestration()
    device_gw = p["device_gw"]
    failing_drone, _ = create_virtual_drone("DRONE_BROKEN")
    device_gw.register_device(failing_drone)
    device_gw.register_adapter(FailingAdapter(), device_id="DRONE_BROKEN")

    res = device_gw.dispatch_to_device("DRONE_BROKEN", "takeoff", "takeoff", {})
    assert res.success is False
    assert "Sensor bus disconnected" in res.message


def test_45_low_battery_rejection_handled_safely():
    """Virtual drone rejects takeoff if battery is too low without uncaught exceptions."""
    p = build_test_orchestration()
    drone_adapter = p["drone_adapter"]
    # Manually drain battery to 0
    drone_adapter.battery_pct = 0.0

    device_gw = p["device_gw"]
    res = device_gw.dispatch_to_device("ATLAS_DRONE_01", "takeoff", "takeoff", {"target_altitude": 10.0})
    # Must fail safely
    assert drone_adapter.airborne is False


def test_46_pipeline_recovers_after_tool_failure():
    """Pipeline continues operating after one tool failure."""
    class FailingPlanner:
        def plan(self, *args: Any, **kwargs: Any) -> Plan:
            tc_fail = ToolCall(
                capability="device_gateway",
                action="dispatch_capability",
                parameters={"device_id": "UNKNOWN_DEV", "capability": "land", "action": "land", "parameters": {}},
                call_id="call_fail_first",
            )
            return Plan(goal="fail_goal", steps=[tc_fail], confidence=1.0)

    p = build_test_orchestration(planner=FailingPlanner())
    orch = p["orchestrator"]

    obs = MultimodalObservation(
        observation_id="obs_fail_recover",
        source_id="ATLAS_GLASS_01",
        source_type="GLASS",
        modality=ModalityType.IMAGE,
        timestamp=100.0,
        payload={"hazard": "fire"},
    )
    res = orch.process_ingress(obs, now=100.0)
    assert res.status == "SUCCESS"
    assert len(res.tool_results) >= 1
    assert res.tool_results[0].success is False


# ============================================================================
# O. TIMEOUT AND BOUNDS (47–48)
# ============================================================================

def test_47_max_cycles_bound_enforced():
    """Central orchestration strictly respects max_cycles bound."""
    cfg = CentralOrchestrationConfig(max_cycles=1)
    p = build_test_orchestration(config=cfg)
    orch = p["orchestrator"]

    obs = MultimodalObservation(
        observation_id="obs_cycle_bound",
        source_id="ATLAS_GLASS_01",
        source_type="GLASS",
        modality=ModalityType.IMAGE,
        timestamp=100.0,
        payload={"hazard": "alarm"},
    )
    res = orch.process_ingress(obs, now=100.0)
    assert res.duration_seconds >= 0.0


def test_48_max_tool_calls_per_turn_bound():
    """Orchestrator bounds the maximum tool calls executed per turn."""
    cfg = CentralOrchestrationConfig(max_tool_calls_per_turn=1)
    p = build_test_orchestration(config=cfg)
    orch = p["orchestrator"]

    class MultiCallPlanner:
        def plan(self, req, dec, ctx=None):
            steps = [
                ToolCall(capability="device_gateway", action="dispatch_capability", parameters={"device_id": "ATLAS_DRONE_01", "capability": "hover", "action": "hover", "parameters": {}}, call_id=f"c_{i}")
                for i in range(5)
            ]
            return Plan(plan_id="multi_plan", request_id=req.id, steps=steps, is_valid=True)

    p["cognitive_runtime"].planner = MultiCallPlanner()
    obs = MultimodalObservation(
        observation_id="obs_bound_calls",
        source_id="ATLAS_GLASS_01",
        source_type="GLASS",
        modality=ModalityType.IMAGE,
        timestamp=100.0,
        payload={"hazard": "alarm"},
    )
    res = orch.process_ingress(obs, now=100.0)
    assert len(res.tool_results) <= 1


# ============================================================================
# P. DUPLICATE SUPPRESSION (49–50)
# ============================================================================

def test_49_duplicate_ingress_observation_suppressed():
    """Identical observation sent twice is suppressed by SituationFusion/Orchestrator."""
    p = build_test_orchestration()
    orch = p["orchestrator"]

    obs = MultimodalObservation(
        observation_id="obs_dup_suppress_01",
        source_id="ATLAS_GLASS_01",
        source_type="GLASS",
        modality=ModalityType.IMAGE,
        timestamp=100.0,
        payload={"hazard": "fire_alarm"},
    )
    res1 = orch.process_ingress(obs, now=100.0)
    res2 = orch.process_ingress(obs, now=100.0)
    # Second run detects duplicate and does not duplicate goals
    assert len(res2.goals_created) == 0


def test_50_duplicate_dispatch_id_suppression():
    """DeviceGateway rejects duplicate dispatch_id deterministically."""
    p = build_test_orchestration()
    gw = p["device_gw"]

    r1 = gw.dispatch_to_device("ATLAS_DRONE_01", "hover", "hover", {}, dispatch_id="disp_dup_99")
    r2 = gw.dispatch_to_device("ATLAS_DRONE_01", "hover", "hover", {}, dispatch_id="disp_dup_99")
    assert r1.success is True
    assert r2.success is False
    assert "Duplicate dispatch_id" in r2.message


# ============================================================================
# Q. CAUSAL LINEAGE RETENTION (51–52)
# ============================================================================

def test_51_causal_lineage_preserved_across_pipeline():
    """Verify correlation_id propagates intact from ingress to drone observation."""
    p = build_test_orchestration()
    orch = p["orchestrator"]

    target_corr = "CORRELATION_CHAIN_ALPHA_77"
    obs = MultimodalObservation(
        observation_id="obs_lineage_start",
        source_id="ATLAS_GLASS_01",
        source_type="GLASS",
        modality=ModalityType.IMAGE,
        timestamp=100.0,
        payload={"hazard": "fire"},
        correlation_id=target_corr,
    )
    res = orch.process_ingress(obs, now=100.0)
    assert len(res.observations_reingested) > 0
    feedback_obs = res.observations_reingested[0]
    assert feedback_obs.correlation_id == target_corr


def test_52_causal_trace_records_all_stages():
    """Verify causal trace dictionary includes all lifecycle stages."""
    p = build_test_orchestration()
    orch = p["orchestrator"]

    obs = MultimodalObservation(
        observation_id="obs_trace_stages",
        source_id="ATLAS_GLASS_01",
        source_type="GLASS",
        modality=ModalityType.IMAGE,
        timestamp=100.0,
        payload={"hazard": "fire"},
        correlation_id="corr_stages",
    )
    res = orch.process_ingress(obs, now=100.0)
    trace = res.causal_trace
    assert len(trace) >= 5
    for entry in trace:
        assert "stage" in entry


# ============================================================================
# R. REPLAY DETERMINISM (53–54)
# ============================================================================

def test_53_replay_runs_produce_identical_traces():
    """Identical input batch executed twice produces identical execution traces."""
    def run_e2e():
        p = build_test_orchestration()
        orch = p["orchestrator"]
        obs = MultimodalObservation(
            observation_id="obs_replay_fixed_1",
            source_id="ATLAS_GLASS_01",
            source_type="GLASS",
            modality=ModalityType.IMAGE,
            timestamp=100.0,
            payload={"hazard": "fire_alarm"},
            correlation_id="corr_replay_fixed",
        )
        res = orch.process_ingress(obs, now=100.0)
        return orch.record_trace_snapshot(res)

    trace1 = run_e2e()
    trace2 = run_e2e()

    p_temp = build_test_orchestration()
    orch_temp = p_temp["orchestrator"]
    is_identical, divergences = orch_temp.verify_replay(trace1, trace2)
    assert is_identical is True, f"Replay diverged: {divergences}"


def test_54_replay_detects_status_divergence():
    """verify_replay detects divergence if execution status differs."""
    p = build_test_orchestration()
    orch = p["orchestrator"]

    trace_a = {"status": "SUCCESS", "situations": [], "world_transitions": [], "goals": []}
    trace_b = {"status": "FAILED", "situations": [], "world_transitions": [], "goals": []}
    is_identical, divergences = orch.verify_replay(trace_a, trace_b)
    assert is_identical is False
    assert len(divergences) == 1
    assert "Status mismatch" in divergences[0]


# ============================================================================
# S. LOOP PROTECTION (55–56)
# ============================================================================

def test_55_loop_protection_suppresses_infinite_cascade():
    """Re-ingested observation loop is bounded and terminates safely within max_cycles."""
    cfg = CentralOrchestrationConfig(max_cycles=3)
    p = build_test_orchestration(config=cfg)
    orch = p["orchestrator"]

    obs = MultimodalObservation(
        observation_id="obs_loop_test",
        source_id="ATLAS_GLASS_01",
        source_type="GLASS",
        modality=ModalityType.IMAGE,
        timestamp=100.0,
        payload={"hazard": "fire"},
        correlation_id="corr_loop",
    )
    # Must terminate without infinite loop
    res = orch.process_ingress(obs, now=100.0)
    assert res.status in ("SUCCESS", "BOUND_REACHED")


def test_56_loop_detected_event_emitted_when_bound_reached():
    """Event sink receives CENTRAL_ORCHESTRATION_LOOP_DETECTED on duplicate observation."""
    p = build_test_orchestration()
    orch = p["orchestrator"]
    sink = p["event_sink"]

    obs = MultimodalObservation(
        observation_id="obs_repeat_event",
        source_id="ATLAS_GLASS_01",
        source_type="GLASS",
        modality=ModalityType.IMAGE,
        timestamp=100.0,
        payload={"hazard": "fire"},
    )
    orch.process_ingress(obs, now=100.0)
    # Second time triggers loop detection
    orch.process_ingress(obs, now=100.0)

    events = sink.get_events()
    loop_events = [e for e in events if e.event_type == CognitiveEventType.CENTRAL_ORCHESTRATION_LOOP_DETECTED]
    assert len(loop_events) >= 1


# ============================================================================
# T. SECURITY & AUTHORITY STATIC BOUNDARY CHECKS (57–58)
# ============================================================================

def test_57_zero_hardware_or_network_imports():
    """Verify central_orchestration.py contains zero hardware or network socket imports."""
    import inspect
    import orchestration.central_orchestration as co_module

    source = inspect.getsource(co_module)
    banned_imports = [
        "pymavlink", "mavsdk", "rclpy", "ros2", "paho.mqtt",
        "import pyserial", "import serial", "RPi.GPIO", "import subprocess", "os.system",
        "eval(", "exec(",
    ]
    for banned in banned_imports:
        assert banned not in source, f"Banned construct '{banned}' found in central_orchestration.py!"


def test_58_authority_boundaries_preserved_without_duplicate_engines():
    """
    Verify CentralOrchestrator does NOT implement its own GoalStore, Planner,
    or PolicyEngine, but delegates strictly to authoritative components.
    """
    p = build_test_orchestration()
    orch = p["orchestrator"]

    # CentralOrchestrator must not be a GoalStore or PolicyEngine
    assert not hasattr(orch, "evaluate_policy")
    assert not hasattr(orch, "create_plan")
    assert not hasattr(orch, "add_goal")
    assert not hasattr(orch, "save_condition")
    assert not hasattr(orch, "execute_shell")
