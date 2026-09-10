import asyncio
import logging
import os
import secrets
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor

from core.connection_manager import BoundedConnectionManager
from config.settings import AtlasSettings, get_settings
from core.models.runtime import CognitiveEvent, CognitiveEventType, CognitiveStage
from runtime.event_sink import InMemoryEventSink
from runtime.cognitive_runtime import CognitiveRuntime
from runtime.file_trace_store import FileTraceStore
from safety.policy_engine import StandardPolicyEngine
from tools.capability_registry import CapabilityRegistry
from tools.tool_orchestrator import ToolOrchestrator
from orchestration.device_gateway import DeviceGateway, DeviceGatewayCapability
from world.store import SQLiteWorldStateStore, InMemoryWorldStateStore
from world.updater import DeterministicWorldStateUpdater
from world.resolver import DeterministicConflictResolver
from goals.store import SQLiteGoalStore, InMemoryGoalStore
from goals.manager import AutonomousGoalManager
from goals.execution_engine import GoalExecutionEngine
from autonomy.classifier import DeterministicEventClassifier
from autonomy.relevance import DeterministicRelevanceEngine
from autonomy.coordinator import EventDrivenAutonomyCoordinator
from anticipation.coordinator import AnticipatoryPlanningCoordinator
from anticipation.analyzer import DeterministicAnticipatoryAnalyzer
from orchestration.input_gateway import CentralInputGateway
from orchestration.fusion_engine import SituationFusionEngine
from orchestration.central_orchestration import CentralOrchestrator, CentralOrchestrationConfig
from orchestration.virtual_devices import (
    create_virtual_drone,
    create_virtual_glass,
    create_virtual_rover,
    create_virtual_vision,
    VirtualDroneAdapter,
    VirtualGlassAdapter,
    VirtualRoverAdapter,
    VirtualVisionAdapter,
)
from perception.registry import PerceptionProviderRegistry
from perception.normalizer import PerceptionObservationNormalizer

logger = logging.getLogger("atlas.runtime")


@dataclass
class AtlasApplicationState:
    """
    Application container holding authoritative singletons for the running server.
    All API endpoints resolve state exclusively from this container.
    """
    settings: AtlasSettings
    ready: bool = False
    shutting_down: bool = False

    # Event sinks & connection managers
    event_sink: Optional[InMemoryEventSink] = None
    telemetry_manager: Optional[BoundedConnectionManager] = None
    hud_manager: Optional[BoundedConnectionManager] = None

    # Authoritative persistence & subsystems
    world_store: Optional[Any] = None
    world_updater: Optional[DeterministicWorldStateUpdater] = None
    goal_store: Optional[Any] = None
    goal_manager: Optional[AutonomousGoalManager] = None
    trace_store: Optional[FileTraceStore] = None

    # Gateways & Orchestration
    device_gateway: Optional[DeviceGateway] = None
    input_gateway: Optional[CentralInputGateway] = None
    fusion_engine: Optional[SituationFusionEngine] = None
    autonomy_coordinator: Optional[EventDrivenAutonomyCoordinator] = None
    anticipation_coordinator: Optional[AnticipatoryPlanningCoordinator] = None

    # Core Execution & Policy
    policy_engine: Optional[StandardPolicyEngine] = None
    tool_orchestrator: Optional[ToolOrchestrator] = None
    cognitive_runtime: Optional[CognitiveRuntime] = None
    central_orchestrator: Optional[CentralOrchestrator] = None

    # Multi-Product Situation & Mission Intelligence (Phase 6.4)
    situation_intelligence: Optional[Any] = None
    mission_coordinator: Optional[Any] = None

    # Multimodal Perception Contracts (Phase 6.5a)
    perception_registry: Optional[PerceptionProviderRegistry] = None
    perception_normalizer: Optional[PerceptionObservationNormalizer] = None

    # Bounded ThreadPool for CPU/LLM offloading (prevent event loop starvation)
    executor: Optional[ThreadPoolExecutor] = None


def initialize_application_state(
    settings: Optional[AtlasSettings] = None,
    in_memory_stores: bool = False,
) -> AtlasApplicationState:
    """
    Build and compose the complete ATLAS production runtime.
    Preserves strict single-authority boundaries across all subsystems.
    """
    cfg = settings or get_settings()
    app_state = AtlasApplicationState(settings=cfg)

    # 1. ThreadPoolExecutor for blocking cognitive operations
    app_state.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="atlas_worker_")

    # 2. Connection Managers
    app_state.telemetry_manager = BoundedConnectionManager(max_connections=100)
    app_state.hud_manager = BoundedConnectionManager(max_connections=50)

    # 3. Observability Event Sink
    # Hook event sink to broadcast to WebSocket telemetry clients
    event_sink = InMemoryEventSink()

    def _on_cognitive_event(event: CognitiveEvent) -> None:
        if app_state.telemetry_manager:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(
                    app_state.telemetry_manager.broadcast({
                        "type": "cognitive_event",
                        "event_id": event.event_id,
                        "event_type": event.event_type.value if hasattr(event.event_type, "value") else str(event.event_type),
                        "turn_id": event.turn_id,
                        "timestamp": event.timestamp,
                        "stage": event.stage.value if hasattr(event.stage, "value") else str(event.stage),
                        "metadata": event.metadata,
                    })
                )
            except RuntimeError:
                pass  # No running event loop in thread

    event_sink.subscribe(_on_cognitive_event)
    app_state.event_sink = event_sink

    # 4. Persistence Stores
    if in_memory_stores or cfg.app_env == "test":
        app_state.world_store = InMemoryWorldStateStore()
        app_state.goal_store = InMemoryGoalStore()
    else:
        db_dir = os.path.abspath(cfg.database_dir)
        os.makedirs(db_dir, exist_ok=True)
        world_db_path = os.path.join(db_dir, "world_state.db")
        goal_db_path = os.path.join(db_dir, "goals.db")
        app_state.world_store = SQLiteWorldStateStore(db_path=world_db_path)
        app_state.goal_store = SQLiteGoalStore(db_path=goal_db_path)

    trace_dir = os.path.abspath(cfg.trace_dir)
    os.makedirs(trace_dir, exist_ok=True)
    app_state.trace_store = FileTraceStore(directory=trace_dir)

    # 5. World State Updater
    app_state.world_updater = DeterministicWorldStateUpdater(
        store=app_state.world_store,
        resolver=DeterministicConflictResolver(),
        event_sink=event_sink,
    )

    # 6. Device Gateway & Capability Registration
    device_gw = DeviceGateway(event_sink=event_sink)
    app_state.device_gateway = device_gw

    # 7. Policy Engine & Tool Orchestrator
    policy_engine = StandardPolicyEngine(demo_mode=cfg.demo_mode)
    app_state.policy_engine = policy_engine

    registry = CapabilityRegistry()
    registry.register("device_gateway", DeviceGatewayCapability(device_gw))
    if cfg.demo_mode:
        from computer.demo_app_capability import DemoAppCapability
        registry.register("computer_app", DemoAppCapability())

    tool_orchestrator = ToolOrchestrator(
        registry=registry,
        policy_engine=policy_engine,
        demo_mode=cfg.demo_mode,
    )
    app_state.tool_orchestrator = tool_orchestrator

    # 8. Execution Engine & Cognitive Runtime
    from brain.router import Router
    from brain.execution import StandardExecutionEngine

    router = Router(
        registry=registry,
        orchestrator=tool_orchestrator,
        policy_engine=policy_engine,
        demo_mode=cfg.demo_mode,
    )
    execution_engine = StandardExecutionEngine(
        router=router,
        policy_engine=policy_engine,
        demo_mode=cfg.demo_mode,
    )

    cognitive_runtime = CognitiveRuntime(
        execution_engine=execution_engine,
        policy_engine=policy_engine,
        event_sink=event_sink,
        demo_mode=cfg.demo_mode,
    )
    app_state.cognitive_runtime = cognitive_runtime

    # 9. Goal Manager
    goal_exec = GoalExecutionEngine(
        runtime=cognitive_runtime,
        store=app_state.goal_store,
        event_sink=event_sink,
    )
    app_state.goal_manager = AutonomousGoalManager(
        store=app_state.goal_store,
        execution_engine=goal_exec,
        event_sink=event_sink,
    )

    # 10. Autonomy & Anticipation
    app_state.autonomy_coordinator = EventDrivenAutonomyCoordinator(
        classifier=DeterministicEventClassifier(),
        relevance_engine=DeterministicRelevanceEngine(),
        goal_manager=app_state.goal_manager,
        goal_store=app_state.goal_store,
        policy_engine=policy_engine,
        world_state_store=app_state.world_store,
        event_sink=event_sink,
    )

    app_state.anticipation_coordinator = AnticipatoryPlanningCoordinator(
        analyzer=DeterministicAnticipatoryAnalyzer(),
        goal_manager=app_state.goal_manager,
        policy_engine=policy_engine,
        world_state_store=app_state.world_store,
        event_sink=event_sink,
    )

    # 11. Ingress & Situation Fusion
    app_state.input_gateway = CentralInputGateway(event_sink=event_sink)
    app_state.fusion_engine = SituationFusionEngine(
        world_state_store=app_state.world_store,
        event_sink=event_sink,
    )

    # 12. Central Orchestrator
    app_state.central_orchestrator = CentralOrchestrator(
        input_gateway=app_state.input_gateway,
        fusion_engine=app_state.fusion_engine,
        world_updater=app_state.world_updater,
        world_store=app_state.world_store,
        autonomy_coordinator=app_state.autonomy_coordinator,
        goal_manager=app_state.goal_manager,
        cognitive_runtime=app_state.cognitive_runtime,
        tool_orchestrator=tool_orchestrator,
        device_gateway=device_gw,
        event_sink=event_sink,
        config=CentralOrchestrationConfig(),
    )

    # 13. Multi-Product Situation & Mission Intelligence (Phase 6.4)
    from mission.situation_intelligence import MultiProductSituationIntelligenceEngine
    from mission.coordinator import MissionCoordinator
    from mission.planner import MissionPlanner
    from mission.role_selector import ProductRoleSelector

    sit_intel = MultiProductSituationIntelligenceEngine()
    role_sel = ProductRoleSelector()
    m_planner = MissionPlanner(role_selector=role_sel)
    m_coord = MissionCoordinator(goal_manager=app_state.goal_manager, planner=m_planner)

    app_state.situation_intelligence = sit_intel
    app_state.mission_coordinator = m_coord

    # 13c. Authoritative Live State Capability (Phase 6.6)
    from tools.live_state_capability import LiveStateCapability
    live_state = LiveStateCapability(
        device_gateway=app_state.device_gateway,
        world_store=app_state.world_store,
        situation_engine=app_state.fusion_engine,
        goal_manager=app_state.goal_manager,
    )
    registry.register("live_state", live_state)
    registry.register("device", live_state)

    # Wire live authorities into CognitiveRuntime
    cognitive_runtime.device_gateway = app_state.device_gateway
    cognitive_runtime.world_store = app_state.world_store
    cognitive_runtime.situation_engine = app_state.fusion_engine
    cognitive_runtime.goal_manager = app_state.goal_manager
    cognitive_runtime.live_state_capability = live_state
    if hasattr(cognitive_runtime.execution_engine, "router") and hasattr(cognitive_runtime.execution_engine.router, "registry"):
        cognitive_runtime.execution_engine.router.registry.register("live_state", live_state)
        cognitive_runtime.execution_engine.router.registry.register("device", live_state)

    # 13d. Multimodal Perception Contracts (Phase 6.5a)
    app_state.perception_registry = PerceptionProviderRegistry()
    app_state.perception_normalizer = PerceptionObservationNormalizer()

    # 14. Register Simulation Devices if enabled
    if cfg.is_simulation():
        logger.info("Initializing Virtual Edge Devices in Simulation Mode...")
        vision, vision_adapter = create_virtual_vision("ATLAS_VISION_01")
        drone, drone_adapter = create_virtual_drone("ATLAS_DRONE_01")
        glass, glass_adapter = create_virtual_glass("ATLAS_GLASS_01")
        rover, rover_adapter = create_virtual_rover("ATLAS_ROVER_01")

        device_gw.register_device(vision)
        device_gw.register_device(drone)
        device_gw.register_device(glass)
        device_gw.register_device(rover)

        if app_state.input_gateway:
            app_state.input_gateway.register_device(vision)
            app_state.input_gateway.register_device(drone)
            app_state.input_gateway.register_device(glass)
            app_state.input_gateway.register_device(rover)

        device_gw.register_adapter(vision_adapter, device_id="ATLAS_VISION_01")
        device_gw.register_adapter(drone_adapter, device_id="ATLAS_DRONE_01")
        device_gw.register_adapter(glass_adapter, device_id="ATLAS_GLASS_01")
        device_gw.register_adapter(rover_adapter, device_id="ATLAS_ROVER_01")

    # Publish lifecycle event
    event_sink.publish(
        CognitiveEvent(
            event_id=f"evt_startup_{int(time.time()*1000)}",
            turn_id="turn_system",
            session_id="session_system",
            stage=CognitiveStage.RECEIVED,
            event_type=CognitiveEventType.APPLICATION_STARTED,
            timestamp=time.time(),
            metadata={"app_env": cfg.app_env, "simulation_mode": cfg.is_simulation()},
        )
    )

    app_state.ready = True
    event_sink.publish(
        CognitiveEvent(
            event_id=f"evt_ready_{int(time.time()*1000)}",
            turn_id="turn_system",
            session_id="session_system",
            stage=CognitiveStage.COMPLETED,
            event_type=CognitiveEventType.APPLICATION_READY,
            timestamp=time.time(),
            metadata={"status": "READY"},
        )
    )
    logger.info("ATLAS Central Runtime initialized and READY.")
    return app_state


async def shutdown_application_state(app_state: AtlasApplicationState) -> None:
    """Cleanly shutdown application resources and active connections."""
    logger.info("Initiating ATLAS Central Runtime shutdown...")
    app_state.shutting_down = True
    app_state.ready = False

    if app_state.event_sink:
        app_state.event_sink.publish(
            CognitiveEvent(
                event_id=f"evt_stopping_{int(time.time()*1000)}",
                turn_id="turn_system",
                session_id="session_system",
                stage=CognitiveStage.RECEIVED,
                event_type=CognitiveEventType.APPLICATION_SHUTTING_DOWN,
                timestamp=time.time(),
                metadata={"reason": "Server shutdown"},
            )
        )

    # Close WebSockets
    if app_state.telemetry_manager:
        await app_state.telemetry_manager.close_all()
    if app_state.hud_manager:
        await app_state.hud_manager.close_all()

    # Shutdown thread executor
    if app_state.executor:
        app_state.executor.shutdown(wait=False)

    if app_state.event_sink:
        app_state.event_sink.publish(
            CognitiveEvent(
                event_id=f"evt_stopped_{int(time.time()*1000)}",
                turn_id="turn_system",
                session_id="session_system",
                stage=CognitiveStage.COMPLETED,
                event_type=CognitiveEventType.APPLICATION_STOPPED,
                timestamp=time.time(),
                metadata={"status": "STOPPED"},
            )
        )
    logger.info("ATLAS Central Runtime shutdown complete.")
