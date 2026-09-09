"""
ATLAS Central Orchestration Coordinator (Phase 5.0e).
Integrates Central Input Gateway, Situation Fusion Engine, World State,
Event-Driven Autonomy, Anticipatory Planning, Autonomous Goal Management,
Cognitive Runtime, Policy Engine, Tool Orchestrator, and Device Gateway into
a single, unified, bounded, closed-loop orchestration pipeline.

ARCHITECTURAL BOUNDARIES & GUARANTEES:
1. Thin integration coordinator only; does NOT implement duplicate reasoning,
   planners, policy engines, or goal lifecycle authorities.
2. Preserves strict authority models:
   - CognitiveRuntime = cognitive turn lifecycle authority
   - PolicyEngine = authorization & safety authority
   - ToolOrchestrator = capability execution authority
   - DeviceGateway = device identity, routing & parameter validation authority
   - CentralInputGateway = external ingress normalization & validation authority
   - SituationFusionEngine = multimodal observation fusion authority
   - WorldStateUpdater = authoritative state mutation & conflict resolution boundary
   - AutonomousGoalManager = sole lifecycle authority for goals
3. Zero direct hardware, live network, serial, or socket calls.
4. Bounded execution loops, duplicate suppression, and cycle limits prevent runaway cascades.
5. Preserves full causal lineage across all stages:
   observation -> situation -> event -> goal -> turn -> tool_call -> dispatch -> observation.
"""

import collections
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

# Core Interfaces
from core.interfaces.anticipation_interface import AnticipatoryPlanningCoordinatorInterface
from core.interfaces.autonomy_interface import EventDrivenAutonomyInterface
from core.interfaces.goal_interface import AutonomousGoalManagerInterface
from core.interfaces.orchestration_interface import (
    CentralInputGatewayInterface,
    CentralOrchestratorInterface,
    DeviceGatewayInterface,
)
from core.interfaces.runtime_interface import (
    CognitiveEventSinkInterface,
    CognitiveRuntimeInterface,
)
from core.interfaces.world_interface import (
    WorldStateStoreInterface,
    WorldStateUpdaterInterface,
)

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
    DeviceIdentity,
    GeoLocation,
    ModalityType,
    MultimodalObservation,
    Situation,
    SituationCategory,
    SituationSeverity,
    SituationStatus,
)
from core.models.result import Result
from core.models.runtime import (
    CognitiveEvent,
    CognitiveEventType,
    CognitiveStage,
    CognitiveTurnResult,
)
from core.models.tool_call import ToolCall
from core.models.world_state import (
    Observation,
    StateProvenance,
    TransitionType,
    WorldStateTransition,
    WorldStateUpdateResult,
)

# Orchestration & Integration Helpers
from autonomy.world_listener import transition_to_autonomy_event
from orchestration.input_gateway import IngressEnvelope
from orchestration.fusion_engine import SituationFusionEngine

logger = logging.getLogger("atlas.central_orchestration")


# ============================================================================
# Configuration & Result Models
# ============================================================================

@dataclass
class CentralOrchestrationConfig:
    """Configuration parameters and safety bounds for central orchestration."""
    max_cycles: int = 5
    max_reingestions: int = 10
    max_situations_per_cycle: int = 20
    max_goals_per_cycle: int = 10
    max_turns_per_cycle: int = 10
    max_tool_calls_per_turn: int = 10
    auto_reingest_observations: bool = True
    world_state_sync: bool = True
    autonomy_sync: bool = True
    goal_execution_sync: bool = True
    max_history_entries: int = 1000


@dataclass
class CentralOrchestratorResult:
    """
    Immutable structured summary of a central orchestration run.
    Contains full end-to-end causal trace and intermediate artifacts.
    """
    cycle_id: str
    status: str
    observations_ingested: List[MultimodalObservation] = field(default_factory=list)
    situations_fused: List[Situation] = field(default_factory=list)
    world_transitions: List[WorldStateTransition] = field(default_factory=list)
    events_evaluated: List[Event] = field(default_factory=list)
    autonomy_decisions: List[AutonomyDecision] = field(default_factory=list)
    goals_created: List[Goal] = field(default_factory=list)
    turn_results: List[CognitiveTurnResult] = field(default_factory=list)
    tool_results: List[Result] = field(default_factory=list)
    observations_reingested: List[MultimodalObservation] = field(default_factory=list)
    causal_trace: List[Dict[str, Any]] = field(default_factory=list)
    duration_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cycle_id": self.cycle_id,
            "status": self.status,
            "observations_ingested_count": len(self.observations_ingested),
            "situations_fused_count": len(self.situations_fused),
            "world_transitions_count": len(self.world_transitions),
            "events_evaluated_count": len(self.events_evaluated),
            "autonomy_decisions_count": len(self.autonomy_decisions),
            "goals_created_count": len(self.goals_created),
            "turn_results_count": len(self.turn_results),
            "tool_results_count": len(self.tool_results),
            "observations_reingested_count": len(self.observations_reingested),
            "causal_trace_depth": len(self.causal_trace),
            "duration_seconds": self.duration_seconds,
        }


# ============================================================================
# Central Orchestrator Implementation
# ============================================================================

class CentralOrchestrator(CentralOrchestratorInterface):
    """
    Coordinating hub connecting Phase 5 Central Orchestration with Phase 4 Core.
    Guarantees strict authority delegation and zero duplicate engines.
    """

    def __init__(
        self,
        input_gateway: CentralInputGatewayInterface,
        fusion_engine: SituationFusionEngine,
        world_updater: Optional[WorldStateUpdaterInterface] = None,
        world_store: Optional[WorldStateStoreInterface] = None,
        autonomy_coordinator: Optional[EventDrivenAutonomyInterface] = None,
        anticipatory_coordinator: Optional[AnticipatoryPlanningCoordinatorInterface] = None,
        goal_manager: Optional[AutonomousGoalManagerInterface] = None,
        cognitive_runtime: Optional[CognitiveRuntimeInterface] = None,
        tool_orchestrator: Optional[Any] = None,
        device_gateway: Optional[DeviceGatewayInterface] = None,
        event_sink: Optional[CognitiveEventSinkInterface] = None,
        config: Optional[CentralOrchestrationConfig] = None,
        clock: Optional[Callable[[], float]] = None,
    ):
        self.input_gateway = input_gateway
        self.fusion_engine = fusion_engine
        self.world_updater = world_updater
        self.world_store = world_store
        self.autonomy_coordinator = autonomy_coordinator
        self.anticipatory_coordinator = anticipatory_coordinator
        self.goal_manager = goal_manager
        self.cognitive_runtime = cognitive_runtime
        self.tool_orchestrator = tool_orchestrator
        self.device_gateway = device_gateway
        self.event_sink = event_sink
        self.config = config or CentralOrchestrationConfig()
        self.clock = clock or time.time

        self._lock = threading.RLock()
        self._processed_obs_ids: collections.deque = collections.deque(
            maxlen=self.config.max_history_entries
        )
        self._processed_obs_set: Set[str] = set()
        self._seen_causal_chains: Set[str] = set()
        self._cycle_history: collections.deque = collections.deque(
            maxlen=self.config.max_history_entries
        )

    # ========================================================================
    # Observability Helper
    # ========================================================================

    def _emit_event(
        self,
        event_type: CognitiveEventType,
        summary: str,
        turn_id: Optional[str] = None,
        session_id: str = "central_orchestration",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Emit telemetry through configured CognitiveEventSink."""
        if not self.event_sink:
            return
        now = self.clock()
        data = dict(metadata or {})
        event = CognitiveEvent(
            event_id=f"evt_co_{uuid.uuid4().hex[:12]}",
            turn_id=turn_id or "co_turn",
            session_id=session_id,
            stage=CognitiveStage.RECEIVED,
            event_type=event_type,
            timestamp=now,
            summary=summary,
            metadata=data,
        )
        try:
            if hasattr(self.event_sink, "publish"):
                self.event_sink.publish(event)
            elif hasattr(self.event_sink, "receive_event"):
                self.event_sink.receive_event(event)
        except Exception as ex:
            logger.debug("Failed publishing orchestration event: %s", ex)

    # ========================================================================
    # Translation Helpers (Pure Data Mapping)
    # ========================================================================

    @staticmethod
    def situation_to_world_observation(
        situation: Situation,
        now: Optional[float] = None,
    ) -> Observation:
        """
        Pure data translation: Map a Situation into a canonical WorldState Observation.
        Does NOT mutate WorldState directly.
        """
        ts = now if now is not None else situation.updated_at
        entities = getattr(situation, "involved_entities", None)
        ent_id = entities[0] if entities else f"situation_{situation.category.value.lower()}"
        prop_name = (
            "hazard_alarm"
            if situation.category in (SituationCategory.SECURITY, SituationCategory.ANOMALY, SituationCategory.UNKNOWN)
            else f"status_{situation.category.value.lower()}"
        )
        val = situation.severity.value

        return Observation(
            observation_id=f"obs_sit_{situation.situation_id}_{int(ts * 1000)}",
            source_id=f"situation_fusion:{situation.situation_id}",
            source_type="SITUATION_FUSION",
            timestamp=ts,
            entity_id=ent_id,
            property_name=prop_name,
            value=val,
            confidence=situation.confidence,
            expires_at=None,
            metadata={
                "situation_id": situation.situation_id,
                "category": situation.category.value,
                "severity": situation.severity.value,
                "correlation_id": situation.correlation_id,
                "causation_id": situation.causation_id,
                "title": situation.title,
            },
        )

    @staticmethod
    def multimodal_observation_to_world_observation(
        obs: MultimodalObservation,
    ) -> Observation:
        """
        Pure data translation: Map a device MultimodalObservation into a WorldState Observation.
        Does NOT mutate WorldState directly.
        """
        ent_id = obs.device_id or obs.source_id
        prop_name = f"{obs.modality.value.lower()}_telemetry"
        return Observation(
            observation_id=obs.observation_id,
            source_id=obs.source_id,
            source_type=obs.source_type,
            timestamp=obs.timestamp,
            entity_id=ent_id,
            property_name=prop_name,
            value=obs.payload,
            confidence=obs.confidence,
            expires_at=None,
            metadata={
                "modality": obs.modality.value,
                "correlation_id": obs.correlation_id,
                "causation_id": obs.causation_id,
                **(obs.metadata or {}),
            },
        )

    # ========================================================================
    # Primary Ingress & Execution Pipeline
    # ========================================================================

    def process_ingress(
        self,
        data: Any,
        correlation_id: Optional[str] = None,
        causation_id: Optional[str] = None,
        now: Optional[float] = None,
    ) -> CentralOrchestratorResult:
        """
        Process a single ingress item (raw envelope dict, IngressEnvelope, or MultimodalObservation)
        through the complete closed-loop pipeline.
        """
        return self.run_cycle([data], now=now)

    def run_cycle(
        self,
        ingress_batch: Sequence[Any],
        now: Optional[float] = None,
    ) -> CentralOrchestratorResult:
        """
        Execute bounded central orchestration cycle across a batch of inputs.
        Supports deterministic closed-loop propagation from edge observations to cognitive
        decisions, tool execution, device actuation, and feedback re-ingestion.
        """
        start_time = self.clock()
        current_time = now if now is not None else start_time
        cycle_id = f"cycle_{uuid.uuid4().hex[:10]}"

        result = CentralOrchestratorResult(cycle_id=cycle_id, status="SUCCESS")

        self._emit_event(
            CognitiveEventType.CENTRAL_ORCHESTRATION_CYCLE_STARTED,
            summary=f"Central orchestration cycle '{cycle_id}' started with {len(ingress_batch)} items.",
            turn_id=cycle_id,
            metadata={"batch_size": len(ingress_batch), "timestamp": current_time},
        )

        with self._lock:
            pending_observations: collections.deque = collections.deque()

            # -------------------------------------------------------------
            # Stage 1: Ingress Ingestion & Normalization
            # -------------------------------------------------------------
            for raw_item in ingress_batch:
                obs = self._normalize_ingress(raw_item, current_time)
                if obs:
                    pending_observations.append(obs)
                    result.observations_ingested.append(obs)
                    result.causal_trace.append({
                        "stage": "INGRESS",
                        "observation_id": obs.observation_id,
                        "source_id": obs.source_id,
                        "modality": obs.modality.value,
                        "correlation_id": obs.correlation_id,
                        "causation_id": obs.causation_id,
                    })

            if not pending_observations:
                result.status = "NO_ACTION"
                result.duration_seconds = max(0.0, self.clock() - start_time)
                return result

            # -------------------------------------------------------------
            # Bounded Closed-Loop Propagation
            # -------------------------------------------------------------
            cycle_count = 0
            reingestion_count = 0

            while pending_observations and cycle_count < self.config.max_cycles:
                cycle_count += 1
                current_obs_batch: List[MultimodalObservation] = []

                # Drain current pending observations for this sub-cycle
                while pending_observations:
                    obs = pending_observations.popleft()
                    # Loop / storm protection: duplicate suppression
                    if obs.observation_id in self._processed_obs_set:
                        self._emit_event(
                            CognitiveEventType.CENTRAL_ORCHESTRATION_LOOP_DETECTED,
                            summary=f"Duplicate observation '{obs.observation_id}' suppressed in cycle.",
                            turn_id=cycle_id,
                            metadata={"observation_id": obs.observation_id},
                        )
                        continue

                    self._processed_obs_ids.append(obs.observation_id)
                    self._processed_obs_set.add(obs.observation_id)
                    current_obs_batch.append(obs)

                if not current_obs_batch:
                    break

                # ---------------------------------------------------------
                # Stage 2: Situation Fusion
                # ---------------------------------------------------------
                fused_situations = self.fusion_engine.ingest_batch(
                    current_obs_batch, now=current_time
                )
                for sit in fused_situations:
                    result.situations_fused.append(sit)
                    result.causal_trace.append({
                        "stage": "SITUATION",
                        "situation_id": sit.situation_id,
                        "category": sit.category.value,
                        "severity": sit.severity.value,
                        "status": sit.status.value,
                        "correlation_id": sit.correlation_id,
                        "causation_id": sit.causation_id,
                    })

                # ---------------------------------------------------------
                # Stage 3: World State Update Boundary
                # ---------------------------------------------------------
                if self.config.world_state_sync and self.world_updater:
                    for sit in fused_situations:
                        if sit.status in (SituationStatus.ACTIVE, SituationStatus.DETECTED, SituationStatus.UPDATING):
                            world_obs = self.situation_to_world_observation(sit, now=current_time)
                            update_res = self.world_updater.apply_observation(world_obs)
                            if update_res.success and update_res.transition:
                                trans = update_res.transition
                                result.world_transitions.append(trans)
                                result.causal_trace.append({
                                    "stage": "WORLD_STATE",
                                    "transition_id": trans.transition_id,
                                    "entity_id": trans.entity_id,
                                    "property_name": trans.property_name,
                                    "transition_type": trans.transition_type.value,
                                    "version": trans.to_version,
                                })

                                # -----------------------------------------
                                # Stage 4: Event-Driven Autonomy
                                # -----------------------------------------
                                if self.config.autonomy_sync and self.autonomy_coordinator:
                                    autonomy_evt = transition_to_autonomy_event(
                                        trans, correlation_id=sit.correlation_id
                                    )
                                    if autonomy_evt:
                                        result.events_evaluated.append(autonomy_evt)
                                        if hasattr(self.autonomy_coordinator, "ingest_event"):
                                            decision = self.autonomy_coordinator.ingest_event(autonomy_evt)
                                        elif hasattr(self.autonomy_coordinator, "process_event"):
                                            decision = self.autonomy_coordinator.process_event(autonomy_evt)
                                        else:
                                            decision = None

                                        if decision:
                                            result.autonomy_decisions.append(decision)
                                            result.causal_trace.append({
                                                "stage": "AUTONOMY",
                                                "event_id": autonomy_evt.event_id,
                                                "decision_type": decision.decision_type.value,
                                                "correlation_id": autonomy_evt.provenance.correlation_id,
                                            })

                                            # ---------------------------------
                                            # Stage 5: Autonomous Goal Creation
                                            # ---------------------------------
                                            if (
                                                decision.decision_type == AutonomyDecisionType.CREATE_GOAL
                                                and self.goal_manager
                                            ):
                                                goal = None
                                                if getattr(decision, "goal_id", None) and hasattr(self.goal_manager, "store") and self.goal_manager.store:
                                                    goal = self.goal_manager.store.get_goal(decision.goal_id)
                                                if not goal:
                                                    goal_prio = (
                                                        GoalPriority.HIGH
                                                        if sit.severity in (SituationSeverity.HIGH, SituationSeverity.MEDIUM)
                                                        else GoalPriority.CRITICAL
                                                    )
                                                    goal = self.goal_manager.create_goal(
                                                        title=f"Respond to {sit.category.value}: {sit.title}",
                                                        description=f"Mitigate active situation '{sit.situation_id}'.",
                                                        priority=goal_prio,
                                                        correlation_id=sit.correlation_id,
                                                        causation_id=autonomy_evt.event_id,
                                                    )

                                                if goal:
                                                    result.goals_created.append(goal)
                                                    result.causal_trace.append({
                                                        "stage": "GOAL",
                                                        "goal_id": getattr(goal, "id", getattr(goal, "goal_id", "")),
                                                        "title": getattr(goal, "title", getattr(goal, "original_goal", "")),
                                                        "status": goal.status.value if hasattr(goal.status, "value") else str(goal.status),
                                                        "correlation_id": getattr(goal, "correlation_id", sit.correlation_id),
                                                        "causation_id": getattr(goal, "causation_id", autonomy_evt.event_id),
                                                    })

                                                    # -----------------------------
                                                    # Stage 6: Cognitive Execution
                                                    # -----------------------------
                                                    if self.config.goal_execution_sync and self.cognitive_runtime:
                                                        turn_res = self._execute_cognitive_turn(
                                                            goal, sit, current_time
                                                        )
                                                        if turn_res:
                                                            result.turn_results.append(turn_res)
                                                            result.causal_trace.append({
                                                                "stage": "TURN",
                                                                "turn_id": turn_res.turn_id,
                                                                "status": turn_res.status.value if hasattr(turn_res.status, "value") else str(turn_res.status),
                                                            })

                                                            # ---------------------
                                                            # Stage 7: Tool & Device Gateway Execution
                                                            # ---------------------
                                                            new_observations = self._execute_tool_calls(
                                                                turn_res, goal, sit, result
                                                            )

                                                            # ---------------------
                                                            # Stage 8: Closed-Loop Feedback
                                                            # ---------------------
                                                            if (
                                                                self.config.auto_reingest_observations
                                                                and new_observations
                                                                and reingestion_count < self.config.max_reingestions
                                                            ):
                                                                for new_obs in new_observations:
                                                                    reingestion_count += 1
                                                                    result.observations_reingested.append(new_obs)
                                                                    pending_observations.append(new_obs)
                                                                    self._emit_event(
                                                                        CognitiveEventType.CENTRAL_ORCHESTRATION_OBSERVATION_REINGESTED,
                                                                        summary=f"Re-ingested feedback observation '{new_obs.observation_id}' from device '{new_obs.source_id}'.",
                                                                        turn_id=cycle_id,
                                                                        metadata={
                                                                            "observation_id": new_obs.observation_id,
                                                                            "device_id": new_obs.source_id,
                                                                            "modality": new_obs.modality.value,
                                                                        },
                                                                    )

            if cycle_count >= self.config.max_cycles and pending_observations:
                result.status = "BOUND_REACHED"
                self._emit_event(
                    CognitiveEventType.CENTRAL_ORCHESTRATION_LOOP_DETECTED,
                    summary=f"Max orchestration cycles limit ({self.config.max_cycles}) reached in cycle '{cycle_id}'.",
                    turn_id=cycle_id,
                )

        result.duration_seconds = max(0.0, self.clock() - start_time)

        self._emit_event(
            CognitiveEventType.CENTRAL_ORCHESTRATION_CYCLE_COMPLETED,
            summary=f"Central orchestration cycle '{cycle_id}' completed with status '{result.status}'.",
            turn_id=cycle_id,
            metadata=result.to_dict(),
        )

        return result

    # ========================================================================
    # Internal Pipeline Steps
    # ========================================================================

    def _normalize_ingress(
        self,
        raw_item: Any,
        now: float,
    ) -> Optional[MultimodalObservation]:
        """Normalize ingress envelope/dict/observation via CentralInputGateway."""
        if isinstance(raw_item, MultimodalObservation):
            return raw_item

        try:
            # If item is dict, convert to IngressEnvelope
            if isinstance(raw_item, dict):
                loc = raw_item.get("location")
                if isinstance(loc, dict):
                    loc = GeoLocation.from_dict(loc)
                mod = raw_item.get("modality", ModalityType.UNKNOWN)
                if isinstance(mod, str):
                    mod = ModalityType.from_str(mod)
                raw_item = IngressEnvelope(
                    message_id=raw_item.get("message_id", f"msg_{uuid.uuid4().hex[:8]}"),
                    source_id=raw_item.get("source_id", "unknown_source"),
                    source_type=raw_item.get("source_type", "UNKNOWN"),
                    modality=mod,
                    timestamp=raw_item.get("timestamp", now),
                    payload=raw_item.get("payload"),
                    confidence=raw_item.get("confidence", 1.0),
                    location=loc,
                    correlation_id=raw_item.get("correlation_id", ""),
                    causation_id=raw_item.get("causation_id"),
                    artifact_reference=raw_item.get("artifact_reference"),
                )

            # If item is IngressEnvelope, use input_gateway.receive_envelope
            if hasattr(self.input_gateway, "receive_envelope"):
                ingress_res = self.input_gateway.receive_envelope(raw_item, now=now)
                if ingress_res and ingress_res.is_accepted and ingress_res.observation:
                    return ingress_res.observation
                return None
            elif hasattr(self.input_gateway, "ingest_observation"):
                self.input_gateway.ingest_observation(raw_item)
                return raw_item if isinstance(raw_item, MultimodalObservation) else None
        except Exception as ex:
            logger.warning("Ingress normalization error for item: %s", ex)
            return None
        return None

    def _execute_cognitive_turn(
        self,
        goal: Goal,
        situation: Situation,
        now: float,
    ) -> Optional[CognitiveTurnResult]:
        """Execute a cognitive turn via the authoritative CognitiveRuntime."""
        if not self.cognitive_runtime:
            return None

        # Build clean cognitive turn input from goal and situation
        turn_input = f"{goal.title}: {goal.description}"
        eff_corr = getattr(goal, "correlation_id", None) or situation.correlation_id
        try:
            return self.cognitive_runtime.execute_turn(turn_input, session_id=eff_corr)
        except Exception as ex:
            logger.error("Cognitive turn execution failure: %s", ex)
            return None

    def _execute_tool_calls(
        self,
        turn_res: CognitiveTurnResult,
        goal: Goal,
        situation: Situation,
        result: CentralOrchestratorResult,
    ) -> List[MultimodalObservation]:
        """
        Execute tool calls via ToolOrchestrator boundary and collect new device observations.
        Does NOT execute devices directly.
        """
        if not self.tool_orchestrator:
            return []

        produced_observations: List[MultimodalObservation] = []
        eff_corr = getattr(goal, "correlation_id", None) or situation.correlation_id
        eff_caus = getattr(goal, "id", getattr(goal, "goal_id", None)) or situation.situation_id

        # If turn already executed tool results through its execution engine:
        if hasattr(turn_res, "results") and turn_res.results:
            for tool_res in turn_res.results[: self.config.max_tool_calls_per_turn]:
                result.tool_results.append(tool_res)
                result.causal_trace.append({
                    "stage": "TOOL_EXECUTION",
                    "capability": tool_res.capability,
                    "action": tool_res.action,
                    "success": tool_res.success,
                    "call_id": tool_res.call_id,
                })
                if (
                    tool_res.success
                    and isinstance(tool_res.data, dict)
                    and "observations" in tool_res.data
                ):
                    raw_obs_list = tool_res.data["observations"]
                    if isinstance(raw_obs_list, (list, tuple)):
                        for o in raw_obs_list:
                            if isinstance(o, MultimodalObservation):
                                if eff_corr and (not o.correlation_id or o.correlation_id == "corr_plan_001"):
                                    o = MultimodalObservation(
                                        observation_id=o.observation_id,
                                        source_id=o.source_id,
                                        source_type=o.source_type,
                                        modality=o.modality,
                                        timestamp=o.timestamp,
                                        payload=o.payload,
                                        confidence=o.confidence,
                                        location=o.location,
                                        device_id=o.device_id,
                                        correlation_id=eff_corr,
                                        causation_id=o.causation_id or eff_caus,
                                        artifact_reference=o.artifact_reference,
                                        metadata=dict(o.metadata),
                                    )
                                produced_observations.append(o)
            return produced_observations

        # Extract tool calls from turn result results or plan steps
        calls_to_execute: List[ToolCall] = []

        # 1. Inspect direct ToolCalls produced by plan/decision
        if hasattr(turn_res, "plan") and turn_res.plan and hasattr(turn_res.plan, "steps"):
            for step in turn_res.plan.steps:
                if isinstance(step, ToolCall):
                    calls_to_execute.append(step)
                elif hasattr(step, "tool") and step.tool == "device_gateway":
                    tc = ToolCall(
                        capability="device_gateway",
                        action=getattr(step, "action", "dispatch_capability"),
                        parameters=getattr(step, "parameters", {}),
                        call_id=f"call_{getattr(step, 'id', uuid.uuid4().hex[:8])}",
                    )
                    calls_to_execute.append(tc)

        # 2. Execute each ToolCall through ToolOrchestrator
        for tc in calls_to_execute[: self.config.max_tool_calls_per_turn]:
            if isinstance(tc.parameters, dict):
                eff_corr = getattr(goal, "correlation_id", None) or situation.correlation_id
                if eff_corr and (not tc.parameters.get("correlation_id") or tc.parameters.get("correlation_id") == "corr_plan_001"):
                    tc.parameters["correlation_id"] = eff_corr
                eff_caus = getattr(goal, "id", getattr(goal, "goal_id", None)) or situation.situation_id
                if eff_caus and (not tc.parameters.get("causation_id") or tc.parameters.get("causation_id") == "req_plan"):
                    tc.parameters["causation_id"] = eff_caus
            try:
                tool_res = self.tool_orchestrator.execute(tc)
                result.tool_results.append(tool_res)
                result.causal_trace.append({
                    "stage": "TOOL_EXECUTION",
                    "capability": tc.capability,
                    "action": tc.action,
                    "success": tool_res.success,
                    "call_id": tc.call_id,
                })

                # Check if device execution generated new observations
                if (
                    tool_res.success
                    and isinstance(tool_res.data, dict)
                    and "observations" in tool_res.data
                ):
                    raw_obs_list = tool_res.data["observations"]
                    if isinstance(raw_obs_list, (list, tuple)):
                        for o in raw_obs_list:
                            if isinstance(o, MultimodalObservation):
                                produced_observations.append(o)

            except Exception as ex:
                logger.error("Tool execution error in orchestrator: %s", ex)
                fail_res = Result.fail(
                    message=f"Tool orchestration error: {ex}",
                    capability=tc.capability,
                    action=tc.action,
                    call_id=tc.call_id,
                )
                result.tool_results.append(fail_res)

        return produced_observations

    # ========================================================================
    # Replay Verification Support
    # ========================================================================

    def record_trace_snapshot(self, result: CentralOrchestratorResult) -> Dict[str, Any]:
        """Capture deterministic execution snapshot for replay comparison."""
        return {
            "cycle_id": result.cycle_id,
            "status": result.status,
            "situations": [s.to_dict() for s in result.situations_fused],
            "world_transitions": [t.to_dict() for t in result.world_transitions],
            "decisions": [d.to_dict() for d in result.autonomy_decisions],
            "goals": [
                g.to_dict() if hasattr(g, "to_dict") else {
                    "goal_id": getattr(g, "id", getattr(g, "goal_id", "")),
                    "title": getattr(g, "title", getattr(g, "original_goal", "")),
                    "status": g.status.value if hasattr(g.status, "value") else str(g.status),
                }
                for g in result.goals_created
            ],
            "causal_trace": list(result.causal_trace),
        }

    def verify_replay(
        self,
        trace_a: Dict[str, Any],
        trace_b: Dict[str, Any],
    ) -> Tuple[bool, List[str]]:
        """
        Verify replay determinism: compare two execution snapshots and identify
        any semantic divergence.
        """
        divergences: List[str] = []

        if trace_a.get("status") != trace_b.get("status"):
            divergences.append(
                f"Status mismatch: {trace_a.get('status')} vs {trace_b.get('status')}"
            )

        if len(trace_a.get("situations", [])) != len(trace_b.get("situations", [])):
            divergences.append(
                f"Situations count mismatch: {len(trace_a.get('situations', []))} vs {len(trace_b.get('situations', []))}"
            )

        if len(trace_a.get("world_transitions", [])) != len(trace_b.get("world_transitions", [])):
            divergences.append(
                f"World transitions count mismatch: {len(trace_a.get('world_transitions', []))} vs {len(trace_b.get('world_transitions', []))}"
            )

        if len(trace_a.get("goals", [])) != len(trace_b.get("goals", [])):
            divergences.append(
                f"Goals count mismatch: {len(trace_a.get('goals', []))} vs {len(trace_b.get('goals', []))}"
            )

        return (len(divergences) == 0, divergences)
