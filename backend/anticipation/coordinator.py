import collections
import logging
import threading
import time
import uuid
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence, Tuple

from core.interfaces.anticipation_interface import (
    AnticipatoryAnalyzerInterface,
    AnticipatoryPlanningCoordinatorInterface,
    EvidenceEvaluatorInterface,
    InvalidationEngineInterface,
)
from core.interfaces.goal_interface import (
    AutonomousGoalManagerInterface,
    GoalStoreInterface,
)
from core.interfaces.policy_interface import PolicyEngineInterface
from core.interfaces.runtime_interface import CognitiveEventSinkInterface
from core.interfaces.world_interface import WorldStateStoreInterface
from core.models.anticipation import (
    Anticipation,
    AnticipationStatus,
    AnticipatoryDecision,
    AnticipatoryDecisionType,
    FutureConditionType,
)
from core.models.goal import Goal, GoalConstraints, GoalPriority, GoalStatus
from core.models.policy import PolicyContext, PolicyDecision, PolicyResult
from core.models.runtime import CognitiveEvent, CognitiveEventType, CognitiveStage
from anticipation.analyzer import DeterministicAnticipatoryAnalyzer
from anticipation.evaluator import DeterministicEvidenceEvaluator
from anticipation.invalidation import DeterministicInvalidationEngine

logger = logging.getLogger(__name__)


class AnticipatoryPlanningCoordinator(AnticipatoryPlanningCoordinatorInterface):
    """
    Central Anticipatory Planning Coordinator for Phase 4.6.
    Evaluates current evidence and world state to identify future-relevant conditions,
    determines appropriate anticipatory actions, evaluates policy, and routes
    decisions strictly into AutonomousGoalManager.

    NON-NEGOTIABLE ARCHITECTURAL INVARIANTS:
    1. Anticipations are HYPOTHESES, NEVER written into World State as observed facts.
    2. Anticipations NEVER directly execute tools, computer actions, models, or shell commands.
    3. GoalStore is persistence/retrieval only; all lifecycle mutations route through AutonomousGoalManager.
    4. Execution is strictly bounded with depth guards and duplicate suppression.
    """

    def __init__(
        self,
        analyzer: Optional[AnticipatoryAnalyzerInterface] = None,
        evaluator: Optional[EvidenceEvaluatorInterface] = None,
        invalidation_engine: Optional[InvalidationEngineInterface] = None,
        goal_manager: Optional[AutonomousGoalManagerInterface] = None,
        goal_store: Optional[GoalStoreInterface] = None,
        world_state_store: Optional[WorldStateStoreInterface] = None,
        policy_engine: Optional[PolicyEngineInterface] = None,
        event_sink: Optional[CognitiveEventSinkInterface] = None,
        clock: Optional[Callable[[], float]] = None,
        max_cascade_depth: int = 3,
        dedup_window_seconds: float = 60.0,
        min_confidence_threshold: float = 0.5,
        min_relevance_threshold: float = 0.4,
    ):
        self.evaluator = evaluator or DeterministicEvidenceEvaluator(clock=clock)
        self.analyzer = analyzer or DeterministicAnticipatoryAnalyzer(evaluator=self.evaluator, clock=clock)
        self.invalidation_engine = invalidation_engine or DeterministicInvalidationEngine()
        self.goal_manager = goal_manager
        self.goal_store = goal_store
        self.world_state_store = world_state_store
        self.policy_engine = policy_engine
        self.event_sink = event_sink
        self.clock = clock or time.time
        self.max_cascade_depth = max_cascade_depth
        self.dedup_window_seconds = dedup_window_seconds
        self.min_confidence_threshold = min_confidence_threshold
        self.min_relevance_threshold = min_relevance_threshold

        self._lock = threading.Lock()
        self._active_anticipations: Dict[str, Anticipation] = {}
        self._seen_signatures: Dict[str, float] = {}
        self._history: Deque[AnticipatoryDecision] = collections.deque(maxlen=1000)

    def _publish_event(
        self,
        event_type: CognitiveEventType,
        summary: str,
        correlation_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Publish structured CognitiveEvent for observability."""
        if not self.event_sink:
            return

        now = self.clock()
        data = {
            "correlation_id": correlation_id,
            "summary": summary,
            **(metadata or {}),
        }

        event = CognitiveEvent(
            event_id=f"evt_ant_{uuid.uuid4().hex[:12]}",
            turn_id=f"ant_{correlation_id}",
            session_id="anticipation_coordinator",
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
            logger.debug("Failed publishing anticipation cognitive event: %s", ex)

    def get_active_anticipations(self) -> List[Anticipation]:
        with self._lock:
            now = self.clock()
            active: List[Anticipation] = []
            for ant in self._active_anticipations.values():
                if ant.status == AnticipationStatus.ACTIVE and not ant.is_stale(now):
                    active.append(ant)
            return active

    def get_decision_history(self, limit: int = 100) -> List[AnticipatoryDecision]:
        with self._lock:
            return list(self._history)[-limit:]

    def evaluate_cycle(
        self,
        max_anticipations: int = 20,
    ) -> List[AnticipatoryDecision]:
        """
        Run a bounded anticipatory evaluation cycle.
        1. Invalidate stale/resolved anticipations against fresh facts.
        2. Analyze state, goals, and events for candidate hypotheses.
        3. Process candidates through policy and goal manager authority.
        """
        with self._lock:
            now = self.clock()
            current_world_state = self.world_state_store.get_current_state() if self.world_state_store else None
            active_goals = self._get_active_goals_locked()

            # 1. Invalidation evaluation for existing active anticipations
            to_remove: List[str] = []
            for ant_id, ant in list(self._active_anticipations.items()):
                new_status, reason = self.invalidation_engine.evaluate_invalidation(
                    ant, current_world_state, active_goals, now
                )
                if new_status != ant.status:
                    if new_status == AnticipationStatus.INVALIDATED:
                        self._publish_event(
                            event_type=CognitiveEventType.ANTICIPATION_INVALIDATED,
                            summary=f"Anticipation '{ant.anticipation_id}' INVALIDATED: {reason}",
                            correlation_id=ant.correlation_id,
                            metadata={"anticipation_id": ant.anticipation_id, "reason": reason},
                        )
                        to_remove.append(ant_id)
                    elif new_status == AnticipationStatus.EXPIRED:
                        self._publish_event(
                            event_type=CognitiveEventType.ANTICIPATION_EXPIRED,
                            summary=f"Anticipation '{ant.anticipation_id}' EXPIRED: {reason}",
                            correlation_id=ant.correlation_id,
                            metadata={"anticipation_id": ant.anticipation_id, "reason": reason},
                        )
                        to_remove.append(ant_id)

            for ant_id in to_remove:
                self._active_anticipations.pop(ant_id, None)

            # 2. Analyze state and goals for new candidates
            candidates = self.analyzer.analyze(
                world_state=current_world_state,
                active_goals=active_goals,
                recent_events=[],
                evidence=[],
                now=now,
            )

            self._publish_event(
                event_type=CognitiveEventType.ANTICIPATION_EVALUATED,
                summary=f"Anticipatory planning cycle evaluated {len(candidates)} candidates.",
                correlation_id=f"cycle_{int(now)}",
                metadata={"candidates_count": len(candidates)},
            )

        # 3. Process candidate anticipations outside the heavy lock
        decisions: List[AnticipatoryDecision] = []
        for candidate in candidates[:max_anticipations]:
            dec = self.process_anticipation(candidate)
            decisions.append(dec)

        return decisions

    def process_anticipation(self, anticipation: Anticipation) -> AnticipatoryDecision:
        """
        Evaluate a candidate anticipation hypothesis, apply loop protection,
        check policy, and dispatch strictly to AutonomousGoalManager.
        """
        with self._lock:
            now = self.clock()

            # -------------------------------------------------------------
            # 1. Loop and Cascade Depth Protection
            # -------------------------------------------------------------
            if anticipation.provenance.depth >= self.max_cascade_depth:
                self._publish_event(
                    event_type=CognitiveEventType.ANTICIPATION_DUPLICATE_SUPPRESSED,
                    summary=f"Anticipation '{anticipation.anticipation_id}' suppressed: depth {anticipation.provenance.depth} >= {self.max_cascade_depth}",
                    correlation_id=anticipation.correlation_id,
                    metadata={"depth": anticipation.provenance.depth},
                )
                decision = AnticipatoryDecision(
                    decision_id=f"dec_ant_{uuid.uuid4().hex[:12]}",
                    anticipation_id=anticipation.anticipation_id,
                    correlation_id=anticipation.correlation_id,
                    decision_type=AnticipatoryDecisionType.NO_ACTION,
                    timestamp=now,
                    reason=f"Cascade depth limit ({self.max_cascade_depth}) reached; suppressed.",
                )
                self._history.append(decision)
                return decision

            # -------------------------------------------------------------
            # 2. Signature Deduplication Window
            # -------------------------------------------------------------
            sig = anticipation.get_signature()
            last_seen = self._seen_signatures.get(sig)
            if last_seen is not None and (now - last_seen) < self.dedup_window_seconds:
                self._publish_event(
                    event_type=CognitiveEventType.ANTICIPATION_DUPLICATE_SUPPRESSED,
                    summary=f"Anticipation suppressed: duplicate signature '{sig}' within window.",
                    correlation_id=anticipation.correlation_id,
                    metadata={"signature": sig, "age": now - last_seen},
                )
                decision = AnticipatoryDecision(
                    decision_id=f"dec_ant_{uuid.uuid4().hex[:12]}",
                    anticipation_id=anticipation.anticipation_id,
                    correlation_id=anticipation.correlation_id,
                    decision_type=AnticipatoryDecisionType.NO_ACTION,
                    timestamp=now,
                    reason=f"Suppressed by duplicate anticipation window ({self.dedup_window_seconds}s).",
                )
                self._history.append(decision)
                return decision

            self._seen_signatures[sig] = now

            # -------------------------------------------------------------
            # 3. Actionability & Threshold Check
            # -------------------------------------------------------------
            if anticipation.condition_type == FutureConditionType.UNKNOWN:
                self._publish_event(
                    event_type=CognitiveEventType.ANTICIPATION_REJECTED,
                    summary=f"Anticipation rejected: unknown condition type.",
                    correlation_id=anticipation.correlation_id,
                )
                decision = AnticipatoryDecision(
                    decision_id=f"dec_ant_{uuid.uuid4().hex[:12]}",
                    anticipation_id=anticipation.anticipation_id,
                    correlation_id=anticipation.correlation_id,
                    decision_type=AnticipatoryDecisionType.NO_ACTION,
                    timestamp=now,
                    reason="Unknown condition type failed safe.",
                )
                self._history.append(decision)
                return decision

            if anticipation.is_stale(now):
                self._publish_event(
                    event_type=CognitiveEventType.ANTICIPATION_EXPIRED,
                    summary=f"Anticipation '{anticipation.anticipation_id}' is stale or expired.",
                    correlation_id=anticipation.correlation_id,
                )
                decision = AnticipatoryDecision(
                    decision_id=f"dec_ant_{uuid.uuid4().hex[:12]}",
                    anticipation_id=anticipation.anticipation_id,
                    correlation_id=anticipation.correlation_id,
                    decision_type=AnticipatoryDecisionType.NO_ACTION,
                    timestamp=now,
                    reason="Anticipation evidence is stale or expired; rejected.",
                )
                self._history.append(decision)
                return decision

            # Check if evidence is sufficient for autonomous intervention
            decision_type = AnticipatoryDecisionType.NO_ACTION
            goal_payload: Optional[Dict[str, Any]] = None
            target_goal_id: Optional[str] = None
            reason: str = ""

            if not anticipation.is_actionable(self.min_confidence_threshold, self.min_relevance_threshold):
                # Low confidence / relevance: monitor instead of taking proactive action
                if anticipation.confidence >= 0.3:
                    decision_type = AnticipatoryDecisionType.MONITOR
                    reason = f"Confidence ({anticipation.confidence:.2f}) or relevance ({anticipation.relevance:.2f}) below autonomous action threshold; set to MONITOR."
                    self._publish_event(
                        event_type=CognitiveEventType.ANTICIPATION_ACCEPTED,
                        summary=f"Anticipation accepted for MONITORING: {anticipation.description}",
                        correlation_id=anticipation.correlation_id,
                    )
                else:
                    decision_type = AnticipatoryDecisionType.NO_ACTION
                    reason = f"Insufficient evidence confidence ({anticipation.confidence:.2f}); ignored."
                    self._publish_event(
                        event_type=CognitiveEventType.ANTICIPATION_REJECTED,
                        summary=f"Anticipation rejected: low confidence ({anticipation.confidence:.2f}).",
                        correlation_id=anticipation.correlation_id,
                    )
            else:
                # Actionable anticipation
                self._publish_event(
                    event_type=CognitiveEventType.ANTICIPATION_ACCEPTED,
                    summary=f"Anticipation accepted: {anticipation.description}",
                    correlation_id=anticipation.correlation_id,
                    metadata={"confidence": anticipation.confidence, "relevance": anticipation.relevance},
                )
                self._active_anticipations[anticipation.anticipation_id] = anticipation

                # Map condition type to proactive decision
                if anticipation.condition_type == FutureConditionType.RESOURCE_DEPLETION_RISK:
                    decision_type = AnticipatoryDecisionType.CREATE_GOAL
                    goal_payload = {
                        "original_goal": f"Proactively mitigate: {anticipation.description}",
                        "priority": "high",
                    }
                    reason = f"High-confidence resource depletion anticipation ({anticipation.horizon.value}): creating proactive goal."

                elif anticipation.condition_type == FutureConditionType.DEADLINE_RISK and anticipation.related_goal_ids:
                    target_goal_id = anticipation.related_goal_ids[0]
                    decision_type = AnticipatoryDecisionType.UPDATE_GOAL
                    goal_payload = {"priority": "critical"}
                    reason = f"Imminent deadline risk for goal '{target_goal_id}': elevating priority to critical."

                elif anticipation.condition_type == FutureConditionType.SAFETY_RISK:
                    decision_type = AnticipatoryDecisionType.CREATE_GOAL
                    goal_payload = {
                        "original_goal": f"Safety mitigation: {anticipation.description}",
                        "priority": "critical",
                    }
                    reason = "Anticipated safety hazard: creating proactive mitigation goal."

                elif anticipation.condition_type == FutureConditionType.ENVIRONMENT_CHANGE_RISK:
                    decision_type = AnticipatoryDecisionType.PREPARE
                    reason = "Environmental stress anticipated: preparing adaptive contingency."

                else:
                    decision_type = AnticipatoryDecisionType.MONITOR
                    reason = f"Anticipation recorded for condition {anticipation.condition_type.value}."

            # -------------------------------------------------------------
            # 4. Policy Engine Enforcement
            # -------------------------------------------------------------
            policy_result: Optional[PolicyResult] = None
            if self.policy_engine and decision_type in (
                AnticipatoryDecisionType.CREATE_GOAL,
                AnticipatoryDecisionType.UPDATE_GOAL,
                AnticipatoryDecisionType.PREPARE,
            ):
                policy_context = PolicyContext(
                    capability="goal",
                    action=decision_type.value.lower(),
                    parameters=goal_payload or {"goal_id": target_goal_id},
                    reason=reason,
                    source="anticipation",
                    call_id=anticipation.correlation_id,
                )
                try:
                    policy_result = self.policy_engine.evaluate(policy_context)
                    if policy_result.decision == PolicyDecision.DENY:
                        self._publish_event(
                            event_type=CognitiveEventType.ANTICIPATION_POLICY_DENIED,
                            summary=f"Proactive action '{decision_type.value}' DENIED by policy: {policy_result.reason}",
                            correlation_id=anticipation.correlation_id,
                            metadata={"action": decision_type.value, "reason": policy_result.reason},
                        )
                        decision = AnticipatoryDecision(
                            decision_id=f"dec_ant_{uuid.uuid4().hex[:12]}",
                            anticipation_id=anticipation.anticipation_id,
                            correlation_id=anticipation.correlation_id,
                            decision_type=AnticipatoryDecisionType.NO_ACTION,
                            policy_result=policy_result,
                            reason=f"Action '{decision_type.value}' denied by policy: {policy_result.reason}",
                            timestamp=now,
                        )
                        self._history.append(decision)
                        return decision

                    elif policy_result.decision in (PolicyDecision.ASK_PERMISSION, PolicyDecision.REQUIRE_CONFIRMATION):
                        decision_type = AnticipatoryDecisionType.ESCALATE_USER
                        reason = f"Policy requires user confirmation: {policy_result.reason}"

                except Exception as ex:
                    logger.warning("Policy evaluation error in anticipation coordinator: %s; failing closed.", ex)
                    decision = AnticipatoryDecision(
                        decision_id=f"dec_ant_{uuid.uuid4().hex[:12]}",
                        anticipation_id=anticipation.anticipation_id,
                        correlation_id=anticipation.correlation_id,
                        decision_type=AnticipatoryDecisionType.NO_ACTION,
                        reason=f"Policy evaluation failed closed: {ex}",
                        timestamp=now,
                    )
                    self._history.append(decision)
                    return decision

            # -------------------------------------------------------------
            # 5. Route strictly through AutonomousGoalManager
            # -------------------------------------------------------------
            executed = False
            created_goal_id: Optional[str] = None

            if decision_type == AnticipatoryDecisionType.CREATE_GOAL and goal_payload:
                if not self.goal_manager:
                    executed = False
                    reason += " (No AutonomousGoalManager configured; goal creation not dispatched)."
                else:
                    created_goal_id = self._dispatch_create_goal_locked(goal_payload, anticipation)
                    executed = created_goal_id is not None
                    if not executed:
                        decision_type = AnticipatoryDecisionType.MONITOR
                        reason += " (Existing goal already active for this anticipation; duplicate prevented)."
                    else:
                        self._publish_event(
                            event_type=CognitiveEventType.ANTICIPATION_ACTION_DISPATCHED,
                            summary=f"Dispatched proactive CREATE_GOAL for '{created_goal_id}' via AutonomousGoalManager.",
                            correlation_id=anticipation.correlation_id,
                            metadata={"goal_id": created_goal_id, "action": "create_goal"},
                        )

            elif decision_type == AnticipatoryDecisionType.UPDATE_GOAL and target_goal_id:
                new_priority = goal_payload.get("priority") if goal_payload else None
                if new_priority and self.goal_manager:
                    try:
                        self.goal_manager.set_goal_priority(target_goal_id, new_priority)
                        executed = True
                        self._publish_event(
                            event_type=CognitiveEventType.ANTICIPATION_ACTION_DISPATCHED,
                            summary=f"Dispatched proactive UPDATE_GOAL for '{target_goal_id}' via AutonomousGoalManager.",
                            correlation_id=anticipation.correlation_id,
                            metadata={"goal_id": target_goal_id, "action": "set_goal_priority"},
                        )
                    except Exception as ex:
                        logger.error("Failed updating priority for goal '%s': %s", target_goal_id, ex)

            self._publish_event(
                event_type=CognitiveEventType.ANTICIPATION_DECISION_PRODUCED,
                summary=f"Anticipation decision produced: {decision_type.value} (executed={executed}).",
                correlation_id=anticipation.correlation_id,
                metadata={
                    "decision_type": decision_type.value,
                    "executed": executed,
                    "goal_id": created_goal_id or target_goal_id,
                },
            )

            decision = AnticipatoryDecision(
                decision_id=f"dec_ant_{uuid.uuid4().hex[:12]}",
                anticipation_id=anticipation.anticipation_id,
                correlation_id=anticipation.correlation_id,
                decision_type=decision_type,
                timestamp=now,
                goal_id=created_goal_id or target_goal_id,
                goal_payload=goal_payload,
                policy_result=policy_result,
                reason=reason,
                executed=executed,
            )
            self._history.append(decision)
            return decision

    def _get_active_goals_locked(self) -> List[Any]:
        store = getattr(self.goal_manager, "store", None) or self.goal_store
        if not store:
            return []
        try:
            return store.list_goals(status=GoalStatus.RUNNING, limit=20)
        except Exception:
            return []

    def _dispatch_create_goal_locked(self, template: Dict[str, Any], anticipation: Anticipation) -> Optional[str]:
        """
        Create a new proactive goal via AutonomousGoalManagerInterface.
        GoalStore is only used for read-only duplicate checks.
        NEVER directly mutates GoalStore.
        """
        if not self.goal_manager:
            return None

        store = getattr(self.goal_manager, "store", None) or self.goal_store
        goal_text = template.get("original_goal", f"Anticipation response to {anticipation.condition_type.value}")
        dedup_goal_key = f"auto_anticipation:{anticipation.get_signature()}"

        # Duplicate Goal Prevention: check if active goal already exists with this key
        if store:
            try:
                active_goals = store.list_goals(limit=50)
                for ag in active_goals:
                    if ag.status in (GoalStatus.CREATED, GoalStatus.RUNNING, GoalStatus.PAUSED):
                        if ag.metadata.get("dedup_key") == dedup_goal_key:
                            return None
            except Exception:
                pass

        raw_pri = template.get("priority", "high")
        pri = GoalPriority.from_str(raw_pri)

        new_goal = Goal(
            original_goal=goal_text,
            priority=pri,
            constraints=GoalConstraints(
                autonomy_level="autonomous",
                max_turns_total=template.get("max_turns", 10),
            ),
            metadata={
                "dedup_key": dedup_goal_key,
                "triggered_by_anticipation": anticipation.anticipation_id,
                "anticipation_id": anticipation.anticipation_id,
                "condition_type": anticipation.condition_type.value,
                "correlation_id": anticipation.correlation_id,
                "causation_id": anticipation.provenance.causation_id or anticipation.anticipation_id,
                "horizon": anticipation.horizon.value,
            },
        )

        try:
            saved = self.goal_manager.create_goal(new_goal)
            return saved.goal_id
        except Exception as ex:
            logger.error("Failed creating anticipatory goal via GoalManager: %s", ex)
            return None
