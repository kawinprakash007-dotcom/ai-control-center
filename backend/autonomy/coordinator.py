import collections
import logging
import threading
import time
import uuid
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence, Tuple

from core.interfaces.autonomy_interface import (
    EventClassifierInterface,
    EventDrivenAutonomyInterface,
    RelevanceEngineInterface,
)
from core.interfaces.goal_interface import (
    AutonomousGoalManagerInterface,
    GoalStoreInterface,
)
from core.interfaces.policy_interface import PolicyEngineInterface
from core.interfaces.runtime_interface import CognitiveEventSinkInterface
from core.interfaces.world_interface import WorldStateStoreInterface
from core.models.autonomy import (
    AutonomyDecision,
    AutonomyDecisionType,
    Event,
    EventCategory,
    EventClassification,
    EventPriority,
    EventRelevance,
    EventTrigger,
    TriggerCondition,
)
from core.models.goal import Goal, GoalConstraints, GoalPriority as GoalModelPriority, GoalStatus
from core.models.policy import AutonomyLevel, PolicyContext, PolicyDecision, PolicyResult
from core.models.runtime import CognitiveEvent, CognitiveEventType, CognitiveStage
from autonomy.classifier import DeterministicEventClassifier
from autonomy.relevance import DeterministicRelevanceEngine

logger = logging.getLogger(__name__)


class EventDrivenAutonomyCoordinator(EventDrivenAutonomyInterface):
    """
    Central Event-Driven Autonomy coordinator for Phase 4.5.
    Evaluates incoming events, classifies them, scores relevance, checks policies,
    and routes decisions strictly into Goal Management / Cognitive Runtime.
    
    NON-NEGOTIABLE ARCHITECTURAL INVARIANT:
    Events NEVER directly execute tools, models, capabilities, or shell commands.
    """

    def __init__(
        self,
        classifier: Optional[EventClassifierInterface] = None,
        relevance_engine: Optional[RelevanceEngineInterface] = None,
        goal_manager: Optional[AutonomousGoalManagerInterface] = None,
        goal_store: Optional[GoalStoreInterface] = None,
        policy_engine: Optional[PolicyEngineInterface] = None,
        world_state_store: Optional[WorldStateStoreInterface] = None,
        event_sink: Optional[CognitiveEventSinkInterface] = None,
        clock: Optional[Callable[[], float]] = None,
        max_cascade_depth: int = 3,
        dedup_window_seconds: float = 60.0,
    ):
        self.classifier = classifier or DeterministicEventClassifier()
        self.relevance_engine = relevance_engine or DeterministicRelevanceEngine(clock=clock)
        self.goal_manager = goal_manager
        self.goal_store = goal_store
        self.policy_engine = policy_engine
        self.world_state_store = world_state_store
        self.event_sink = event_sink
        self.clock = clock or time.time
        self.max_cascade_depth = max_cascade_depth
        self.dedup_window_seconds = dedup_window_seconds

        self._lock = threading.Lock()
        self._triggers: Dict[str, EventTrigger] = {}
        self._seen_dedup_keys: Dict[str, float] = {}
        self._trigger_history: Dict[str, List[float]] = {}
        self._history: Deque[AutonomyDecision] = collections.deque(maxlen=1000)
        self._pending_events: Deque[Event] = collections.deque(maxlen=1000)

    def _publish_event(
        self,
        event_type: CognitiveEventType,
        summary: str,
        correlation_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Publish structured CognitiveEvent for auditable observability."""
        if not self.event_sink:
            return

        now = self.clock()
        data = {
            "correlation_id": correlation_id,
            "summary": summary,
            **(metadata or {}),
        }

        event = CognitiveEvent(
            event_id=f"evt_auto_{uuid.uuid4().hex[:12]}",
            turn_id=f"auto_{correlation_id}",
            session_id="autonomy_coordinator",
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
            logger.debug("Failed publishing autonomy cognitive event: %s", ex)

    def register_trigger(self, trigger: EventTrigger) -> None:
        with self._lock:
            self._triggers[trigger.trigger_id] = trigger

    def get_history(self, limit: int = 100) -> List[AutonomyDecision]:
        with self._lock:
            return list(self._history)[-limit:]

    def ingest_event(self, event: Event) -> AutonomyDecision:
        with self._lock:
            now = self.clock()

            self._publish_event(
                event_type=CognitiveEventType.AUTONOMY_EVENT_INGESTED,
                summary=f"Autonomy event '{event.event_type}' ingested from {event.source.value}.",
                correlation_id=event.correlation_id,
                metadata={"event_id": event.event_id, "priority": event.priority.value},
            )

            # -------------------------------------------------------------
            # 1. Loop and Storm Protection Checks
            # -------------------------------------------------------------

            # A. Cascade Depth Limit (prevents infinite recursive triggering)
            if event.provenance.depth >= self.max_cascade_depth:
                self._publish_event(
                    event_type=CognitiveEventType.AUTONOMY_TRIGGER_SUPPRESSED,
                    summary=f"Event '{event.event_id}' suppressed: cascade depth {event.provenance.depth} >= {self.max_cascade_depth}",
                    correlation_id=event.correlation_id,
                    metadata={"event_id": event.event_id, "depth": event.provenance.depth},
                )
                decision = AutonomyDecision(
                    decision_id=f"dec_{uuid.uuid4().hex[:12]}",
                    event_id=event.event_id,
                    correlation_id=event.correlation_id,
                    decision_type=AutonomyDecisionType.IGNORE,
                    classification=EventClassification(
                        category=EventCategory.UNKNOWN,
                        priority=event.priority,
                        severity="low",
                        is_novel=False,
                        requires_action=False,
                        rationale="Cascade depth limit exceeded.",
                    ),
                    relevance=EventRelevance(score=0.0, is_relevant=False, rationale="Suppressed by cascade limit."),
                    reason=f"Cascade depth limit ({self.max_cascade_depth}) reached; loop suppressed.",
                    timestamp=now,
                )
                self._history.append(decision)
                return decision

            # B. Duplicate Suppression Window
            dedup_key = event.get_dedup_key()
            last_seen = self._seen_dedup_keys.get(dedup_key)
            if last_seen is not None and (now - last_seen) < self.dedup_window_seconds:
                self._publish_event(
                    event_type=CognitiveEventType.AUTONOMY_TRIGGER_SUPPRESSED,
                    summary=f"Event '{event.event_id}' suppressed by deduplication key '{dedup_key}'.",
                    correlation_id=event.correlation_id,
                    metadata={"dedup_key": dedup_key, "age": now - last_seen},
                )
                decision = AutonomyDecision(
                    decision_id=f"dec_{uuid.uuid4().hex[:12]}",
                    event_id=event.event_id,
                    correlation_id=event.correlation_id,
                    decision_type=AutonomyDecisionType.IGNORE,
                    classification=EventClassification(
                        category=EventCategory.UNKNOWN,
                        priority=event.priority,
                        severity="low",
                        is_novel=False,
                        requires_action=False,
                        rationale="Duplicate storm suppression.",
                    ),
                    relevance=EventRelevance(score=0.0, is_relevant=False, rationale="Suppressed by duplicate window."),
                    reason=f"Suppressed by duplicate suppression window ({self.dedup_window_seconds}s).",
                    timestamp=now,
                )
                self._history.append(decision)
                return decision

            self._seen_dedup_keys[dedup_key] = now

            # -------------------------------------------------------------
            # 2. Classification
            # -------------------------------------------------------------
            classification = self.classifier.classify(event)

            self._publish_event(
                event_type=CognitiveEventType.AUTONOMY_EVENT_CLASSIFIED,
                summary=f"Event classified as {classification.category.value} (severity: {classification.severity}).",
                correlation_id=event.correlation_id,
                metadata={
                    "category": classification.category.value,
                    "severity": classification.severity,
                    "requires_action": classification.requires_action,
                },
            )

            # Unknown events fail safely with IGNORE
            if classification.category == EventCategory.UNKNOWN:
                decision = AutonomyDecision(
                    decision_id=f"dec_{uuid.uuid4().hex[:12]}",
                    event_id=event.event_id,
                    correlation_id=event.correlation_id,
                    decision_type=AutonomyDecisionType.IGNORE,
                    classification=classification,
                    relevance=EventRelevance(score=0.0, is_relevant=False, rationale="Unknown event."),
                    reason="Unknown event type failed safe.",
                    timestamp=now,
                )
                self._history.append(decision)
                return decision

            # -------------------------------------------------------------
            # 3. Relevance Evaluation
            # -------------------------------------------------------------
            active_goals = self._get_active_goals_locked()
            current_world_state = self.world_state_store.get_current_state() if self.world_state_store else None

            relevance = self.relevance_engine.evaluate_relevance(
                event=event,
                classification=classification,
                current_state=current_world_state,
                active_goals=active_goals,
            )

            self._publish_event(
                event_type=CognitiveEventType.AUTONOMY_RELEVANCE_EVALUATED,
                summary=f"Event relevance evaluated: score={relevance.score:.2f}, is_relevant={relevance.is_relevant}",
                correlation_id=event.correlation_id,
                metadata={"score": relevance.score, "is_relevant": relevance.is_relevant},
            )

            # -------------------------------------------------------------
            # 4. Trigger Matching & Decision Derivation
            # -------------------------------------------------------------
            decision_type = AutonomyDecisionType.RECORD_ONLY
            matched_trigger: Optional[EventTrigger] = None
            goal_payload: Optional[Dict[str, Any]] = None
            target_goal_id: Optional[str] = None
            reason: str = ""

            for trig in self._triggers.values():
                if not trig.is_active:
                    continue
                if trig.condition.matches(event, classification, relevance):
                    # Check trigger cooldown
                    if self._is_trigger_rate_limited(trig, now):
                        continue
                    matched_trigger = trig
                    decision_type = trig.action_type
                    goal_payload = trig.target_goal_template
                    target_goal_id = event.payload.get("goal_id") or (relevance.matched_goals[0] if relevance.matched_goals else None)
                    reason = f"Matched explicit trigger '{trig.trigger_id}'."
                    self._record_trigger_execution(trig, now)
                    self._publish_event(
                        event_type=CognitiveEventType.AUTONOMY_TRIGGER_MATCHED,
                        summary=f"Event matched trigger '{trig.trigger_id}' -> action {decision_type.value}",
                        correlation_id=event.correlation_id,
                        metadata={"trigger_id": trig.trigger_id, "action_type": decision_type.value},
                    )
                    break

            # Default heuristic if no explicit trigger matched
            if matched_trigger is None:
                if not relevance.is_relevant and not classification.requires_action:
                    decision_type = AutonomyDecisionType.RECORD_ONLY
                    reason = "Event not sufficiently relevant for autonomous action."
                elif classification.category == EventCategory.SAFETY_ALERT or event.priority == EventPriority.CRITICAL:
                    # Safety alert: if an active goal matches, pause it; else escalate to user or protective goal
                    if target_goal_id:
                        decision_type = AutonomyDecisionType.PAUSE_GOAL
                        reason = f"Safety alert on active goal '{target_goal_id}': pausing goal."
                    else:
                        decision_type = AutonomyDecisionType.CREATE_GOAL
                        goal_payload = {
                            "original_goal": f"Mitigate safety alert: {event.payload.get('summary', event.event_type)}",
                            "priority": "critical",
                        }
                        reason = "Critical safety alert: creating protective goal."
                elif classification.category == EventCategory.THRESHOLD_BREACH:
                    decision_type = AutonomyDecisionType.CREATE_GOAL
                    goal_payload = {
                        "original_goal": f"Resolve threshold breach: {event.payload.get('summary', event.event_type)}",
                        "priority": "high",
                    }
                    reason = "Threshold breach: creating corrective goal."
                else:
                    decision_type = AutonomyDecisionType.RECORD_ONLY
                    reason = "Event recorded; no action triggered."

            # -------------------------------------------------------------
            # 5. Policy Enforcement
            # -------------------------------------------------------------
            policy_result: Optional[PolicyResult] = None
            if self.policy_engine and decision_type in (
                AutonomyDecisionType.CREATE_GOAL,
                AutonomyDecisionType.UPDATE_GOAL,
                AutonomyDecisionType.PAUSE_GOAL,
                AutonomyDecisionType.RESUME_GOAL,
                AutonomyDecisionType.CANCEL_GOAL,
            ):
                policy_context = PolicyContext(
                    capability="goal",
                    action=decision_type.value.lower(),
                    parameters=goal_payload or {"goal_id": target_goal_id},
                    reason=reason,
                    source="autonomy",
                    call_id=event.correlation_id,
                )
                try:
                    policy_result = self.policy_engine.evaluate(policy_context)
                    if policy_result.decision == PolicyDecision.DENY:
                        self._publish_event(
                            event_type=CognitiveEventType.AUTONOMY_POLICY_DENIED,
                            summary=f"Autonomous action '{decision_type.value}' DENIED by policy: {policy_result.reason}",
                            correlation_id=event.correlation_id,
                            metadata={"action": decision_type.value, "reason": policy_result.reason},
                        )
                        decision = AutonomyDecision(
                            decision_id=f"dec_{uuid.uuid4().hex[:12]}",
                            event_id=event.event_id,
                            correlation_id=event.correlation_id,
                            decision_type=AutonomyDecisionType.IGNORE,
                            classification=classification,
                            relevance=relevance,
                            matched_trigger_id=matched_trigger.trigger_id if matched_trigger else None,
                            policy_result=policy_result,
                            reason=f"Action '{decision_type.value}' denied by policy: {policy_result.reason}",
                            timestamp=now,
                        )
                        self._history.append(decision)
                        return decision

                    elif policy_result.decision in (PolicyDecision.ASK_PERMISSION, PolicyDecision.REQUIRE_CONFIRMATION):
                        decision_type = AutonomyDecisionType.ESCALATE_TO_USER
                        reason = f"Policy requires user confirmation: {policy_result.reason}"

                except Exception as ex:
                    logger.warning("Policy evaluation error in autonomy coordinator: %s; failing closed.", ex)
                    decision = AutonomyDecision(
                        decision_id=f"dec_{uuid.uuid4().hex[:12]}",
                        event_id=event.event_id,
                        correlation_id=event.correlation_id,
                        decision_type=AutonomyDecisionType.IGNORE,
                        classification=classification,
                        relevance=relevance,
                        reason=f"Policy evaluation failed: {ex}; failing closed.",
                        timestamp=now,
                    )
                    self._history.append(decision)
                    return decision

            # -------------------------------------------------------------
            # 6. Execution Routing strictly into Goal Management
            # -------------------------------------------------------------
            executed = False
            created_goal_id: Optional[str] = None

            if decision_type == AutonomyDecisionType.CREATE_GOAL and goal_payload:
                if not self.goal_manager:
                    executed = False
                    reason += " (No AutonomousGoalManager configured; goal creation not dispatched)."
                else:
                    created_goal_id = self._dispatch_create_goal_locked(goal_payload, event)
                    executed = created_goal_id is not None
                    if not executed:
                        decision_type = AutonomyDecisionType.RECORD_ONLY
                        reason += " (Existing goal already active for this condition; duplicate prevented)."
                    else:
                        self._publish_event(
                            event_type=CognitiveEventType.AUTONOMY_ACTION_DISPATCHED,
                            summary=f"Dispatched CREATE_GOAL for '{created_goal_id}' via AutonomousGoalManager.",
                            correlation_id=event.correlation_id,
                            metadata={"goal_id": created_goal_id, "action": "create_goal"},
                        )

            elif decision_type == AutonomyDecisionType.PAUSE_GOAL and target_goal_id:
                if self.goal_manager:
                    try:
                        self.goal_manager.pause_goal(target_goal_id, reason=reason)
                        executed = True
                        self._publish_event(
                            event_type=CognitiveEventType.AUTONOMY_ACTION_DISPATCHED,
                            summary=f"Dispatched PAUSE_GOAL for '{target_goal_id}' via AutonomousGoalManager.",
                            correlation_id=event.correlation_id,
                            metadata={"goal_id": target_goal_id, "action": "pause_goal"},
                        )
                    except Exception as ex:
                        logger.error("Failed pausing goal '%s': %s", target_goal_id, ex)

            elif decision_type == AutonomyDecisionType.RESUME_GOAL and target_goal_id:
                if self.goal_manager:
                    try:
                        self.goal_manager.resume_goal(target_goal_id)
                        executed = True
                        self._publish_event(
                            event_type=CognitiveEventType.AUTONOMY_ACTION_DISPATCHED,
                            summary=f"Dispatched RESUME_GOAL for '{target_goal_id}' via AutonomousGoalManager.",
                            correlation_id=event.correlation_id,
                            metadata={"goal_id": target_goal_id, "action": "resume_goal"},
                        )
                    except Exception as ex:
                        logger.error("Failed resuming goal '%s': %s", target_goal_id, ex)

            elif decision_type == AutonomyDecisionType.CANCEL_GOAL and target_goal_id:
                if self.goal_manager:
                    try:
                        self.goal_manager.cancel_goal(target_goal_id, reason=reason)
                        executed = True
                        self._publish_event(
                            event_type=CognitiveEventType.AUTONOMY_ACTION_DISPATCHED,
                            summary=f"Dispatched CANCEL_GOAL for '{target_goal_id}' via AutonomousGoalManager.",
                            correlation_id=event.correlation_id,
                            metadata={"goal_id": target_goal_id, "action": "cancel_goal"},
                        )
                    except Exception as ex:
                        logger.error("Failed cancelling goal '%s': %s", target_goal_id, ex)

            elif decision_type == AutonomyDecisionType.UPDATE_GOAL and target_goal_id:
                new_priority = goal_payload.get("priority") if goal_payload else None
                if new_priority and self.goal_manager:
                    try:
                        self.goal_manager.set_goal_priority(target_goal_id, new_priority)
                        executed = True
                        self._publish_event(
                            event_type=CognitiveEventType.AUTONOMY_ACTION_DISPATCHED,
                            summary=f"Dispatched UPDATE_GOAL for '{target_goal_id}' via AutonomousGoalManager.",
                            correlation_id=event.correlation_id,
                            metadata={"goal_id": target_goal_id, "action": "set_goal_priority"},
                        )
                    except Exception as ex:
                        logger.error("Failed updating priority for goal '%s': %s", target_goal_id, ex)

            self._publish_event(
                event_type=CognitiveEventType.AUTONOMY_DECISION_PRODUCED,
                summary=f"Autonomy decision produced: {decision_type.value} (executed={executed}).",
                correlation_id=event.correlation_id,
                metadata={
                    "decision_type": decision_type.value,
                    "executed": executed,
                    "goal_id": created_goal_id or target_goal_id,
                },
            )

            decision = AutonomyDecision(
                decision_id=f"dec_{uuid.uuid4().hex[:12]}",
                event_id=event.event_id,
                correlation_id=event.correlation_id,
                decision_type=decision_type,
                classification=classification,
                relevance=relevance,
                matched_trigger_id=matched_trigger.trigger_id if matched_trigger else None,
                goal_id=created_goal_id or target_goal_id,
                goal_payload=goal_payload,
                policy_result=policy_result,
                reason=reason,
                timestamp=now,
                executed=executed,
            )
            self._history.append(decision)
            return decision

    def process_pending_events(self, max_events: int = 10) -> List[AutonomyDecision]:
        """Process a bounded batch of queued events to prevent runaway execution."""
        decisions: List[AutonomyDecision] = []
        with self._lock:
            count = min(max_events, len(self._pending_events))
            events_to_process = [self._pending_events.popleft() for _ in range(count)]

        for ev in events_to_process:
            dec = self.ingest_event(ev)
            decisions.append(dec)

        return decisions

    def queue_event(self, event: Event) -> None:
        with self._lock:
            self._pending_events.append(event)

    # ------------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------------

    def _get_active_goals_locked(self) -> List[Any]:
        store = getattr(self.goal_manager, "store", None) or self.goal_store
        if not store:
            return []
        try:
            return store.list_goals(status=GoalStatus.RUNNING, limit=20)
        except Exception:
            return []

    def _is_trigger_rate_limited(self, trigger: EventTrigger, now: float) -> bool:
        history = self._trigger_history.setdefault(trigger.trigger_id, [])
        # Prune older than window
        cutoff = now - trigger.window_seconds
        valid_history = [t for t in history if t >= cutoff]
        self._trigger_history[trigger.trigger_id] = valid_history
        if len(valid_history) >= trigger.max_triggers_per_window:
            return True
        if valid_history and (now - valid_history[-1]) < trigger.cooldown_seconds:
            return True
        return False

    def _record_trigger_execution(self, trigger: EventTrigger, now: float) -> None:
        history = self._trigger_history.setdefault(trigger.trigger_id, [])
        history.append(now)

    def _dispatch_create_goal_locked(self, template: Dict[str, Any], event: Event) -> Optional[str]:
        """
        Create a new goal via AutonomousGoalManagerInterface, with duplicate goal prevention.
        GoalStore is only used for read/retrieval; all lifecycle mutations are delegated to AutonomousGoalManager.
        """
        if not self.goal_manager:
            logger.warning("No AutonomousGoalManager configured; cannot create autonomous goal.")
            return None

        store = getattr(self.goal_manager, "store", None) or self.goal_store

        goal_text = template.get("original_goal", f"Autonomy response to {event.event_type}")
        dedup_goal_key = f"auto_goal:{event.get_dedup_key()}"

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

        raw_pri = template.get("priority", "normal")
        pri = GoalModelPriority.from_str(raw_pri)

        new_goal = Goal(
            original_goal=goal_text,
            priority=pri,
            constraints=GoalConstraints(
                autonomy_level="autonomous",
                max_turns_total=template.get("max_turns", 10),
            ),
            metadata={
                "dedup_key": dedup_goal_key,
                "triggered_by_event": event.event_id,
                "event_id": event.event_id,
                "correlation_id": event.correlation_id,
                "causation_id": event.provenance.metadata.get("causation_id") or event.event_id,
                "event_source": event.source.value,
            },
        )

        try:
            saved = self.goal_manager.create_goal(new_goal)
            return saved.goal_id
        except Exception as ex:
            logger.error("Failed creating autonomous goal via GoalManager: %s", ex)
            return None
