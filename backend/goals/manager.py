import logging
import threading
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from core.interfaces.goal_interface import (
    AutonomousGoalManagerInterface,
    GoalExecutionEngineInterface,
    GoalSchedulerInterface,
    GoalStoreInterface,
)
from core.interfaces.runtime_interface import CognitiveEventSinkInterface
from core.models.goal import Goal, GoalConstraints, GoalStatus
from core.models.goal_management import (
    GoalFreshnessStatus,
    GoalManagementState,
    GoalPriority,
    QuantumResult,
    SchedulerConfig,
    SchedulingDecision,
)
from core.models.runtime import CognitiveEvent, CognitiveEventType, CognitiveStage
from goals.scheduler import DeterministicGoalScheduler

logger = logging.getLogger(__name__)


class AutonomousGoalManager(AutonomousGoalManagerInterface):
    """
    Model-neutral, deterministic Autonomous Goal Manager for Phase 4.3.
    Coordinates multiple goals over time, determines scheduling priority,
    enforces a single active goal invariant, and drives bounded execution quanta.

    Hierarchy:
    AutonomousGoalManager
        ↓
    GoalExecutionEngine
        ↓
    CognitiveRuntime
        ↓
    Bounded cognitive turn
    """

    def __init__(
        self,
        store: GoalStoreInterface,
        execution_engine: GoalExecutionEngineInterface,
        scheduler: Optional[GoalSchedulerInterface] = None,
        event_sink: Optional[CognitiveEventSinkInterface] = None,
        config: Optional[SchedulerConfig] = None,
        clock: Optional[Callable[[], float]] = None,
        manager_id: Optional[str] = None,
    ):
        self.store = store
        self.execution_engine = execution_engine
        self.scheduler = scheduler or DeterministicGoalScheduler()
        self.event_sink = event_sink
        self.config = config or SchedulerConfig()
        self.clock = clock or time.time
        self.manager_id = manager_id or f"mgr_{uuid.uuid4().hex[:8]}"

        self._lock = threading.RLock()
        self._state = GoalManagementState(
            scheduler_version=self.config.scheduler_version,
        )

    # ------------------------------------------------------------------------
    # SCHEDULING QUANTA
    # ------------------------------------------------------------------------

    def schedule_once(self) -> Optional[QuantumResult]:
        """
        Evaluate all candidate goals and execute at most one bounded quantum
        for the selected goal. Regains control immediately after the step.
        """
        now = self.clock()

        with self._lock:
            # 1. Inspect and synchronize current active claim state
            current_claim = self.store.get_claimed_goal()
            if current_claim is not None:
                claimed_goal_id, claimed_owner, claim_exp = current_claim
                if claimed_owner == self.manager_id and claim_exp > now:
                    self._state.active_goal_id = claimed_goal_id
                    self._state.active_goal_owner = claimed_owner
                    self._state.active_goal_lease_expires = claim_exp
                elif claim_exp <= now:
                    self.store.release_goal(claimed_goal_id, claimed_owner)
                    self._state.active_goal_id = None
                    self._state.active_goal_owner = None
                    self._state.active_goal_lease_expires = None
            else:
                self._state.active_goal_id = None
                self._state.active_goal_owner = None
                self._state.active_goal_lease_expires = None

            # 2. Retrieve all known goals from store
            all_goals = self.store.list_goals(limit=200)

            # 3. Detect stale goals and update queues
            self._update_state_partitions(all_goals, now)

            # 4. Invoke deterministic scheduler
            decision: SchedulingDecision = self.scheduler.select_next_goal(
                goals=all_goals,
                state=self._state,
                config=self.config,
                clock=self.clock,
            )
            self._state.last_selection_time = now

            if not decision.is_selection_made():
                # Emit skipped events for auditability if candidates were considered
                if decision.skipped_candidates:
                    for g_id, skip_reason in decision.skipped_candidates.items():
                        self._publish_event(
                            event_type=CognitiveEventType.GOAL_SCHEDULING_SKIPPED,
                            goal_id=g_id,
                            summary=f"Goal '{g_id}' skipped: {skip_reason}",
                            payload={"reason": skip_reason},
                        )
                return None

            selected_id = decision.selected_goal_id
            if selected_id is None:
                return None

            # 5. Acquire atomic active-goal lease
            claim_success = self.store.claim_goal(
                goal_id=selected_id,
                owner_id=self.manager_id,
                lease_duration=self.config.lease_duration_seconds,
            )

            if not claim_success:
                logger.warning(
                    "AutonomousGoalManager '%s' could not claim selected goal '%s' (active lease held elsewhere).",
                    self.manager_id,
                    selected_id,
                )
                return None

            self._state.active_goal_id = selected_id
            self._state.active_goal_owner = self.manager_id
            self._state.active_goal_lease_expires = now + self.config.lease_duration_seconds

            self._publish_event(
                event_type=CognitiveEventType.GOAL_SELECTED,
                goal_id=selected_id,
                summary=f"Goal '{selected_id}' selected for execution quantum.",
                payload={
                    "decision_reason": decision.reason,
                    "factors": decision.scheduling_factors.get(selected_id, {}),
                },
            )

            # 6. Update fairness counters
            # Selected goal resets to 0
            self._state.fairness_counters[selected_id] = 0
            # Skipped eligible candidates increment by 1
            for cand_id in decision.candidates_considered:
                if cand_id != selected_id and cand_id not in decision.skipped_candidates:
                    self._state.fairness_counters[cand_id] = self._state.fairness_counters.get(cand_id, 0) + 1

        # 7. Execute exactly ONE bounded quantum outside manager lock
        # This prevents lock contention during cognitive runtime execution
        turn_executed = False
        goal_after_step: Optional[Goal] = None
        try:
            goal_after_step = self.execution_engine.step_goal(selected_id)
            turn_executed = True
        except Exception as ex:
            logger.error("Error executing goal quantum for '%s': %s", selected_id, ex, exc_info=True)
            goal_after_step = self.store.get_goal(selected_id)

        with self._lock:
            final_status = goal_after_step.status.value if goal_after_step else "unknown"

            # 8. Release lease if goal reached non-runnable or terminal state
            should_release = False
            if goal_after_step and (
                goal_after_step.is_terminal()
                or goal_after_step.status in (
                    GoalStatus.PAUSED,
                    GoalStatus.BLOCKED,
                    GoalStatus.WAITING_FOR_USER,
                )
            ):
                should_release = True

            if should_release:
                self.store.release_goal(selected_id, self.manager_id)
                self._state.active_goal_id = None
                self._state.active_goal_owner = None
                self._state.active_goal_lease_expires = None
                self._publish_event(
                    event_type=CognitiveEventType.GOAL_DESELECTED,
                    goal_id=selected_id,
                    summary=f"Goal '{selected_id}' released active lease (status={final_status}).",
                    payload={"status": final_status},
                )

            return QuantumResult(
                goal_id=selected_id,
                decision=decision,
                goal_status=final_status,
                turn_executed=turn_executed,
                details={
                    "percentage": goal_after_step.progress.percentage if goal_after_step else 0.0,
                    "active_objective_id": goal_after_step.active_objective_id if goal_after_step else None,
                },
            )

    def run_next_quantum(self) -> Optional[QuantumResult]:
        """Alias for schedule_once()."""
        return self.schedule_once()

    # ------------------------------------------------------------------------
    # ATOMIC ACTIVE GOAL MANAGEMENT
    # ------------------------------------------------------------------------

    def claim_active_goal(self, goal_id: str) -> bool:
        with self._lock:
            now = self.clock()
            success = self.store.claim_goal(
                goal_id=goal_id,
                owner_id=self.manager_id,
                lease_duration=self.config.lease_duration_seconds,
            )
            if success:
                self._state.active_goal_id = goal_id
                self._state.active_goal_owner = self.manager_id
                self._state.active_goal_lease_expires = now + self.config.lease_duration_seconds
            return success

    def _release_active_goal_locked(self, goal_id: str) -> None:
        """Internal helper for releasing active goal when caller already holds self._lock."""
        self.store.release_goal(goal_id, self.manager_id)
        if self._state.active_goal_id == goal_id:
            self._state.active_goal_id = None
            self._state.active_goal_owner = None
            self._state.active_goal_lease_expires = None

    def release_active_goal(self, goal_id: str) -> None:
        with self._lock:
            self._release_active_goal_locked(goal_id)

    def get_active_goal(self) -> Optional[str]:
        with self._lock:
            claim = self.store.get_claimed_goal()
            return claim[0] if claim is not None else None

    # ------------------------------------------------------------------------
    # GOAL LIFECYCLE CONTROLS
    # ------------------------------------------------------------------------

    def create_goal(
        self,
        goal: Optional[Union[Goal, str]] = None,
        *,
        title: Optional[str] = None,
        description: Optional[str] = None,
        priority: Union[GoalPriority, str] = GoalPriority.NORMAL,
        constraints: Optional[GoalConstraints] = None,
        correlation_id: Optional[str] = None,
        causation_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Goal:
        """
        Create and persist a new goal under management authority.
        Emits CognitiveEventType.GOAL_CREATED with structured metadata.
        Supports passing either a Goal instance or individual parameters.
        """
        with self._lock:
            if not isinstance(goal, Goal):
                goal_text = title or description or (goal if isinstance(goal, str) else "Autonomous Goal")
                meta = dict(kwargs.get("metadata", {}))
                if correlation_id:
                    meta["correlation_id"] = correlation_id
                if causation_id:
                    meta["causation_id"] = causation_id
                goal = Goal(
                    original_goal=goal_text,
                    priority=priority,
                    constraints=constraints or GoalConstraints(),
                    metadata=meta,
                )
            created = self.store.create_goal(goal)
            self._publish_event(
                event_type=CognitiveEventType.GOAL_CREATED,
                goal_id=created.goal_id,
                summary=f"Goal '{created.goal_id}' created under manager authority.",
                payload={
                    "original_goal": created.original_goal,
                    "priority": created.priority.value if hasattr(created.priority, "value") else str(created.priority),
                    **(created.metadata or {}),
                },
            )
            return created

    def pause_goal(self, goal_id: str, reason: str = "") -> Goal:
        with self._lock:
            if self._state.active_goal_id == goal_id:
                self._release_active_goal_locked(goal_id)

        # Delegate execution engine to pause (updates store and emits GOAL_PAUSED)
        return self.execution_engine.pause_goal(goal_id, reason)

    def resume_goal(self, goal_id: str) -> Goal:
        # Delegate execution engine to resume (updates store and emits GOAL_RESUMED)
        resumed = self.execution_engine.resume_goal(goal_id)
        with self._lock:
            # Reset fairness starvation counter upon resume
            self._state.fairness_counters[goal_id] = 0
        return resumed

    def cancel_goal(self, goal_id: str, reason: str = "") -> Goal:
        with self._lock:
            if self._state.active_goal_id == goal_id:
                self._release_active_goal_locked(goal_id)

            goal = self.store.get_goal(goal_id)
            if goal is None:
                raise KeyError(f"Goal '{goal_id}' not found in goal store.")

            goal.status = GoalStatus.CANCELLED
            if hasattr(goal, "_update_progress"):
                goal._update_progress(f"Goal cancelled: {reason or 'Explicit cancellation'}")
            self.store.update_goal(goal)

            self._publish_event(
                event_type=CognitiveEventType.GOAL_CANCELLED,
                goal_id=goal_id,
                summary=f"Goal cancelled: {reason}",
                payload={"reason": reason},
            )
            return goal

    def set_goal_priority(self, goal_id: str, priority: Any) -> Goal:
        norm_priority = priority if isinstance(priority, GoalPriority) else GoalPriority.from_str(str(priority))
        updated = self.store.update_priority(goal_id, norm_priority)
        self._publish_event(
            event_type=CognitiveEventType.GOAL_PRIORITY_CHANGED,
            goal_id=goal_id,
            summary=f"Goal '{goal_id}' priority changed to {norm_priority.value}.",
            payload={"priority": norm_priority.value},
        )
        return updated

    def set_goal_deadline(self, goal_id: str, deadline: Optional[float]) -> Goal:
        updated = self.store.update_deadline(goal_id, deadline)
        self._publish_event(
            event_type=CognitiveEventType.GOAL_DEADLINE_CHANGED,
            goal_id=goal_id,
            summary=f"Goal '{goal_id}' deadline changed to {deadline}.",
            payload={"deadline": deadline},
        )
        return updated

    def get_management_state(self) -> GoalManagementState:
        with self._lock:
            # Refresh active goal status
            claim = self.store.get_claimed_goal()
            if claim:
                self._state.active_goal_id = claim[0]
                self._state.active_goal_owner = claim[1]
                self._state.active_goal_lease_expires = claim[2]
            else:
                self._state.active_goal_id = None
                self._state.active_goal_owner = None
                self._state.active_goal_lease_expires = None

            # Return deepcopy/snapshot
            return GoalManagementState(
                active_goal_id=self._state.active_goal_id,
                active_goal_owner=self._state.active_goal_owner,
                active_goal_lease_expires=self._state.active_goal_lease_expires,
                queued_goal_ids=self._state.queued_goal_ids,
                paused_goal_ids=self._state.paused_goal_ids,
                waiting_goal_ids=self._state.waiting_goal_ids,
                blocked_goal_ids=self._state.blocked_goal_ids,
                completed_goal_ids=self._state.completed_goal_ids,
                stale_goal_ids=self._state.stale_goal_ids,
                fairness_counters=dict(self._state.fairness_counters),
                last_selection_time=self._state.last_selection_time,
                scheduler_version=self._state.scheduler_version,
                metadata=dict(self._state.metadata),
            )

    # ------------------------------------------------------------------------
    # INTERNAL HELPERS
    # ------------------------------------------------------------------------

    def _update_state_partitions(self, goals: Sequence[Goal], now: float) -> None:
        """Partition goal IDs into management state categories."""
        queued: List[str] = []
        paused: List[str] = []
        waiting: List[str] = []
        blocked: List[str] = []
        completed: List[str] = []
        stale: List[str] = []

        for g in goals:
            gid = g.goal_id
            if g.status == GoalStatus.PAUSED:
                paused.append(gid)
            elif g.status == GoalStatus.WAITING_FOR_USER:
                waiting.append(gid)
            elif g.status == GoalStatus.BLOCKED:
                blocked.append(gid)
            elif g.status == GoalStatus.COMPLETED:
                completed.append(gid)
            elif g.status in (GoalStatus.CREATED, GoalStatus.RUNNING):
                queued.append(gid)

            # Check staleness: if goal has not updated for longer than stale threshold
            time_since_update = now - g.updated_at
            if time_since_update >= self.config.stale_threshold_seconds and not g.is_terminal():
                stale.append(gid)
                self._publish_event(
                    event_type=CognitiveEventType.GOAL_STALE,
                    goal_id=gid,
                    summary=f"Goal '{gid}' identified as STALE (inactive for {int(time_since_update)}s).",
                    payload={"time_since_update": time_since_update},
                )

        self._state.queued_goal_ids = tuple(queued)
        self._state.paused_goal_ids = tuple(paused)
        self._state.waiting_goal_ids = tuple(waiting)
        self._state.blocked_goal_ids = tuple(blocked)
        self._state.completed_goal_ids = tuple(completed)
        self._state.stale_goal_ids = tuple(stale)

    def _publish_event(
        self,
        event_type: CognitiveEventType,
        goal_id: str,
        summary: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Publish structured event to event sink with goal correlation metadata."""
        if not self.event_sink:
            return

        now = self.clock()
        data = {
            "goal_id": goal_id,
            "manager_id": self.manager_id,
            "summary": summary,
            **(payload or {}),
        }

        event = CognitiveEvent(
            event_id=f"evt_{uuid.uuid4().hex[:12]}",
            turn_id=f"goal_{goal_id}",
            session_id=f"goal_{goal_id}",
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
            logger.debug("Failed publishing goal management event: %s", ex)
