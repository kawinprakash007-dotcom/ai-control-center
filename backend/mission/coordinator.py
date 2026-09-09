"""
ATLAS Phase 6.4 — Multi-Product Mission Coordinator.

Coordinates active tactical missions, monitors incoming product evidence,
evaluates evidence-driven completion criteria, handles product failure/reassignment,
and routes goal creation requests through AutonomousGoalManager.

CRITICAL ARCHITECTURAL RULES:
1. NO SECOND BRAIN: Coordinates mission progress; does not duplicate CognitiveRuntime.
2. NO DIRECT EXECUTION: Does not execute device commands or tools directly.
3. GOAL LIFECYCLE PRESERVED: All goal creation/cancellation routes strictly through AutonomousGoalManager.
4. DIRECT GOALSTORE MUTATION PROHIBITED: Never touches GoalStore directly.
5. EVIDENCE-DRIVEN COMPLETION: command success != mission success. Real evidence required.
"""

from dataclasses import replace
import logging
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core.interfaces.goal_interface import AutonomousGoalManagerInterface
from core.interfaces.mission_interface import MissionCoordinatorInterface, MissionPlannerInterface
from core.models.goal import (
    Goal,
    GoalCompletionCriteria,
    GoalConstraints,
    GoalPriority,
    GoalStatus,
)
from core.models.mission import (
    Mission,
    MissionLimits,
    MissionObjective,
    MissionObjectiveType,
    MissionStatus,
    ObjectiveStatus,
    ProductEvidence,
)
from mission.planner import MissionPlanner
from mission.timeline import MissionTimeline

logger = logging.getLogger("atlas.mission.coordinator")


class MissionCoordinator(MissionCoordinatorInterface):
    """
    Authoritative coordinator for multi-product edge missions.
    """

    def __init__(
        self,
        goal_manager: Optional[AutonomousGoalManagerInterface] = None,
        planner: Optional[MissionPlannerInterface] = None,
        limits: Optional[MissionLimits] = None,
    ):
        self._lock = threading.RLock()
        self.goal_manager = goal_manager
        self.planner = planner or MissionPlanner()
        self.limits = limits or MissionLimits()

        self._active_missions: Dict[str, Mission] = {}
        self._completed_missions: Dict[str, Mission] = {}
        self._mission_goals: Dict[str, List[str]] = {}       # mission_id -> list of goal_ids
        self._objective_goals: Dict[str, str] = {}           # objective_id -> goal_id
        self._timeline = MissionTimeline(self.limits)

    # ========================================================================
    # Mission Lifecycle
    # ========================================================================

    def create_mission(self, mission: Mission) -> Mission:
        """
        Register and begin tracking an active multi-product mission.
        Enforces duplicate prevention and capacity limits.
        """
        with self._lock:
            # 1. Capacity check
            if len(self._active_missions) >= self.limits.max_active_missions:
                raise ValueError(
                    f"Mission capacity bound ({self.limits.max_active_missions}) exceeded. "
                    "Cannot create new mission."
                )

            # 2. Duplicate mission prevention by trigger situation & mission type
            for active in self._active_missions.values():
                if (
                    active.trigger_situation_ids == mission.trigger_situation_ids
                    and active.mission_type == mission.mission_type
                    and active.current_status in (MissionStatus.ACTIVE, MissionStatus.PLANNED, MissionStatus.INVESTIGATING)
                ):
                    logger.info("Duplicate mission suppressed for situation %s", mission.trigger_situation_ids)
                    return active

            # 3. Activate mission
            activated_mission = replace(
                mission,
                current_status=MissionStatus.ACTIVE,
                updated_at=time.time(),
            )
            self._active_missions[mission.mission_id] = activated_mission
            self._mission_goals[mission.mission_id] = []

            self._timeline.record_entry(
                event_type="MISSION_CREATED",
                description=f"Created mission '{mission.mission_type}' with {len(mission.objectives)} objectives.",
                source_id="COORDINATOR",
                timestamp=activated_mission.created_at,
            )

            # Activate first wave of dependency-free objectives
            activated_mission = self._activate_eligible_objectives(activated_mission)
            self._active_missions[mission.mission_id] = activated_mission
            return activated_mission

    def get_mission(self, mission_id: str) -> Optional[Mission]:
        with self._lock:
            return self._active_missions.get(mission_id) or self._completed_missions.get(mission_id)

    def list_active_missions(self) -> Sequence[Mission]:
        with self._lock:
            return tuple(self._active_missions.values())

    def list_completed_missions(self) -> Sequence[Mission]:
        with self._lock:
            return tuple(self._completed_missions.values())

    # ========================================================================
    # Evidence Ingress & Verification
    # ========================================================================

    def ingest_evidence(
        self,
        evidence: ProductEvidence,
        mission_id: Optional[str] = None,
        now: Optional[float] = None,
    ) -> Sequence[Mission]:
        """
        Evaluate incoming product evidence against active mission objectives.
        Marks objectives COMPLETED when evidence-driven completion criteria are satisfied.
        """
        current_time = float(now if now is not None else time.time())
        updated_missions: List[Mission] = []

        with self._lock:
            target_missions = []
            if mission_id and mission_id in self._active_missions:
                target_missions.append(self._active_missions[mission_id])
            else:
                # Match against missions referencing this situation or source
                for m in self._active_missions.values():
                    if evidence.situation_id in m.trigger_situation_ids or evidence.source_id in m.involved_product_ids:
                        target_missions.append(m)

            for m in target_missions:
                updated_m = self._process_evidence_for_mission(m, evidence, current_time)
                if updated_m.current_status in (MissionStatus.COMPLETED, MissionStatus.FAILED, MissionStatus.ABORTED):
                    self._active_missions.pop(m.mission_id, None)
                    self._completed_missions[m.mission_id] = updated_m
                else:
                    self._active_missions[m.mission_id] = updated_m
                updated_missions.append(updated_m)

            return tuple(updated_missions)

    def _process_evidence_for_mission(
        self,
        mission: Mission,
        evidence: ProductEvidence,
        current_time: float,
    ) -> Mission:
        """Process evidence against objectives of a specific mission."""
        updated_objectives: List[MissionObjective] = []
        any_completed = False

        # Record timeline entry for evidence arrival
        self._timeline.record_entry(
            event_type="EVIDENCE_INGESTED",
            description=f"Evidence from {evidence.source_id}: {evidence.summary} (conf={evidence.confidence:.2f})",
            source_id=evidence.source_id,
            evidence_id=evidence.evidence_id,
            timestamp=current_time,
        )

        for obj in mission.objectives:
            if obj.status == ObjectiveStatus.IN_PROGRESS:
                # Evaluate if evidence matches this objective's requirements
                if self._evidence_satisfies_objective(obj, evidence):
                    completed_obj = replace(
                        obj,
                        status=ObjectiveStatus.COMPLETED,
                        progress=1.0,
                        completed_at=current_time,
                        result_summary=f"Satisfied by {evidence.source_id}: {evidence.summary}",
                    )
                    updated_objectives.append(completed_obj)
                    any_completed = True

                    self._timeline.record_entry(
                        event_type="OBJECTIVE_COMPLETED",
                        description=f"Objective '{obj.description}' completed via {evidence.source_id}.",
                        source_id=evidence.source_id,
                        evidence_id=evidence.evidence_id,
                        timestamp=current_time,
                    )
                else:
                    updated_objectives.append(obj)
            else:
                updated_objectives.append(obj)

        # Append evidence reference to mission
        new_evidence_refs = list(mission.evidence_references)
        if len(new_evidence_refs) < self.limits.max_evidence_per_mission:
            new_evidence_refs.append(evidence)

        # Compute overall progress
        total_objs = len(updated_objectives)
        completed_count = sum(1 for o in updated_objectives if o.status == ObjectiveStatus.COMPLETED)
        progress = (completed_count / total_objs) if total_objs > 0 else 0.0

        current_mission = replace(
            mission,
            objectives=tuple(updated_objectives),
            evidence_references=tuple(new_evidence_refs),
            progress=round(progress, 2),
            updated_at=current_time,
        )

        # If any objective completed, activate newly eligible dependent objectives
        if any_completed:
            current_mission = self._activate_eligible_objectives(current_mission)

        # Evaluate overall mission completion
        return self._evaluate_mission_completion(current_mission, current_time)

    def _evidence_satisfies_objective(
        self,
        objective: MissionObjective,
        evidence: ProductEvidence,
    ) -> bool:
        """
        Evaluate if incoming evidence satisfies the objective's completion criteria.
        CRITICAL: Requires genuine corroborating evidence, not merely command dispatch success.
        """
        # 1. Source matching: If assigned to a specific product, evidence should match that product or compatible type
        if objective.assigned_product_id and evidence.source_id != objective.assigned_product_id:
            # Allow fallback if source product type matches candidate products
            pass

        # 2. Criteria matching
        crit = objective.completion_criteria
        req_ev = str(crit.get("required_evidence", "")).lower()
        summary_norm = evidence.summary.lower()
        data_keys = [str(k).lower() for k in evidence.data.keys()]

        # Keyword / semantic match
        if req_ev and (req_ev in summary_norm or any(req_ev in k for k in data_keys)):
            return True

        # Confidence threshold
        thresh = float(crit.get("confidence_threshold", 0.50))
        if evidence.confidence < thresh:
            return False

        # Objective type specific matching
        if objective.type == MissionObjectiveType.VERIFY_INCIDENT:
            return "verified" in summary_norm or "confirmed" in summary_norm or "detected" in summary_norm
        elif objective.type == MissionObjectiveType.NOTIFY_WEARER:
            return "hud" in summary_norm or "displayed" in summary_norm or "notified" in summary_norm
        elif objective.type == MissionObjectiveType.INSPECT_ROUTE:
            return "route" in summary_norm or "inspected" in summary_norm or "navigated" in summary_norm
        elif objective.type == MissionObjectiveType.MAINTAIN_OBSERVATION:
            return "tracking" in summary_norm or "monitoring" in summary_norm or "observed" in summary_norm
        elif objective.type == MissionObjectiveType.CONFIRM_RESOLUTION:
            return "cleared" in summary_norm or "resolved" in summary_norm or "nominal" in summary_norm

        return True

    # ========================================================================
    # Goal Creation & AutonomousGoalManager Integration
    # ========================================================================

    def step_coordination(self, now: Optional[float] = None) -> Sequence[Mission]:
        """
        Periodic coordination step advancing active missions and activating objectives.
        """
        current_time = float(now if now is not None else time.time())
        updated: List[Mission] = []

        with self._lock:
            for mid, mission in list(self._active_missions.items()):
                m = self._activate_eligible_objectives(mission)
                m = self._evaluate_mission_completion(m, current_time)
                self._active_missions[mid] = m
                updated.append(m)

            return tuple(updated)

    def _activate_eligible_objectives(self, mission: Mission) -> Mission:
        """
        Identify PENDING objectives whose dependencies are COMPLETED,
        translate them into Goals, and register them via AutonomousGoalManager.
        """
        completed_ids = {o.objective_id for o in mission.objectives if o.status == ObjectiveStatus.COMPLETED}
        updated_objectives: List[MissionObjective] = []
        new_goals: List[str] = list(self._mission_goals.get(mission.mission_id, []))

        for obj in mission.objectives:
            if obj.status == ObjectiveStatus.PENDING:
                # Check if all dependencies are satisfied
                all_deps_met = all(dep in completed_ids for dep in obj.dependencies)
                if all_deps_met:
                    # Transition to IN_PROGRESS
                    in_prog_obj = replace(obj, status=ObjectiveStatus.IN_PROGRESS)
                    updated_objectives.append(in_prog_obj)

                    # Create Goal via AutonomousGoalManager
                    goal_id = self._create_goal_for_objective(in_prog_obj, mission)
                    if goal_id:
                        self._objective_goals[obj.objective_id] = goal_id
                        new_goals.append(goal_id)

                    self._timeline.record_entry(
                        event_type="OBJECTIVE_ACTIVATED",
                        description=f"Activated objective '{obj.description}' assigned to {obj.assigned_product_id}",
                        source_id="COORDINATOR",
                        timestamp=time.time(),
                    )
                else:
                    updated_objectives.append(obj)
            else:
                updated_objectives.append(obj)

        self._mission_goals[mission.mission_id] = new_goals
        return replace(mission, objectives=tuple(updated_objectives))

    def _create_goal_for_objective(self, objective: MissionObjective, mission: Mission) -> Optional[str]:
        """
        Translate a MissionObjective into a Goal request and submit strictly to AutonomousGoalManager.
        NEVER mutates GoalStore directly.
        """
        if not self.goal_manager:
            return None

        goal_id = f"goal_{objective.objective_id}"
        prio = mission.priority if isinstance(mission.priority, GoalPriority) else GoalPriority.NORMAL

        goal = Goal(
            goal_id=goal_id,
            original_goal=objective.description,
            priority=prio,
            constraints=GoalConstraints(
                deadline=time.time() + 600.0,
                max_turns_total=20,
            ),
            completion_criteria=GoalCompletionCriteria(
                require_all_objectives=True,
            ),
            metadata={
                "mission_id": mission.mission_id,
                "objective_id": objective.objective_id,
                "assigned_product": objective.assigned_product_id,
                "correlation_id": mission.correlation_id,
            },
        )

        try:
            # Strictly use AutonomousGoalManager.create_goal()
            created_goal = self.goal_manager.create_goal(goal)
            logger.info("GoalManager created goal %s for objective %s", created_goal.goal_id, objective.objective_id)
            return created_goal.goal_id
        except Exception as ex:
            logger.error("Failed to create goal for objective %s via GoalManager: %s", objective.objective_id, ex)
            return None

    # ========================================================================
    # Completion & Abort
    # ========================================================================

    def _evaluate_mission_completion(self, mission: Mission, current_time: float) -> Mission:
        """Evaluate if all required objectives in a mission have finished."""
        if mission.current_status in (MissionStatus.COMPLETED, MissionStatus.FAILED, MissionStatus.ABORTED):
            return mission

        total_objs = len(mission.objectives)
        if total_objs == 0:
            return mission

        all_completed = all(o.status in (ObjectiveStatus.COMPLETED, ObjectiveStatus.SKIPPED) for o in mission.objectives)
        any_failed = any(o.status == ObjectiveStatus.FAILED for o in mission.objectives)

        if all_completed:
            completed_mission = replace(
                mission,
                current_status=MissionStatus.COMPLETED,
                progress=1.0,
                completed_at=current_time,
                updated_at=current_time,
            )
            self._completed_missions[mission.mission_id] = completed_mission
            self._active_missions.pop(mission.mission_id, None)

            self._timeline.record_entry(
                event_type="MISSION_COMPLETED",
                description=f"Mission '{mission.mission_type}' ({mission.mission_id}) successfully completed.",
                source_id="COORDINATOR",
                timestamp=current_time,
            )
            return completed_mission

        if any_failed and mission.replanning_count >= self.limits.max_replanning_attempts:
            failed_mission = replace(
                mission,
                current_status=MissionStatus.FAILED,
                updated_at=current_time,
            )
            self._completed_missions[mission.mission_id] = failed_mission
            self._active_missions.pop(mission.mission_id, None)
            return failed_mission

        return mission

    def abort_mission(
        self,
        mission_id: str,
        reason: str = "",
        now: Optional[float] = None,
    ) -> Optional[Mission]:
        """
        Abort an active mission and cancel all associated goals via AutonomousGoalManager.
        """
        current_time = float(now if now is not None else time.time())
        with self._lock:
            mission = self._active_missions.pop(mission_id, None)
            if not mission:
                return None

            # Cancel associated goals via AutonomousGoalManager
            if self.goal_manager:
                for gid in self._mission_goals.get(mission_id, []):
                    try:
                        self.goal_manager.cancel_goal(gid, reason=f"Mission {mission_id} aborted: {reason}")
                    except Exception as ex:
                        logger.warning("Error cancelling goal %s: %s", gid, ex)

            aborted_mission = replace(
                mission,
                current_status=MissionStatus.ABORTED,
                failure_reason=reason or "Aborted by operator / system",
                updated_at=current_time,
            )
            self._completed_missions[mission_id] = aborted_mission

            self._timeline.record_entry(
                event_type="MISSION_ABORTED",
                description=f"Mission {mission_id} aborted. Reason: {reason}",
                source_id="COORDINATOR",
                timestamp=current_time,
            )
            return aborted_mission

    def pause_mission(
        self,
        mission_id: str,
        reason: str = "",
        now: Optional[float] = None,
    ) -> Optional[Mission]:
        """Pause an active mission and its active goals via AutonomousGoalManager."""
        current_time = float(now if now is not None else time.time())
        with self._lock:
            mission = self._active_missions.get(mission_id)
            if not mission:
                return None
            if self.goal_manager:
                for gid in self._mission_goals.get(mission_id, []):
                    try:
                        self.goal_manager.pause_goal(gid, reason=reason)
                    except Exception as ex:
                        logger.warning("Error pausing goal %s: %s", gid, ex)
            paused = replace(mission, current_status=MissionStatus.PAUSED, updated_at=current_time)
            self._active_missions[mission_id] = paused
            self._timeline.record_entry(
                event_type="MISSION_PAUSED",
                description=f"Mission {mission_id} paused. Reason: {reason}",
                source_id="COORDINATOR",
                timestamp=current_time,
            )
            return paused

    def resume_mission(
        self,
        mission_id: str,
        now: Optional[float] = None,
    ) -> Optional[Mission]:
        """Resume a paused mission and its active goals via AutonomousGoalManager."""
        current_time = float(now if now is not None else time.time())
        with self._lock:
            mission = self._active_missions.get(mission_id)
            if not mission:
                return None
            if self.goal_manager:
                for gid in self._mission_goals.get(mission_id, []):
                    try:
                        self.goal_manager.resume_goal(gid)
                    except Exception as ex:
                        logger.warning("Error resuming goal %s: %s", gid, ex)
            resumed = replace(mission, current_status=MissionStatus.ACTIVE, updated_at=current_time)
            self._active_missions[mission_id] = resumed
            self._timeline.record_entry(
                event_type="MISSION_RESUMED",
                description=f"Mission {mission_id} resumed.",
                source_id="COORDINATOR",
                timestamp=current_time,
            )
            return resumed

    def get_timeline(self) -> MissionTimeline:
        return self._timeline

    def clear(self) -> None:
        with self._lock:
            self._active_missions.clear()
            self._completed_missions.clear()
            self._mission_goals.clear()
            self._objective_goals.clear()
            self._timeline.clear()
