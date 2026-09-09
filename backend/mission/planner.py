"""
ATLAS Phase 6.4 — Deterministic Mission Planner.

Constructs dependency-ordered tactical mission plans from MultiProductSituation
instances and orchestrates bounded replanning upon product or objective failures.

CRITICAL ARCHITECTURAL RULES:
1. NO SECOND BRAIN: Plans semantic mission structures only.
2. NO DIRECT EXECUTION: Does not execute tools or commands.
3. GOAL LIFECYCLE PRESERVED: Objectives will be converted to Goals by MissionCoordinator.
4. BOUNDED REPLANNING: Prevents infinite replanning loops with strict max_replanning_attempts.
"""

from dataclasses import replace
import hashlib
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core.interfaces.mission_interface import MissionPlannerInterface
from core.models.device_contract import ProductType
from core.models.goal import GoalPriority
from core.models.mission import (
    Mission,
    MissionLimits,
    MissionObjective,
    MissionObjectiveType,
    MissionStatus,
    MultiProductSituation,
    ObjectiveStatus,
)
from core.models.orchestration import SituationCategory, SituationSeverity
from mission.role_selector import ProductRoleSelector


class MissionPlanner(MissionPlannerInterface):
    """
    Deterministic tactical mission planner.
    """

    def __init__(
        self,
        role_selector: Optional[ProductRoleSelector] = None,
        limits: Optional[MissionLimits] = None,
    ):
        self.role_selector = role_selector or ProductRoleSelector()
        self.limits = limits or MissionLimits()

    def plan_mission(
        self,
        situation: MultiProductSituation,
        available_devices: Sequence[Any],
        constraints: Optional[Dict[str, Any]] = None,
        now: Optional[float] = None,
    ) -> Mission:
        """
        Generate a structured, dependency-ordered Mission plan based on a MultiProductSituation.
        """
        current_time = float(now if now is not None else time.time())
        seed = f"{situation.situation_id}:{int(current_time)}"
        mission_id = f"msn_{hashlib.sha256(seed.encode()).hexdigest()[:10]}"

        # Select mission type
        mission_type = self._determine_mission_type(situation)
        priority = self._map_severity_to_priority(situation.severity)

        # Generate tactical objectives based on situation category and profile
        raw_objectives = self._build_tactical_objectives(mission_id, situation, current_time)

        # Assign products to objectives via ProductRoleSelector
        assigned_objectives: List[MissionObjective] = []
        involved_devices: List[str] = []

        for obj in raw_objectives:
            candidates = self.role_selector.select_candidate_products(obj, available_devices, constraints)
            assigned_id = candidates[0] if candidates else None
            if assigned_id and assigned_id not in involved_devices:
                involved_devices.append(assigned_id)

            assigned_obj = replace(
                obj,
                candidate_products=tuple(candidates),
                assigned_product_id=assigned_id,
            )
            assigned_objectives.append(assigned_obj)

        desc = (
            f"Multi-product mission '{mission_type}' responding to {situation.situation_id} "
            f"({situation.category.value}, {situation.severity.value}). "
            f"Objectives: {len(assigned_objectives)}."
        )

        return Mission(
            mission_id=mission_id,
            mission_type=mission_type,
            description=desc,
            trigger_situation_ids=(situation.situation_id,),
            involved_product_ids=tuple(involved_devices),
            objectives=tuple(assigned_objectives[:self.limits.max_objectives_per_mission]),
            current_status=MissionStatus.PLANNED,
            priority=priority,
            constraints=constraints or {},
            evidence_references=situation.evidence_references,
            created_at=current_time,
            updated_at=current_time,
            correlation_id=situation.correlation_id,
            causation_id=situation.situation_id,
        )

    def replan_mission(
        self,
        mission: Mission,
        failed_objective_id: str,
        reason: str,
        available_devices: Sequence[Any],
        now: Optional[float] = None,
    ) -> Mission:
        """
        Replan an active mission when an objective fails or an assigned product becomes unavailable.
        Substitutes candidate devices or adapts tactical objectives up to max_replanning_attempts.
        """
        current_time = float(now if now is not None else time.time())

        # Check replanning limit to prevent infinite loops
        if mission.replanning_count >= self.limits.max_replanning_attempts:
            return replace(
                mission,
                current_status=MissionStatus.FAILED,
                failure_reason=f"Exceeded max replanning attempts ({self.limits.max_replanning_attempts}). Last error: {reason}",
                updated_at=current_time,
            )

        updated_objectives: List[MissionObjective] = []
        reassigned_any = False
        new_involved = list(mission.involved_product_ids)

        for obj in mission.objectives:
            if obj.objective_id == failed_objective_id:
                # Select alternative candidate product excluding currently failed/unavailable product
                old_assigned = obj.assigned_product_id
                remaining_candidates = [c for c in obj.candidate_products if c != old_assigned]

                # If no remaining candidates in initial list, re-evaluate with current available devices
                if not remaining_candidates:
                    fresh_candidates = self.role_selector.select_candidate_products(obj, available_devices)
                    remaining_candidates = [c for c in fresh_candidates if c != old_assigned]

                if remaining_candidates:
                    new_assigned = remaining_candidates[0]
                    if new_assigned not in new_involved:
                        new_involved.append(new_assigned)
                    if old_assigned in new_involved and not any(o.assigned_product_id == old_assigned for o in mission.objectives if o.objective_id != failed_objective_id):
                        new_involved.remove(old_assigned)

                    reassigned_obj = replace(
                        obj,
                        assigned_product_id=new_assigned,
                        candidate_products=tuple(remaining_candidates),
                        status=ObjectiveStatus.PENDING,
                        result_summary=f"Reassigned from {old_assigned} to {new_assigned} due to: {reason}",
                    )
                    updated_objectives.append(reassigned_obj)
                    reassigned_any = True
                else:
                    # No viable alternate product available
                    failed_obj = replace(
                        obj,
                        status=ObjectiveStatus.FAILED,
                        result_summary=f"No viable alternative device available. Reason: {reason}",
                    )
                    updated_objectives.append(failed_obj)
            else:
                updated_objectives.append(obj)

        if not reassigned_any:
            # If unable to reassign, mission fails
            return replace(
                mission,
                objectives=tuple(updated_objectives),
                current_status=MissionStatus.FAILED,
                failure_reason=f"Failed to find alternate device for objective '{failed_objective_id}'. Reason: {reason}",
                updated_at=current_time,
                replanning_count=mission.replanning_count + 1,
            )

        return replace(
            mission,
            objectives=tuple(updated_objectives),
            involved_product_ids=tuple(new_involved),
            current_status=MissionStatus.ACTIVE,
            replanning_count=mission.replanning_count + 1,
            updated_at=current_time,
        )

    # ========================================================================
    # Tactical Objective Generation Templates
    # ========================================================================

    def _build_tactical_objectives(
        self,
        mission_id: str,
        situation: MultiProductSituation,
        now: float,
    ) -> List[MissionObjective]:
        """Synthesize dependency-aware tactical objectives for the situation."""
        cat = situation.category
        objs: List[MissionObjective] = []

        if cat == SituationCategory.SECURITY:
            # 1. Aerial or rapid target localization & verification
            o1_id = f"{mission_id}_obj_1"
            objs.append(
                MissionObjective(
                    objective_id=o1_id,
                    mission_id=mission_id,
                    type=MissionObjectiveType.VERIFY_INCIDENT,
                    description=f"Verify perimeter incident reported in {situation.situation_id}",
                    dependencies=(),
                    required_capabilities=("flight", "locomotion", "camera"),
                    completion_criteria={"required_evidence": "TARGET_VERIFIED", "confidence_threshold": 0.70},
                    created_at=now,
                )
            )

            # 2. Ground route inspection / perimeter containment (depends on verification)
            o2_id = f"{mission_id}_obj_2"
            objs.append(
                MissionObjective(
                    objective_id=o2_id,
                    mission_id=mission_id,
                    type=MissionObjectiveType.INSPECT_ROUTE,
                    description="Inspect and secure perimeter access route",
                    dependencies=(o1_id,),
                    required_capabilities=("locomotion", "navigation"),
                    completion_criteria={"required_evidence": "ROUTE_INSPECTED"},
                    created_at=now,
                )
            )

            # 3. Wearer HUD alert / confirmation (depends on verification)
            o3_id = f"{mission_id}_obj_3"
            objs.append(
                MissionObjective(
                    objective_id=o3_id,
                    mission_id=mission_id,
                    type=MissionObjectiveType.NOTIFY_WEARER,
                    description="Deliver HUD tactical alert to wearer via Glass",
                    dependencies=(o1_id,),
                    required_capabilities=("hud", "notification"),
                    completion_criteria={"required_evidence": "NOTIFICATION_DISPLAYED"},
                    created_at=now,
                )
            )

            # 4. Continuous persistent surveillance
            o4_id = f"{mission_id}_obj_4"
            objs.append(
                MissionObjective(
                    objective_id=o4_id,
                    mission_id=mission_id,
                    type=MissionObjectiveType.MAINTAIN_OBSERVATION,
                    description="Maintain stationary optical tracking on incident zone",
                    dependencies=(),
                    required_capabilities=("camera", "detection"),
                    completion_criteria={"required_evidence": "CONTINUOUS_TRACKING"},
                    created_at=now,
                )
            )

        elif cat == SituationCategory.ANOMALY:
            o1_id = f"{mission_id}_obj_1"
            objs.append(
                MissionObjective(
                    objective_id=o1_id,
                    mission_id=mission_id,
                    type=MissionObjectiveType.LOCATE_TARGET,
                    description=f"Inspect spatial anomaly reported in {situation.situation_id}",
                    dependencies=(),
                    required_capabilities=("camera", "locomotion", "flight"),
                    completion_criteria={"required_evidence": "ANOMALY_INSPECTED"},
                    created_at=now,
                )
            )
            o2_id = f"{mission_id}_obj_2"
            objs.append(
                MissionObjective(
                    objective_id=o2_id,
                    mission_id=mission_id,
                    type=MissionObjectiveType.CONFIRM_RESOLUTION,
                    description="Confirm anomaly cleared or resolved",
                    dependencies=(o1_id,),
                    required_capabilities=("camera", "detection"),
                    completion_criteria={"required_evidence": "ANOMALY_CLEARED"},
                    created_at=now,
                )
            )

        else:
            # Default general monitoring pattern
            o1_id = f"{mission_id}_obj_1"
            objs.append(
                MissionObjective(
                    objective_id=o1_id,
                    mission_id=mission_id,
                    type=MissionObjectiveType.MONITOR_AREA,
                    description=f"Monitor incident area for {situation.situation_id}",
                    dependencies=(),
                    required_capabilities=("camera", "detection"),
                    completion_criteria={"required_evidence": "ZONE_MONITORED"},
                    created_at=now,
                )
            )

        return objs

    def _determine_mission_type(self, situation: MultiProductSituation) -> str:
        if situation.recommended_missions:
            return situation.recommended_missions[0]
        if situation.category == SituationCategory.SECURITY:
            return "PERIMETER_SECURITY_RESPONSE"
        if situation.category == SituationCategory.ANOMALY:
            return "ANOMALY_INVESTIGATION"
        if situation.category == SituationCategory.ENVIRONMENTAL:
            return "HAZARD_RESPONSE"
        return "GENERAL_MONITORING"

    def _map_severity_to_priority(self, severity: SituationSeverity) -> GoalPriority:
        if severity == SituationSeverity.CRITICAL:
            return GoalPriority.CRITICAL
        if severity == SituationSeverity.HIGH:
            return GoalPriority.HIGH
        if severity == SituationSeverity.MEDIUM:
            return GoalPriority.NORMAL
        return GoalPriority.LOW
