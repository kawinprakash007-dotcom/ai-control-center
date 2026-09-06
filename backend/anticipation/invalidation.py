from typing import Any, Optional, Sequence, Tuple

from core.interfaces.anticipation_interface import InvalidationEngineInterface
from core.models.anticipation import Anticipation, AnticipationStatus, FutureConditionType


class DeterministicInvalidationEngine(InvalidationEngineInterface):
    """
    Model-neutral, deterministic invalidation engine for Phase 4.6.
    Evaluates whether an anticipation hypothesis remains valid or has been
    INVALIDATED by fresh facts or EXPIRED by temporal horizon boundaries.
    Never mutates World State or executes tools.
    """

    def evaluate_invalidation(
        self,
        anticipation: Anticipation,
        current_world_state: Optional[Any],
        active_goals: Sequence[Any],
        now: float,
    ) -> Tuple[AnticipationStatus, str]:
        # 1. If already in terminal status, preserve it
        if anticipation.status in (
            AnticipationStatus.INVALIDATED,
            AnticipationStatus.EXPIRED,
            AnticipationStatus.CONFIRMED,
            AnticipationStatus.DISMISSED,
        ):
            return (anticipation.status, f"Preserved prior terminal status '{anticipation.status.value}'.")

        # -------------------------------------------------------------
        # 2. Check Temporal Expiration
        # -------------------------------------------------------------
        max_deadline = anticipation.provenance.created_at + anticipation.horizon_window_seconds[1]
        if now > max_deadline:
            return (
                AnticipationStatus.EXPIRED,
                f"Time horizon exceeded: age {now - anticipation.provenance.created_at:.1f}s > max {anticipation.horizon_window_seconds[1]:.1f}s.",
            )

        # Check evidence expiration
        expirable_evidence = [e for e in anticipation.evidence_items if e.expires_at is not None]
        if expirable_evidence and all(now >= e.expires_at for e in expirable_evidence):
            return (
                AnticipationStatus.EXPIRED,
                "All supporting evidence items have expired.",
            )

        # -------------------------------------------------------------
        # 3. Check World State Invalidation (Facts contradicted hypothesis)
        # -------------------------------------------------------------
        if current_world_state is not None:
            entity_id = anticipation.target_entity_id
            if entity_id:
                hyp_state = anticipation.hypothetical_state

                # A. RESOURCE_DEPLETION_RISK
                if anticipation.condition_type == FutureConditionType.RESOURCE_DEPLETION_RISK:
                    for prop_name, expected_val in hyp_state.items():
                        cond = None
                        if hasattr(current_world_state, "get_condition"):
                            cond = current_world_state.get_condition(entity_id, prop_name)

                        if cond is not None:
                            current_val = cond.value if hasattr(cond, "value") else cond
                            threshold = anticipation.metadata.get("risk_threshold", 25.0)
                            if isinstance(current_val, (int, float)) and current_val > threshold * 1.5:
                                return (
                                    AnticipationStatus.INVALIDATED,
                                    f"Resource '{prop_name}' recovered to {current_val} (safe > {threshold * 1.5}); depletion risk invalidated.",
                                )

                # B. SAFETY_RISK
                elif anticipation.condition_type == FutureConditionType.SAFETY_RISK:
                    for prop_name, expected_val in hyp_state.items():
                        cond = None
                        if hasattr(current_world_state, "get_condition"):
                            cond = current_world_state.get_condition(entity_id, prop_name)

                        if cond is not None:
                            current_val = cond.value if hasattr(cond, "value") else cond
                            if current_val in ("safe", "clear", "nominal", "resolved", False):
                                return (
                                    AnticipationStatus.INVALIDATED,
                                    f"Safety property '{prop_name}' is currently '{current_val}'; safety risk invalidated.",
                                )

                # C. DEPENDENCY_RISK
                elif anticipation.condition_type == FutureConditionType.DEPENDENCY_RISK:
                    for prop_name, expected_val in hyp_state.items():
                        cond = None
                        if hasattr(current_world_state, "get_condition"):
                            cond = current_world_state.get_condition(entity_id, prop_name)

                        if cond is not None:
                            current_val = cond.value if hasattr(cond, "value") else cond
                            if current_val in ("available", "ready", "connected", True):
                                return (
                                    AnticipationStatus.INVALIDATED,
                                    f"Dependency '{prop_name}' is now available ({current_val}); risk invalidated.",
                                )

        # -------------------------------------------------------------
        # 4. Check Goal Completion Invalidation
        # -------------------------------------------------------------
        if anticipation.related_goal_ids:
            active_ids = {getattr(g, "goal_id", "") for g in active_goals}
            for gid in anticipation.related_goal_ids:
                if gid and gid not in active_ids:
                    return (
                        AnticipationStatus.INVALIDATED,
                        f"Target goal '{gid}' is no longer active; goal-related risk invalidated.",
                    )

        # Still valid
        return (AnticipationStatus.ACTIVE, "Anticipation hypothesis remains supported by current evidence.")
