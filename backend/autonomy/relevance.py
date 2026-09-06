import time
from typing import Any, Callable, List, Optional, Sequence, Tuple

from core.interfaces.autonomy_interface import RelevanceEngineInterface
from core.models.autonomy import (
    Event,
    EventCategory,
    EventClassification,
    EventPriority,
    EventRelevance,
)


class DeterministicRelevanceEngine(RelevanceEngineInterface):
    """
    Deterministic relevance evaluation engine for Phase 4.5.
    Evaluates how critical/relevant an event is to current system goals and state.
    Never calls external models or tools.
    """

    def __init__(self, clock: Optional[Callable[[], float]] = None):
        self.clock = clock or time.time

    def evaluate_relevance(
        self,
        event: Event,
        classification: EventClassification,
        current_state: Optional[Any] = None,
        active_goals: Optional[Sequence[Any]] = None,
    ) -> EventRelevance:
        now = self.clock()

        # 1. Base category score
        cat_scores = {
            EventCategory.SAFETY_ALERT: 0.95,
            EventCategory.THRESHOLD_BREACH: 0.90,
            EventCategory.FAILURE: 0.80,
            EventCategory.TIMEOUT: 0.70,
            EventCategory.STATE_CHANGE: 0.60,
            EventCategory.EXTERNAL_SIGNAL: 0.50,
            EventCategory.GOAL_PROGRESS: 0.40,
            EventCategory.UNKNOWN: 0.0,
        }
        base_score = cat_scores.get(classification.category, 0.0)

        # 2. Priority weight factor
        priority_factors = {
            EventPriority.CRITICAL: 1.0,
            EventPriority.HIGH: 0.8,
            EventPriority.NORMAL: 0.5,
            EventPriority.LOW: 0.2,
        }
        priority_factor = priority_factors.get(classification.priority, 0.5)

        # 3. Freshness decay factor
        age = max(0.0, now - event.timestamp)
        if age <= 30.0:
            freshness_factor = 1.0
        elif age <= 120.0:
            freshness_factor = 0.8
        elif age <= 300.0:
            freshness_factor = 0.5
        else:
            freshness_factor = 0.2

        # 4. Active Goal Correlation
        matched_goals: List[str] = []
        goal_boost = 0.0
        target_goal_id = event.payload.get("goal_id")
        target_entity_id = event.payload.get("entity_id")

        if active_goals:
            for g in active_goals:
                gid = getattr(g, "goal_id", None) or (g.get("goal_id") if isinstance(g, dict) else None)
                if not gid:
                    continue
                if target_goal_id and gid == target_goal_id:
                    matched_goals.append(gid)
                    goal_boost = 0.2
                    break
                elif target_entity_id:
                    g_meta = getattr(g, "metadata", None) or (g.get("metadata") if isinstance(g, dict) else {})
                    if target_entity_id in str(g_meta) or target_entity_id in getattr(g, "original_goal", ""):
                        matched_goals.append(gid)
                        goal_boost = 0.15

        # 5. Composite score computation
        weighted_base = (0.5 * base_score) + (0.3 * priority_factor) + (0.2 * (1.0 if goal_boost > 0 else 0.0))
        final_score = min(1.0, max(0.0, weighted_base * freshness_factor))
        final_score = round(final_score, 4)

        is_relevant = (final_score >= 0.5) and (classification.category != EventCategory.UNKNOWN)

        rationale = (
            f"Base category '{classification.category.value}'={base_score:.2f}, "
            f"priority '{classification.priority.name}'={priority_factor:.2f}, "
            f"freshness={freshness_factor:.2f} (age={int(age)}s), "
            f"matched_goals={matched_goals} -> final_score={final_score:.2f}"
        )

        return EventRelevance(
            score=final_score,
            is_relevant=is_relevant,
            matched_goals=tuple(matched_goals),
            freshness_factor=freshness_factor,
            priority_factor=priority_factor,
            rationale=rationale,
        )
