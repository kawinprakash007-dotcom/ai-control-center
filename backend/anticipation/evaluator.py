import math
import time
from typing import Any, List, Optional, Sequence, Tuple

from core.interfaces.anticipation_interface import EvidenceEvaluatorInterface
from core.models.anticipation import EvidenceItem, EvidenceSourceType


class DeterministicEvidenceEvaluator(EvidenceEvaluatorInterface):
    """
    Model-neutral, deterministic evidence evaluator for Phase 4.6.
    Evaluates confidence, freshness, and relevance as orthogonal dimensions in [0.0, 1.0].
    Never combines these metrics into a single ambiguous number.
    Never executes tools or models.
    """

    def __init__(
        self,
        half_life_seconds: float = 3600.0,
        clock: Optional[Any] = None,
    ):
        self.half_life_seconds = max(1.0, half_life_seconds)
        self.clock = clock or time.time

    def evaluate_evidence(
        self,
        items: Sequence[EvidenceItem],
        active_goals: Sequence[Any],
        now: Optional[float] = None,
    ) -> Tuple[float, float, float]:
        """
        Calculates:
        1. Confidence: average confidence of evidence items, boosted by corroborating count.
        2. Freshness: recency of evidence items via exponential half-life decay. Expired evidence scores 0.0.
        3. Relevance: degree of match with current active goals and system criticality.

        Returns:
            (confidence, freshness, relevance) each clamped to [0.0, 1.0].
        """
        current_time = now if now is not None else self.clock()

        if not items:
            return (0.0, 0.0, 0.0)

        # -------------------------------------------------------------
        # 1. Confidence Evaluation
        # -------------------------------------------------------------
        # Average base confidence
        base_conf = sum(e.confidence for e in items) / len(items)
        # Multi-evidence corroboration boost (+0.05 per corroborating item beyond the first, max +0.15)
        count_boost = min(0.15, (len(items) - 1) * 0.05) if len(items) > 1 else 0.0
        confidence = max(0.0, min(1.0, base_conf + count_boost))

        # -------------------------------------------------------------
        # 2. Freshness Evaluation
        # -------------------------------------------------------------
        freshness_scores: List[float] = []
        for e in items:
            if not e.is_fresh(current_time):
                # Stale/expired evidence receives 0.0 freshness
                freshness_scores.append(0.0)
                continue

            age = max(0.0, current_time - e.observed_at)
            # Exponential decay: 0.5 ^ (age / half_life)
            score = math.pow(0.5, age / self.half_life_seconds)
            freshness_scores.append(score)

        avg_freshness = sum(freshness_scores) / len(freshness_scores) if freshness_scores else 0.0
        freshness = max(0.0, min(1.0, avg_freshness))

        # -------------------------------------------------------------
        # 3. Relevance Evaluation
        # -------------------------------------------------------------
        # Baseline relevance
        relevance_score = 0.3

        # Extract active goal titles/descriptions
        active_goal_keywords: List[str] = []
        for g in active_goals:
            text = getattr(g, "original_goal", "") or getattr(g, "title", "") or ""
            active_goal_keywords.extend(text.lower().split())

        # Check keyword matches against evidence description and source IDs
        match_count = 0
        for e in items:
            e_text = f"{e.description} {e.source_id} {e.metadata}".lower()
            for kw in active_goal_keywords:
                if len(kw) > 3 and kw in e_text:
                    match_count += 1

        if match_count > 0:
            relevance_score += min(0.5, match_count * 0.1)

        # High-confidence world-state evidence inherently carries baseline relevance
        has_world_state = any(e.source_type == EvidenceSourceType.WORLD_STATE for e in items)
        if has_world_state:
            relevance_score += 0.1

        relevance = max(0.0, min(1.0, relevance_score))

        return (confidence, freshness, relevance)
