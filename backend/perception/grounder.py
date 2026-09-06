import time
from typing import Optional, List, Tuple, Dict, Any

from core.interfaces.perception_interface import TargetGrounderInterface
from core.models.computer import ComputerObservation
from core.models.perception import (
    VisualScene,
    VisualElement,
    GroundingRequest,
    GroundedTarget,
    GroundingAmbiguity,
    GroundingResult,
    SpatialRelation,
)


class TargetGrounder(TargetGrounderInterface):
    """
    Model-neutral target grounding engine.
    Resolves GroundingRequest against a VisualScene using deterministic scoring,
    spatial relationships, and ambiguity detection.

    CRITICAL ARCHITECTURAL BOUNDARY:
    TargetGrounder produces GroundedTarget data; it NEVER triggers clicks or communicates
    directly with the ComputerBackend.
    """

    def __init__(
        self,
        ambiguity_margin: float = 0.08,
        stale_max_age_seconds: float = 30.0,
    ):
        self.ambiguity_margin = ambiguity_margin
        self.stale_max_age_seconds = stale_max_age_seconds

    def ground(
        self,
        request: GroundingRequest,
        scene: VisualScene,
        current_observation: Optional[ComputerObservation] = None,
    ) -> GroundingResult:
        """
        Ground a target request against a VisualScene.
        """
        # 1. Stale target check against current observation
        if current_observation is not None:
            current_obs_id = current_observation.observation_id or f"obs_{int(current_observation.timestamp * 1000)}"
            if scene.source_observation_id != current_obs_id:
                return GroundingResult(
                    success=False,
                    error_code="TARGET_STALE",
                    reason=f"VisualScene observation ID '{scene.source_observation_id}' does not match current observation '{current_obs_id}'. Re-observation required.",
                )

        # 2. Check observation age
        if scene.timestamp > 0:
            current_time = time.time()
            if (current_time - scene.timestamp) > self.stale_max_age_seconds:
                return GroundingResult(
                    success=False,
                    error_code="TARGET_STALE",
                    reason=f"Observation is stale (age: {round(current_time - scene.timestamp, 1)}s > {self.stale_max_age_seconds}s). Re-observation required.",
                )

        if not scene.elements:
            return GroundingResult(
                success=False,
                error_code="TARGET_NOT_FOUND",
                reason="VisualScene contains no detected elements.",
            )

        # 3. Resolve spatial relation reference element if requested
        reference_el: Optional[VisualElement] = None
        if request.relation and request.relation_target_text:
            ref_matches = scene.find_text(request.relation_target_text, exact=False)
            if ref_matches:
                reference_el = ref_matches[0]

        # 4. Score all elements
        scored_candidates: List[Tuple[float, VisualElement]] = []
        for el in scene.elements:
            score = self._score_element(el, request, scene, reference_el)
            if score >= request.min_confidence:
                scored_candidates.append((score, el))

        # Sort descending by score, then area ascending (prefer tighter target bounding box)
        scored_candidates.sort(key=lambda item: (-item[0], item[1].bounding_box.area, item[1].element_id))

        if not scored_candidates:
            return GroundingResult(
                success=False,
                error_code="TARGET_NOT_FOUND",
                reason=f"No element met minimum confidence {request.min_confidence} for criteria: text='{request.text}', type='{request.element_type}'.",
            )

        # Bound candidates
        candidates_list = [item[1] for item in scored_candidates[: request.max_candidates]]
        top_score, top_el = scored_candidates[0]

        # 5. Ambiguity detection: check if multiple elements tie closely
        if len(scored_candidates) > 1:
            second_score, second_el = scored_candidates[1]
            if (top_score - second_score) <= self.ambiguity_margin and top_score < 0.95:
                # Find all tied candidates
                ambiguous_els = [
                    item[1] for item in scored_candidates if (top_score - item[0]) <= self.ambiguity_margin
                ]
                return GroundingResult(
                    success=False,
                    ambiguity=GroundingAmbiguity(
                        is_ambiguous=True,
                        candidate_count=len(ambiguous_els),
                        candidates=tuple(ambiguous_els),
                        reason=f"Detected {len(ambiguous_els)} ambiguous candidates with near-identical confidence ({round(top_score, 2)} vs {round(second_score, 2)}).",
                    ),
                    error_code="AMBIGUOUS_TARGET",
                    reason="Grounding query is ambiguous. Multiple elements match with similar confidence.",
                    candidates=tuple(candidates_list),
                )

        # 6. Build GroundedTarget
        click_coord = top_el.center
        grounded_target = GroundedTarget(
            target_id=f"target_{top_el.element_id}",
            element=top_el,
            click_coordinate=click_coord,
            confidence=round(top_score, 3),
            source_observation_id=scene.source_observation_id,
            observation_timestamp=scene.timestamp,
            provenance={
                "scoring_rank": 1,
                "confidence": round(top_score, 3),
                "matched_text": top_el.text,
                "element_type": top_el.element_type.value,
                "bounding_box": (
                    top_el.bounding_box.x,
                    top_el.bounding_box.y,
                    top_el.bounding_box.width,
                    top_el.bounding_box.height,
                ),
            },
        )

        return GroundingResult(
            success=True,
            target=grounded_target,
            candidates=tuple(candidates_list),
            metadata={
                "total_candidates": len(scored_candidates),
                "top_score": round(top_score, 3),
            },
        )

    def _score_element(
        self,
        el: VisualElement,
        req: GroundingRequest,
        scene: VisualScene,
        reference_el: Optional[VisualElement],
    ) -> float:
        base_score = 0.0

        # Text matching
        if req.text:
            if not el.text:
                return 0.0
            el_text = el.text.strip().lower()
            query_text = req.text.strip().lower()

            if el_text == query_text:
                base_score += 0.85
            elif query_text in el_text:
                base_score += 0.65
            elif any(word in el_text for word in query_text.split()):
                base_score += 0.45
            else:
                return 0.0
        else:
            # If no text specified, start from a baseline
            base_score = 0.35

        # Element type match
        if req.element_type:
            if el.element_type == req.element_type:
                base_score += 0.20
            elif req.element_type.value in el.element_type.value:
                base_score += 0.10

        # Semantic role match
        if req.semantic_role and "role" in el.properties:
            if req.semantic_role.lower() in str(el.properties["role"]).lower():
                base_score += 0.15

        # Spatial relationship constraint
        if req.relation and reference_el:
            rel = el.bounding_box.relation_to(reference_el.bounding_box)
            if rel == req.relation:
                base_score += 0.25
            else:
                base_score -= 0.20

        # Approximate region constraint
        if req.approximate_region:
            if req.approximate_region.contains_box(el.bounding_box):
                base_score += 0.15
            elif req.approximate_region.intersects(el.bounding_box):
                base_score += 0.05
            else:
                base_score -= 0.15

        # Application context boost
        if req.application_context and scene.active_window_title:
            if req.application_context.lower() in scene.active_window_title.lower():
                base_score += 0.10

        # Factor in intrinsic detection confidence
        final_score = base_score * el.confidence
        return max(0.0, min(1.0, final_score))
