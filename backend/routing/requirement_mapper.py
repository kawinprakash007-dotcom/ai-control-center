from typing import List

from core.models.reasoning import (
    ReasoningRequest,
    ModelCapabilityType,
)
from core.models.model_router import (
    ModelRequirements,
    LocalityRequirement,
    PrivacyClass,
)


def derive_requirements_from_request(req: ReasoningRequest) -> ModelRequirements:
    """
    Deterministic mapping from a structured ReasoningRequest context to ModelRequirements.
    Uses existing domain signals (visual scene, grounded targets, task description)
    without performing LLM or intent classification calls.
    """
    required_caps: List[ModelCapabilityType] = [ModelCapabilityType.TEXT]
    preferred_caps: List[ModelCapabilityType] = []
    structured_out = False
    locality = LocalityRequirement.CLOUD_ALLOWED
    privacy = PrivacyClass.PUBLIC_ALLOWED

    # 1. Visual desktop reasoning
    if req.visual_scene is not None or req.screenshot_ref is not None:
        required_caps.append(ModelCapabilityType.VISION)
        structured_out = True

    # 2. Grounded action proposals
    if req.grounded_candidates or (req.available_capabilities and "computer" in req.available_capabilities):
        structured_out = True
        required_caps.append(ModelCapabilityType.TOOL_REASONING)

    # 3. Privacy sensitivity
    # If memory context indicates sensitive or internal data
    if req.memory_context and any("private" in m.lower() or "secret" in m.lower() for m in req.memory_context):
        locality = LocalityRequirement.LOCAL_ONLY
        privacy = PrivacyClass.STRICT_LOCAL

    # Deduplicate required caps preserving order
    deduped_required: List[ModelCapabilityType] = []
    for c in required_caps:
        if c not in deduped_required:
            deduped_required.append(c)

    return ModelRequirements(
        required_capabilities=tuple(deduped_required),
        preferred_capabilities=tuple(preferred_caps),
        locality_requirement=locality,
        privacy_requirement=privacy,
        structured_output_required=structured_out,
        minimum_context_window=4096 if (req.history and len(req.history) > 5) else 0,
        metadata={"derived_from_goal": req.goal},
    )
