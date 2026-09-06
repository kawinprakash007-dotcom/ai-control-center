from typing import List, Optional, Tuple

from core.models.model_router import (
    ModelDescriptor,
    ModelRequirements,
    RoutingResult,
    LocalityRequirement,
    PrivacyClass,
)
from core.models.reasoning import ModelCapabilityType
from core.interfaces.reasoning_interface import ReasoningProviderInterface
from core.interfaces.model_router_interface import (
    ModelRouterInterface,
    ModelProviderRegistryInterface,
)
from routing.registry import ModelProviderRegistry


class StandardModelRouter(ModelRouterInterface):
    """
    Deterministic runtime Model Router.
    Selects the most suitable ReasoningProvider based on explicit task ModelRequirements.

    Design Principles:
    1. The model router is NOT the brain, planner, or executor.
    2. Deterministic: Given identical requirements and registry state, produces identical output.
    3. Fail-Closed: Never silently downgrades required capabilities.
    4. No network or remote model calls during routing.
    """

    def __init__(self, registry: Optional[ModelProviderRegistryInterface] = None):
        self._registry = registry or ModelProviderRegistry()

    @property
    def registry(self) -> ModelProviderRegistryInterface:
        return self._registry

    def get_provider(self, provider_id: str) -> Optional[ReasoningProviderInterface]:
        return self._registry.get_provider(provider_id)

    def route(self, requirements: ModelRequirements) -> RoutingResult:
        """
        Evaluate all registered and enabled descriptors against requirements.
        Rank and return the best match or report structured failure.
        """
        all_descriptors = self._registry.list_descriptors(enabled_only=True)
        if not all_descriptors:
            return RoutingResult(
                success=False,
                error_reason="No registered reasoning providers are currently enabled.",
            )

        # 1. Filter candidates by Hard Constraints
        compatible: List[Tuple[ModelDescriptor, List[ModelCapabilityType], List[str]]] = []

        for desc in all_descriptors:
            is_compat, matched_caps, reject_reasons = self._check_compatibility(desc, requirements)
            if is_compat:
                compatible.append((desc, matched_caps, reject_reasons))

        if not compatible:
            reasons = self._diagnose_incompatibility(all_descriptors, requirements)
            return RoutingResult(
                success=False,
                error_reason=f"No compatible provider found: {'; '.join(reasons)}",
            )

        # 2. Deterministic Ranking
        # Ranks candidates based on:
        # a. Preferred provider/model match (highest weight)
        # b. Number of preferred capabilities matched
        # c. Locality preference match
        # d. Descriptor priority (descending)
        # e. Lexicographical tie-breaker (provider_id, model_id)
        def score_candidate(item: Tuple[ModelDescriptor, List[ModelCapabilityType], List[str]]) -> Tuple:
            d, matched_caps, _ = item
            is_pref_provider = (requirements.preferred_provider_id == d.provider_id)
            is_pref_model = (requirements.preferred_model_id == d.model_id)

            pref_caps_matched = sum(1 for c in requirements.preferred_capabilities if d.has_capability(c))

            locality_bonus = 0
            if requirements.locality_requirement == LocalityRequirement.PREFER_LOCAL and d.is_local:
                locality_bonus = 1

            return (
                1 if (is_pref_provider and is_pref_model) else 0,
                1 if is_pref_provider else 0,
                pref_caps_matched,
                locality_bonus,
                d.priority,
                # Negative of strings for reverse sorting if needed, or sort tuple directly
                -(len(d.provider_id)),
                d.provider_id,
                d.model_id,
            )

        # Sort descending by primary score, with deterministic tie-breaking
        compatible.sort(
            key=lambda x: (
                1 if (requirements.preferred_provider_id == x[0].provider_id and requirements.preferred_model_id == x[0].model_id) else 0,
                1 if requirements.preferred_provider_id == x[0].provider_id else 0,
                sum(1 for c in requirements.preferred_capabilities if x[0].has_capability(c)),
                1 if (requirements.locality_requirement == LocalityRequirement.PREFER_LOCAL and x[0].is_local) else 0,
                x[0].priority,
                x[0].provider_id,
                x[0].model_id,
            ),
            reverse=True,
        )

        selected_desc, matched_caps, _ = compatible[0]

        # Calculate unmet soft preferences
        unmet: List[str] = []
        for pref in requirements.preferred_capabilities:
            if not selected_desc.has_capability(pref):
                unmet.append(f"capability:{pref.value}")
        if requirements.locality_requirement == LocalityRequirement.PREFER_LOCAL and not selected_desc.is_local:
            unmet.append("locality:local")
        if requirements.preferred_provider_id and requirements.preferred_provider_id != selected_desc.provider_id:
            unmet.append(f"provider:{requirements.preferred_provider_id}")

        fallback_used = len(compatible) > 1 and bool(requirements.preferred_provider_id and requirements.preferred_provider_id != selected_desc.provider_id)

        rationale = (
            f"Selected model '{selected_desc.model_id}' from provider '{selected_desc.provider_id}' "
            f"matching required capabilities {[c.value for c in requirements.required_capabilities]} "
            f"(priority={selected_desc.priority})."
        )

        return RoutingResult(
            success=True,
            provider_id=selected_desc.provider_id,
            model_id=selected_desc.model_id,
            descriptor=selected_desc,
            matched_capabilities=tuple(matched_caps),
            unmet_preferences=tuple(unmet),
            routing_rationale=rationale,
            fallback_used=fallback_used,
            metadata={"candidate_count": len(compatible)},
        )

    def _check_compatibility(
        self,
        desc: ModelDescriptor,
        req: ModelRequirements,
    ) -> Tuple[bool, List[ModelCapabilityType], List[str]]:
        reject_reasons: List[str] = []
        matched_caps: List[ModelCapabilityType] = []

        # 1. Hard Required Capabilities
        for cap in req.required_capabilities:
            if desc.has_capability(cap):
                matched_caps.append(cap)
            else:
                reject_reasons.append(f"Missing required capability '{cap.value}'")

        # 2. Structured Output
        if req.structured_output_required and not desc.has_capability(ModelCapabilityType.STRUCTURED_OUTPUT):
            reject_reasons.append("Missing required 'structured_output' capability")

        # 3. Context Window
        if req.minimum_context_window > 0 and desc.context_window < req.minimum_context_window:
            reject_reasons.append(
                f"Context window too small ({desc.context_window} < required {req.minimum_context_window})"
            )

        # 4. Locality Requirement (Hard)
        if req.locality_requirement == LocalityRequirement.LOCAL_ONLY and not desc.is_local:
            reject_reasons.append("Provider is not local (LOCAL_ONLY required)")

        # 5. Privacy Requirement (Hard)
        if req.privacy_requirement == PrivacyClass.STRICT_LOCAL and not desc.is_local:
            reject_reasons.append("STRICT_LOCAL privacy requires local execution")

        return len(reject_reasons) == 0, matched_caps, reject_reasons

    def _diagnose_incompatibility(
        self,
        descriptors: List[ModelDescriptor],
        req: ModelRequirements,
    ) -> List[str]:
        diagnostics: List[str] = []
        for d in descriptors:
            _, _, reasons = self._check_compatibility(d, req)
            diagnostics.append(f"{d.provider_id}/{d.model_id}: {', '.join(reasons)}")
        return diagnostics
