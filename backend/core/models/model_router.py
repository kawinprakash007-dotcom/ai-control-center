from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Set

from core.models.reasoning import ModelCapabilityType


class LocalityRequirement(str, Enum):
    """
    Locality constraint or preference for model execution.
    """
    LOCAL_ONLY = "local_only"
    PREFER_LOCAL = "prefer_local"
    CLOUD_ALLOWED = "cloud_allowed"


class CostClass(str, Enum):
    """
    Relative cost profile of a model.
    """
    FREE = "free"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class LatencyClass(str, Enum):
    """
    Relative latency profile of a model.
    """
    FAST = "fast"
    STANDARD = "standard"
    BATCH = "batch"


class PrivacyClass(str, Enum):
    """
    Privacy tier required or offered.
    """
    STRICT_LOCAL = "strict_local"
    ORGANIZATION_INTERNAL = "organization_internal"
    PUBLIC_ALLOWED = "public_allowed"


@dataclass(frozen=True)
class ModelDescriptor:
    """
    Immutable metadata descriptor of a model available through a reasoning provider.
    Separates provider identity from model identity and declares deterministic attributes.
    """
    provider_id: str
    model_id: str
    display_name: str
    capabilities: Tuple[ModelCapabilityType, ...] = (
        ModelCapabilityType.TEXT,
        ModelCapabilityType.STRUCTURED_OUTPUT,
    )
    context_window: int = 8192
    is_local: bool = True
    cost_class: CostClass = CostClass.FREE
    latency_class: LatencyClass = LatencyClass.STANDARD
    privacy_class: PrivacyClass = PrivacyClass.STRICT_LOCAL
    priority: int = 50
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.context_window < 0:
            raise ValueError(f"context_window must be non-negative, got {self.context_window}")
        object.__setattr__(self, "provider_id", str(self.provider_id).strip())
        object.__setattr__(self, "model_id", str(self.model_id).strip())

    @property
    def key(self) -> Tuple[str, str]:
        return (self.provider_id, self.model_id)

    def has_capability(self, cap: ModelCapabilityType) -> bool:
        return cap in self.capabilities

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider_id": self.provider_id,
            "model_id": self.model_id,
            "display_name": self.display_name,
            "capabilities": [c.value for c in self.capabilities],
            "context_window": self.context_window,
            "is_local": self.is_local,
            "cost_class": self.cost_class.value,
            "latency_class": self.latency_class.value,
            "privacy_class": self.privacy_class.value,
            "priority": self.priority,
            "enabled": self.enabled,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ModelRequirements:
    """
    Explicit specification of what reasoning capabilities a task requires.
    Distinguishes hard constraints from soft preferences.
    """
    required_capabilities: Tuple[ModelCapabilityType, ...] = (ModelCapabilityType.TEXT,)
    preferred_capabilities: Tuple[ModelCapabilityType, ...] = ()
    minimum_context_window: int = 0
    locality_requirement: LocalityRequirement = LocalityRequirement.CLOUD_ALLOWED
    privacy_requirement: PrivacyClass = PrivacyClass.PUBLIC_ALLOWED
    structured_output_required: bool = False
    preferred_provider_id: Optional[str] = None
    preferred_model_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "required_capabilities": [c.value for c in self.required_capabilities],
            "preferred_capabilities": [c.value for c in self.preferred_capabilities],
            "minimum_context_window": self.minimum_context_window,
            "locality_requirement": self.locality_requirement.value,
            "privacy_requirement": self.privacy_requirement.value,
            "structured_output_required": self.structured_output_required,
            "preferred_provider_id": self.preferred_provider_id,
            "preferred_model_id": self.preferred_model_id,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class RoutingResult:
    """
    Structured outcome of a deterministic routing decision.
    """
    success: bool
    provider_id: Optional[str] = None
    model_id: Optional[str] = None
    descriptor: Optional[ModelDescriptor] = None
    matched_capabilities: Tuple[ModelCapabilityType, ...] = ()
    unmet_preferences: Tuple[str, ...] = ()
    routing_rationale: str = ""
    fallback_used: bool = False
    error_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "provider_id": self.provider_id,
            "model_id": self.model_id,
            "descriptor": self.descriptor.to_dict() if self.descriptor else None,
            "matched_capabilities": [c.value for c in self.matched_capabilities],
            "unmet_preferences": list(self.unmet_preferences),
            "routing_rationale": self.routing_rationale,
            "fallback_used": self.fallback_used,
            "error_reason": self.error_reason,
            "metadata": dict(self.metadata),
        }
