from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Set

from core.models.computer import ComputerObservation
from core.models.orchestration import ModalityType, MultimodalObservation
from core.models.perception import (
    GroundingRequest,
    GroundingResult,
    PerceptionCapability,
    PerceptionRequest,
    PerceptionResult,
    PerceptionSource,
    VisualElement,
    VisualScene,
)


class VisualPerceptionProvider(ABC):
    """
    Abstract contract for perception providers (OCR, CV, Accessibility, Multimodal).
    Never controls device directly or executes clicks.
    """
    source: PerceptionSource

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if backend dependencies and hardware are accessible."""
        pass

    @abstractmethod
    def perceive(self, observation: ComputerObservation) -> List[VisualElement]:
        """Extract visual elements from a desktop observation."""
        pass


class PerceptionEngineInterface(ABC):
    """
    Contract for perception engine that aggregates and deduplicates multi-source perception.
    """

    @abstractmethod
    def perceive(self, observation: ComputerObservation) -> VisualScene:
        """Process observation into a unified, deduplicated VisualScene."""
        pass


class TargetGrounderInterface(ABC):
    """
    Contract for resolving user intent / grounding requests to visual targets.
    Produces GroundedTarget data; NEVER triggers mouse/keyboard actions.
    """

    @abstractmethod
    def ground(
        self,
        request: GroundingRequest,
        scene: VisualScene,
        current_observation: Optional[ComputerObservation] = None,
    ) -> GroundingResult:
        """Resolve a GroundingRequest against the VisualScene."""
        pass


class PerceptionProviderInterface(ABC):
    """
    Contract for multimodal perception providers in Phase 6.5a.
    Perception providers extract structured evidence from raw modality inputs.
    They NEVER execute actions, trigger device controls, or make mission decisions.
    """

    @property
    @abstractmethod
    def provider_id(self) -> str:
        """Unique provider identifier."""
        pass

    @property
    @abstractmethod
    def provider_version(self) -> str:
        """Provider semantic version."""
        pass

    @property
    @abstractmethod
    def supported_capabilities(self) -> Set[PerceptionCapability]:
        """Set of perception capabilities supported by this provider."""
        pass

    @property
    @abstractmethod
    def supported_modalities(self) -> Set[ModalityType]:
        """Set of input modalities supported by this provider."""
        pass

    @abstractmethod
    def perceive(self, request: PerceptionRequest) -> PerceptionResult:
        """Process perception request and return structured perception result."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if provider is operational and available."""
        pass

    @abstractmethod
    def get_health(self) -> Dict[str, Any]:
        """Return health status and telemetry for this provider."""
        pass


class PerceptionProviderRegistryInterface(ABC):
    """
    Contract for registering, unregistering, and selecting perception providers.
    """

    @abstractmethod
    def register_provider(self, provider: PerceptionProviderInterface) -> None:
        """Register a perception provider."""
        pass

    @abstractmethod
    def unregister_provider(self, provider_id: str) -> bool:
        """Unregister a perception provider by ID."""
        pass

    @abstractmethod
    def get_provider(self, provider_id: str) -> Optional[PerceptionProviderInterface]:
        """Retrieve provider by ID."""
        pass

    @abstractmethod
    def list_providers(self) -> List[PerceptionProviderInterface]:
        """List all registered perception providers."""
        pass

    @abstractmethod
    def list_capabilities(self) -> Set[PerceptionCapability]:
        """List all capabilities supported across registered providers."""
        pass

    @abstractmethod
    def find_providers_for_capability(
        self, capability: PerceptionCapability
    ) -> List[PerceptionProviderInterface]:
        """Find providers capable of performing the requested capability."""
        pass

    @abstractmethod
    def select_provider(
        self,
        capability: PerceptionCapability,
        modality: Optional[ModalityType] = None,
    ) -> Optional[PerceptionProviderInterface]:
        """Deterministically select best provider matching capability and modality."""
        pass


class PerceptionObservationNormalizerInterface(ABC):
    """
    Contract for normalizing perception results into canonical MultimodalObservation
    instances ready for CentralInputGateway intake.
    """

    @abstractmethod
    def normalize(
        self,
        result: PerceptionResult,
        evidence_index: int = 0,
    ) -> MultimodalObservation:
        """Normalize a single evidence item into a canonical MultimodalObservation."""
        pass

    @abstractmethod
    def normalize_all(
        self,
        result: PerceptionResult,
    ) -> Sequence[MultimodalObservation]:
        """Normalize all evidence items into canonical MultimodalObservation objects."""
        pass
