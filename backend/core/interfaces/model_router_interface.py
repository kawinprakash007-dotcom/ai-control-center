from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Tuple

from core.models.model_router import (
    ModelDescriptor,
    ModelRequirements,
    RoutingResult,
)
from core.interfaces.reasoning_interface import ReasoningProviderInterface


class ModelProviderRegistryInterface(ABC):
    """
    Contract for managing model descriptors and associating them with ReasoningProviderInterface instances.
    Responsible for cataloging available providers without executing reasoning.
    """

    @abstractmethod
    def register(
        self,
        descriptor: ModelDescriptor,
        provider: ReasoningProviderInterface,
    ) -> None:
        """Register a model descriptor and its reasoning provider implementation."""
        pass

    @abstractmethod
    def unregister(self, provider_id: str, model_id: Optional[str] = None) -> bool:
        """Unregister a provider or a specific model under a provider."""
        pass

    @abstractmethod
    def get_provider(self, provider_id: str) -> Optional[ReasoningProviderInterface]:
        """Retrieve the reasoning provider instance for a given provider_id."""
        pass

    @abstractmethod
    def get_descriptor(self, provider_id: str, model_id: Optional[str] = None) -> Optional[ModelDescriptor]:
        """Retrieve the descriptor for a registered provider/model."""
        pass

    @abstractmethod
    def list_descriptors(self, enabled_only: bool = True) -> List[ModelDescriptor]:
        """List all registered descriptors."""
        pass

    @abstractmethod
    def set_enabled(self, provider_id: str, enabled: bool, model_id: Optional[str] = None) -> bool:
        """Enable or disable a provider or specific model."""
        pass


class ModelRouterInterface(ABC):
    """
    Contract for deterministically selecting a ReasoningProvider based on ModelRequirements.
    NEVER executes tools, calls OS APIs, or modifies state.
    """

    @property
    @abstractmethod
    def registry(self) -> ModelProviderRegistryInterface:
        """Access underlying model provider registry."""
        pass

    @abstractmethod
    def route(self, requirements: ModelRequirements) -> RoutingResult:
        """
        Deterministically evaluate available providers and select the best match.

        Args:
            requirements: Structured task requirements and preferences.

        Returns:
            RoutingResult indicating success, selected provider_id, model_id,
            and rationale, or failure reason.
        """
        pass

    @abstractmethod
    def get_provider(self, provider_id: str) -> Optional[ReasoningProviderInterface]:
        """Retrieve the provider instance for the routed provider_id."""
        pass
