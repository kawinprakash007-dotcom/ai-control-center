from typing import Dict, List, Optional, Tuple, Set
import threading

from core.models.model_router import ModelDescriptor
from core.interfaces.reasoning_interface import ReasoningProviderInterface
from core.interfaces.model_router_interface import ModelProviderRegistryInterface


class DuplicateProviderError(Exception):
    """Raised when registering a provider or model ID that already exists."""
    pass


class ModelProviderRegistry(ModelProviderRegistryInterface):
    """
    Thread-safe registry for ModelDescriptors and their associated ReasoningProviderInterface implementations.
    Responsibilities:
    - Register (ModelDescriptor, ReasoningProviderInterface)
    - Enable/disable providers or models
    - Look up providers by provider_id
    - List active/all descriptors
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._descriptors: Dict[Tuple[str, str], ModelDescriptor] = {}
        self._providers: Dict[str, ReasoningProviderInterface] = {}

    def register(
        self,
        descriptor: ModelDescriptor,
        provider: ReasoningProviderInterface,
    ) -> None:
        with self._lock:
            key = descriptor.key
            if key in self._descriptors:
                raise DuplicateProviderError(
                    f"Model '{descriptor.model_id}' under provider '{descriptor.provider_id}' is already registered."
                )

            self._descriptors[key] = descriptor
            # A provider instance is keyed by provider_id.
            # Multiple models can share the same provider instance.
            self._providers[descriptor.provider_id] = provider

    def unregister(self, provider_id: str, model_id: Optional[str] = None) -> bool:
        with self._lock:
            removed = False
            if model_id is not None:
                key = (provider_id, model_id)
                if key in self._descriptors:
                    del self._descriptors[key]
                    removed = True
                # If no models remain for this provider, remove provider instance
                has_remaining = any(p == provider_id for p, m in self._descriptors.keys())
                if not has_remaining and provider_id in self._providers:
                    del self._providers[provider_id]
            else:
                # Remove all models under provider_id
                keys_to_remove = [k for k in self._descriptors if k[0] == provider_id]
                for k in keys_to_remove:
                    del self._descriptors[k]
                    removed = True
                if provider_id in self._providers:
                    del self._providers[provider_id]

            return removed

    def get_provider(self, provider_id: str) -> Optional[ReasoningProviderInterface]:
        with self._lock:
            return self._providers.get(provider_id)

    def get_descriptor(self, provider_id: str, model_id: Optional[str] = None) -> Optional[ModelDescriptor]:
        with self._lock:
            if model_id is not None:
                return self._descriptors.get((provider_id, model_id))
            # If model_id not specified, find first enabled model under provider_id or first match
            matches = [d for (p, m), d in self._descriptors.items() if p == provider_id]
            if not matches:
                return None
            enabled_matches = [d for d in matches if d.enabled]
            return enabled_matches[0] if enabled_matches else matches[0]

    def list_descriptors(self, enabled_only: bool = True) -> List[ModelDescriptor]:
        with self._lock:
            if enabled_only:
                return [d for d in self._descriptors.values() if d.enabled]
            return list(self._descriptors.values())

    def set_enabled(self, provider_id: str, enabled: bool, model_id: Optional[str] = None) -> bool:
        with self._lock:
            found = False
            for (p, m), desc in list(self._descriptors.items()):
                if p == provider_id and (model_id is None or m == model_id):
                    # Replace with updated descriptor (frozen dataclass)
                    updated = ModelDescriptor(
                        provider_id=desc.provider_id,
                        model_id=desc.model_id,
                        display_name=desc.display_name,
                        capabilities=desc.capabilities,
                        context_window=desc.context_window,
                        is_local=desc.is_local,
                        cost_class=desc.cost_class,
                        latency_class=desc.latency_class,
                        privacy_class=desc.privacy_class,
                        priority=desc.priority,
                        enabled=enabled,
                        metadata=desc.metadata,
                    )
                    self._descriptors[(p, m)] = updated
                    found = True
            return found
