"""
Perception Provider Registry (Phase 6.5a)

Thread-safe registry for multimodal perception providers.
Supports registration, capability-based and modality-based provider discovery,
and deterministic provider selection.
"""

from __future__ import annotations

import logging
import threading
from typing import Dict, List, Optional, Set

from core.interfaces.perception_interface import (
    PerceptionProviderInterface,
    PerceptionProviderRegistryInterface,
)
from core.models.orchestration import ModalityType
from core.models.perception import PerceptionCapability, PerceptionLimits

logger = logging.getLogger(__name__)


class PerceptionProviderRegistry(PerceptionProviderRegistryInterface):
    """
    Thread-safe registry managing perception provider instances.
    Provides deterministic selection and enforces capacity bounds.
    """

    def __init__(self, max_providers: int = PerceptionLimits.max_provider_count) -> None:
        self._max_providers = max_providers
        self._providers: Dict[str, PerceptionProviderInterface] = {}
        self._lock = threading.RLock()

    def register_provider(self, provider: PerceptionProviderInterface) -> None:
        """
        Register a perception provider.

        Raises:
            ValueError: If provider is invalid or already registered.
            RuntimeError: If maximum provider count is reached.
        """
        if provider is None or not isinstance(provider, PerceptionProviderInterface):
            raise ValueError(f"Invalid provider instance: {type(provider)}")

        provider_id = provider.provider_id
        if not provider_id or not isinstance(provider_id, str):
            raise ValueError("Provider must have a non-empty string provider_id")

        if not provider.provider_version or not isinstance(provider.provider_version, str):
            raise ValueError(f"Provider '{provider_id}' must have a valid provider_version")

        if not provider.supported_capabilities:
            raise ValueError(f"Provider '{provider_id}' must declare at least one supported capability")

        if not provider.supported_modalities:
            raise ValueError(f"Provider '{provider_id}' must declare at least one supported modality")

        with self._lock:
            if provider_id in self._providers:
                raise ValueError(f"Perception provider '{provider_id}' is already registered")

            if len(self._providers) >= self._max_providers:
                raise RuntimeError(
                    f"Maximum perception provider limit ({self._max_providers}) exceeded"
                )

            self._providers[provider_id] = provider
            logger.info(
                "Registered perception provider '%s' (v%s, capabilities: %s)",
                provider_id,
                provider.provider_version,
                [c.value for c in provider.supported_capabilities],
            )

    def unregister_provider(self, provider_id: str) -> bool:
        """
        Unregister a perception provider by ID.
        Returns True if provider was removed, False if not found.
        """
        with self._lock:
            if provider_id in self._providers:
                del self._providers[provider_id]
                logger.info("Unregistered perception provider '%s'", provider_id)
                return True
            return False

    def get_provider(self, provider_id: str) -> Optional[PerceptionProviderInterface]:
        """Retrieve provider by ID or return None."""
        with self._lock:
            return self._providers.get(provider_id)

    def list_providers(self) -> List[PerceptionProviderInterface]:
        """Return a snapshot list of all registered providers."""
        with self._lock:
            return list(self._providers.values())

    def list_capabilities(self) -> Set[PerceptionCapability]:
        """Return union of all capabilities supported across registered providers."""
        with self._lock:
            caps: Set[PerceptionCapability] = set()
            for p in self._providers.values():
                caps.update(p.supported_capabilities)
            return caps

    def find_providers_for_capability(
        self, capability: PerceptionCapability
    ) -> List[PerceptionProviderInterface]:
        """Find all registered providers that support the given capability."""
        if not isinstance(capability, PerceptionCapability):
            raise ValueError(f"Expected PerceptionCapability, got {type(capability)}")

        with self._lock:
            return [
                p for p in self._providers.values()
                if capability in p.supported_capabilities
            ]

    def select_provider(
        self,
        capability: PerceptionCapability,
        modality: Optional[ModalityType] = None,
    ) -> Optional[PerceptionProviderInterface]:
        """
        Deterministically select the best available provider matching capability and modality.
        Returns None if no available provider satisfies the criteria.
        """
        if not isinstance(capability, PerceptionCapability):
            raise ValueError(f"Expected PerceptionCapability, got {type(capability)}")

        with self._lock:
            candidates: List[PerceptionProviderInterface] = []
            for p in self._providers.values():
                if capability not in p.supported_capabilities:
                    continue
                if modality is not None and modality not in p.supported_modalities:
                    continue
                if not p.is_available():
                    continue
                candidates.append(p)

            if not candidates:
                return None

            # Deterministic sorting by provider_id
            candidates.sort(key=lambda p: p.provider_id)
            return candidates[0]

    def clear(self) -> None:
        """Clear all registered providers (primarily for test teardown)."""
        with self._lock:
            self._providers.clear()
