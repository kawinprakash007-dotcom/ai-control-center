from typing import Optional, Dict, Any, List
from tools.tool_registry import TOOL_REGISTRY


class CapabilityRegistry:
    """
    Authoritative registry resolving capability names to their implementing callables/objects.
    Supports dependency injection for isolated testing without live external dependencies.
    Dynamically delegates to TOOL_REGISTRY by default to preserve legacy tool registration.
    """

    def __init__(self, capabilities: Optional[Dict[str, Any]] = None):
        """
        Initialize CapabilityRegistry.

        Args:
            capabilities: Optional dictionary mapping capability names to callables/objects.
                          If omitted, delegates dynamically to system TOOL_REGISTRY.
        """
        self._custom_capabilities: Optional[Dict[str, Any]] = (
            dict(capabilities) if capabilities is not None else None
        )

    def get_executor(self, tool_name: str) -> Optional[Any]:
        """
        Retrieve capability implementation by name.
        Maintains complete backward compatibility with legacy Router, tests, and TOOL_REGISTRY.
        """
        if not tool_name:
            return None
        key = str(tool_name).strip().lower()
        if self._custom_capabilities is not None:
            return self._custom_capabilities.get(key)
        return TOOL_REGISTRY.get(key) or TOOL_REGISTRY.get(tool_name)

    def get(self, capability_name: str) -> Optional[Any]:
        """
        Retrieve capability implementation by name.
        Delegates to get_executor for uniform resolution and mocking compatibility.
        """
        return self.get_executor(capability_name)

    def register(self, name: str, capability: Any) -> None:
        """
        Register or override a capability in this registry.
        """
        if self._custom_capabilities is None:
            self._custom_capabilities = dict(TOOL_REGISTRY)
        self._custom_capabilities[str(name).strip().lower()] = capability

    def has_capability(self, name: str) -> bool:
        """
        Check if a capability is registered.
        """
        if not name:
            return False
        key = str(name).strip().lower()
        if self._custom_capabilities is not None:
            return key in self._custom_capabilities
        return key in TOOL_REGISTRY or name in TOOL_REGISTRY

    def list_capabilities(self) -> List[str]:
        """
        List all registered capability names.
        """
        if self._custom_capabilities is not None:
            return list(self._custom_capabilities.keys())
        return list(TOOL_REGISTRY.keys())