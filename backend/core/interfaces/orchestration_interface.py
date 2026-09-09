from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core.models.orchestration import (
    DeviceCapabilityDescriptor,
    DeviceIdentity,
    ModalityType,
    MultimodalObservation,
    Situation,
)
from core.models.tool_call import ToolCall


class CentralInputGatewayInterface(ABC):
    """
    Contract for Central Input Gateway.
    Responsible for normalizing, validating, and ingesting multimodal observations
    from edge devices, perception pipelines, and external streams.
    """

    @abstractmethod
    def ingest_observation(self, observation: MultimodalObservation) -> None:
        """
        Validate and ingest an immutable MultimodalObservation.
        Routes to situation fusion or world state without executing side-effects.
        """
        pass

    @abstractmethod
    def get_recent_observations(
        self,
        modality: Optional[ModalityType] = None,
        limit: int = 100,
    ) -> Sequence[MultimodalObservation]:
        """Retrieve recent observations filtered by modality if specified."""
        pass


class SituationFusionInterface(ABC):
    """
    Contract for Situation Fusion.
    Responsible for correlating related MultimodalObservations into semantic Situations.
    CRITICAL: Contains NO execution engine; outputs declarative Situations only.
    """

    @abstractmethod
    def evaluate_observations(
        self,
        observations: Sequence[MultimodalObservation],
    ) -> Sequence[Situation]:
        """
        Correlate and fuse multimodal observations into candidate or updated Situations.
        """
        pass

    @abstractmethod
    def get_active_situations(self) -> Sequence[Situation]:
        """Retrieve all currently active and unexpired Situations."""
        pass


class DeviceGatewayInterface(ABC):
    """
    Architectural boundary for device registration, identity management, and adapter routing.

    CRITICAL ARCHITECTURAL RULES:
    1. ToolOrchestrator remains the SOLE execution authority.
    2. DeviceGateway is ONLY a routing, identity, and adapter boundary.
    3. DeviceGateway NEVER directly executes hardware commands or accesses physical buses.
    """

    @abstractmethod
    def register_device(self, device: DeviceIdentity) -> None:
        """Register or update an edge participant device identity."""
        pass

    @abstractmethod
    def get_device(self, device_id: str) -> Optional[DeviceIdentity]:
        """Retrieve device identity by device_id."""
        pass

    @abstractmethod
    def list_devices(self) -> Sequence[DeviceIdentity]:
        """List all registered device identities."""
        pass

    @abstractmethod
    def resolve_adapter(self, device_id: str) -> "DeviceAdapterInterface":
        """
        Resolve the appropriate protocol adapter for an identified device.
        Does NOT execute commands.
        """
        pass


class DeviceAdapterInterface(ABC):
    """
    Protocol and transport translation boundary for edge devices.

    CRITICAL ARCHITECTURAL RULES:
    1. Translates model-neutral ToolCalls into device-specific payloads.
    2. NEVER directly triggered by CognitiveRuntime; only invoked via ToolOrchestrator boundary.
    """

    @abstractmethod
    def get_protocol_name(self) -> str:
        """Return canonical protocol name (e.g. 'simulated', 'mqtt', 'mavlink', 'ros2')."""
        pass

    @abstractmethod
    def format_command(self, tool_call: ToolCall) -> Dict[str, Any]:
        """
        Translate a ToolCall into a device-specific payload representation.
        Pure translation; does not perform network transmission in model layer.
        """
        pass
