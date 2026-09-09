from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core.models.orchestration import (
    ConnectivityStatus,
    DeviceCapabilityDescriptor,
    DeviceIdentity,
    DeviceType,
    ModalityType,
    MultimodalObservation,
    Situation,
)
from core.models.result import Result
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

    def receive_envelope(self, envelope: Any, now: Optional[float] = None) -> Any:
        """Receive, validate, and normalize an IngressEnvelope."""
        raise NotImplementedError

    def ingest_batch(
        self,
        envelopes: Sequence[Any],
        now: Optional[float] = None,
    ) -> Sequence[Any]:
        """Bounded batch ingestion of envelopes or observations."""
        raise NotImplementedError

    def get_metrics(self) -> Any:
        """Query gateway operational counters and status."""
        raise NotImplementedError


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
    def unregister_device(self, device_id: str) -> bool:
        """Unregister an edge participant."""
        pass

    @abstractmethod
    def get_device(self, device_id: str) -> Optional[DeviceIdentity]:
        """Retrieve device identity by device_id."""
        pass

    @abstractmethod
    def list_devices(self, device_type: Optional[DeviceType] = None) -> Sequence[DeviceIdentity]:
        """List all registered device identities, optionally filtered by type."""
        pass

    @abstractmethod
    def update_device_status(
        self,
        device_id: str,
        status: ConnectivityStatus,
        timestamp: Optional[float] = None,
    ) -> bool:
        """Update operational connectivity status of a registered device."""
        pass

    @abstractmethod
    def query_device_status(self, device_id: str, now: Optional[float] = None) -> ConnectivityStatus:
        """Query semantic connectivity status of a registered device."""
        pass

    @abstractmethod
    def list_device_capabilities(self, device_id: str) -> Sequence[DeviceCapabilityDescriptor]:
        """List declared capability descriptors of a registered device."""
        pass

    @abstractmethod
    def register_adapter(
        self,
        adapter: "DeviceAdapterInterface",
        device_type: Optional[DeviceType] = None,
        device_id: Optional[str] = None,
    ) -> None:
        """Register a protocol adapter for a device type or specific device."""
        pass

    @abstractmethod
    def resolve_adapter(self, device_id: str) -> "DeviceAdapterInterface":
        """
        Resolve the appropriate protocol adapter for an identified device.
        Does NOT execute commands.
        """
        pass

    @abstractmethod
    def dispatch_to_device(
        self,
        device_id: str,
        capability: str,
        action: str,
        parameters: Dict[str, Any],
        dispatch_id: Optional[str] = None,
        correlation_id: str = "",
        causation_id: Optional[str] = None,
        now: Optional[float] = None,
    ) -> Result:
        """
        Validate capability/action/parameters and route command to the registered adapter.
        """
        pass


class DeviceAdapterInterface(ABC):
    """
    Protocol and transport translation boundary for edge devices.

    CRITICAL ARCHITECTURAL RULES:
    1. Translates model-neutral ToolCalls or semantic commands into device-specific payloads.
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

    @abstractmethod
    def execute_command(self, command: Any) -> Result:
        """
        Execute a semantic DeviceCommand on the underlying transport/virtual device.
        """
        pass


class CentralOrchestratorInterface(ABC):
    """
    Contract for Central Orchestration Coordinator (Phase 5.0e).
    Coordinates ingress, situation fusion, world state updates, event autonomy,
    goal management, cognitive runtime turns, and device gateway dispatch.

    CRITICAL BOUNDARIES:
    - Pure integration coordinator; does NOT contain duplicate reasoning/planners/policy engines.
    - Preserves all authority boundaries across CognitiveRuntime, PolicyEngine, ToolOrchestrator,
      AutonomousGoalManager, and WorldState.
    - Enforces bounded cycles, loop protection, and full causal lineage.
    """

    @abstractmethod
    def process_ingress(
        self,
        data: Any,
        correlation_id: Optional[str] = None,
        causation_id: Optional[str] = None,
        now: Optional[float] = None,
    ) -> Any:
        """Process ingress message through central orchestration pipeline."""
        pass

    @abstractmethod
    def run_cycle(
        self,
        ingress_batch: Sequence[Any],
        now: Optional[float] = None,
    ) -> Any:
        """Run bounded orchestration cycle over an ingress batch."""
        pass
