"""
ATLAS Phase 6.3 — Digital Twin & Simulation Interfaces.

Defines abstract contracts for SimulationClock, DigitalTwin, SimulationWorld,
and ScenarioRunner.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core.models.orchestration import MultimodalObservation
from core.models.result import Result
from core.models.simulation import (
    TwinFault,
    TwinState,
    TwinTelemetry,
)


class SimulationClockInterface(ABC):
    """Abstract interface for deterministic simulation clock."""

    @abstractmethod
    def now(self) -> float:
        """Get current deterministic simulation timestamp in seconds."""
        raise NotImplementedError

    @abstractmethod
    def advance(self, seconds: float) -> float:
        """Advance the simulation time forward by the given positive delta. Returns new time."""
        raise NotImplementedError

    @abstractmethod
    def set_time(self, timestamp: float) -> float:
        """Explicitly advance clock to a future timestamp. Rejects backwards jumps."""
        raise NotImplementedError


class DigitalTwinInterface(ABC):
    """Abstract interface for an edge product digital twin."""

    @property
    @abstractmethod
    def twin_id(self) -> str:
        """Unique identifier of the digital twin."""
        raise NotImplementedError

    @abstractmethod
    def get_state(self) -> TwinState:
        """Retrieve current immutable twin state."""
        raise NotImplementedError

    @abstractmethod
    def get_telemetry(self) -> TwinTelemetry:
        """Generate current telemetry snapshot."""
        raise NotImplementedError

    @abstractmethod
    def execute_command(self, command: Any) -> Result:
        """Execute a semantic device command and return execution Result."""
        raise NotImplementedError

    @abstractmethod
    def generate_observation(self, observation_type: str, **kwargs) -> Optional[MultimodalObservation]:
        """Synthesize a canonical MultimodalObservation for central ingress."""
        raise NotImplementedError

    @abstractmethod
    def tick(self, now: float, delta_seconds: float) -> Sequence[MultimodalObservation]:
        """Advance internal twin state by delta_seconds and produce any autonomous observations."""
        raise NotImplementedError

    @abstractmethod
    def inject_fault(self, fault: TwinFault) -> None:
        """Inject a controlled fault into the twin."""
        raise NotImplementedError

    @abstractmethod
    def remove_fault(self, fault_id: str) -> bool:
        """Remove an injected fault by ID. Returns True if removed."""
        raise NotImplementedError

    @abstractmethod
    def clear_faults(self) -> None:
        """Clear all active faults on this twin."""
        raise NotImplementedError


class SimulationWorldInterface(ABC):
    """Abstract interface for simulation environment and twin container."""

    @abstractmethod
    def get_clock(self) -> SimulationClockInterface:
        """Retrieve the simulation clock driving this world."""
        raise NotImplementedError

    @abstractmethod
    def register_twin(self, twin: DigitalTwinInterface) -> None:
        """Register a digital twin into the simulation world."""
        raise NotImplementedError

    @abstractmethod
    def get_twin(self, twin_id: str) -> Optional[DigitalTwinInterface]:
        """Lookup a registered digital twin by ID."""
        raise NotImplementedError

    @abstractmethod
    def list_twins(self) -> Sequence[DigitalTwinInterface]:
        """List all registered digital twins."""
        raise NotImplementedError

    @abstractmethod
    def step(self, delta_seconds: float = 1.0) -> Any:
        """Step the simulation forward by delta_seconds, ticking all entities and twins."""
        raise NotImplementedError

    @abstractmethod
    def flush_observations(self) -> Sequence[MultimodalObservation]:
        """Drain and return all pending observations generated during simulation steps."""
        raise NotImplementedError

    @abstractmethod
    def get_snapshot(self) -> Any:
        """Produce an immutable snapshot of the entire simulation world."""
        raise NotImplementedError


class ScenarioRunnerInterface(ABC):
    """Abstract interface for executing scenarios deterministically."""

    @abstractmethod
    def run_scenario(self, scenario: Any) -> Any:
        """Execute the specified Scenario and return a ScenarioResult."""
        raise NotImplementedError
