"""
ATLAS Phase 6.3 — Digital Twin & Simulation Package.

Exposes deterministic simulation clock, digital twins (Vision, Glass, Drone, Rover),
simulation world and environment, fault injection manager, scenario engine,
and scenario runner.
"""

from simulation.clock import SimulationClock
from simulation.environment import (
    SimulatedEntity,
    SimulatedEnvironment,
    SimulatedHazard,
)
from simulation.fault_injection import FaultInjectionManager
from simulation.twin import (
    BaseDigitalTwin,
    DroneDigitalTwin,
    GlassDigitalTwin,
    RoverDigitalTwin,
    VisionDigitalTwin,
)
from simulation.adapters import (
    DigitalTwinAdapter,
    create_digital_twin_device,
)
from simulation.world import (
    SimulationWorld,
    SimulationWorldSnapshot,
    SimulationWorldStepResult,
)
from simulation.scenario import (
    Scenario,
    ScenarioAssertion,
    ScenarioBuilder,
    ScenarioResult,
    ScenarioStep,
)
from simulation.runner import ScenarioRunner

__all__ = [
    "SimulationClock",
    "SimulatedEnvironment",
    "SimulatedEntity",
    "SimulatedHazard",
    "FaultInjectionManager",
    "BaseDigitalTwin",
    "VisionDigitalTwin",
    "GlassDigitalTwin",
    "DroneDigitalTwin",
    "RoverDigitalTwin",
    "DigitalTwinAdapter",
    "create_digital_twin_device",
    "SimulationWorld",
    "SimulationWorldStepResult",
    "SimulationWorldSnapshot",
    "Scenario",
    "ScenarioStep",
    "ScenarioAssertion",
    "ScenarioResult",
    "ScenarioBuilder",
    "ScenarioRunner",
]
