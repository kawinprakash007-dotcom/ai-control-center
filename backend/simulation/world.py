"""
ATLAS Phase 6.3 — Simulation World.

Container orchestrating SimulationClock, SimulatedEnvironment, FaultInjectionManager,
and registered DigitalTwins.
CRITICAL ARCHITECTURAL BOUNDARY:
SimulationWorld NEVER directly mutates production WorldState.
Observations are queued and flushed exclusively into CentralInputGateway.
"""

from collections import deque
from dataclasses import dataclass, field
import threading
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core.interfaces.simulation_interface import (
    DigitalTwinInterface,
    SimulationWorldInterface,
)
from core.models.orchestration import MultimodalObservation
from core.models.simulation import (
    SimulationLimits,
    TwinPosition,
    TwinState,
)
from simulation.clock import SimulationClock
from simulation.environment import SimulatedEnvironment
from simulation.fault_injection import FaultInjectionManager


@dataclass(frozen=True)
class SimulationWorldStepResult:
    """Outcome of a single world simulation tick."""
    simulated_time: float
    delta_seconds: float
    observations_generated: int
    active_twins_count: int
    active_faults_count: int


@dataclass(frozen=True)
class SimulationWorldSnapshot:
    """Immutable snapshot of complete simulation state at a point in time."""
    simulated_time: float
    twins: Dict[str, TwinState]
    environment: Dict[str, Any]
    active_faults: List[Dict[str, Any]]
    pending_observations_count: int


class SimulationWorld(SimulationWorldInterface):
    """
    Authoritative container for deterministic multi-product simulation.
    Coordinates time progression, spatial environment, fault injection,
    and observation emission.
    """

    def __init__(
        self,
        clock: Optional[SimulationClock] = None,
        environment: Optional[SimulatedEnvironment] = None,
        fault_manager: Optional[FaultInjectionManager] = None,
        limits: Optional[SimulationLimits] = None,
    ):
        self._lock = threading.RLock()
        self.limits = limits or SimulationLimits()
        self.clock = clock or SimulationClock()
        self.environment = environment or SimulatedEnvironment(limits=self.limits)
        self.fault_manager = fault_manager or FaultInjectionManager(self.limits)

        self._twins: Dict[str, DigitalTwinInterface] = {}
        self._pending_observations: deque = deque(maxlen=self.limits.max_pending_observations)

    def get_clock(self) -> SimulationClock:
        return self.clock

    def get_environment(self) -> SimulatedEnvironment:
        return self.environment

    def get_fault_manager(self) -> FaultInjectionManager:
        return self.fault_manager

    def register_twin(self, twin: DigitalTwinInterface) -> None:
        """Register a digital twin into the world."""
        with self._lock:
            if len(self._twins) >= self.limits.max_twins and twin.twin_id not in self._twins:
                raise ValueError(f"Simulation world twin capacity bound ({self.limits.max_twins}) exceeded")
            self._twins[twin.twin_id] = twin

    def unregister_twin(self, twin_id: str) -> bool:
        """Remove a digital twin from the world."""
        with self._lock:
            return self._twins.pop(twin_id, None) is not None

    def get_twin(self, twin_id: str) -> Optional[DigitalTwinInterface]:
        with self._lock:
            return self._twins.get(twin_id)

    def list_twins(self) -> List[DigitalTwinInterface]:
        with self._lock:
            return list(self._twins.values())

    def step(self, delta_seconds: float = 1.0) -> SimulationWorldStepResult:
        """
        Advance simulation time, update environment entities, tick digital twins,
        and synthesize environmental observations into the pending queue.
        """
        if delta_seconds < 0.0:
            raise ValueError(f"Cannot step simulation world by negative delta: {delta_seconds}")

        with self._lock:
            # 1. Advance clock
            now = self.clock.advance(delta_seconds)

            # 2. Advance environment entities
            self.environment.step(delta_seconds)

            generated_observations: List[MultimodalObservation] = []

            # 3. Tick each twin
            twins = list(self._twins.values())
            for twin in twins:
                obs_list = twin.tick(now, delta_seconds)
                generated_observations.extend(obs_list)

            # 4. Synthesize proximity-based perception
            for twin in twins:
                state = twin.get_state()
                if not state.position:
                    continue

                # Nearby hazards trigger observations
                hazards = self.environment.get_hazards_near(state.position, radius_meters=30.0)
                for hazard in hazards:
                    obs = twin.generate_observation(
                        observation_type=hazard.hazard_type,
                        timestamp=now,
                        payload={"hazard_id": hazard.hazard_id, "severity": hazard.severity},
                    )
                    if obs:
                        generated_observations.append(obs)

                # Nearby entities trigger observations
                entities = self.environment.get_entities_near(state.position, radius_meters=25.0)
                for entity in entities:
                    obs = twin.generate_observation(
                        observation_type=f"{entity.entity_type}_DETECTED",
                        timestamp=now,
                        payload={"entity_id": entity.entity_id, "entity_type": entity.entity_type},
                    )
                    if obs:
                        generated_observations.append(obs)

            # 5. Enqueue observations
            for obs in generated_observations:
                self._pending_observations.append(obs)

            active_faults = self.fault_manager.get_active_faults(current_time=now)

            return SimulationWorldStepResult(
                simulated_time=now,
                delta_seconds=delta_seconds,
                observations_generated=len(generated_observations),
                active_twins_count=len(twins),
                active_faults_count=len(active_faults),
            )

    def queue_observation(self, observation: MultimodalObservation) -> None:
        """Manually queue an observation into the pending ingress queue."""
        with self._lock:
            self._pending_observations.append(observation)

    def flush_observations(self) -> List[MultimodalObservation]:
        """Drain and return all pending observations for central ingress."""
        with self._lock:
            flushed = list(self._pending_observations)
            self._pending_observations.clear()
            return flushed

    def get_snapshot(self) -> SimulationWorldSnapshot:
        """Produce an immutable snapshot of current world state."""
        with self._lock:
            now = self.clock.now()
            twin_states = {tid: twin.get_state() for tid, twin in self._twins.items()}
            env_dict = self.environment.to_dict()
            faults = [f.to_dict() for f in self.fault_manager.get_active_faults(current_time=now)]
            return SimulationWorldSnapshot(
                simulated_time=now,
                twins=twin_states,
                environment=env_dict,
                active_faults=faults,
                pending_observations_count=len(self._pending_observations),
            )
