"""
ATLAS Phase 6.3 — Scenario Runner.

Deterministic execution coordinator for simulation scenarios.
Coordinates clock, world, twins, fault injection, and optional central pipeline ingress.
DOES NOT duplicate reasoning, cognitive, or policy authority.
"""

from collections import deque
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core.interfaces.orchestration_interface import (
    CentralInputGatewayInterface,
    DeviceGatewayInterface,
)
from core.interfaces.simulation_interface import (
    DigitalTwinInterface,
    ScenarioRunnerInterface,
)
from core.models.device_contract import ProductType
from core.models.orchestration import MultimodalObservation
from core.models.result import Result
from core.models.simulation import (
    SimulationLimits,
    TwinConfiguration,
    TwinFault,
    TwinFaultType,
    TwinPosition,
)
from orchestration.device_gateway import DeviceCommand
from simulation.adapters import create_digital_twin_device
from simulation.clock import SimulationClock
from simulation.environment import SimulatedEntity, SimulatedHazard
from simulation.scenario import (
    Scenario,
    ScenarioAssertion,
    ScenarioResult,
    ScenarioStep,
)
from simulation.twin import (
    DroneDigitalTwin,
    GlassDigitalTwin,
    RoverDigitalTwin,
    VisionDigitalTwin,
)
from simulation.world import SimulationWorld

logger = logging.getLogger("atlas.simulation.runner")


class ScenarioRunner(ScenarioRunnerInterface):
    """
    Deterministic runner executing scenarios against digital twins and the simulation world.
    """

    def __init__(
        self,
        input_gateway: Optional[CentralInputGatewayInterface] = None,
        device_gateway: Optional[DeviceGatewayInterface] = None,
        central_orchestrator: Optional[Any] = None,
        limits: Optional[SimulationLimits] = None,
    ):
        self.input_gateway = input_gateway
        self.device_gateway = device_gateway
        self.central_orchestrator = central_orchestrator
        self.limits = limits or SimulationLimits()

    def run_scenario(self, scenario: Scenario) -> ScenarioResult:
        """
        Execute a Scenario completely and deterministically.
        """
        # 1. Initialize clock and world
        clock = SimulationClock(initial_time=scenario.initial_time)
        world = SimulationWorld(clock=clock, limits=self.limits)

        # 2. Instantiate and register digital twins
        twins: Dict[str, DigitalTwinInterface] = {}
        for twin_id, cfg in scenario.initial_states.items():
            twin = self._create_twin_from_config(cfg, world)
            world.register_twin(twin)
            twins[twin_id] = twin

            if self.device_gateway:
                identity, adapter = create_digital_twin_device(twin)
                self.device_gateway.register_device(identity)
                self.device_gateway.register_adapter(adapter, device_id=identity.device_id)

        # 3. Setup initial environment if provided
        for entity_data in scenario.environment_config.get("entities", []):
            world.environment.add_entity(SimulatedEntity.from_dict(entity_data))
        for hazard_data in scenario.environment_config.get("hazards", []):
            world.environment.add_hazard(SimulatedHazard.from_dict(hazard_data))

        trace: List[Dict[str, Any]] = []
        command_results: Dict[str, Result] = {}
        total_ingested_observations: List[MultimodalObservation] = []
        current_offset = 0.0

        # 4. Execute steps in order
        for step in scenario.steps:
            # Advance time to step offset if needed
            target_offset = step.time_offset
            if target_offset > current_offset:
                delta = target_offset - current_offset
                world.step(delta_seconds=delta)
                current_offset = target_offset

            # Execute step action
            action_result = self._execute_step(
                step=step,
                world=world,
                twins=twins,
                command_results=command_results,
            )

            # Flush observations from world
            flushed = world.flush_observations()
            for obs in flushed:
                total_ingested_observations.append(obs)
                if self.input_gateway:
                    self.input_gateway.ingest_observation(obs)

            # Optionally trigger central orchestration
            if self.central_orchestrator and flushed:
                try:
                    self.central_orchestrator.run_cycle()
                except Exception as ex:
                    logger.warning("Central orchestrator cycle exception: %s", ex)

            trace.append({
                "step_id": step.step_id,
                "time_offset": step.time_offset,
                "action_type": step.action_type,
                "target_id": step.target_id,
                "result": action_result,
                "observations_count": len(flushed),
            })

        # 5. Evaluate assertions
        assertions_evaluated = len(scenario.assertions)
        passed_count = 0
        failures: List[Dict[str, Any]] = []

        for assertion in scenario.assertions:
            passed, failure_detail = self._evaluate_assertion(
                assertion=assertion,
                world=world,
                twins=twins,
                command_results=command_results,
                total_observations_count=len(total_ingested_observations),
            )
            if passed:
                passed_count += 1
            else:
                failures.append(failure_detail)

        success = (passed_count == assertions_evaluated)
        end_time = clock.now()
        sim_duration = end_time - scenario.initial_time

        summary = (
            f"Scenario '{scenario.name}' completed: {passed_count}/{assertions_evaluated} assertions passed "
            f"in {sim_duration:.1f}s simulated time."
        )

        return ScenarioResult(
            scenario_id=scenario.scenario_id,
            success=success,
            total_steps_executed=len(scenario.steps),
            start_time=scenario.initial_time,
            end_time=end_time,
            simulated_duration=sim_duration,
            assertions_evaluated=assertions_evaluated,
            assertions_passed=passed_count,
            assertion_failures=tuple(failures),
            trace=tuple(trace),
            summary=summary,
        )

    def _create_twin_from_config(
        self,
        cfg: TwinConfiguration,
        world: SimulationWorld,
    ) -> DigitalTwinInterface:
        ptype = cfg.product_type
        if ptype == ProductType.VISION:
            return VisionDigitalTwin(config=cfg, fault_manager=world.fault_manager, limits=self.limits)
        elif ptype == ProductType.GLASS:
            return GlassDigitalTwin(config=cfg, fault_manager=world.fault_manager, limits=self.limits)
        elif ptype == ProductType.DRONE:
            return DroneDigitalTwin(config=cfg, fault_manager=world.fault_manager, limits=self.limits)
        elif ptype == ProductType.ROVER:
            return RoverDigitalTwin(config=cfg, fault_manager=world.fault_manager, limits=self.limits)
        else:
            return VisionDigitalTwin(config=cfg, fault_manager=world.fault_manager, limits=self.limits)

    def _execute_step(
        self,
        step: ScenarioStep,
        world: SimulationWorld,
        twins: Dict[str, DigitalTwinInterface],
        command_results: Dict[str, Result],
    ) -> Dict[str, Any]:
        atype = step.action_type.upper()
        target_id = step.target_id

        if atype == "DISPATCH_COMMAND":
            if not target_id or target_id not in twins:
                return {"error": f"Target twin '{target_id}' not found"}
            twin = twins[target_id]
            cap = str(step.payload.get("capability", "default"))
            act = str(step.payload.get("action", ""))
            params = dict(step.payload.get("parameters", {}))
            cmd = DeviceCommand(
                dispatch_id=f"cmd_{step.step_id}",
                device_id=target_id,
                capability=cap,
                action=act,
                parameters=params,
                timestamp=world.clock.now(),
            )
            res = twin.execute_command(cmd)
            command_results[step.step_id] = res
            return {"success": res.success, "message": res.message, "data": res.data}

        elif atype == "EMIT_OBSERVATION":
            if not target_id or target_id not in twins:
                return {"error": f"Target twin '{target_id}' not found"}
            twin = twins[target_id]
            obs_type = str(step.payload.get("observation_type", "GENERIC_EVENT"))
            obs = twin.generate_observation(observation_type=obs_type, payload=step.payload.get("payload", {}))
            if obs:
                world.queue_observation(obs)
                return {"success": True, "observation_id": obs.observation_id}
            return {"success": False, "error": "Observation generation returned None"}

        elif atype == "INJECT_FAULT":
            if not target_id:
                return {"error": "Target twin required for fault injection"}
            ftype = TwinFaultType.from_str(step.payload.get("fault_type", "COMMAND_FAILURE"))
            fault = TwinFault(
                fault_id=str(step.payload.get("fault_id", f"fault_{step.step_id}")),
                twin_id=target_id,
                fault_type=ftype,
                parameters=dict(step.payload.get("parameters", {})),
                injected_at=world.clock.now(),
                duration_seconds=step.payload.get("duration_seconds"),
            )
            world.fault_manager.inject_fault(fault)
            return {"success": True, "fault_id": fault.fault_id}

        elif atype == "REMOVE_FAULT":
            fid = str(step.payload.get("fault_id", ""))
            removed = world.fault_manager.remove_fault(fid)
            return {"success": removed, "fault_id": fid}

        elif atype == "SPAWN_ENTITY":
            entity = SimulatedEntity.from_dict(step.payload)
            world.environment.add_entity(entity)
            return {"success": True, "entity_id": entity.entity_id}

        elif atype == "ADD_HAZARD":
            hazard = SimulatedHazard.from_dict(step.payload)
            world.environment.add_hazard(hazard)
            return {"success": True, "hazard_id": hazard.hazard_id}

        elif atype == "ADVANCE_TIME":
            secs = float(step.payload.get("seconds", 1.0))
            world.step(delta_seconds=secs)
            return {"success": True, "advanced_seconds": secs}

        return {"error": f"Unknown action_type '{step.action_type}'"}

    def _evaluate_assertion(
        self,
        assertion: ScenarioAssertion,
        world: SimulationWorld,
        twins: Dict[str, DigitalTwinInterface],
        command_results: Dict[str, Result],
        total_observations_count: int,
    ) -> Tuple[bool, Dict[str, Any]]:
        target_type = assertion.target_type.upper()
        target_id = assertion.target_id
        field_name = assertion.expected_field
        expected = assertion.expected_value
        op = assertion.operator.upper()

        actual: Any = None

        if target_type == "TWIN_STATE":
            if target_id not in twins:
                return False, {"assertion_id": assertion.assertion_id, "error": f"Twin '{target_id}' not found"}
            state = twins[target_id].get_state()
            state_dict = state.to_dict()
            actual = state_dict.get(field_name)

        elif target_type == "COMMAND_RESULT":
            if target_id not in command_results:
                return False, {"assertion_id": assertion.assertion_id, "error": f"Command step '{target_id}' not found"}
            res = command_results[target_id]
            if field_name == "success":
                actual = res.success
            elif field_name == "error_code":
                actual = res.error_code
            elif field_name == "message":
                actual = res.message
            else:
                actual = res.data.get(field_name) if res.data else None

        elif target_type == "OBSERVATION_COUNT":
            actual = total_observations_count

        elif target_type == "ACTIVE_FAULT":
            faults = world.fault_manager.get_active_faults(target_id, current_time=world.clock.now())
            if field_name == "count":
                actual = len(faults)
            elif field_name == "has_fault":
                actual = any(f.fault_type.value == str(expected) for f in faults)

        passed = self._check_operator(actual, expected, op)
        detail = {
            "assertion_id": assertion.assertion_id,
            "target_type": target_type,
            "target_id": target_id,
            "field": field_name,
            "expected": expected,
            "actual": actual,
            "operator": op,
            "passed": passed,
        }
        return passed, detail

    def _check_operator(self, actual: Any, expected: Any, operator: str) -> bool:
        if operator == "EQUALS":
            return actual == expected
        elif operator == "CONTAINS":
            if actual is None:
                return False
            return str(expected) in str(actual)
        elif operator == "GREATER_THAN":
            return actual is not None and actual > expected
        elif operator == "LESS_THAN":
            return actual is not None and actual < expected
        elif operator == "IS_NONE":
            return actual is None
        elif operator == "NOT_NONE":
            return actual is not None
        return actual == expected
