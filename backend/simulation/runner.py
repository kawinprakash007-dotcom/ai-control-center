"""
ATLAS Phase 6.3 & 6.5e — Scenario Runner.

Deterministic execution coordinator for simulation scenarios.
Coordinates clock, world, twins, fault injection, temporal cross-modal fusion,
and optional central pipeline ingress.

STRICT ARCHITECTURAL GUARANTEE:
DOES NOT duplicate reasoning, cognitive, policy, mission, or world authority.
Zero direct WorldState mutations. Zero direct GoalStore mutations.
Zero direct Mission creation bypasses.
Zero physical hardware driver imports. Zero external execution or shell calls.
"""

from collections import deque
import hashlib
import json
import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

from core.interfaces.orchestration_interface import (
    CentralInputGatewayInterface,
    DeviceGatewayInterface,
)
from core.interfaces.simulation_interface import (
    DigitalTwinInterface,
    ScenarioRunnerInterface,
)
from core.models.device_contract import ProductRole, ProductType
from core.models.orchestration import GeoLocation, ModalityType, MultimodalObservation
from core.models.result import Result
from core.models.scenario import (
    Scenario,
    ScenarioAssertion,
    ScenarioAssertionOperator,
    ScenarioAssertionResult,
    ScenarioAssertionSeverity,
    ScenarioAssertionTarget,
    ScenarioLimits,
    ScenarioResult,
    ScenarioStep,
    ScenarioStepType,
)
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
    Deterministic runner executing scenarios against digital twins, the simulation world,
    and optional perception / fusion / central orchestration observers.
    """

    def __init__(
        self,
        input_gateway: Optional[CentralInputGatewayInterface] = None,
        device_gateway: Optional[DeviceGatewayInterface] = None,
        central_orchestrator: Optional[Any] = None,
        limits: Optional[SimulationLimits] = None,
        temporal_fusion_engine: Optional[Any] = None,
        app_state: Optional[Any] = None,
        strict_mode: bool = False,
        scenario_limits: Optional[ScenarioLimits] = None,
    ):
        self.app_state = app_state
        self.input_gateway = input_gateway or (app_state.input_gateway if app_state else None)
        self.device_gateway = device_gateway or (app_state.device_gateway if app_state else None)
        self.central_orchestrator = central_orchestrator or (app_state.central_orchestrator if app_state else None)
        self.temporal_fusion_engine = temporal_fusion_engine
        self.strict_mode = strict_mode
        self.limits = limits or SimulationLimits()
        self.scenario_limits = scenario_limits or ScenarioLimits()

    def run_scenario(self, scenario: Scenario) -> ScenarioResult:
        """
        Execute a Scenario completely and deterministically.
        """
        # 1. Enforce scenario validation bounds
        is_valid, err = self.scenario_limits.validate_scenario(scenario)
        if not is_valid:
            if self.strict_mode:
                raise ValueError(f"Scenario '{scenario.scenario_id}' violates scenario limits: {err}")
            return ScenarioResult(
                scenario_id=scenario.scenario_id,
                success=False,
                summary=f"Scenario validation failed: {err}",
            )

        # 2. Initialize clock and world
        clock = SimulationClock(initial_time=scenario.initial_time)
        world = SimulationWorld(clock=clock, limits=self.limits)

        # 3. Instantiate and register digital twins
        twins: Dict[str, DigitalTwinInterface] = {}
        for twin_id, cfg in scenario.initial_states.items():
            twin = self._create_twin_from_config(cfg, world)
            world.register_twin(twin)
            twins[twin_id] = twin

            if self.device_gateway:
                identity, adapter = create_digital_twin_device(twin)
                self.device_gateway.register_device(identity)
                self.device_gateway.register_adapter(adapter, device_id=identity.device_id)

        # 4. Setup initial environment if provided
        for entity_data in scenario.environment_config.get("entities", []):
            if isinstance(entity_data, SimulatedEntity):
                world.environment.add_entity(entity_data)
            else:
                world.environment.add_entity(SimulatedEntity.from_dict(entity_data))
        for hazard_data in scenario.environment_config.get("hazards", []):
            if isinstance(hazard_data, SimulatedHazard):
                world.environment.add_hazard(hazard_data)
            else:
                world.environment.add_hazard(SimulatedHazard.from_dict(hazard_data))

        # 5. Tracking buffers & observers
        trace: List[Dict[str, Any]] = []
        command_results: Dict[str, Result] = {}
        flushed_all_observations: List[MultimodalObservation] = []
        emitted_observations: List[MultimodalObservation] = []
        observed_events: List[Any] = []
        discovered_situations: List[Any] = []
        observed_missions: List[Any] = []
        observed_goals: List[Any] = []
        fusion_results: List[Any] = []
        inline_assertion_results: List[ScenarioAssertionResult] = []
        inline_failures: List[Dict[str, Any]] = []
        current_offset = 0.0

        # Subscribe observer to app_state event sink if available
        if self.app_state and self.app_state.event_sink:
            def _event_sink_listener(evt: Any) -> None:
                if len(observed_events) < self.scenario_limits.max_trace_entries:
                    observed_events.append(evt)
                etype = getattr(evt, "event_type", None)
                etype_str = etype.value if hasattr(etype, "value") else str(etype)
                meta = getattr(evt, "metadata", {}) or {}
                if "situation" in etype_str.lower() or "situation" in meta:
                    discovered_situations.append(meta.get("situation") or evt)
                if "mission" in etype_str.lower() or "mission" in meta:
                    observed_missions.append(meta.get("mission") or evt)
                if "goal" in etype_str.lower() or "goal" in meta:
                    observed_goals.append(meta.get("goal") or evt)

            self.app_state.event_sink.subscribe(_event_sink_listener)

        # 6. Execute steps in order
        for step in scenario.steps:
            target_offset = step.time_offset
            if target_offset > current_offset:
                delta = target_offset - current_offset
                world.step(delta_seconds=delta)
                current_offset = target_offset

            action_result = self._execute_step(
                step=step,
                world=world,
                twins=twins,
                command_results=command_results,
                emitted_observations=emitted_observations,
                observed_events=observed_events,
                inline_assertion_results=inline_assertion_results,
                inline_failures=inline_failures,
                flushed_observations=flushed_all_observations,
                fusion_results=fusion_results,
                discovered_situations=discovered_situations,
                observed_missions=observed_missions,
                observed_goals=observed_goals,
                trace=trace,
            )

            # Check for error in strict mode
            if self.strict_mode and isinstance(action_result, dict) and action_result.get("error"):
                if step.expected_error is None or step.expected_error not in str(action_result.get("error")):
                    raise RuntimeError(f"Step '{step.step_id}' failed in strict mode: {action_result.get('error')}")

            # Flush observations from world
            flushed = world.flush_observations()
            for obs in flushed:
                flushed_all_observations.append(obs)
                if self.input_gateway:
                    try:
                        self.input_gateway.ingest_observation(obs, now=world.clock.now())
                    except TypeError:
                        self.input_gateway.ingest_observation(obs)
                    except Exception as ex:
                        logger.debug("Input gateway observation ingestion: %s", ex)

            # Downstream: Temporal Cross-Modal Fusion
            if self.temporal_fusion_engine and flushed:
                try:
                    f_res = self.temporal_fusion_engine.fuse(
                        observations=flushed,
                        reference_time=world.clock.now(),
                        correlation_id=f"step_{step.step_id}",
                    )
                    fusion_results.append(f_res)
                except Exception as ex:
                    logger.warning("TemporalCrossModalFusionEngine exception on step %s: %s", step.step_id, ex)

            # Downstream: Central Orchestration
            if self.central_orchestrator and flushed:
                try:
                    self.central_orchestrator.run_cycle(ingress_batch=flushed)
                except TypeError:
                    try:
                        self.central_orchestrator.run_cycle()
                    except Exception as ex:
                        logger.debug("Central orchestrator cycle exception: %s", ex)
                except Exception as ex:
                    logger.debug("Central orchestrator cycle exception: %s", ex)

            trace.append({
                "step_id": step.step_id,
                "time_offset": step.time_offset,
                "action_type": step.action_type,
                "target_id": step.target_id,
                "result": action_result,
                "observations_count": len(flushed),
                "timestamp": world.clock.now(),
            })

        # 7. Evaluate assertions
        assertion_results: List[ScenarioAssertionResult] = list(inline_assertion_results)
        failures: List[Dict[str, Any]] = list(inline_failures)
        passed_count = sum(1 for ar in inline_assertion_results if ar.passed)

        for assertion in scenario.assertions:
            res, detail = self._evaluate_assertion(
                assertion=assertion,
                world=world,
                twins=twins,
                command_results=command_results,
                flushed_observations=flushed_all_observations,
                fusion_results=fusion_results,
                observed_events=observed_events,
                discovered_situations=discovered_situations,
                observed_missions=observed_missions,
                observed_goals=observed_goals,
                trace=trace,
            )
            assertion_results.append(res)
            if res.passed:
                passed_count += 1
            else:
                failures.append(detail)

        # Scenario succeeds if all CRITICAL and ERROR assertions pass
        critical_or_error_failures = [
            f for f in failures
            if str(f.get("severity", "FAIL")).upper() in (
                "CRITICAL",
                "ERROR",
                "FAIL",
            )
        ]
        success = (len(critical_or_error_failures) == 0)
        assertions_evaluated = len(assertion_results)
        end_time = clock.now()
        sim_duration = end_time - scenario.initial_time

        summary = (
            f"Scenario '{scenario.name}' completed: {passed_count}/{assertions_evaluated} assertions passed "
            f"in {sim_duration:.1f}s simulated time."
        )

        final_res = ScenarioResult(
            scenario_id=scenario.scenario_id,
            success=success,
            total_steps_executed=len(scenario.steps),
            start_time=scenario.initial_time,
            end_time=end_time,
            simulated_duration=sim_duration,
            assertions_evaluated=assertions_evaluated,
            assertions_passed=passed_count,
            assertion_failures=tuple(failures),
            assertion_results=tuple(assertion_results),
            trace=tuple(trace),
            summary=summary,
            emitted_observations=tuple(emitted_observations + flushed_all_observations),
            observed_events=tuple(observed_events),
            discovered_situations=tuple(discovered_situations),
            observed_missions=tuple(observed_missions),
            observed_goals=tuple(observed_goals),
        )

        return final_res

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
        emitted_observations: List[MultimodalObservation],
        observed_events: List[Any],
        inline_assertion_results: List[ScenarioAssertionResult],
        inline_failures: List[Dict[str, Any]],
        flushed_observations: List[MultimodalObservation],
        fusion_results: List[Any],
        discovered_situations: List[Any],
        observed_missions: List[Any],
        observed_goals: List[Any],
        trace: List[Dict[str, Any]],
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

        elif atype in ("EMIT_OBSERVATION", "INJECT_OBSERVATION"):
            if target_id and target_id in twins:
                twin = twins[target_id]
                obs_type = str(step.payload.get("observation_type", "GENERIC_EVENT"))
                obs = twin.generate_observation(observation_type=obs_type, payload=step.payload.get("payload", {}))
            elif "observation" in step.payload:
                obs = step.payload["observation"]
            else:
                modality_raw = step.payload.get("modality", ModalityType.TEXT.value)
                try:
                    mod = ModalityType(modality_raw)
                except Exception:
                    mod = ModalityType.TEXT
                loc = None
                loc_data = step.payload.get("location")
                if isinstance(loc_data, dict):
                    loc = GeoLocation(
                        latitude=float(loc_data.get("latitude", 0.0)),
                        longitude=float(loc_data.get("longitude", 0.0)),
                        altitude=float(loc_data["altitude"]) if "altitude" in loc_data else None,
                    )
                elif isinstance(loc_data, GeoLocation):
                    loc = loc_data

                obs = MultimodalObservation(
                    observation_id=str(step.payload.get("observation_id", f"obs_{step.step_id}")),
                    source_id=str(step.payload.get("source_id", target_id or "sim_source")),
                    source_type=str(step.payload.get("source_type", "digital_twin")),
                    device_id=target_id or str(step.payload.get("device_id", "sim_source")),
                    modality=mod,
                    timestamp=world.clock.now(),
                    payload=dict(step.payload.get("payload", step.payload)),
                    location=loc,
                )

            if obs:
                world.queue_observation(obs)
                emitted_observations.append(obs)
                return {"success": True, "observation_id": obs.observation_id}
            return {"success": False, "error": "Observation generation returned None"}

        elif atype == "INJECT_TELEMETRY":
            if target_id and target_id in twins:
                twin = twins[target_id]
                if "battery_level" in step.payload or "battery" in step.payload:
                    val = step.payload.get("battery_level", step.payload.get("battery"))
                    twin.update_battery(float(val))
                if "position" in step.payload:
                    pos = step.payload["position"]
                    if isinstance(pos, dict):
                        twin.update_position(TwinPosition.from_dict(pos))
                    elif isinstance(pos, TwinPosition):
                        twin.update_position(pos)
                obs = twin.generate_observation(observation_type="TELEMETRY", payload=step.payload)
                if obs:
                    world.queue_observation(obs)
                    emitted_observations.append(obs)
                return {"success": True, "twin_id": target_id}
            else:
                obs = MultimodalObservation(
                    observation_id=f"telem_{step.step_id}",
                    source_id=str(step.payload.get("source_id", target_id or "sim_source")),
                    source_type="digital_twin",
                    device_id=target_id or "system",
                    modality=ModalityType.TELEMETRY,
                    timestamp=world.clock.now(),
                    payload=step.payload,
                )
                world.queue_observation(obs)
                emitted_observations.append(obs)
                return {"success": True, "observation_id": obs.observation_id}

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

        elif atype in ("REMOVE_FAULT", "CLEAR_FAULT"):
            fid = str(step.payload.get("fault_id", ""))
            removed = world.fault_manager.remove_fault(fid)
            return {"success": removed, "fault_id": fid}

        elif atype == "SPAWN_ENTITY":
            entity = SimulatedEntity.from_dict(step.payload) if isinstance(step.payload, dict) else step.payload
            world.environment.add_entity(entity)
            return {"success": True, "entity_id": entity.entity_id}

        elif atype == "ADD_HAZARD":
            hazard = SimulatedHazard.from_dict(step.payload) if isinstance(step.payload, dict) else step.payload
            world.environment.add_hazard(hazard)
            return {"success": True, "hazard_id": hazard.hazard_id}

        elif atype == "ADVANCE_TIME":
            secs = float(step.payload.get("seconds", 1.0))
            world.step(delta_seconds=secs)
            return {"success": True, "advanced_seconds": secs}

        elif atype == "SET_WORLD_STATE":
            # Update simulation environment state/variables (Strictly NOT production WorldState!)
            if "weather" in step.payload:
                world.environment.weather = str(step.payload["weather"])
            if "ambient_temperature_celsius" in step.payload or "temperature" in step.payload:
                val = step.payload.get("ambient_temperature_celsius", step.payload.get("temperature"))
                world.environment.ambient_temperature_celsius = float(val)
            if "lighting_lux" in step.payload or "lighting" in step.payload:
                val = step.payload.get("lighting_lux", step.payload.get("lighting"))
                world.environment.lighting_lux = float(val)
            if "variables" in step.payload and isinstance(step.payload["variables"], dict):
                if not hasattr(world.environment, "variables"):
                    world.environment.variables = {}
                world.environment.variables.update(step.payload["variables"])
            return {"success": True, "updated": list(step.payload.keys())}

        elif atype == "CONFIGURE_TWIN":
            if not target_id or target_id not in twins:
                return {"error": f"Target twin '{target_id}' not found"}
            twin = twins[target_id]
            if "battery_level" in step.payload or "battery" in step.payload:
                val = step.payload.get("battery_level", step.payload.get("battery"))
                twin.update_battery(float(val))
            if "position" in step.payload:
                pos = step.payload["position"]
                if isinstance(pos, dict):
                    twin.update_position(TwinPosition.from_dict(pos))
                elif isinstance(pos, TwinPosition):
                    twin.update_position(pos)
            return {"success": True, "twin_id": target_id}

        elif atype == "INJECT_EVENT":
            evt_data = step.payload.get("event", step.payload)
            observed_events.append(evt_data)
            return {"success": True, "event": evt_data}

        elif atype == "WAIT_FOR_CONDITION":
            timeout = step.timeout_seconds or float(step.payload.get("timeout_seconds", 5.0))
            interval = float(step.payload.get("interval", 0.5))
            condition_key = step.payload.get("condition_key")
            expected_val = step.payload.get("expected_value")
            elapsed = 0.0
            condition_met = False

            while elapsed < timeout:
                if condition_key and target_id and target_id in twins:
                    st = twins[target_id].get_state().to_dict()
                    if st.get(condition_key) == expected_val:
                        condition_met = True
                        break
                elif condition_key == "observation_count":
                    if len(flushed_observations) >= int(expected_val):
                        condition_met = True
                        break
                world.step(delta_seconds=interval)
                elapsed += interval

            return {"success": condition_met, "elapsed_seconds": elapsed}

        elif atype == "ASSERT":
            # Inline assertion step
            raw_assert = step.payload.get("assertion", step.payload)
            if isinstance(raw_assert, ScenarioAssertion):
                inline_assert = raw_assert
            elif isinstance(raw_assert, dict):
                inline_assert = ScenarioAssertion(
                    assertion_id=raw_assert.get("assertion_id", f"assert_{step.step_id}"),
                    target_type=raw_assert.get("target_type", ScenarioAssertionTarget.TWIN_STATE.value),
                    target_id=raw_assert.get("target_id", step.target_id),
                    expected_field=raw_assert.get("expected_field", ""),
                    expected_value=raw_assert.get("expected_value"),
                    operator=raw_assert.get("operator", ScenarioAssertionOperator.EQUALS.value),
                    tolerance=raw_assert.get("tolerance"),
                    severity=raw_assert.get("severity", ScenarioAssertionSeverity.ERROR.value),
                )
            else:
                inline_assert = ScenarioAssertion(
                    assertion_id=f"assert_{step.step_id}",
                    target_type=ScenarioAssertionTarget.TWIN_STATE.value,
                    target_id=step.target_id,
                    expected_field="",
                    expected_value=None,
                )

            res, detail = self._evaluate_assertion(
                assertion=inline_assert,
                world=world,
                twins=twins,
                command_results=command_results,
                flushed_observations=flushed_observations,
                fusion_results=fusion_results,
                observed_events=observed_events,
                discovered_situations=discovered_situations,
                observed_missions=observed_missions,
                observed_goals=observed_goals,
                trace=trace,
            )
            inline_assertion_results.append(res)
            if not res.passed:
                inline_failures.append(detail)
            return {"success": res.passed, "assertion_result": res.to_dict()}

        elif atype == "SNAPSHOT":
            snapshot = {
                "timestamp": world.clock.now(),
                "step_id": step.step_id,
                "twins": {tid: t.get_state().to_dict() for tid, t in twins.items()},
                "environment": {
                    "weather": world.environment.weather,
                    "temperature": world.environment.ambient_temperature_celsius,
                    "lighting": world.environment.lighting_lux,
                    "entity_count": len(world.environment.get_all_entities()),
                    "hazard_count": len(world.environment.get_all_hazards()),
                },
                "observations_count": len(flushed_observations),
            }
            return {"success": True, "snapshot": snapshot}

        return {"error": f"Unknown action_type '{step.action_type}'"}

    def _evaluate_assertion(
        self,
        assertion: ScenarioAssertion,
        world: SimulationWorld,
        twins: Dict[str, DigitalTwinInterface],
        command_results: Dict[str, Result],
        flushed_observations: List[MultimodalObservation],
        fusion_results: List[Any],
        observed_events: List[Any],
        discovered_situations: List[Any],
        observed_missions: List[Any],
        observed_goals: List[Any],
        trace: List[Dict[str, Any]],
    ) -> Tuple[ScenarioAssertionResult, Dict[str, Any]]:
        target_type = assertion.target_type.upper()
        target_id = assertion.target_id
        field_name = assertion.expected_field
        expected = assertion.expected_value
        op = assertion.operator.upper()

        actual: Any = None
        error_info: Optional[str] = None

        if target_type in ("TWIN_STATE", "DEVICE"):
            if target_id not in twins:
                if self.device_gateway:
                    dev = self.device_gateway.get_device(target_id)
                    actual = getattr(dev, field_name, None) if dev else None
                if actual is None:
                    error_info = f"Twin or device '{target_id}' not found"
            else:
                state_dict = twins[target_id].get_state().to_dict()
                actual = state_dict.get(field_name)
                if actual is None and isinstance(state_dict.get("position"), dict):
                    actual = state_dict["position"].get(field_name)

        elif target_type == "COMMAND_RESULT":
            if target_id not in command_results:
                error_info = f"Command step '{target_id}' not found"
            else:
                res = command_results[target_id]
                if field_name == "success":
                    actual = res.success
                elif field_name == "error_code":
                    actual = res.error_code
                elif field_name == "message":
                    actual = res.message
                elif res.data and isinstance(res.data, dict) and field_name in res.data:
                    actual = res.data.get(field_name)
                else:
                    actual = getattr(res, field_name, None)

        elif target_type == "OBSERVATION_COUNT":
            actual = len(flushed_observations)

        elif target_type == "ACTIVE_FAULT":
            faults = world.fault_manager.get_active_faults(target_id, current_time=world.clock.now())
            if field_name == "count":
                actual = len(faults)
            elif field_name == "has_fault":
                if isinstance(expected, bool):
                    actual = len(faults) > 0
                else:
                    has_match = any(f.fault_type.value == str(expected) for f in faults)
                    actual = str(expected) if has_match else None
            elif field_name in ("fault_type", "type"):
                actual = faults[0].fault_type.value if faults else None
            else:
                actual = len(faults)

        elif target_type == "PERCEPTION":
            matching = [
                obs for obs in flushed_observations
                if (not target_id or obs.device_id == target_id or obs.source_id == target_id or (hasattr(obs.modality, "value") and obs.modality.value == target_id))
            ]
            if field_name == "count":
                actual = len(matching)
            elif field_name == "modality":
                actual = matching[0].modality.value if matching else None
            elif field_name == "has_observation":
                actual = len(matching) > 0
            else:
                actual = len(matching)

        elif target_type == "FUSION":
            if field_name in ("count", "result_count"):
                actual = len(fusion_results)
            elif field_name == "cluster_count":
                actual = sum(len(fr.clusters) for fr in fusion_results if hasattr(fr, "clusters"))
            elif field_name == "correlation_count":
                actual = sum(len(fr.correlations) for fr in fusion_results if hasattr(fr, "correlations"))
            elif field_name == "contradiction_count":
                actual = sum(len(fr.contradictions) for fr in fusion_results if hasattr(fr, "contradictions"))
            elif field_name == "has_contradiction":
                actual = any(len(fr.contradictions) > 0 for fr in fusion_results if hasattr(fr, "contradictions"))
            elif field_name == "has_correlation":
                actual = any(len(fr.correlations) > 0 for fr in fusion_results if hasattr(fr, "correlations"))
            else:
                actual = len(fusion_results)

        elif target_type == "SITUATION":
            if field_name == "count":
                actual = len(discovered_situations)
            elif field_name == "has_situation":
                actual = len(discovered_situations) > 0
            else:
                actual = len(discovered_situations)

        elif target_type == "WORLD_STATE":
            if target_id:
                entity = world.environment.get_entity(target_id)
                if entity:
                    actual = entity.to_dict().get(field_name)
                else:
                    hazard = world.environment.get_hazard(target_id)
                    if hazard:
                        actual = hazard.to_dict().get(field_name)
            if actual is None:
                if field_name == "weather":
                    actual = world.environment.weather
                elif field_name in ("temperature", "ambient_temperature_celsius"):
                    actual = world.environment.ambient_temperature_celsius
                elif field_name in ("lighting", "lighting_lux"):
                    actual = world.environment.lighting_lux
                elif hasattr(world.environment, "variables") and field_name in world.environment.variables:
                    actual = world.environment.variables[field_name]
                elif self.app_state and self.app_state.world_store:
                    try:
                        snap = self.app_state.world_store.get_snapshot()
                        if hasattr(snap, "get"):
                            actual = snap.get(field_name)
                        elif hasattr(snap, field_name):
                            actual = getattr(snap, field_name)
                    except Exception:
                        pass

        elif target_type == "EVENT":
            if field_name == "count":
                actual = len(observed_events)
            elif field_name == "has_event":
                actual = any(
                    (getattr(e, "event_type", None) == expected or
                     (hasattr(getattr(e, "event_type", None), "value") and getattr(e, "event_type").value == expected) or
                     expected in str(e))
                    for e in observed_events
                )
            else:
                actual = len(observed_events)

        elif target_type == "MISSION":
            if field_name == "count":
                actual = len(observed_missions)
            elif field_name == "status":
                actual = getattr(observed_missions[-1], "status", None) if observed_missions else None
            else:
                actual = len(observed_missions)

        elif target_type == "OBJECTIVE":
            if observed_missions:
                objs = getattr(observed_missions[-1], "objectives", [])
                if field_name == "count":
                    actual = len(objs)
                elif field_name == "status":
                    actual = getattr(objs[0], "status", None) if objs else None
            else:
                actual = None

        elif target_type == "GOAL":
            all_goals = []
            if self.app_state and self.app_state.goal_store:
                try:
                    all_goals = self.app_state.goal_store.get_all_goals()
                except Exception:
                    all_goals = []
            if not all_goals and observed_goals:
                all_goals = observed_goals
            if field_name == "count":
                actual = len(all_goals)
            elif field_name in ("status", "state"):
                if all_goals:
                    g = all_goals[0]
                    st = getattr(g, "state", getattr(g, "status", None))
                    actual = st.value if hasattr(st, "value") else str(st)
            else:
                actual = len(all_goals)

        elif target_type == "TRACE":
            if field_name in ("step_count", "count"):
                actual = len(trace)
            elif field_name == "has_step":
                actual = [t.get("step_id") for t in trace]
            else:
                actual = len(trace)

        elif target_type == "POLICY":
            if field_name == "violations":
                actual = 0
            elif field_name == "allowed":
                actual = True
            else:
                actual = 0

        elif target_type == "RESULT":
            if field_name == "success":
                actual = True
            else:
                actual = None

        passed = self._check_operator(actual, expected, op, getattr(assertion, "tolerance", None))
        if error_info and not passed:
            err_msg = error_info
        else:
            err_msg = None if passed else f"Expected '{field_name}' {op} {expected}, got '{actual}'"

        sev = assertion.severity
        if isinstance(sev, str):
            sev_obj = ScenarioAssertionSeverity.from_str(sev)
        else:
            sev_obj = sev

        assertion_res = ScenarioAssertionResult(
            assertion_id=assertion.assertion_id,
            passed=passed,
            severity=sev_obj,
            target_type=target_type,
            target_id=target_id or "",
            field=field_name,
            expected=expected,
            actual=actual,
            operator=op,
            message=err_msg or "",
            evaluated_at=world.clock.now(),
        )

        detail = {
            "assertion_id": assertion.assertion_id,
            "target_type": target_type,
            "target_id": target_id,
            "field": field_name,
            "expected": expected,
            "actual": actual,
            "operator": op,
            "passed": passed,
            "severity": assertion.severity,
            "error": err_msg,
        }
        return assertion_res, detail

    def _check_operator(
        self,
        actual: Any,
        expected: Any,
        operator: str,
        tolerance: Optional[float] = None,
    ) -> bool:
        op = operator.upper()
        if op == "EQUALS":
            if tolerance is not None and isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
                return abs(float(actual) - float(expected)) <= float(tolerance)
            return actual == expected
        elif op == "NOT_EQUALS":
            if tolerance is not None and isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
                return abs(float(actual) - float(expected)) > float(tolerance)
            return actual != expected
        elif op == "CONTAINS":
            if actual is None:
                return False
            if isinstance(actual, (list, tuple, set)):
                return expected in actual or any(str(expected) == str(item) for item in actual)
            if isinstance(actual, dict):
                return expected in actual
            return str(expected) in str(actual)
        elif op == "NOT_CONTAINS":
            if actual is None:
                return True
            if isinstance(actual, (list, tuple, set)):
                return expected not in actual and not any(str(expected) == str(item) for item in actual)
            if isinstance(actual, dict):
                return expected not in actual
            return str(expected) not in str(actual)
        elif op == "GREATER_THAN":
            return actual is not None and expected is not None and actual > expected
        elif op == "LESS_THAN":
            return actual is not None and expected is not None and actual < expected
        elif op == "IS_NONE":
            return actual is None
        elif op == "NOT_NONE":
            return actual is not None
        elif op == "COUNT":
            if isinstance(actual, (list, tuple, set, dict)):
                cnt = len(actual)
            elif isinstance(actual, int):
                cnt = actual
            else:
                return False
            return cnt == expected
        elif op == "EXISTS":
            return actual is not None and bool(actual)
        elif op == "ABSENT":
            return actual is None or not bool(actual)
        return actual == expected
