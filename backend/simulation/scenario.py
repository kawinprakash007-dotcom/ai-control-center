"""
ATLAS Phase 6.3 — Scenario Definition & Result Models.

Provides domain models for defining, building, and asserting outcomes of
deterministic multi-agent edge product scenarios.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core.models.simulation import (
    SimulationLimits,
    TwinConfiguration,
    TwinPosition,
)


@dataclass(frozen=True)
class ScenarioStep:
    """
    Single discrete action or simulation event occurring at a specified time offset.
    """
    step_id: str
    time_offset: float  # Seconds relative to scenario initial_time
    action_type: str    # "DISPATCH_COMMAND", "EMIT_OBSERVATION", "INJECT_FAULT", "REMOVE_FAULT", "ADVANCE_TIME", "SPAWN_ENTITY", "ADD_HAZARD"
    target_id: Optional[str] = None  # e.g. twin_id or entity_id
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "time_offset": self.time_offset,
            "action_type": self.action_type,
            "target_id": self.target_id,
            "payload": dict(self.payload),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScenarioStep":
        return cls(
            step_id=str(data["step_id"]),
            time_offset=float(data.get("time_offset", 0.0)),
            action_type=str(data["action_type"]),
            target_id=str(data["target_id"]) if data.get("target_id") else None,
            payload=dict(data.get("payload", {})),
        )


@dataclass(frozen=True)
class ScenarioAssertion:
    """
    Post-condition assertion evaluated at the conclusion of scenario execution.
    """
    assertion_id: str
    target_type: str  # "TWIN_STATE", "COMMAND_RESULT", "OBSERVATION_COUNT", "ACTIVE_FAULT"
    target_id: str
    expected_field: str
    expected_value: Any
    operator: str = "EQUALS"  # "EQUALS", "CONTAINS", "GREATER_THAN", "LESS_THAN", "IS_NONE", "NOT_NONE"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "assertion_id": self.assertion_id,
            "target_type": self.target_type,
            "target_id": self.target_id,
            "expected_field": self.expected_field,
            "expected_value": self.expected_value,
            "operator": self.operator,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScenarioAssertion":
        return cls(
            assertion_id=str(data["assertion_id"]),
            target_type=str(data["target_type"]),
            target_id=str(data["target_id"]),
            expected_field=str(data["expected_field"]),
            expected_value=data["expected_value"],
            operator=str(data.get("operator", "EQUALS")),
        )


@dataclass(frozen=True)
class Scenario:
    """
    Immutable specification of a complete simulation scenario.
    """
    scenario_id: str
    name: str
    description: str
    initial_time: float = 1000000.0
    initial_states: Dict[str, TwinConfiguration] = field(default_factory=dict)
    environment_config: Dict[str, Any] = field(default_factory=dict)
    steps: Tuple[ScenarioStep, ...] = ()
    assertions: Tuple[ScenarioAssertion, ...] = ()
    termination_condition: Optional[str] = None
    max_duration_seconds: float = 3600.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "name": self.name,
            "description": self.description,
            "initial_time": self.initial_time,
            "initial_states": {k: v.to_dict() for k, v in self.initial_states.items()},
            "environment_config": dict(self.environment_config),
            "steps": [s.to_dict() for s in self.steps],
            "assertions": [a.to_dict() for a in self.assertions],
            "termination_condition": self.termination_condition,
            "max_duration_seconds": self.max_duration_seconds,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Scenario":
        init_states = {
            k: TwinConfiguration.from_dict(v) for k, v in data.get("initial_states", {}).items()
        }
        steps = tuple(ScenarioStep.from_dict(s) for s in data.get("steps", []))
        assertions = tuple(ScenarioAssertion.from_dict(a) for a in data.get("assertions", []))
        return cls(
            scenario_id=str(data["scenario_id"]),
            name=str(data.get("name", "")),
            description=str(data.get("description", "")),
            initial_time=float(data.get("initial_time", 1000000.0)),
            initial_states=init_states,
            environment_config=dict(data.get("environment_config", {})),
            steps=steps,
            assertions=assertions,
            termination_condition=data.get("termination_condition"),
            max_duration_seconds=float(data.get("max_duration_seconds", 3600.0)),
        )


@dataclass(frozen=True)
class ScenarioResult:
    """
    Immutable summary of scenario execution with assertion outcomes and execution trace.
    """
    scenario_id: str
    success: bool
    total_steps_executed: int
    start_time: float
    end_time: float
    simulated_duration: float
    assertions_evaluated: int
    assertions_passed: int
    assertion_failures: Tuple[Dict[str, Any], ...] = ()
    trace: Tuple[Dict[str, Any], ...] = ()
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "success": self.success,
            "total_steps_executed": self.total_steps_executed,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "simulated_duration": self.simulated_duration,
            "assertions_evaluated": self.assertions_evaluated,
            "assertions_passed": self.assertions_passed,
            "assertion_failures": list(self.assertion_failures),
            "trace_depth": len(self.trace),
            "summary": self.summary,
        }


class ScenarioBuilder:
    """
    Fluent builder for constructing deterministic scenarios.
    """

    def __init__(self, scenario_id: str, name: str, description: str = ""):
        self.scenario_id = scenario_id
        self.name = name
        self.description = description
        self.initial_time: float = 1000000.0
        self.initial_states: Dict[str, TwinConfiguration] = {}
        self.environment_config: Dict[str, Any] = {}
        self.steps: List[ScenarioStep] = []
        self.assertions: List[ScenarioAssertion] = []
        self.max_duration_seconds: float = 3600.0

    def with_initial_time(self, t: float) -> "ScenarioBuilder":
        self.initial_time = t
        return self

    def add_twin_config(self, config: TwinConfiguration) -> "ScenarioBuilder":
        self.initial_states[config.twin_id] = config
        return self

    def add_step(
        self,
        step_id: str,
        time_offset: float,
        action_type: str,
        target_id: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> "ScenarioBuilder":
        self.steps.append(
            ScenarioStep(
                step_id=step_id,
                time_offset=time_offset,
                action_type=action_type,
                target_id=target_id,
                payload=payload or {},
            )
        )
        return self

    def add_assertion(
        self,
        assertion_id: str,
        target_type: str,
        target_id: str,
        expected_field: str,
        expected_value: Any,
        operator: str = "EQUALS",
    ) -> "ScenarioBuilder":
        self.assertions.append(
            ScenarioAssertion(
                assertion_id=assertion_id,
                target_type=target_type,
                target_id=target_id,
                expected_field=expected_field,
                expected_value=expected_value,
                operator=operator,
            )
        )
        return self

    def build(self) -> Scenario:
        # Sort steps deterministically by time_offset
        sorted_steps = sorted(self.steps, key=lambda s: (s.time_offset, s.step_id))
        return Scenario(
            scenario_id=self.scenario_id,
            name=self.name,
            description=self.description,
            initial_time=self.initial_time,
            initial_states=dict(self.initial_states),
            environment_config=dict(self.environment_config),
            steps=tuple(sorted_steps),
            assertions=tuple(self.assertions),
            max_duration_seconds=self.max_duration_seconds,
        )
