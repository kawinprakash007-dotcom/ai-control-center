"""
ATLAS Phase 6.5e — Scenario Domain Models & Limits.

Provides transport-neutral, immutable domain models for:
- Discrete scenario steps and semantic actions
- Multi-target observational scenario assertions with severity levels
- Execution results, traces, and deterministic SHA-256 hashing
- Bounded operational limits preventing runaway memory or infinite loops

CRITICAL ARCHITECTURAL RULES:
1. Scenario infrastructure is an OBSERVER/CONTROLLER of the simulation environment.
2. NEVER directly mutates production WorldState, GoalStore, or application state.
3. NEVER bypasses PolicyEngine, CognitiveRuntime, or DeviceGateway.
4. ZERO physical hardware dependencies or device driver imports.
5. ZERO external execution calls, shells, or dynamic evaluation.
6. Purely transport-neutral, deterministic, and replay-compatible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import math
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

from core.models.device_contract import sanitize_contract_metadata
from core.models.orchestration import MultimodalObservation
from core.models.simulation import (
    SimulationLimits,
    TwinConfiguration,
    TwinPosition,
)


# ============================================================================
# Enums
# ============================================================================

class ScenarioStepType(str, Enum):
    """Semantic action types supported by scenario steps."""
    ADVANCE_TIME = "ADVANCE_TIME"
    SET_WORLD_STATE = "SET_WORLD_STATE"
    CONFIGURE_TWIN = "CONFIGURE_TWIN"
    INJECT_OBSERVATION = "INJECT_OBSERVATION"
    INJECT_TELEMETRY = "INJECT_TELEMETRY"
    INJECT_EVENT = "INJECT_EVENT"
    INJECT_FAULT = "INJECT_FAULT"
    CLEAR_FAULT = "CLEAR_FAULT"
    WAIT_FOR_CONDITION = "WAIT_FOR_CONDITION"
    ASSERT = "ASSERT"
    SNAPSHOT = "SNAPSHOT"
    DISPATCH_COMMAND = "DISPATCH_COMMAND"
    EMIT_OBSERVATION = "EMIT_OBSERVATION"
    SPAWN_ENTITY = "SPAWN_ENTITY"
    ADD_HAZARD = "ADD_HAZARD"

    @classmethod
    def from_str(cls, val: Any) -> "ScenarioStepType":
        if isinstance(val, cls):
            return val
        norm = str(val or "").strip().upper()
        for member in cls:
            if member.value == norm or member.name == norm:
                return member
        raise ValueError(f"Unknown ScenarioStepType: '{val}'")


class ScenarioAssertionTarget(str, Enum):
    """Observational targets evaluated by scenario assertions."""
    PERCEPTION = "PERCEPTION"
    FUSION = "FUSION"
    SITUATION = "SITUATION"
    WORLD_STATE = "WORLD_STATE"
    EVENT = "EVENT"
    MISSION = "MISSION"
    OBJECTIVE = "OBJECTIVE"
    GOAL = "GOAL"
    DEVICE = "DEVICE"
    TRACE = "TRACE"
    POLICY = "POLICY"
    RESULT = "RESULT"
    TWIN_STATE = "TWIN_STATE"
    COMMAND_RESULT = "COMMAND_RESULT"
    OBSERVATION_COUNT = "OBSERVATION_COUNT"
    ACTIVE_FAULT = "ACTIVE_FAULT"

    @classmethod
    def from_str(cls, val: Any) -> "ScenarioAssertionTarget":
        if isinstance(val, cls):
            return val
        norm = str(val or "").strip().upper()
        for member in cls:
            if member.value == norm or member.name == norm:
                return member
        return cls.TWIN_STATE


class ScenarioAssertionOperator(str, Enum):
    """Comparison operators for scenario assertions."""
    EQUALS = "EQUALS"
    NOT_EQUALS = "NOT_EQUALS"
    CONTAINS = "CONTAINS"
    NOT_CONTAINS = "NOT_CONTAINS"
    GREATER_THAN = "GREATER_THAN"
    LESS_THAN = "LESS_THAN"
    IS_NONE = "IS_NONE"
    NOT_NONE = "NOT_NONE"
    COUNT = "COUNT"
    EXISTS = "EXISTS"
    ABSENT = "ABSENT"

    @classmethod
    def from_str(cls, val: Any) -> "ScenarioAssertionOperator":
        if isinstance(val, cls):
            return val
        norm = str(val or "").strip().upper()
        for member in cls:
            if member.value == norm or member.name == norm:
                return member
        return cls.EQUALS


class ScenarioAssertionSeverity(str, Enum):
    """Severity classification of assertion failures."""
    INFO = "INFO"
    WARNING = "WARNING"
    FAIL = "FAIL"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

    @classmethod
    def from_str(cls, val: Any) -> "ScenarioAssertionSeverity":
        if isinstance(val, cls):
            return val
        norm = str(val or "").strip().upper()
        for member in cls:
            if member.value == norm or member.name == norm:
                return member
        return cls.FAIL


# ============================================================================
# Limits and Capacity Bounds
# ============================================================================

@dataclass(frozen=True)
class ScenarioLimits:
    """Explicit bounds preventing runaway resource consumption during scenarios."""
    max_scenarios: int = 100
    max_steps: int = 500
    max_scenario_duration: float = 86400.0
    max_assertions: int = 200
    max_injected_observations: int = 500
    max_faults: int = 100
    max_output_records: int = 1000
    max_timeline_entries: int = 1000
    max_devices: int = 50
    max_entities: int = 100
    max_string_length: int = 128

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_scenarios": self.max_scenarios,
            "max_steps": self.max_steps,
            "max_scenario_duration": self.max_scenario_duration,
            "max_assertions": self.max_assertions,
            "max_injected_observations": self.max_injected_observations,
            "max_faults": self.max_faults,
            "max_output_records": self.max_output_records,
            "max_timeline_entries": self.max_timeline_entries,
            "max_devices": self.max_devices,
            "max_entities": self.max_entities,
            "max_string_length": self.max_string_length,
        }

    def validate_scenario(self, scenario: Any) -> Tuple[bool, Optional[str]]:
        """Validate whether a Scenario satisfies these bounds."""
        if hasattr(scenario, "steps") and len(scenario.steps) > self.max_steps:
            return False, f"Steps count {len(scenario.steps)} exceeds max_steps {self.max_steps}"
        if hasattr(scenario, "assertions") and len(scenario.assertions) > self.max_assertions:
            return False, f"Assertions count {len(scenario.assertions)} exceeds max_assertions {self.max_assertions}"
        if hasattr(scenario, "initial_states") and len(scenario.initial_states) > self.max_devices:
            return False, f"Devices count {len(scenario.initial_states)} exceeds max_devices {self.max_devices}"
        if hasattr(scenario, "max_duration_seconds") and scenario.max_duration_seconds > self.max_scenario_duration:
            return False, f"Duration {scenario.max_duration_seconds} exceeds max_scenario_duration {self.max_scenario_duration}"
        return True, None

    def model_dump(self) -> Dict[str, Any]:
        return self.to_dict()

    def model_dump_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScenarioLimits":
        return cls(
            max_scenarios=int(data.get("max_scenarios", 100)),
            max_steps=int(data.get("max_steps", 500)),
            max_scenario_duration=float(data.get("max_scenario_duration", 86400.0)),
            max_assertions=int(data.get("max_assertions", 200)),
            max_injected_observations=int(data.get("max_injected_observations", 500)),
            max_faults=int(data.get("max_faults", 100)),
            max_output_records=int(data.get("max_output_records", 1000)),
            max_timeline_entries=int(data.get("max_timeline_entries", 1000)),
            max_devices=int(data.get("max_devices", 50)),
            max_entities=int(data.get("max_entities", 100)),
            max_string_length=int(data.get("max_string_length", 128)),
        )

    @classmethod
    def model_validate(cls, obj: Any) -> "ScenarioLimits":
        if isinstance(obj, cls):
            return obj
        if isinstance(obj, dict):
            return cls.from_dict(obj)
        raise TypeError(f"Cannot validate {type(obj)} as ScenarioLimits")


# ============================================================================
# Scenario Step
# ============================================================================

@dataclass(frozen=True)
class ScenarioStep:
    """
    Single discrete action or simulation event occurring at a specified time offset.
    """
    step_id: str
    time_offset: float  # Seconds relative to scenario initial_time
    action_type: str    # ScenarioStepType or legacy string
    target_id: Optional[str] = None  # e.g. twin_id or entity_id
    payload: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.step_id or not isinstance(self.step_id, str) or not self.step_id.strip():
            raise ValueError("ScenarioStep step_id must be a non-empty string.")
        t = float(self.time_offset)
        if math.isnan(t) or math.isinf(t) or t < 0.0:
            raise ValueError(f"ScenarioStep time_offset must be a finite non-negative number, got {t}")
        object.__setattr__(self, "time_offset", round(t, 4))
        object.__setattr__(self, "payload", sanitize_contract_metadata(dict(self.payload)))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "time_offset": self.time_offset,
            "action_type": str(self.action_type),
            "target_id": self.target_id,
            "payload": dict(self.payload),
        }

    def model_dump(self) -> Dict[str, Any]:
        return self.to_dict()

    def model_dump_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScenarioStep":
        return cls(
            step_id=str(data["step_id"]),
            time_offset=float(data.get("time_offset", 0.0)),
            action_type=str(data["action_type"]),
            target_id=str(data["target_id"]) if data.get("target_id") else None,
            payload=dict(data.get("payload", {})),
        )

    @classmethod
    def model_validate(cls, obj: Any) -> "ScenarioStep":
        if isinstance(obj, cls):
            return obj
        if isinstance(obj, dict):
            return cls.from_dict(obj)
        raise TypeError(f"Cannot validate {type(obj)} as ScenarioStep")


# ============================================================================
# Scenario Assertion & Assertion Result
# ============================================================================

@dataclass(frozen=True)
class ScenarioAssertion:
    """
    Observational assertion evaluated during or at the conclusion of scenario execution.
    Inspects real outputs; NEVER writes to systems or mutates state.
    """
    assertion_id: str
    target_type: str  # ScenarioAssertionTarget or string
    target_id: str
    expected_field: str
    expected_value: Any
    operator: str = "EQUALS"  # ScenarioAssertionOperator or string
    severity: str = "FAIL"    # ScenarioAssertionSeverity ("INFO", "WARNING", "FAIL")
    tolerance: Optional[float] = None

    def __post_init__(self):
        if not self.assertion_id or not isinstance(self.assertion_id, str):
            raise ValueError("ScenarioAssertion assertion_id must be a non-empty string.")
        sev = self.severity.upper() if isinstance(self.severity, str) else self.severity.value
        if sev not in ("INFO", "WARNING", "FAIL", "ERROR", "CRITICAL"):
            sev = "FAIL"
        object.__setattr__(self, "severity", sev)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "assertion_id": self.assertion_id,
            "target_type": str(self.target_type),
            "target_id": self.target_id,
            "expected_field": self.expected_field,
            "expected_value": self.expected_value,
            "operator": str(self.operator),
            "severity": str(self.severity),
            "tolerance": self.tolerance,
        }

    def model_dump(self) -> Dict[str, Any]:
        return self.to_dict()

    def model_dump_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScenarioAssertion":
        return cls(
            assertion_id=str(data["assertion_id"]),
            target_type=str(data["target_type"]),
            target_id=str(data["target_id"]),
            expected_field=str(data["expected_field"]),
            expected_value=data["expected_value"],
            operator=str(data.get("operator", "EQUALS")),
            severity=str(data.get("severity", "FAIL")),
            tolerance=float(data["tolerance"]) if data.get("tolerance") is not None else None,
        )

    @classmethod
    def model_validate(cls, obj: Any) -> "ScenarioAssertion":
        if isinstance(obj, cls):
            return obj
        if isinstance(obj, dict):
            return cls.from_dict(obj)
        raise TypeError(f"Cannot validate {type(obj)} as ScenarioAssertion")


@dataclass(frozen=True)
class ScenarioAssertionResult:
    """
    Detailed evaluation outcome of a single ScenarioAssertion.
    """
    assertion_id: str
    passed: bool
    severity: ScenarioAssertionSeverity
    target_type: str
    target_id: str
    field: str
    expected: Any
    actual: Any
    operator: str
    message: str = ""
    evaluated_at: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "assertion_id": self.assertion_id,
            "passed": self.passed,
            "severity": self.severity.value if hasattr(self.severity, "value") else str(self.severity),
            "target_type": self.target_type,
            "target_id": self.target_id,
            "field": self.field,
            "expected": self.expected,
            "actual": self.actual,
            "operator": self.operator,
            "message": self.message,
            "evaluated_at": self.evaluated_at,
        }

    def model_dump(self) -> Dict[str, Any]:
        return self.to_dict()

    def model_dump_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScenarioAssertionResult":
        return cls(
            assertion_id=str(data["assertion_id"]),
            passed=bool(data["passed"]),
            severity=ScenarioAssertionSeverity.from_str(data.get("severity", "FAIL")),
            target_type=str(data["target_type"]),
            target_id=str(data["target_id"]),
            field=str(data["field"]),
            expected=data["expected"],
            actual=data.get("actual"),
            operator=str(data.get("operator", "EQUALS")),
            message=str(data.get("message", "")),
            evaluated_at=float(data.get("evaluated_at", 0.0)),
        )


# ============================================================================
# Scenario Definition
# ============================================================================

@dataclass(frozen=True)
class Scenario:
    """
    Immutable specification of a complete simulation scenario.
    Exercises the ATLAS ecosystem using Digital Twins and synthetic multimodal inputs.
    """
    scenario_id: str
    name: str
    description: str = ""
    initial_time: float = 1000000.0
    initial_states: Dict[str, TwinConfiguration] = field(default_factory=dict)
    environment_config: Dict[str, Any] = field(default_factory=dict)
    initial_devices: Dict[str, Any] = field(default_factory=dict)
    steps: Tuple[ScenarioStep, ...] = ()
    assertions: Tuple[ScenarioAssertion, ...] = ()
    termination_condition: Optional[str] = None
    max_duration_seconds: float = 3600.0
    configuration: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.scenario_id or not isinstance(self.scenario_id, str) or not self.scenario_id.strip():
            raise ValueError("Scenario scenario_id must be a non-empty string.")
        if not self.name or not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Scenario name must be a non-empty string.")

        t = float(self.initial_time)
        if math.isnan(t) or math.isinf(t) or t < 0.0:
            raise ValueError(f"Scenario initial_time must be a finite non-negative number, got {t}")
        object.__setattr__(self, "initial_time", round(t, 4))

        md = float(self.max_duration_seconds)
        if math.isnan(md) or math.isinf(md) or md <= 0.0:
            raise ValueError(f"Scenario max_duration_seconds must be positive, got {md}")
        object.__setattr__(self, "max_duration_seconds", round(md, 4))

        # Sort steps deterministically by time_offset, then step_id
        sorted_steps = tuple(sorted(self.steps, key=lambda s: (s.time_offset, s.step_id)))
        object.__setattr__(self, "steps", sorted_steps)
        object.__setattr__(self, "provenance", sanitize_contract_metadata(dict(self.provenance)))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "name": self.name,
            "description": self.description,
            "initial_time": self.initial_time,
            "initial_states": {k: v.to_dict() for k, v in self.initial_states.items()},
            "environment_config": dict(self.environment_config),
            "initial_devices": dict(self.initial_devices),
            "steps": [s.to_dict() for s in self.steps],
            "assertions": [a.to_dict() for a in self.assertions],
            "termination_condition": self.termination_condition,
            "max_duration_seconds": self.max_duration_seconds,
            "configuration": dict(self.configuration),
            "provenance": dict(self.provenance),
        }

    def model_dump(self) -> Dict[str, Any]:
        return self.to_dict()

    def model_dump_json(self) -> str:
        return json.dumps(self.to_dict())

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
            initial_devices=dict(data.get("initial_devices", {})),
            steps=steps,
            assertions=assertions,
            termination_condition=data.get("termination_condition"),
            max_duration_seconds=float(data.get("max_duration_seconds", 3600.0)),
            configuration=dict(data.get("configuration", {})),
            provenance=dict(data.get("provenance", {})),
        )

    @classmethod
    def model_validate(cls, obj: Any) -> "Scenario":
        if isinstance(obj, cls):
            return obj
        if isinstance(obj, dict):
            return cls.from_dict(obj)
        raise TypeError(f"Cannot validate {type(obj)} as Scenario")


# ============================================================================
# Scenario Result & Replay
# ============================================================================

@dataclass(frozen=True)
class ScenarioResult:
    """
    Immutable execution outcome of a simulation scenario.
    Provides complete observable history, assertion outcomes, and deterministic SHA-256 hash.
    """
    scenario_id: str
    success: bool
    total_steps_executed: int = 0
    start_time: float = 0.0
    end_time: float = 0.0
    simulated_duration: float = 0.0
    completed_steps: int = 0
    failed_steps: int = 0
    assertions_evaluated: int = 0
    assertions_passed: int = 0
    assertion_results: Tuple[ScenarioAssertionResult, ...] = ()
    assertion_failures: Tuple[Dict[str, Any], ...] = ()
    emitted_observations: Tuple[MultimodalObservation, ...] = ()
    observed_events: Tuple[Any, ...] = ()
    discovered_situations: Tuple[Any, ...] = ()
    observed_missions: Tuple[Any, ...] = ()
    observed_goals: Tuple[Any, ...] = ()
    device_results: Dict[str, Any] = field(default_factory=dict)
    trace: Tuple[Dict[str, Any], ...] = ()
    trace_reference: Optional[str] = None
    deterministic_hash: str = ""
    summary: str = ""

    def __post_init__(self):
        if not self.scenario_id or not isinstance(self.scenario_id, str):
            raise ValueError("ScenarioResult scenario_id must be a non-empty string.")

        if not self.deterministic_hash:
            # Deterministically compute SHA-256 hash across stable fields
            pass_rate = f"{self.assertions_passed}/{self.assertions_evaluated}"
            trace_sig = f"{len(self.trace)}:{self.total_steps_executed}"
            h_input = (
                f"{self.scenario_id}|{self.success}|{self.simulated_duration:.2f}|"
                f"{pass_rate}|{trace_sig}|{len(self.emitted_observations)}"
            ).encode("utf-8")
            h = hashlib.sha256(h_input).hexdigest()
            object.__setattr__(self, "deterministic_hash", h)

    def compute_deterministic_hash(self) -> str:
        """Compute stable SHA-256 digest over outcome invariants."""
        pass_rate = f"{self.assertions_passed}/{self.assertions_evaluated}"
        trace_sig = f"{len(self.trace)}:{self.total_steps_executed}"
        h_input = (
            f"{self.scenario_id}|{self.success}|{self.simulated_duration:.2f}|"
            f"{pass_rate}|{trace_sig}|{len(self.emitted_observations)}"
        ).encode("utf-8")
        return hashlib.sha256(h_input).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "success": self.success,
            "total_steps_executed": self.total_steps_executed,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "simulated_duration": self.simulated_duration,
            "completed_steps": self.completed_steps,
            "failed_steps": self.failed_steps,
            "assertions_evaluated": self.assertions_evaluated,
            "assertions_passed": self.assertions_passed,
            "assertion_results": [r.to_dict() for r in self.assertion_results],
            "assertion_failures": list(self.assertion_failures),
            "emitted_observation_count": len(self.emitted_observations),
            "observed_event_count": len(self.observed_events),
            "discovered_situation_count": len(self.discovered_situations),
            "observed_mission_count": len(self.observed_missions),
            "observed_goal_count": len(self.observed_goals),
            "device_results": dict(self.device_results),
            "trace_depth": len(self.trace),
            "trace_reference": self.trace_reference,
            "deterministic_hash": self.deterministic_hash,
            "summary": self.summary,
        }

    def model_dump(self) -> Dict[str, Any]:
        return self.to_dict()

    def model_dump_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScenarioResult":
        a_results = tuple(
            ScenarioAssertionResult.from_dict(r) for r in data.get("assertion_results", [])
        )
        return cls(
            scenario_id=str(data["scenario_id"]),
            success=bool(data["success"]),
            total_steps_executed=int(data.get("total_steps_executed", 0)),
            start_time=float(data.get("start_time", 0.0)),
            end_time=float(data.get("end_time", 0.0)),
            simulated_duration=float(data.get("simulated_duration", 0.0)),
            completed_steps=int(data.get("completed_steps", 0)),
            failed_steps=int(data.get("failed_steps", 0)),
            assertions_evaluated=int(data.get("assertions_evaluated", 0)),
            assertions_passed=int(data.get("assertions_passed", 0)),
            assertion_results=a_results,
            assertion_failures=tuple(data.get("assertion_failures", ())),
            device_results=dict(data.get("device_results", {})),
            trace=tuple(data.get("trace", ())),
            trace_reference=data.get("trace_reference"),
            deterministic_hash=str(data.get("deterministic_hash", "")),
            summary=str(data.get("summary", "")),
        )

    @classmethod
    def model_validate(cls, obj: Any) -> "ScenarioResult":
        if isinstance(obj, cls):
            return obj
        if isinstance(obj, dict):
            return cls.from_dict(obj)
        raise TypeError(f"Cannot validate {type(obj)} as ScenarioResult")


# ============================================================================
# Scenario Builder
# ============================================================================

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
        self.initial_devices: Dict[str, Any] = {}
        self.steps: List[ScenarioStep] = []
        self.assertions: List[ScenarioAssertion] = []
        self.termination_condition: Optional[str] = None
        self.max_duration_seconds: float = 3600.0
        self.configuration: Dict[str, Any] = {}
        self.provenance: Dict[str, Any] = {}

    def with_initial_time(self, t: float) -> "ScenarioBuilder":
        self.initial_time = t
        return self

    def add_twin_config(self, config: TwinConfiguration) -> "ScenarioBuilder":
        self.initial_states[config.twin_id] = config
        return self

    def with_environment_config(self, env_cfg: Dict[str, Any]) -> "ScenarioBuilder":
        self.environment_config = dict(env_cfg)
        return self

    def with_initial_devices(self, devices: Dict[str, Any]) -> "ScenarioBuilder":
        self.initial_devices = dict(devices)
        return self

    def add_step(
        self,
        step_id: str,
        time_offset: float,
        action_type: Union[ScenarioStepType, str],
        target_id: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> "ScenarioBuilder":
        act = action_type.value if hasattr(action_type, "value") else str(action_type)
        self.steps.append(
            ScenarioStep(
                step_id=step_id,
                time_offset=time_offset,
                action_type=act,
                target_id=target_id,
                payload=payload or {},
            )
        )
        return self

    def add_assertion(
        self,
        assertion_id: str,
        target_type: Union[ScenarioAssertionTarget, str],
        target_id: str,
        expected_field: str,
        expected_value: Any,
        operator: Union[ScenarioAssertionOperator, str] = ScenarioAssertionOperator.EQUALS,
        severity: Union[ScenarioAssertionSeverity, str] = ScenarioAssertionSeverity.FAIL,
        tolerance: Optional[float] = None,
    ) -> "ScenarioBuilder":
        tt = target_type.value if hasattr(target_type, "value") else str(target_type)
        op = operator.value if hasattr(operator, "value") else str(operator)
        sev = severity.value if hasattr(severity, "value") else str(severity)
        self.assertions.append(
            ScenarioAssertion(
                assertion_id=assertion_id,
                target_type=tt,
                target_id=target_id,
                expected_field=expected_field,
                expected_value=expected_value,
                operator=op,
                severity=sev,
                tolerance=tolerance,
            )
        )
        return self

    def with_termination_condition(self, condition: str) -> "ScenarioBuilder":
        self.termination_condition = condition
        return self

    def with_max_duration(self, seconds: float) -> "ScenarioBuilder":
        self.max_duration_seconds = seconds
        return self

    def with_configuration(self, config: Dict[str, Any]) -> "ScenarioBuilder":
        self.configuration = dict(config)
        return self

    def with_provenance(self, provenance: Dict[str, Any]) -> "ScenarioBuilder":
        self.provenance = dict(provenance)
        return self

    def build(self) -> Scenario:
        sorted_steps = sorted(self.steps, key=lambda s: (s.time_offset, s.step_id))
        return Scenario(
            scenario_id=self.scenario_id,
            name=self.name,
            description=self.description,
            initial_time=self.initial_time,
            initial_states=dict(self.initial_states),
            environment_config=dict(self.environment_config),
            initial_devices=dict(self.initial_devices),
            steps=tuple(sorted_steps),
            assertions=tuple(self.assertions),
            termination_condition=self.termination_condition,
            max_duration_seconds=self.max_duration_seconds,
            configuration=dict(self.configuration),
            provenance=dict(self.provenance),
        )
