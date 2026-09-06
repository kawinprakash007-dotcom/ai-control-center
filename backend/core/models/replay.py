from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from core.models.runtime import CognitiveStage, CognitiveTrace, TurnStatus


class ReplayMode(str, Enum):
    """
    Operating mode for deterministic turn replay.
    """
    OFFLINE_REPLAY = "OFFLINE_REPLAY"        # Uses recorded outputs; no external calls or side effects.
    SIMULATION_REPLAY = "SIMULATION_REPLAY"  # Uses injected mock/stub subsystems; no real side effects.


class ReplayDivergenceType(str, Enum):
    """
    Categorization of behavioral divergences between original and replay traces.
    """
    MISSING_EVENT = "MISSING_EVENT"
    EXTRA_EVENT = "EXTRA_EVENT"
    STAGE_MISMATCH = "STAGE_MISMATCH"
    RESULT_MISMATCH = "RESULT_MISMATCH"
    POLICY_MISMATCH = "POLICY_MISMATCH"
    PROVIDER_MISMATCH = "PROVIDER_MISMATCH"
    STATE_MISMATCH = "STATE_MISMATCH"
    METADATA_MISMATCH = "METADATA_MISMATCH"


class DivergenceSeverity(str, Enum):
    """
    Severity level of an identified divergence.
    """
    LOW = "LOW"            # Non-blocking diagnostic difference (e.g. non-critical metadata)
    MEDIUM = "MEDIUM"      # Benign divergence (e.g. alternate equivalent provider)
    HIGH = "HIGH"          # Behavioral deviation (e.g. unexpected result, recovery triggered)
    CRITICAL = "CRITICAL"  # Policy or security violation (e.g. policy DENY vs ALLOW)


@dataclass(frozen=True)
class ReplayDivergence:
    """
    Structured record of a single divergence point between original and replay traces.
    """
    event_index: int
    stage: CognitiveStage
    divergence_type: ReplayDivergenceType
    expected: Any
    actual: Any
    severity: DivergenceSeverity = DivergenceSeverity.HIGH
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ReplayLimits:
    """
    Safety limits and bounds for replay execution.
    Prevents runaway simulation loops and unbounded memory consumption.
    """
    max_events: int = 100
    max_replay_steps: int = 20
    max_divergences: int = 50
    max_stored_results: int = 200
    max_artifact_size: int = 1024 * 1024  # 1 MB


@dataclass(frozen=True)
class ReplayRequest:
    """
    Structured request for replaying a historical cognitive turn.
    """
    source_trace: CognitiveTrace
    replay_run_id: str
    replay_mode: ReplayMode = ReplayMode.OFFLINE_REPLAY
    policy_reevaluation: bool = False
    deterministic_seed: Optional[int] = None
    limits: ReplayLimits = field(default_factory=ReplayLimits)
    component_overrides: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ComparisonResult:
    """
    Deterministic comparison between an original trace and its replay trace.
    """
    status: str  # "IDENTICAL", "SEMANTICALLY_EQUIVALENT", "DIVERGENT"
    divergences: Tuple[ReplayDivergence, ...] = ()
    matched_events_count: int = 0
    total_original_events: int = 0
    total_replay_events: int = 0
    summary: str = ""

    @property
    def is_equivalent(self) -> bool:
        return self.status in ("IDENTICAL", "SEMANTICALLY_EQUIVALENT")


@dataclass(frozen=True)
class ReplayResult:
    """
    Complete structured outcome of a turn replay run.
    """
    success: bool
    replay_run_id: str
    source_turn_id: str
    completed_stages: Tuple[CognitiveStage, ...]
    replay_trace: CognitiveTrace
    comparison: Optional[ComparisonResult] = None
    final_status: TurnStatus = TurnStatus.SUCCEEDED
    diagnostics: Dict[str, Any] = field(default_factory=dict)
