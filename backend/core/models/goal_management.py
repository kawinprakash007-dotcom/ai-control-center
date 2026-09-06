from dataclasses import dataclass, field
from enum import Enum
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


class GoalPriority(str, Enum):
    """
    Discrete priority levels for goal scheduling preferences.
    Priority is a scheduling preference only and never overrides policy, safety, or constraints.
    """
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"

    @property
    def score_weight(self) -> int:
        weights = {
            GoalPriority.LOW: 1000,
            GoalPriority.NORMAL: 2000,
            GoalPriority.HIGH: 3000,
            GoalPriority.CRITICAL: 4000,
        }
        return weights.get(self, 2000)

    @classmethod
    def from_str(cls, val: str) -> "GoalPriority":
        if isinstance(val, cls):
            return val
        s = str(val).strip().lower()
        for p in cls:
            if p.value == s:
                return p
        return cls.NORMAL


class DeadlineStatus(str, Enum):
    """
    Deterministic classification of a goal's deadline urgency.
    """
    NO_DEADLINE = "no_deadline"
    ON_TRACK = "on_track"
    DUE_SOON = "due_soon"
    OVERDUE = "overdue"


class GoalFreshnessStatus(str, Enum):
    """
    Freshness classification distinguishing active from stale goals.
    Stale does NOT equal failed.
    """
    FRESH = "fresh"
    STALE = "stale"


@dataclass(frozen=True)
class SchedulerConfig:
    """
    Configuration parameters for deterministic goal scheduling.
    """
    due_soon_window_seconds: float = 300.0  # 5 minutes
    stale_threshold_seconds: float = 3600.0  # 1 hour without update
    fairness_starvation_limit: int = 5       # Max quanta bypassed before aging boost
    fairness_boost_per_skip: int = 250       # Points added per skipped eligible quantum
    active_continuation_boost: int = 200     # Points added to currently active/partial goal
    lease_duration_seconds: float = 60.0     # Default active goal claim lease
    scheduler_version: str = "v1.0"


@dataclass(frozen=True)
class SchedulingDecision:
    """
    Structured outcome of a deterministic goal scheduling cycle.
    Provides complete explanation for replay and auditability.
    """
    selected_goal_id: Optional[str]
    reason: str
    candidates_considered: Tuple[str, ...] = field(default_factory=tuple)
    skipped_candidates: Dict[str, str] = field(default_factory=dict)
    scheduling_factors: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def is_selection_made(self) -> bool:
        return self.selected_goal_id is not None


@dataclass
class GoalManagementState:
    """
    Operational metadata representing the high-level state of managed goals.
    Does NOT duplicate the full Goal domain model.
    """
    active_goal_id: Optional[str] = None
    active_goal_owner: Optional[str] = None
    active_goal_lease_expires: Optional[float] = None
    queued_goal_ids: Tuple[str, ...] = field(default_factory=tuple)
    paused_goal_ids: Tuple[str, ...] = field(default_factory=tuple)
    waiting_goal_ids: Tuple[str, ...] = field(default_factory=tuple)
    blocked_goal_ids: Tuple[str, ...] = field(default_factory=tuple)
    completed_goal_ids: Tuple[str, ...] = field(default_factory=tuple)
    stale_goal_ids: Tuple[str, ...] = field(default_factory=tuple)
    fairness_counters: Dict[str, int] = field(default_factory=dict)
    last_selection_time: Optional[float] = None
    scheduler_version: str = "v1.0"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class QuantumResult:
    """
    Result of executing a single bounded goal execution quantum.
    """
    goal_id: str
    decision: SchedulingDecision
    goal_status: str
    turn_executed: bool
    details: Dict[str, Any] = field(default_factory=dict)
