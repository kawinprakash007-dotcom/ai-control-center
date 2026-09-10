import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from core.models.goal_management import GoalPriority


class GoalStatus(str, Enum):
    """
    Lifecycle status of a high-level ATLAS goal.
    Explicit vocabulary distinguishing active, paused, blocked, waiting, and terminal states.
    """
    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    BLOCKED = "blocked"
    WAITING_FOR_USER = "waiting_for_user"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"
    CANCELLED = "cancelled"


class ObjectiveStatus(str, Enum):
    """
    Lifecycle status of an individual objective within a goal.
    """
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    PARTIAL = "partial"
    BLOCKED = "blocked"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class GoalConstraints:
    """
    Bounded constraints describing goal execution limits and parameters.
    """
    deadline: Optional[float] = None
    allowed_capabilities: Optional[Tuple[str, ...]] = None
    max_turns_total: int = 20
    max_turns_per_objective: int = 5
    max_failed_objectives: int = 2
    max_consecutive_no_progress: int = 3
    privacy_requirement: Optional[str] = None
    autonomy_level: str = "supervised"


@dataclass(frozen=True)
class GoalCompletionCriteria:
    """
    Explicit, deterministic criteria determining when a goal is completed.
    """
    require_all_objectives: bool = True
    required_objective_ids: Optional[Tuple[str, ...]] = None
    min_completion_percentage: float = 100.0
    user_confirmation_required: bool = False


@dataclass(frozen=True)
class GoalProgress:
    """
    Structured progress model with explicit semantics.
    """
    completed_objectives: int = 0
    total_objectives: int = 0
    percentage: float = 0.0
    active_objective_id: Optional[str] = None
    blockers: Tuple[str, ...] = field(default_factory=tuple)
    last_update: float = field(default_factory=time.time)
    progress_reason: str = ""


@dataclass
class Objective:
    """
    Bounded objective representing a concrete step towards achieving a Goal.
    Does NOT contain raw cognitive traces.
    """
    objective_id: str
    description: str
    order: int = 1
    dependencies: Tuple[str, ...] = field(default_factory=tuple)
    status: ObjectiveStatus = ObjectiveStatus.PENDING
    completion_criteria: Optional[str] = None
    attempts: int = 0
    max_attempts: int = 3
    progress: float = 0.0
    result_summary: Optional[str] = None
    blocker_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "objective_id": self.objective_id,
            "description": self.description,
            "order": self.order,
            "dependencies": list(self.dependencies),
            "status": self.status.value if hasattr(self.status, "value") else str(self.status),
            "completion_criteria": self.completion_criteria,
            "attempts": self.attempts,
            "max_attempts": self.max_attempts,
            "progress": self.progress,
            "result_summary": self.result_summary,
            "blocker_reason": self.blocker_reason,
            "metadata": dict(self.metadata),
        }

    def is_terminal(self) -> bool:
        """Check if objective is in a terminal state."""
        return self.status in (
            ObjectiveStatus.COMPLETED,
            ObjectiveStatus.FAILED,
            ObjectiveStatus.SKIPPED,
            ObjectiveStatus.CANCELLED,
        )

    def is_ready(self, completed_objective_ids: Tuple[str, ...]) -> bool:
        """Check if all prerequisite dependencies are completed."""
        if self.status not in (ObjectiveStatus.PENDING, ObjectiveStatus.READY, ObjectiveStatus.PARTIAL):
            return False
        return all(dep in completed_objective_ids for dep in self.dependencies)


@dataclass
class Goal:
    """
    First-class goal domain model.
    Encapsulates necessary state to pursue a goal across multiple cognitive turns.
    Guarantees that original_goal remains strictly immutable.
    Preserves backward compatibility for legacy callers.
    """
    original_goal: str = ""
    goal_id: str = ""
    objectives: Tuple[Objective, ...] = field(default_factory=tuple)
    status: GoalStatus = GoalStatus.CREATED
    constraints: GoalConstraints = field(default_factory=GoalConstraints)
    completion_criteria: GoalCompletionCriteria = field(default_factory=GoalCompletionCriteria)
    progress: GoalProgress = field(default_factory=GoalProgress)
    active_objective_id: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Legacy Phase 1 compatibility attributes
    goal: str = ""
    priority: Union[GoalPriority, str] = GoalPriority.NORMAL
    confidence: float = 1.0
    reason: str = ""
    query: str = ""

    def __post_init__(self):
        # Support legacy Goal(goal="...") initialization
        if not self.original_goal and self.goal:
            super().__setattr__("original_goal", self.goal)
        elif self.original_goal and not self.goal:
            super().__setattr__("goal", self.original_goal)

        if not self.goal_id:
            super().__setattr__("goal_id", f"goal_{uuid.uuid4().hex[:10]}")

        # Ensure status is GoalStatus enum
        if isinstance(self.status, str) and not isinstance(self.status, GoalStatus):
            try:
                super().__setattr__("status", GoalStatus(self.status.lower()))
            except ValueError:
                super().__setattr__("status", GoalStatus.CREATED)

        # Normalize priority to GoalPriority enum
        if isinstance(self.priority, str) and not isinstance(self.priority, GoalPriority):
            super().__setattr__("priority", GoalPriority.from_str(self.priority))

        super().__setattr__("_initialized", True)

    def __setattr__(self, name: str, value: Any) -> None:
        # Enforce original_goal immutability invariant after initialization
        if getattr(self, "_initialized", False):
            if name in ("original_goal", "goal") and getattr(self, "original_goal", None) != value:
                raise AttributeError("original_goal is strictly immutable and cannot be modified")
        super().__setattr__(name, value)

    def is_terminal(self) -> bool:
        """Check if goal is in a terminal state."""
        return self.status in (
            GoalStatus.COMPLETED,
            GoalStatus.FAILED,
            GoalStatus.ABORTED,
            GoalStatus.CANCELLED,
        )

    @property
    def id(self) -> str:
        return self.goal_id

    @property
    def title(self) -> str:
        return self.original_goal

    @property
    def description(self) -> str:
        return self.original_goal

    @property
    def correlation_id(self) -> str:
        return self.metadata.get("correlation_id", "") if self.metadata else ""

    @property
    def causation_id(self) -> str:
        return self.metadata.get("causation_id", "") if self.metadata else ""

    def get_objective(self, objective_id: str) -> Optional[Objective]:
        """Find an objective by ID."""
        for obj in self.objectives:
            if obj.objective_id == objective_id:
                return obj
        return None

    def get_completed_objective_ids(self) -> Tuple[str, ...]:
        """Return tuple of completed objective IDs."""
        return tuple(obj.objective_id for obj in self.objectives if obj.status == ObjectiveStatus.COMPLETED)

    def to_dict(self) -> Dict[str, Any]:
        prio = self.priority.value if hasattr(self.priority, "value") else str(self.priority)
        st = self.status.value if hasattr(self.status, "value") else str(self.status)
        prog = {}
        if hasattr(self, "progress") and self.progress:
            prog = {
                "percentage": getattr(self.progress, "percentage", 0.0),
                "completed_objectives": getattr(self.progress, "completed_objectives", 0),
                "total_objectives": getattr(self.progress, "total_objectives", 0),
                "progress_reason": getattr(self.progress, "progress_reason", ""),
            }
        return {
            "goal_id": self.goal_id,
            "original_goal": self.original_goal,
            "goal": self.goal,
            "status": st,
            "priority": prio,
            "confidence": self.confidence,
            "reason": self.reason,
            "query": self.query,
            "active_objective_id": self.active_objective_id,
            "progress": prog,
            "objectives": [obj.to_dict() if hasattr(obj, "to_dict") else str(obj) for obj in self.objectives],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": dict(self.metadata),
        }