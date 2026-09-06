from abc import ABC, abstractmethod
from typing import Any, Callable, List, Optional, Sequence, Tuple

from core.models.goal import Goal, GoalStatus, Objective


class GoalDecomposerInterface(ABC):
    """
    Model-neutral boundary for decomposing a high-level Goal into bounded Objectives.
    Decomposition operates purely on Goal specifications and does not execute tools or reasoning loops.
    """

    @abstractmethod
    def decompose(self, goal: Goal) -> Tuple[Objective, ...]:
        """
        Decompose a Goal into a tuple of bounded Objective instances.

        Args:
            goal: The Goal to decompose.

        Returns:
            Tuple of Objective instances with dependency and order information.
        """
        pass


class GoalStoreInterface(ABC):
    """
    Persistence contract for operational Goal state.
    Separated from conversation/preference memory stores.
    """

    @abstractmethod
    def create_goal(self, goal: Goal) -> Goal:
        """Persist a new goal."""
        pass

    @abstractmethod
    def get_goal(self, goal_id: str) -> Optional[Goal]:
        """Retrieve a goal by ID, returning None if not found."""
        pass

    @abstractmethod
    def update_goal(self, goal: Goal) -> Goal:
        """Update existing goal state."""
        pass

    @abstractmethod
    def list_goals(
        self,
        status: Optional[GoalStatus] = None,
        limit: int = 50,
    ) -> List[Goal]:
        """List goals, optionally filtered by status."""
        pass

    @abstractmethod
    def delete_goal(self, goal_id: str) -> bool:
        """Delete or archive a goal by ID."""
        pass

    @abstractmethod
    def claim_goal(self, goal_id: str, owner_id: str, lease_duration: float = 60.0) -> bool:
        """
        Atomically claim a lease on a goal.
        Returns True if claim succeeded, False if already claimed by another owner.
        """
        pass

    @abstractmethod
    def release_goal(self, goal_id: str, owner_id: str) -> bool:
        """
        Release a previously acquired lease on a goal.
        """
        pass

    @abstractmethod
    def get_claimed_goal(self) -> Optional[Tuple[str, str, float]]:
        """
        Return the currently claimed goal tuple (goal_id, owner_id, expires_at), or None.
        """
        pass

    @abstractmethod
    def update_priority(self, goal_id: str, priority: Any) -> Goal:
        """Update goal priority."""
        pass

    @abstractmethod
    def update_deadline(self, goal_id: str, deadline: Optional[float]) -> Goal:
        """Update goal deadline."""
        pass


class GoalExecutionEngineInterface(ABC):
    """
    Contract for coordinating long-horizon goal pursuit across cognitive turns.
    Coordinates existing CognitiveRuntime instances without bypassing policy, routing, or recovery.
    """

    @abstractmethod
    def execute_goal(self, goal_id: str, max_steps: Optional[int] = None) -> Goal:
        """
        Pursue a goal across multiple cognitive turns until completed, blocked, paused, or aborted.
        """
        pass

    @abstractmethod
    def step_goal(self, goal_id: str) -> Goal:
        """
        Execute exactly one bounded cognitive turn for the next ready objective of a goal.
        """
        pass

    @abstractmethod
    def pause_goal(self, goal_id: str, reason: str = "") -> Goal:
        """
        Pause goal execution, preventing further automated cognitive turns while preserving state.
        """
        pass

    @abstractmethod
    def resume_goal(self, goal_id: str) -> Goal:
        """
        Resume a paused or waiting goal from its preserved state.
        """
        pass

    @abstractmethod
    def abort_goal(self, goal_id: str, reason: str = "") -> Goal:
        """
        Abort goal execution immediately due to unrecoverable conditions or user request.
        """
        pass


class GoalSchedulerInterface(ABC):
    """
    Contract for deterministic goal scheduling across multiple candidates.
    Purely decision-making: does not execute turns, tools, or models.
    """

    @abstractmethod
    def select_next_goal(
        self,
        goals: Sequence[Goal],
        state: Any,
        config: Optional[Any] = None,
        clock: Optional[Callable[[], float]] = None,
    ) -> Any:
        """
        Deterministically select the next eligible goal to receive an execution quantum.

        Args:
            goals: Sequence of available Candidate goals.
            state: Current GoalManagementState.
            config: Optional SchedulerConfig.
            clock: Optional injectable time provider.

        Returns:
            SchedulingDecision detailing selection, ranking factors, and skipped reasons.
        """
        pass


class AutonomousGoalManagerInterface(ABC):
    """
    High-level management contract overseeing multiple goals over time.
    Manages eligibility, priority, single-active-goal leases, and bounded quanta.
    """

    @abstractmethod
    def create_goal(self, goal: Goal) -> Goal:
        """
        Create and persist a new goal under management authority.
        """
        pass

    @abstractmethod
    def schedule_once(self) -> Optional[Any]:
        """
        Evaluate candidate goals and execute at most one bounded quantum for the selected goal.
        """
        pass

    @abstractmethod
    def run_next_quantum(self) -> Optional[Any]:
        """
        Alias for schedule_once().
        """
        pass

    @abstractmethod
    def claim_active_goal(self, goal_id: str) -> bool:
        """
        Atomically claim active execution rights for a goal.
        """
        pass

    @abstractmethod
    def release_active_goal(self, goal_id: str) -> None:
        """
        Release active execution rights for a goal.
        """
        pass

    @abstractmethod
    def get_active_goal(self) -> Optional[str]:
        """
        Get the ID of the currently active goal, or None.
        """
        pass

    @abstractmethod
    def pause_goal(self, goal_id: str, reason: str = "") -> Goal:
        """
        Pause a goal and remove it from scheduling eligibility.
        """
        pass

    @abstractmethod
    def resume_goal(self, goal_id: str) -> Goal:
        """
        Resume a paused, blocked, or waiting goal.
        """
        pass

    @abstractmethod
    def cancel_goal(self, goal_id: str, reason: str = "") -> Goal:
        """
        Cancel a goal permanently without deleting historical state.
        """
        pass

    @abstractmethod
    def set_goal_priority(self, goal_id: str, priority: Any) -> Goal:
        """
        Update priority of a goal.
        """
        pass

    @abstractmethod
    def set_goal_deadline(self, goal_id: str, deadline: Optional[float]) -> Goal:
        """
        Update deadline timestamp of a goal.
        """
        pass

    @abstractmethod
    def get_management_state(self) -> Any:
        """
        Retrieve snapshot of current GoalManagementState.
        """
        pass
