from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

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
