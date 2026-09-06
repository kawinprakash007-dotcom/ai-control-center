from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from core.models.world_state import (
    ConflictPolicy,
    ConflictResolution,
    FreshnessConfig,
    FreshnessStatus,
    Observation,
    StateConflict,
    WorldCondition,
    WorldEntity,
    WorldRelationship,
    WorldState,
    WorldStateTransition,
    WorldStateUpdateResult,
)


class ObservationSourceInterface(ABC):
    """
    Contract for sources producing observations (sensors, perception, tools, users).
    Ensures clear identity and modality attribution.
    """

    @abstractmethod
    def get_source_id(self) -> str:
        """Return unique identifier for this observation source."""
        pass

    @abstractmethod
    def get_source_type(self) -> str:
        """Return modality/type category (e.g. 'sensor', 'camera', 'tool_result', 'user')."""
        pass


class WorldStateStoreInterface(ABC):
    """
    Persistence and retrieval contract for persistent world state.
    Maintains monotonic state versions, entity/condition index, and auditable transitions.
    """

    @abstractmethod
    def get_current_state(self) -> WorldState:
        """Retrieve the latest immutable WorldState snapshot."""
        pass

    @abstractmethod
    def get_condition(self, entity_id: str, property_name: str) -> Optional[WorldCondition]:
        """Retrieve a specific condition by entity ID and property name from current state."""
        pass

    @abstractmethod
    def get_entity(self, entity_id: str) -> Optional[WorldEntity]:
        """Retrieve a specific entity by entity ID from current state."""
        pass

    @abstractmethod
    def apply_transition(self, transition: WorldStateTransition, next_state: WorldState) -> WorldState:
        """
        Atomically record a transition and commit the resulting next WorldState.
        Enforces monotonic version increments (next_state.version == current.version + 1).
        """
        pass

    @abstractmethod
    def get_state_version(self, version: int) -> Optional[WorldState]:
        """Retrieve a historical WorldState snapshot by version number."""
        pass

    @abstractmethod
    def get_transition_history(self, since_version: int = 0, limit: int = 100) -> List[WorldStateTransition]:
        """Retrieve auditable transition history starting from a given version."""
        pass

    @abstractmethod
    def get_conflicts(self, unresolved_only: bool = True) -> List[StateConflict]:
        """Retrieve recorded state conflicts."""
        pass

    @abstractmethod
    def save_conflict(self, conflict: StateConflict) -> StateConflict:
        """Persist or update an identified state conflict."""
        pass

    @abstractmethod
    def resolve_conflict(self, resolution: ConflictResolution) -> None:
        """Record conflict resolution and update conflict status to RESOLVED."""
        pass


class ConflictResolverInterface(ABC):
    """
    Contract for deterministic conflict resolution between competing observations.
    Never executes external models or tools.
    """

    @abstractmethod
    def resolve(
        self,
        conflict: StateConflict,
        policy: Optional[ConflictPolicy] = None,
    ) -> Optional[ConflictResolution]:
        """
        Deterministically resolve competing condition values using explicit policy.
        Returns ConflictResolution if resolvable, or None if unresolvable.
        """
        pass


class WorldStateUpdaterInterface(ABC):
    """
    Contract for the explicit state-update boundary.
    All accepted observations must pass through this boundary.
    """

    @abstractmethod
    def apply_observation(self, observation: Observation) -> WorldStateUpdateResult:
        """
        Validate observation, evaluate freshness, detect/resolve conflicts,
        and atomically commit state transition if accepted.
        """
        pass

    @abstractmethod
    def evaluate_freshness(self, condition: WorldCondition, now: Optional[float] = None) -> FreshnessStatus:
        """Evaluate freshness status for a given condition according to active policy."""
        pass

    @abstractmethod
    def check_expirations(self, now: Optional[float] = None) -> List[WorldCondition]:
        """
        Identify currently expired conditions in active world state.
        Emits expiration events if configured.
        """
        pass
