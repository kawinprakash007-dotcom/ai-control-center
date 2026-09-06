from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence

from core.models.autonomy import (
    AutonomyDecision,
    Event,
    EventClassification,
    EventRelevance,
    EventTrigger,
)


class EventClassifierInterface(ABC):
    """
    Contract for deterministic, model-neutral event classification.
    Identifies category, priority, and whether an event requires action evaluation.
    Never executes external models or tools.
    """

    @abstractmethod
    def classify(self, event: Event) -> EventClassification:
        """
        Classify an incoming autonomy event.
        Unknown events must fail safely with requires_action=False.
        """
        pass


class RelevanceEngineInterface(ABC):
    """
    Contract for deterministic relevance scoring.
    Evaluates an event against system state, freshness, and active goals.
    Never executes external models or tools.
    """

    @abstractmethod
    def evaluate_relevance(
        self,
        event: Event,
        classification: EventClassification,
        current_state: Optional[Any] = None,
        active_goals: Optional[Sequence[Any]] = None,
    ) -> EventRelevance:
        """
        Deterministically evaluate relevance score in range [0.0, 1.0].
        """
        pass


class EventDrivenAutonomyInterface(ABC):
    """
    Contract for the central Event-Driven Autonomy coordinator.
    Routes evaluated events into Goal Management / Cognitive Runtime.
    Events NEVER directly execute tools, models, or shell commands.
    """

    @abstractmethod
    def ingest_event(self, event: Event) -> AutonomyDecision:
        """
        Ingest, classify, evaluate, and process a single autonomy event.
        Enforces storm suppression, loop bounds, and policy checks.
        """
        pass

    @abstractmethod
    def process_pending_events(self, max_events: int = 10) -> List[AutonomyDecision]:
        """
        Process a bounded batch of queued events.
        Guarantees bounded execution without infinite loops.
        """
        pass

    @abstractmethod
    def register_trigger(self, trigger: EventTrigger) -> None:
        """
        Register an autonomous trigger condition and rule.
        """
        pass

    @abstractmethod
    def get_history(self, limit: int = 100) -> List[AutonomyDecision]:
        """
        Retrieve auditable history of recent autonomy decisions.
        """
        pass
