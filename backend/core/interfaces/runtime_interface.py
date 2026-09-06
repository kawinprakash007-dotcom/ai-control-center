from abc import ABC, abstractmethod
from typing import Any, List, Optional

from core.models.runtime import (
    CognitiveEvent,
    CognitiveTurnResult,
)


class CognitiveEventSinkInterface(ABC):
    """
    Interface for publishing structured domain events emitted by CognitiveRuntime.
    Implementations include InMemoryEventSink (test/debug), LoggingEventSink, and future sinks.
    """

    @abstractmethod
    def publish(self, event: CognitiveEvent) -> None:
        """Publish an immutable structured cognitive event."""
        pass

    @abstractmethod
    def get_events(self, turn_id: Optional[str] = None) -> List[CognitiveEvent]:
        """Retrieve collected events, optionally filtered by turn_id."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all stored events."""
        pass


class CognitiveRuntimeInterface(ABC):
    """
    Interface for the unified ATLAS cognitive runtime.
    Coordinates the full end-to-end cognitive turn lifecycle across Phase 2 and 3 subsystems.
    """

    @abstractmethod
    def execute_turn(
        self,
        input_data: Any,
        session_id: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
    ) -> CognitiveTurnResult:
        """
        Execute one complete cognitive turn:
        Understanding -> Decision -> Planning -> Context -> Routing ->
        Reasoning -> Proposal -> Validation -> Policy -> Execution ->
        Observation -> Verification -> (Recovery) -> Memory -> Response -> Completed.

        Returns a structured CognitiveTurnResult.
        """
        pass

    @abstractmethod
    def run(
        self,
        input_data: Any,
        session_id: Optional[str] = None,
    ) -> str:
        """
        Execute one cognitive turn and return only the final response text.
        """
        pass
