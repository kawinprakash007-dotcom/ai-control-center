from collections import defaultdict
import logging
import threading
from typing import Any, Callable, Dict, List, Optional

from core.interfaces.runtime_interface import CognitiveEventSinkInterface
from core.models.runtime import CognitiveEvent

logger = logging.getLogger("atlas.cognitive_runtime")


class InMemoryEventSink(CognitiveEventSinkInterface):
    """
    Thread-safe, bounded in-memory event collector.
    Used for testing, diagnostics, and building bounded CognitiveTrace snapshots.
    """

    def __init__(self, max_events_per_turn: int = 100):
        self._max_events = max_events_per_turn
        self._events: List[CognitiveEvent] = []
        self._by_turn: Dict[str, List[CognitiveEvent]] = defaultdict(list)
        self._subscribers: List[Any] = []
        self._lock = threading.Lock()

    def subscribe(self, callback: Any) -> None:
        """Register a subscriber callback for published events."""
        with self._lock:
            self._subscribers.append(callback)

    def publish(self, event: CognitiveEvent) -> None:
        """Publish and store a structured cognitive event within bounded limits."""
        with self._lock:
            # Enforce per-turn bounded limit
            turn_events = self._by_turn[event.turn_id]
            if len(turn_events) < self._max_events:
                turn_events.append(event)
                self._events.append(event)
            else:
                logger.warning(
                    "Event sink reached maximum event limit (%d) for turn %s. Dropping event: %s",
                    self._max_events,
                    event.turn_id,
                    event.event_type.value,
                )
            callbacks = list(self._subscribers)
        for cb in callbacks:
            try:
                cb(event)
            except Exception as e:
                logger.debug("Error in event sink subscriber: %s", e)

    def get_events(self, turn_id: Optional[str] = None) -> List[CognitiveEvent]:
        """Return a copy of events, optionally filtered by turn_id."""
        with self._lock:
            if turn_id is not None:
                return list(self._by_turn.get(turn_id, []))
            return list(self._events)

    def clear(self) -> None:
        """Clear all stored events."""
        with self._lock:
            self._events.clear()
            self._by_turn.clear()


class LoggingEventSink(CognitiveEventSinkInterface):
    """
    Structured domain logging event sink.
    Emits events to standard logger without storing unbounded history.
    """

    def __init__(self, log_level: int = logging.INFO):
        self.log_level = log_level

    def publish(self, event: CognitiveEvent) -> None:
        logger.log(
            self.log_level,
            "COGNITIVE_EVENT [%s] turn=%s stage=%s status=%s comp=%s summary='%s'",
            event.event_type.value,
            event.turn_id,
            event.stage.value,
            event.status,
            event.component,
            event.summary,
        )

    def get_events(self, turn_id: Optional[str] = None) -> List[CognitiveEvent]:
        return []

    def clear(self) -> None:
        pass


class CompositeEventSink(CognitiveEventSinkInterface):
    """
    Broadcasts events to multiple sinks (e.g. InMemory + Logging).
    """

    def __init__(self, sinks: List[CognitiveEventSinkInterface]):
        self._sinks = list(sinks)

    def publish(self, event: CognitiveEvent) -> None:
        for sink in self._sinks:
            sink.publish(event)

    def get_events(self, turn_id: Optional[str] = None) -> List[CognitiveEvent]:
        for sink in self._sinks:
            evs = sink.get_events(turn_id)
            if evs:
                return evs
        return []

    def clear(self) -> None:
        for sink in self._sinks:
            sink.clear()
