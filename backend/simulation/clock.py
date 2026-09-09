"""
ATLAS Phase 6.3 — Deterministic Simulation Clock.

Provides a strictly monotonic, manually controllable simulation clock.
Guarantees deterministic progression with ZERO time.sleep() or daemon threads.
"""

import threading
from typing import Callable, List, Optional

from core.interfaces.simulation_interface import SimulationClockInterface


class SimulationClock(SimulationClockInterface):
    """
    Deterministic, manually controllable simulation clock.
    Thread-safe and strictly monotonic. Rejects negative time transitions.
    """

    def __init__(self, initial_time: float = 1000000.0):
        if initial_time < 0.0:
            raise ValueError(f"initial_time must be non-negative, got {initial_time}")
        self._lock = threading.RLock()
        self._current_time: float = float(initial_time)
        self._monotonic_ticks: int = 0
        self._listeners: List[Callable[[float, float], None]] = []

    def now(self) -> float:
        """Get current deterministic simulation time in seconds."""
        with self._lock:
            return self._current_time

    def advance(self, seconds: float) -> float:
        """
        Advance simulation time forward by the given positive delta.
        Returns the updated simulation timestamp.
        """
        if seconds < 0.0:
            raise ValueError(f"Cannot advance simulation clock by negative delta: {seconds}")

        with self._lock:
            old_time = self._current_time
            self._current_time += seconds
            self._monotonic_ticks += 1
            listeners = list(self._listeners)

        for listener in listeners:
            try:
                listener(old_time, self._current_time)
            except Exception:
                pass

        return self._current_time

    def tick(self, delta: float = 1.0) -> float:
        """Convenience alias for advancing the clock by delta seconds (default: 1.0)."""
        return self.advance(delta)

    def set_time(self, timestamp: float) -> float:
        """
        Explicitly advance simulation time to a future timestamp.
        Rejects timestamps that would cause the clock to move backwards.
        """
        with self._lock:
            if timestamp < self._current_time:
                raise ValueError(
                    f"Simulation clock cannot move backwards: current={self._current_time}, target={timestamp}"
                )
            old_time = self._current_time
            self._current_time = float(timestamp)
            self._monotonic_ticks += 1
            listeners = list(self._listeners)

        for listener in listeners:
            try:
                listener(old_time, self._current_time)
            except Exception:
                pass

        return self._current_time

    def get_ticks(self) -> int:
        """Return the number of time advances that have occurred."""
        with self._lock:
            return self._monotonic_ticks

    def register_listener(self, callback: Callable[[float, float], None]) -> None:
        """Register a callback to be notified when simulation time advances."""
        with self._lock:
            if callback not in self._listeners:
                self._listeners.append(callback)

    def unregister_listener(self, callback: Callable[[float, float], None]) -> None:
        """Unregister a previously registered listener."""
        with self._lock:
            if callback in self._listeners:
                self._listeners.remove(callback)
