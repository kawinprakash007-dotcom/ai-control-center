"""
ATLAS Phase 6.4 — Bounded Mission Timeline.

Manages chronological milestones and audit trace records for active missions.
Enforces strict capacity bounds and thread-safe operations.
"""

from collections import deque
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core.models.mission import MissionLimits, MissionTimelineEntry


class MissionTimeline:
    """
    Thread-safe, bounded container for chronological mission timeline entries.
    """

    def __init__(self, limits: Optional[MissionLimits] = None, max_entries: Optional[int] = None):
        self._lock = threading.RLock()
        self.limits = limits or MissionLimits()
        capacity = max_entries if max_entries is not None else self.limits.max_timeline_entries
        self._entries: deque = deque(maxlen=capacity)
        self._counter: int = 0

    def record_entry(
        self,
        event_type: str,
        description: str,
        source_id: str = "CENTRAL",
        evidence_id: Optional[str] = None,
        timestamp: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MissionTimelineEntry:
        """Record a new timeline entry with bounded capacity."""
        with self._lock:
            self._counter += 1
            ts = float(timestamp if timestamp is not None else time.time())
            entry_id = f"mte_{self._counter}_{int(ts)}"
            entry = MissionTimelineEntry(
                entry_id=entry_id,
                timestamp=ts,
                source_id=source_id,
                event_type=event_type,
                description=description,
                evidence_id=evidence_id,
                metadata=metadata or {},
            )
            self._entries.append(entry)
            return entry

    def add_entry(self, entry: MissionTimelineEntry) -> None:
        """Append an existing MissionTimelineEntry directly."""
        with self._lock:
            self._entries.append(entry)

    def get_entries(
        self,
        event_type: Optional[str] = None,
        source_id: Optional[str] = None,
    ) -> Tuple[MissionTimelineEntry, ...]:
        """Return all current timeline entries in chronological order, with optional filtering."""
        with self._lock:
            res = list(self._entries)
            if event_type:
                res = [e for e in res if e.event_type == event_type]
            if source_id:
                res = [e for e in res if e.source_id == source_id]
            return tuple(res)

    def get_entries_by_event_type(self, event_type: str) -> Tuple[MissionTimelineEntry, ...]:
        """Filter entries by event_type."""
        with self._lock:
            return tuple(e for e in self._entries if e.event_type == event_type)

    def get_entries_by_source(self, source_id: str) -> Tuple[MissionTimelineEntry, ...]:
        """Filter entries by source_id."""
        with self._lock:
            return tuple(e for e in self._entries if e.source_id == source_id)

    def clear(self) -> None:
        """Clear all timeline entries."""
        with self._lock:
            self._entries.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)
