"""
ATLAS Phase 6.3 — Controlled Fault Injection Manager.

Provides explicit, observable, bounded, and removable fault injection for
digital twins (OFFLINE, LOW_BATTERY, GPS_LOSS, SENSOR_FAILURE, TELEMETRY_STALE,
COMMAND_TIMEOUT, COMMAND_FAILURE, DUPLICATE_OBSERVATION, CONFLICTING_OBSERVATION).
"""

from collections import deque
import threading
from typing import Dict, List, Optional

from core.models.simulation import SimulationLimits, TwinFault, TwinFaultType


class FaultInjectionManager:
    """
    Manages active and historical faults injected into digital twins.
    Ensures faults are explicit, observable, bounded, and removable.
    """

    def __init__(self, limits: Optional[SimulationLimits] = None):
        self._lock = threading.RLock()
        self.limits = limits or SimulationLimits()
        self._active_faults: Dict[str, TwinFault] = {}
        self._history: deque = deque(maxlen=self.limits.max_fault_records)

    def inject_fault(self, fault: TwinFault) -> None:
        """Inject an observable fault into a digital twin."""
        with self._lock:
            if len(self._active_faults) >= self.limits.max_fault_records and fault.fault_id not in self._active_faults:
                raise ValueError(f"Active fault capacity bound ({self.limits.max_fault_records}) exceeded")
            self._active_faults[fault.fault_id] = fault
            self._history.append(fault)

    def remove_fault(self, fault_id: str) -> bool:
        """Remove an active fault by ID. Returns True if removed."""
        with self._lock:
            return self._active_faults.pop(fault_id, None) is not None

    def clear_faults(self, twin_id: Optional[str] = None) -> int:
        """Clear active faults. If twin_id specified, clears faults for that twin only."""
        with self._lock:
            if twin_id is None:
                count = len(self._active_faults)
                self._active_faults.clear()
                return count

            to_remove = [fid for fid, f in self._active_faults.items() if f.twin_id == twin_id]
            for fid in to_remove:
                del self._active_faults[fid]
            return len(to_remove)

    def get_active_faults(
        self,
        twin_id: Optional[str] = None,
        current_time: Optional[float] = None,
    ) -> List[TwinFault]:
        """
        List currently active, non-expired faults.
        Cleans up expired faults if current_time is provided.
        """
        with self._lock:
            expired_ids = []
            active = []
            for fid, fault in self._active_faults.items():
                if current_time is not None and fault.is_expired(current_time):
                    expired_ids.append(fid)
                    continue
                if twin_id is None or fault.twin_id == twin_id:
                    active.append(fault)

            for fid in expired_ids:
                del self._active_faults[fid]

            return active

    def list_active_faults(
        self,
        twin_id: Optional[str] = None,
        current_time: Optional[float] = None,
    ) -> List[TwinFault]:
        """Alias for get_active_faults."""
        return self.get_active_faults(twin_id=twin_id, current_time=current_time)

    def has_fault(
        self,
        twin_id: str,
        fault_type: TwinFaultType,
        current_time: Optional[float] = None,
    ) -> bool:
        """Check if twin currently has an active fault of the given type."""
        return self.get_fault(twin_id, fault_type, current_time) is not None

    def get_fault(
        self,
        twin_id: str,
        fault_type: TwinFaultType,
        current_time: Optional[float] = None,
    ) -> Optional[TwinFault]:
        """Get the specific active fault for a twin, if present and non-expired."""
        with self._lock:
            for fault in self.get_active_faults(twin_id, current_time):
                if fault.fault_type == fault_type:
                    return fault
            return None

    def get_history(self) -> List[TwinFault]:
        """Return historical fault records up to limits.max_fault_records."""
        with self._lock:
            return list(self._history)
