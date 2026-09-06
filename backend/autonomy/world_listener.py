import time
import uuid
from typing import Optional

from core.models.autonomy import (
    Event,
    EventPriority,
    EventProvenance,
    EventSource,
)
from core.models.world_state import TransitionType, WorldStateTransition


def transition_to_autonomy_event(
    transition: WorldStateTransition,
    correlation_id: Optional[str] = None,
) -> Optional[Event]:
    """
    Deterministically convert a meaningful Phase 4.4 WorldStateTransition into an autonomy Event.
    Filters out noise/benign state changes unless they are relevant for autonomy evaluation.
    """
    if transition.transition_type == TransitionType.UNCHANGED:
        # Benign refresh; do not generate autonomy event
        return None

    prop = transition.property_name.lower()
    entity = transition.entity_id.lower()

    # Determine priority based on domain property semantics
    priority = EventPriority.NORMAL
    event_type = f"world_state.{transition.property_name}.{transition.transition_type.value.lower()}"

    if any(k in prop for k in ("battery", "power", "fuel", "temp", "pressure", "load", "error", "alarm", "hazard")):
        if any(k in prop for k in ("alarm", "hazard", "fire", "breach", "critical")):
            priority = EventPriority.CRITICAL
            event_type = f"world_state.safety_alert.{transition.property_name}"
        elif any(k in prop for k in ("battery", "power", "fuel", "temp")):
            # Check if numeric value indicates critical threshold
            try:
                val = float(transition.new_value)
                if ("battery" in prop or "power" in prop or "fuel" in prop) and val < 20.0:
                    priority = EventPriority.HIGH
                    event_type = f"world_state.threshold_breach.{transition.property_name}"
                elif "temp" in prop and val > 85.0:
                    priority = EventPriority.HIGH
                    event_type = f"world_state.threshold_breach.{transition.property_name}"
            except (ValueError, TypeError):
                priority = EventPriority.NORMAL
    elif transition.transition_type == TransitionType.ADD:
        priority = EventPriority.NORMAL
    elif transition.transition_type == TransitionType.UNCHANGED:
        # Benign refresh; do not generate autonomy event
        return None

    corr_id = correlation_id or f"corr_ws_{transition.transition_id}"

    provenance = EventProvenance(
        source_id=transition.provenance.source_id,
        source_type=EventSource.WORLD_STATE,
        origin_timestamp=transition.timestamp,
        correlation_id=corr_id,
        depth=0,
        metadata={"transition_id": transition.transition_id, "version": transition.to_version},
    )

    return Event(
        event_id=f"ev_ws_{uuid.uuid4().hex[:12]}",
        source=EventSource.WORLD_STATE,
        event_type=event_type,
        priority=priority,
        timestamp=transition.timestamp,
        payload={
            "entity_id": transition.entity_id,
            "property_name": transition.property_name,
            "old_value": transition.old_value,
            "new_value": transition.new_value,
            "transition_type": transition.transition_type.value,
            "version": transition.to_version,
        },
        provenance=provenance,
        correlation_id=corr_id,
        world_state_version=transition.to_version,
        deduplication_key=f"ws:{transition.entity_id}:{transition.property_name}:{transition.new_value}",
    )
