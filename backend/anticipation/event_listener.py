from typing import Optional

from core.models.anticipation import (
    Anticipation,
    AnticipationProvenance,
    AnticipatoryDecision,
    EvidenceItem,
    EvidenceSourceType,
    FutureConditionType,
    TimeHorizon,
)
from core.models.autonomy import Event, EventCategory, EventPriority
from anticipation.coordinator import AnticipatoryPlanningCoordinator


def event_to_anticipation_candidate(event: Event) -> Optional[Anticipation]:
    """
    Model-neutral bridge transforming a Phase 4.5 autonomy Event into a
    candidate future condition hypothesis if the event portends future risk.
    Does NOT mutate World State or GoalStore.
    """
    # Only transform events that imply prospective conditions rather than static facts
    is_prospective = False
    condition_type = FutureConditionType.UNKNOWN
    horizon = TimeHorizon.NEAR_TERM

    event_type_lower = event.event_type.lower()
    payload = event.payload or {}

    if any(k in event_type_lower for k in ("pre_warning", "degrading", "discharge", "leak", "depleting")):
        condition_type = FutureConditionType.RESOURCE_DEPLETION_RISK
        horizon = TimeHorizon.NEAR_TERM
        is_prospective = True
    elif any(k in event_type_lower for k in ("deadline_approaching", "timeout_warning")):
        condition_type = FutureConditionType.DEADLINE_RISK
        horizon = TimeHorizon.IMMEDIATE
        is_prospective = True
    elif any(k in event_type_lower for k in ("anomaly", "hazard_detected", "stress")):
        condition_type = FutureConditionType.SAFETY_RISK
        horizon = TimeHorizon.IMMEDIATE
        is_prospective = True

    if not is_prospective:
        return None

    ev_item = EvidenceItem(
        evidence_id=f"ev_bridge_{event.event_id}",
        source_type=EvidenceSourceType.EVENT,
        source_id=event.event_id,
        description=f"Prospective signal from event '{event.event_type}': {payload.get('summary', '')}",
        confidence=0.85 if event.priority >= EventPriority.HIGH else 0.7,
        observed_at=event.timestamp,
        metadata=dict(payload),
    )

    entity_id = payload.get("entity_id") or payload.get("goal_id")
    window = horizon.get_default_window_seconds()

    return Anticipation(
        anticipation_id=f"ant_bridge_{event.event_id}",
        condition_type=condition_type,
        description=f"Anticipated {condition_type.value} prompted by event '{event.event_type}'.",
        target_entity_id=entity_id,
        hypothetical_state={"projected_hazard": True, "source_event": event.event_id},
        horizon=horizon,
        horizon_window_seconds=window,
        confidence=ev_item.confidence,
        relevance=0.8,
        freshness=1.0,
        evidence_items=(ev_item,),
        related_event_ids=(event.event_id,),
        provenance=AnticipationProvenance(
            source_entity="event_to_anticipation_candidate",
            created_at=event.timestamp,
            correlation_id=event.correlation_id,
            depth=event.provenance.depth + 1,
            causation_id=event.event_id,
        ),
        correlation_id=event.correlation_id,
        metadata={"event_type": event.event_type},
    )
