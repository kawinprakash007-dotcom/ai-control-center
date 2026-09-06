from typing import Dict, Set

from core.interfaces.autonomy_interface import EventClassifierInterface
from core.models.autonomy import (
    Event,
    EventCategory,
    EventClassification,
    EventPriority,
)


class DeterministicEventClassifier(EventClassifierInterface):
    """
    Deterministic, model-neutral event classifier for Phase 4.5.
    Inspects event types, sources, and payloads according to explicit rules.
    Never executes models, prompts, or tools.
    Unknown events fail safely with requires_action=False.
    """

    def __init__(self):
        self._seen_types: Set[str] = set()

    def classify(self, event: Event) -> EventClassification:
        event_type_lower = event.event_type.lower()
        payload = event.payload or {}

        # 1. Check explicit payload category override if valid
        explicit_cat = payload.get("category")
        if explicit_cat and isinstance(explicit_cat, str):
            try:
                category = EventCategory(explicit_cat.upper())
                is_novel = event.event_type not in self._seen_types
                self._seen_types.add(event.event_type)
                severity = self._derive_severity(category, event.priority)
                requires_action = self._determine_requires_action(category, event.priority)
                return EventClassification(
                    category=category,
                    priority=event.priority,
                    severity=severity,
                    is_novel=is_novel,
                    requires_action=requires_action,
                    rationale=f"Classified via explicit payload category: {category.value}",
                )
            except ValueError:
                pass

        # 2. Pattern-based deterministic categorization
        category = EventCategory.UNKNOWN
        if any(k in event_type_lower for k in ("safety", "hazard", "threat", "danger", "alarm")):
            category = EventCategory.SAFETY_ALERT
        elif any(k in event_type_lower for k in ("threshold", "breach", "limit", "exceed", "critical_low", "battery_low")):
            category = EventCategory.THRESHOLD_BREACH
        elif any(k in event_type_lower for k in ("fail", "failed", "failure", "error", "aborted", "crash", "fault")):
            category = EventCategory.FAILURE
        elif any(k in event_type_lower for k in ("timeout", "expired", "stale", "deadline")):
            category = EventCategory.TIMEOUT
        elif any(k in event_type_lower for k in ("progress", "objective_completed", "goal_completed", "step_done")):
            category = EventCategory.GOAL_PROGRESS
        elif any(k in event_type_lower for k in ("changed", "state_change", "transition", "condition", "updated")):
            category = EventCategory.STATE_CHANGE
        elif any(k in event_type_lower for k in ("external", "sensor_signal", "signal", "ping", "user_input")):
            category = EventCategory.EXTERNAL_SIGNAL

        # 3. Fail-safe handling of UNKNOWN events
        if category == EventCategory.UNKNOWN:
            return EventClassification(
                category=EventCategory.UNKNOWN,
                priority=event.priority,
                severity="low",
                is_novel=event.event_type not in self._seen_types,
                requires_action=False,
                rationale=f"Event type '{event.event_type}' is unknown; failing safe with no autonomous action required.",
            )

        is_novel = event.event_type not in self._seen_types
        self._seen_types.add(event.event_type)

        severity = self._derive_severity(category, event.priority)
        requires_action = self._determine_requires_action(category, event.priority)

        return EventClassification(
            category=category,
            priority=event.priority,
            severity=severity,
            is_novel=is_novel,
            requires_action=requires_action,
            rationale=f"Classified deterministically as {category.value} with severity {severity}.",
        )

    def _derive_severity(self, category: EventCategory, priority: EventPriority) -> str:
        if priority == EventPriority.CRITICAL or category == EventCategory.SAFETY_ALERT:
            return "critical"
        elif priority == EventPriority.HIGH or category in (EventCategory.THRESHOLD_BREACH, EventCategory.FAILURE):
            return "high"
        elif priority == EventPriority.NORMAL:
            return "medium"
        return "low"

    def _determine_requires_action(self, category: EventCategory, priority: EventPriority) -> bool:
        if category in (EventCategory.SAFETY_ALERT, EventCategory.THRESHOLD_BREACH, EventCategory.FAILURE):
            return True
        if priority >= EventPriority.HIGH:
            return True
        if category == EventCategory.GOAL_PROGRESS:
            return False
        if priority == EventPriority.LOW:
            return False
        return True
