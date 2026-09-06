import time
from typing import List, Optional

from core.models.context import (
    ContextItem,
    ContextPriority,
    ContextSource,
    SensitivityLevel,
)
from core.models.world_state import (
    FreshnessConfig,
    FreshnessStatus,
    WorldCondition,
    WorldState,
)


def world_condition_to_context_item(
    condition: WorldCondition,
    freshness: FreshnessStatus,
    relevance: float = 0.8,
    priority: ContextPriority = ContextPriority.HIGH,
) -> ContextItem:
    """
    Deterministically convert a WorldCondition into an immutable ContextItem for ContextManager.
    Guarantees provenance preservation and freshness visibility.
    """
    content = (
        f"World Belief: {condition.entity_id}.{condition.property_name} = {condition.value} "
        f"[confidence={condition.confidence:.2f}, freshness={freshness.value}, "
        f"source={condition.provenance.source_type}:{condition.provenance.source_id}]"
    )

    return ContextItem(
        item_id=f"ctx_ws_{condition.entity_id}_{condition.property_name}",
        source=ContextSource.WORLD_STATE,
        content=content,
        relevance=relevance,
        priority=priority,
        recency=condition.observed_at,
        confidence=condition.confidence,
        timestamp=condition.observed_at,
        sensitivity=SensitivityLevel.PUBLIC,
        metadata={
            "entity_id": condition.entity_id,
            "property_name": condition.property_name,
            "freshness": freshness.value,
            "provenance": condition.provenance.to_dict(),
        },
    )


def world_state_to_context_items(
    world_state: WorldState,
    freshness_config: Optional[FreshnessConfig] = None,
    now: Optional[float] = None,
    include_expired: bool = False,
) -> List[ContextItem]:
    """
    Extract active world conditions as candidate ContextItems for the immediate cognitive turn.
    By default, excludes EXPIRED conditions to prevent stale beliefs from masquerading as current truth.
    """
    effective_config = freshness_config or FreshnessConfig()
    current_time = now if now is not None else time.time()
    items: List[ContextItem] = []

    for cond in world_state.conditions:
        freshness = effective_config.evaluate(
            observed_at=cond.observed_at,
            expires_at=cond.expires_at,
            now=current_time,
        )

        if freshness == FreshnessStatus.EXPIRED and not include_expired:
            continue

        priority = ContextPriority.HIGH if freshness == FreshnessStatus.FRESH else ContextPriority.MEDIUM
        item = world_condition_to_context_item(cond, freshness=freshness, priority=priority)
        items.append(item)

    return items
