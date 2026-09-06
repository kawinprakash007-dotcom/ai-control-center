from typing import Dict, List, Tuple, Optional, Set, Any
import hashlib

from core.models.context import (
    ContextItem,
    ContextSource,
    ContextPriority,
    AttentionFocus,
    ContextBudget,
    SensitivityLevel,
)


# Priority adjustment multipliers based on AttentionFocus
FOCUS_SOURCE_MULTIPLIERS: Dict[AttentionFocus, Dict[ContextSource, float]] = {
    AttentionFocus.GOAL: {
        ContextSource.CURRENT_GOAL: 1.3,
        ContextSource.CURRENT_TASK: 1.2,
        ContextSource.RECOVERY: 1.1,
    },
    AttentionFocus.PLANNING: {
        ContextSource.CURRENT_GOAL: 1.2,
        ContextSource.CURRENT_TASK: 1.2,
        ContextSource.CAPABILITY_STATE: 1.15,
        ContextSource.MEMORY: 1.1,
        ContextSource.KNOWLEDGE: 1.1,
    },
    AttentionFocus.EXECUTION: {
        ContextSource.CURRENT_TASK: 1.3,
        ContextSource.CAPABILITY_STATE: 1.2,
        ContextSource.EXECUTION_RESULT: 1.15,
        ContextSource.VISUAL_SCENE: 1.15,
    },
    AttentionFocus.VERIFICATION: {
        ContextSource.VERIFICATION: 1.4,
        ContextSource.EXECUTION_RESULT: 1.3,
        ContextSource.CURRENT_TASK: 1.1,
    },
    AttentionFocus.RECOVERY: {
        ContextSource.RECOVERY: 1.4,
        ContextSource.VERIFICATION: 1.3,
        ContextSource.CURRENT_GOAL: 1.25,
        ContextSource.CURRENT_TASK: 1.2,
        ContextSource.EXECUTION_RESULT: 1.15,
    },
    AttentionFocus.VISUAL_TARGET: {
        ContextSource.VISUAL_SCENE: 1.4,
        ContextSource.CURRENT_TASK: 1.2,
        ContextSource.CAPABILITY_STATE: 1.1,
    },
    AttentionFocus.RESEARCH: {
        ContextSource.WEB_EVIDENCE: 1.35,
        ContextSource.RESEARCH: 1.35,
        ContextSource.CURRENT_TASK: 1.2,
        ContextSource.KNOWLEDGE: 1.1,
    },
    AttentionFocus.MEMORY_RECALL: {
        ContextSource.MEMORY: 1.4,
        ContextSource.RECENT_CONVERSATION: 1.25,
        ContextSource.CURRENT_GOAL: 1.15,
    },
}


def apply_attention_focus_weights(
    item: ContextItem,
    focus: AttentionFocus,
) -> float:
    """
    Compute an adjusted composite ranking score incorporating priority,
    relevance, confidence, and attention focus multiplier.
    """
    multipliers = FOCUS_SOURCE_MULTIPLIERS.get(focus, {})
    multiplier = multipliers.get(item.source, 1.0)

    # Base priority weight (1.0 to 4.0)
    priority_weight = float(item.priority)

    # Composite score formula:
    # 40% priority + 40% relevance + 20% confidence, scaled by focus multiplier
    composite = (
        (0.40 * (priority_weight / 4.0)) +
        (0.40 * item.relevance) +
        (0.20 * item.confidence)
    ) * multiplier

    # Protected items receive an unbreachable ranking floor
    if item.is_protected:
        composite += 10.0

    return round(composite, 5)


def deduplicate_items_deterministic(items: List[ContextItem]) -> List[ContextItem]:
    """
    Deduplicate items with identical or strictly overlapping content.
    Preserves items with higher priority or higher confidence.
    For web evidence, items with distinct URLs or citation IDs are preserved independently.
    """
    deduped: List[ContextItem] = []
    seen_content_hashes: Dict[str, int] = {}  # hash -> index in deduped

    for item in items:
        # Check if item is independent evidence (distinguished by url / citation id)
        evidence_key = item.metadata.get("url") or item.metadata.get("id") or item.metadata.get("evidence_id")
        if evidence_key and item.source in (ContextSource.WEB_EVIDENCE, ContextSource.RESEARCH):
            content_key = f"evidence:{item.source.value}:{evidence_key}"
        else:
            # Normalized content hash for general text
            normalized = " ".join(item.content.strip().lower().split())
            content_key = hashlib.sha256(normalized.encode("utf-8")).hexdigest()

        if content_key in seen_content_hashes:
            existing_idx = seen_content_hashes[content_key]
            existing_item = deduped[existing_idx]

            # Compare to choose which to retain
            # Keep protected item if either is protected
            is_prot = existing_item.is_protected or item.is_protected
            # Choose item with higher priority, then higher confidence, then higher relevance
            should_replace = (
                (item.priority > existing_item.priority) or
                (item.priority == existing_item.priority and item.confidence > existing_item.confidence) or
                (item.priority == existing_item.priority and item.confidence == existing_item.confidence and item.relevance > existing_item.relevance)
            )

            if should_replace:
                # Merge metadata from both
                merged_meta = dict(existing_item.metadata)
                merged_meta.update(item.metadata)
                merged_meta["deduplicated_sources"] = list(set(
                    merged_meta.get("deduplicated_sources", [existing_item.source.value]) + [item.source.value]
                ))
                deduped[existing_idx] = ContextItem(
                    item_id=item.item_id,
                    source=item.source,
                    content=item.content,
                    relevance=max(item.relevance, existing_item.relevance),
                    priority=item.priority,
                    recency=max(item.recency, existing_item.recency),
                    confidence=max(item.confidence, existing_item.confidence),
                    timestamp=item.timestamp or existing_item.timestamp,
                    sensitivity=item.sensitivity if item.is_sensitive else existing_item.sensitivity,
                    token_estimate=item.token_estimate,
                    is_protected=is_prot,
                    metadata=merged_meta,
                )
            else:
                # Update existing item metadata and protection
                merged_meta = dict(existing_item.metadata)
                merged_meta.update(item.metadata)
                merged_meta["deduplicated_sources"] = list(set(
                    merged_meta.get("deduplicated_sources", [existing_item.source.value]) + [item.source.value]
                ))
                if is_prot and not existing_item.is_protected:
                    deduped[existing_idx] = ContextItem(
                        item_id=existing_item.item_id,
                        source=existing_item.source,
                        content=existing_item.content,
                        relevance=existing_item.relevance,
                        priority=existing_item.priority,
                        recency=existing_item.recency,
                        confidence=existing_item.confidence,
                        timestamp=existing_item.timestamp,
                        sensitivity=existing_item.sensitivity,
                        token_estimate=existing_item.token_estimate,
                        is_protected=True,
                        metadata=merged_meta,
                    )
        else:
            seen_content_hashes[content_key] = len(deduped)
            deduped.append(item)

    return deduped


def compact_and_budget_items(
    ranked_items: List[ContextItem],
    budget: ContextBudget,
) -> Tuple[List[ContextItem], List[ContextItem]]:
    """
    Enforce budget boundaries deterministically:
    1. Truncate oversized individual items to budget.max_content_length_per_item.
    2. Preserve all protected items.
    3. Allocate remaining budget (items limit and token limit) to top-ranked items.
    4. Return (selected_items, omitted_items).
    """
    selected: List[ContextItem] = []
    omitted: List[ContextItem] = []

    # 1. Compact individual oversized items
    compacted_items: List[ContextItem] = [
        item.compact(budget.max_content_length_per_item)
        for item in ranked_items
    ]

    # 2. Extract protected items first
    protected_items = [it for it in compacted_items if it.is_protected]
    candidate_pool = [it for it in compacted_items if not it.is_protected]

    current_tokens = sum(it.token_estimate for it in protected_items)
    selected.extend(protected_items)

    # 3. Allocate budget to candidate items
    for item in candidate_pool:
        # Check item count boundary
        if len(selected) >= budget.max_items:
            omitted.append(item)
            continue

        # Check token boundary
        if (current_tokens + item.token_estimate) > budget.max_tokens:
            omitted.append(item)
            continue

        selected.append(item)
        current_tokens += item.token_estimate

    return selected, omitted
