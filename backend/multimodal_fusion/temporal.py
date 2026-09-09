"""ATLAS Phase 6.5d — Temporal Alignment Subsystem.

Provides deterministic temporal evaluation, relation categorization, and windowing.
Decoupled completely from wall-clock time; caller or simulation provides all timestamps.
"""

from __future__ import annotations

import math
from typing import Any, Callable, List, Optional, Sequence, Tuple, TypeVar

from core.models.multimodal_fusion import (
    FusionLimits,
    TemporalRelation,
    TemporalWindow,
)

T = TypeVar("T")


def evaluate_temporal_relation(
    t1: Union[float, Tuple[float, float]],
    t2: Union[float, Tuple[float, float]],
    duration1: float = 0.0,
    duration2: float = 0.0,
    tolerance_seconds: float = 0.1,
    coincidence_tolerance: Optional[float] = None,
    sequence_tolerance: float = 0.0,
) -> TemporalRelation:
    """
    Deterministically evaluate the temporal relation between two points or intervals in time.

    Relations:
    - COINCIDENT: Absolute difference is within tolerance.
    - OVERLAPS: For non-zero durations, intervals intersect without being coincident.
    - BEFORE: Point/interval 1 strictly precedes point/interval 2.
    - AFTER: Point/interval 1 strictly succeeds point/interval 2.
    - SEQUENCE: Consecutive ordered timestamps from the same source within sequence tolerance.
    - WITHIN_WINDOW: Point or interval 1 is strictly within interval 2.
    """
    tol = max(0.0, float(coincidence_tolerance if coincidence_tolerance is not None else tolerance_seconds))

    if isinstance(t1, (tuple, list)):
        start1, end1 = float(t1[0]), float(t1[1])
    else:
        start1 = float(t1)
        end1 = start1 + max(0.0, float(duration1))

    if isinstance(t2, (tuple, list)):
        start2, end2 = float(t2[0]), float(t2[1])
    else:
        start2 = float(t2)
        end2 = start2 + max(0.0, float(duration2))

    # Point or Interval Coincidence
    if abs(start1 - start2) <= tol and abs(end1 - end2) <= tol:
        return TemporalRelation.COINCIDENT

    # Strictly within window check (t1 is inside t2)
    if start1 >= (start2 - tol) and end1 <= (end2 + tol) and (start2 != start1 or end2 != end1):
        return TemporalRelation.WITHIN_WINDOW

    # Sequence check (before, but within sequence tolerance)
    if sequence_tolerance > 0.0 and end1 <= start2 and (start2 - end1) <= sequence_tolerance:
        return TemporalRelation.SEQUENCE

    # Before / After
    if end1 < start2 - tol:
        return TemporalRelation.BEFORE

    if start1 > end2 + tol:
        return TemporalRelation.AFTER

    # Overlapping intervals
    if max(start1, start2) <= min(end1, end2) + tol:
        return TemporalRelation.OVERLAPS

    return TemporalRelation.WITHIN_WINDOW


def build_temporal_window(
    timestamps: Sequence[float],
    max_duration: float = 300.0,
    reference_timestamp: Optional[float] = None,
) -> TemporalWindow:
    """
    Construct a bounded TemporalWindow from a sequence of observation timestamps.
    Clips span to max_duration if timestamps exceed bounds, or falls back to reference_timestamp if empty.
    """
    valid_ts = [float(t) for t in timestamps if not (math.isnan(t) or math.isinf(t))] if timestamps else []
    if not valid_ts:
        ref = float(reference_timestamp) if reference_timestamp is not None else 0.0
        return TemporalWindow(
            start_time=ref,
            end_time=ref,
            max_duration=max_duration,
            source_timestamps=(),
            reference_timestamp=reference_timestamp,
        )

    min_t = min(valid_ts)
    max_t = max(valid_ts)

    if (max_t - min_t) > max_duration:
        min_t = max_t - max_duration

    return TemporalWindow(
        start_time=round(min_t, 4),
        end_time=round(max_t, 4),
        max_duration=max_duration,
        source_timestamps=tuple(sorted(valid_ts)),
        reference_timestamp=reference_timestamp if reference_timestamp is not None else max_t,
    )


def filter_by_temporal_window(
    items: Sequence[T],
    window: TemporalWindow,
    get_timestamp: Optional[Callable[[T], float]] = None,
    tolerance: float = 0.0,
) -> List[T]:
    """
    Filter observations or evidence items to those falling strictly within the TemporalWindow.
    Does NOT silently expand the window.
    """
    retained: List[T] = []
    for item in items:
        if get_timestamp is not None:
            ts = get_timestamp(item)
        elif isinstance(item, dict):
            ts = float(item.get("timestamp", 0.0))
        else:
            ts = float(getattr(item, "timestamp", 0.0))
        if window.contains(ts, tolerance=tolerance):
            retained.append(item)
    return retained
