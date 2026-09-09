"""ATLAS Phase 6.5d — Spatial Correlation Subsystem.

Bridges spatial observations into cross-modal evidence links.
Reuses geodesy algorithms from Phase 6.5c (spatial_telemetry.spatial) without duplication.
"""

from __future__ import annotations

import math
from typing import Optional, Union

from core.models.multimodal_fusion import SpatialEvidenceLink
from core.models.orchestration import GeoLocation
from core.models.spatial_telemetry import SpatialRelationType
from spatial_telemetry.spatial import (
    calculate_bearing,
    calculate_distance,
    evaluate_relative_position,
)


def correlate_spatial_evidence(
    source_id_1: str,
    source_id_2: str,
    location_1: Optional[GeoLocation],
    location_2: Optional[GeoLocation],
    max_distance_meters: float = 500.0,
) -> Optional[SpatialEvidenceLink]:
    """
    Evaluate spatial correlation between two observed entities or positions.
    Reuses Phase 6.5c Haversine and relative positioning functions.

    Returns:
        SpatialEvidenceLink if both locations are valid and within max_distance_meters,
        None otherwise.
    """
    if location_1 is None or location_2 is None:
        return None

    # Calculate great-circle distance reusing Phase 6.5c Haversine implementation
    dist = calculate_distance(location_1, location_2)
    if dist > max_distance_meters:
        return None

    # Evaluate relative position category reusing Phase 6.5c
    rel_pos = evaluate_relative_position(
        source_lat=location_1.latitude,
        source_lon=location_1.longitude,
        target_lat=location_2.latitude,
        target_lon=location_2.longitude,
        target_id=source_id_2,
        reference_id=source_id_1,
    )

    # Compute deterministic spatial confidence based on distance
    if dist < 5.0:
        conf = 1.0
    elif dist < 50.0:
        conf = 0.95
    elif dist < 200.0:
        conf = 0.85
    else:
        conf = max(0.1, round(1.0 - (dist / max_distance_meters), 3))

    return SpatialEvidenceLink(
        source_id_1=source_id_1,
        source_id_2=source_id_2,
        distance_meters=dist,
        spatial_relation=rel_pos.relation,
        spatial_confidence=conf,
        bearing_degrees=round(rel_pos.bearing_degrees, 2) if rel_pos else None,
        location_1=location_1,
        location_2=location_2,
    )
