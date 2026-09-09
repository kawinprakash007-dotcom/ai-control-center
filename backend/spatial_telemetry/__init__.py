"""ATLAS Spatial and Telemetry Perception Module.

Phase 6.5c: Semantic spatial normalization, relative positioning, geofencing,
and telemetry ingestion/quality assessment for ATLAS Central.
"""

from backend.core.models.spatial_telemetry import (
    GeofenceArea,
    GeofenceRelationType,
    MovementState,
    NavigationObservation,
    PositionObservation,
    RelativePosition,
    SpatialObservation,
    SpatialRelationType,
    SpatialTelemetryLimits,
    TelemetryHealth,
    TelemetryMetric,
    TelemetryObservation,
    TelemetryQuality,
)
from backend.spatial_telemetry.provider import SpatialTelemetryPerceptionProvider
from backend.spatial_telemetry.reference_provider import (
    MockSpatialTelemetryProvider,
    ReferenceSpatialTelemetryProvider,
)
from backend.spatial_telemetry.spatial import (
    SpatialProcessor,
    calculate_bearing,
    calculate_distance,
    evaluate_relative_position,
    validate_coordinates,
)
from backend.spatial_telemetry.telemetry import (
    TelemetryProcessor,
    evaluate_freshness,
    normalize_metric,
    process_telemetry,
)

__all__ = [
    # Models & Enums
    "SpatialTelemetryLimits",
    "TelemetryQuality",
    "SpatialRelationType",
    "GeofenceRelationType",
    "MovementState",
    "TelemetryMetric",
    "TelemetryHealth",
    "TelemetryObservation",
    "PositionObservation",
    "RelativePosition",
    "NavigationObservation",
    "GeofenceArea",
    "SpatialObservation",
    # Spatial functions & processor
    "validate_coordinates",
    "calculate_distance",
    "calculate_bearing",
    "evaluate_relative_position",
    "SpatialProcessor",
    # Telemetry functions & processor
    "evaluate_freshness",
    "normalize_metric",
    "process_telemetry",
    "TelemetryProcessor",
    # Providers
    "SpatialTelemetryPerceptionProvider",
    "MockSpatialTelemetryProvider",
    "ReferenceSpatialTelemetryProvider",
]
