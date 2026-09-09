"""ATLAS Phase 6.5c — Spatial Processing Subsystem.

Provides model-neutral, deterministic geographic algorithms for:
- Coordinate validation (WGS-84 constraints, NaN/Inf rejection)
- Great-circle distance calculations via Haversine formula
- Initial forward azimuth / bearing calculations in [0, 360)
- Relative position evaluation and spatial relation categorization
- Lightweight geofence boundary evaluation (circular and box)
- Conversion into structured PerceptionEvidence domain models

CRITICAL RULES:
1. Pure math only: zero external GIS frameworks or hardware GPS drivers.
2. Perception evidence only: does NOT make operational or tactical decisions.
3. Fully deterministic: coincident points and edge angles are handled deterministically.
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from core.models.orchestration import GeoLocation, ModalityType
from core.models.perception import (
    PerceptionEvidence,
    SpatialEvidence,
)
from core.models.spatial_telemetry import (
    GeofenceArea,
    GeofenceRelationType,
    PositionObservation,
    RelativePosition,
    SpatialObservation,
    SpatialRelationType,
    SpatialTelemetryLimits,
)


def validate_coordinates(
    latitude: Any,
    longitude: Any,
    altitude: Optional[Any] = None,
    accuracy: Optional[Any] = None,
) -> Tuple[bool, Optional[str]]:
    """
    Validate WGS-84 coordinates, altitude, and horizontal accuracy.
    Returns (is_valid, error_message).
    """
    if latitude is None or longitude is None:
        return False, "Coordinates cannot be None"

    try:
        lat = float(latitude)
        lon = float(longitude)
    except (ValueError, TypeError):
        return False, "Coordinates must be numeric"

    if math.isnan(lat) or math.isinf(lat):
        return False, "Latitude cannot be NaN or Inf"
    if math.isnan(lon) or math.isinf(lon):
        return False, "Longitude cannot be NaN or Inf"

    if not (-90.0 <= lat <= 90.0):
        return False, f"Latitude out of bounds [-90.0, 90.0]: {lat}"
    if not (-180.0 <= lon <= 180.0):
        return False, f"Longitude out of bounds [-180.0, 180.0]: {lon}"

    if altitude is not None:
        try:
            alt = float(altitude)
            if math.isnan(alt) or math.isinf(alt):
                return False, "Altitude cannot be NaN or Inf"
            if not (-1000.0 <= alt <= 100000.0):
                return False, f"Altitude out of bounds [-1000.0, 100000.0]: {alt}"
        except (ValueError, TypeError):
            return False, "Altitude must be numeric"

    if accuracy is not None:
        try:
            acc = float(accuracy)
            if math.isnan(acc) or math.isinf(acc) or acc < 0.0:
                return False, "Accuracy must be finite and non-negative"
        except (ValueError, TypeError):
            return False, "Accuracy must be numeric"

    return True, None


def calculate_distance(
    p1_or_lat1: Union[GeoLocation, float],
    p2_or_lon1: Union[GeoLocation, float],
    lat2: Optional[float] = None,
    lon2: Optional[float] = None,
) -> float:
    """
    Calculate great-circle distance in meters between two points using the Haversine formula.
    Supports either calculate_distance(geo1, geo2) or calculate_distance(lat1, lon1, lat2, lon2).
    Returns 0.0 for coincident locations.
    """
    if isinstance(p1_or_lat1, GeoLocation):
        lat1 = p1_or_lat1.latitude
        lon1 = p1_or_lat1.longitude
        if not isinstance(p2_or_lon1, GeoLocation):
            raise TypeError("Second argument must be GeoLocation when first is GeoLocation")
        lat2_val = p2_or_lon1.latitude
        lon2_val = p2_or_lon1.longitude
    else:
        lat1 = float(p1_or_lat1)
        lon1 = float(p2_or_lon1)
        if lat2 is None or lon2 is None:
            raise ValueError("lat2 and lon2 must be provided when passing coordinates as floats")
        lat2_val = float(lat2)
        lon2_val = float(lon2)

    # Coincident coordinates fast-path
    if lat1 == lat2_val and lon1 == lon2_val:
        return 0.0

    r = 6371000.0  # Earth mean radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2_val)
    delta_phi = math.radians(lat2_val - lat1)
    delta_lambda = math.radians(lon2_val - lon1)

    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return round(r * c, 2)


def calculate_bearing(
    p1_or_lat1: Union[GeoLocation, float],
    p2_or_lon1: Union[GeoLocation, float],
    lat2: Optional[float] = None,
    lon2: Optional[float] = None,
) -> float:
    """
    Calculate initial forward azimuth / bearing from point 1 to point 2 in degrees [0.0, 360.0).
    Supports either calculate_bearing(geo1, geo2) or calculate_bearing(lat1, lon1, lat2, lon2).
    For coincident points, returns 0.0 deterministically.
    """
    if isinstance(p1_or_lat1, GeoLocation):
        lat1 = p1_or_lat1.latitude
        lon1 = p1_or_lat1.longitude
        if not isinstance(p2_or_lon1, GeoLocation):
            raise TypeError("Second argument must be GeoLocation when first is GeoLocation")
        lat2_val = p2_or_lon1.latitude
        lon2_val = p2_or_lon1.longitude
    else:
        lat1 = float(p1_or_lat1)
        lon1 = float(p2_or_lon1)
        if lat2 is None or lon2 is None:
            raise ValueError("lat2 and lon2 must be provided when passing coordinates as floats")
        lat2_val = float(lat2)
        lon2_val = float(lon2)

    if lat1 == lat2_val and lon1 == lon2_val:
        return 0.0

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2_val)
    delta_lambda = math.radians(lon2_val - lon1)

    y = math.sin(delta_lambda) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(delta_lambda)

    theta = math.atan2(y, x)
    bearing = (math.degrees(theta) + 360.0) % 360.0
    return round(bearing, 2)


def evaluate_relative_position(
    source_lat: Optional[float] = None,
    source_lon: Optional[float] = None,
    target_lat: Optional[float] = None,
    target_lon: Optional[float] = None,
    target_id: str = "target",
    reference_id: str = "self",
    source_heading: Optional[float] = None,
    reference_location: Optional[GeoLocation] = None,
    target_location: Optional[GeoLocation] = None,
    **kwargs: Any,
) -> RelativePosition:
    """
    Compute relative spatial metrics and categorize spatial relationship.
    Categorization rules:
    - < 5m: SAME_LOCATION
    - 5m to 50m: NEARBY
    - 50m to 500m: VICINITY
    - 500m to 5000m: DISTANT
    - >= 5000m: REMOTE
    """
    if reference_location is not None and target_location is not None:
        s_lat, s_lon = reference_location.latitude, reference_location.longitude
        t_lat, t_lon = target_location.latitude, target_location.longitude
    else:
        s_lat = float(source_lat if source_lat is not None else 0.0)
        s_lon = float(source_lon if source_lon is not None else 0.0)
        t_lat = float(target_lat if target_lat is not None else 0.0)
        t_lon = float(target_lon if target_lon is not None else 0.0)

    dist = calculate_distance(s_lat, s_lon, t_lat, t_lon)
    bearing = calculate_bearing(s_lat, s_lon, t_lat, t_lon)

    if dist < 5.0:
        relation = SpatialRelationType.SAME_LOCATION
    elif dist < 50.0:
        relation = SpatialRelationType.NEARBY
    elif dist < 500.0:
        relation = SpatialRelationType.VICINITY
    elif dist < 5000.0:
        relation = SpatialRelationType.DISTANT
    else:
        relation = SpatialRelationType.REMOTE

    return RelativePosition(
        reference_entity_id=reference_id,
        target_entity_id=target_id,
        distance=dist,
        bearing=bearing,
        relation=relation,
        target_id=target_id,
        reference_id=reference_id,
        distance_meters=dist,
        bearing_degrees=bearing,
    )


class SpatialProcessor:
    """
    Stateful or stateless spatial observation processor.
    Normalizes coordinates, evaluates relative trajectories, and checks geofences.
    """

    def __init__(
        self,
        limits: Optional[SpatialTelemetryLimits] = None,
        max_relations: Optional[int] = None,
        max_geofences: Optional[int] = None,
    ) -> None:
        base = limits or SpatialTelemetryLimits()
        if max_relations is not None or max_geofences is not None:
            self.limits = SpatialTelemetryLimits(
                max_spatial_relations=max_relations if max_relations is not None else base.max_spatial_relations,
                max_geofences=max_geofences if max_geofences is not None else base.max_geofences,
                max_metrics_per_result=base.max_metrics_per_result,
                max_attributes_per_metric=base.max_attributes_per_metric,
                max_tracked_sources=base.max_tracked_sources,
                max_history_entries=base.max_history_entries,
                max_staleness_seconds=base.max_staleness_seconds,
                max_batch_size=base.max_batch_size,
                max_string_length=base.max_string_length,
                max_route_ref_length=base.max_route_ref_length,
            )
        else:
            self.limits = base

        self._geofences: Dict[str, GeofenceArea] = {}

    def create_position(
        self,
        lat: float,
        lon: float,
        altitude: Optional[float] = None,
        accuracy: Optional[float] = None,
        heading: Optional[float] = None,
        speed: Optional[float] = None,
        source_id: str = "default",
    ) -> PositionObservation:
        """Helper to create and validate a PositionObservation."""
        valid, err = validate_coordinates(lat, lon, altitude=altitude, accuracy=accuracy)
        if not valid:
            raise ValueError(err)

        return PositionObservation(
            latitude=lat,
            longitude=lon,
            altitude=altitude,
            accuracy=accuracy,
            heading=heading,
            speed=speed,
            source_id=source_id,
        )

    def evaluate_geofence(
        self,
        lat: float,
        lon: float,
        geofence: GeofenceArea,
    ) -> Tuple[GeofenceRelationType, float]:
        """Evaluate a single coordinate against a geofence."""
        rel = geofence.evaluate((lat, lon))
        dist = 0.0
        if geofence.center_latitude is not None and geofence.center_longitude is not None:
            dist = calculate_distance(lat, lon, geofence.center_latitude, geofence.center_longitude)
        elif geofence.center is not None:
            dist = calculate_distance(lat, lon, geofence.center.latitude, geofence.center.longitude)
        return rel, dist

    def evaluate_geofences(
        self,
        lat: float,
        lon: float,
        geofences: Optional[Sequence[GeofenceArea]] = None,
    ) -> Dict[str, GeofenceRelationType]:
        """Evaluate a coordinate against multiple geofences."""
        target_fences = geofences or list(self._geofences.values())
        results: Dict[str, GeofenceRelationType] = {}
        for gf in target_fences[:self.limits.max_geofences]:
            gid = gf.geofence_id or gf.name
            rel, _ = self.evaluate_geofence(lat, lon, gf)
            results[gid] = rel
        return results

    def evaluate_multiple_relative_positions(
        self,
        source_lat: float,
        source_lon: float,
        targets: Sequence[Dict[str, Any]],
        source_id: str = "self",
    ) -> List[RelativePosition]:
        """Evaluate relative positions for multiple targets, bounded by max_spatial_relations."""
        results: List[RelativePosition] = []
        for t in targets[:self.limits.max_spatial_relations]:
            t_id = t.get("target_id", "target")
            t_lat = float(t.get("latitude", 0.0))
            t_lon = float(t.get("longitude", 0.0))
            rel = evaluate_relative_position(
                source_lat=source_lat,
                source_lon=source_lon,
                target_lat=t_lat,
                target_lon=t_lon,
                target_id=t_id,
                reference_id=source_id,
            )
            results.append(rel)
        return results

    def process_position(
        self,
        source_id: str,
        raw_payload: Dict[str, Any],
        captured_at: float,
        provenance: Dict[str, Any],
        request_id: str,
        input_id: str,
        correlation_id: str = "",
        causation_id: Optional[str] = None,
    ) -> Tuple[PositionObservation, List[PerceptionEvidence]]:
        """Normalize raw position payload and produce PositionObservation and PerceptionEvidence."""
        lat_raw = raw_payload.get("latitude", raw_payload.get("lat"))
        lon_raw = raw_payload.get("longitude", raw_payload.get("lon", raw_payload.get("lng")))

        valid, err = validate_coordinates(lat_raw, lon_raw)
        if not valid:
            raise ValueError(f"Invalid coordinates: {err}")

        lat = float(lat_raw)
        lon = float(lon_raw)
        alt_raw = raw_payload.get("altitude", raw_payload.get("alt"))
        acc_raw = raw_payload.get("accuracy", raw_payload.get("horizontal_accuracy"))
        vacc_raw = raw_payload.get("vertical_accuracy")
        hdg_raw = raw_payload.get("heading")
        spd_raw = raw_payload.get("speed")

        pos_obs = PositionObservation(
            source_id=source_id,
            latitude=lat,
            longitude=lon,
            altitude=float(alt_raw) if alt_raw is not None else None,
            accuracy=float(acc_raw) if acc_raw is not None else None,
            vertical_accuracy=float(vacc_raw) if vacc_raw is not None else None,
            heading=float(hdg_raw) if hdg_raw is not None else None,
            speed=float(spd_raw) if spd_raw is not None else None,
            captured_at=captured_at,
            observed_at=captured_at,
            provenance=provenance,
            correlation_id=correlation_id or request_id,
            causation_id=causation_id or input_id,
            metadata=dict(raw_payload.get("metadata", {})),
        )

        geo = pos_obs.to_geo_location()

        # Build primary GPS perception evidence
        spatial_evidence = SpatialEvidence(
            point=(lat, lon),
            location=geo,
            spatial_confidence=1.0,
        )

        ev_attributes: Dict[str, Any] = {
            "latitude": lat,
            "longitude": lon,
        }
        if pos_obs.altitude is not None:
            ev_attributes["altitude"] = pos_obs.altitude
        if pos_obs.accuracy is not None:
            ev_attributes["accuracy"] = pos_obs.accuracy
        if pos_obs.heading is not None:
            ev_attributes["heading"] = pos_obs.heading
        if pos_obs.speed is not None:
            ev_attributes["speed"] = pos_obs.speed

        ev = PerceptionEvidence(
            evidence_id=f"ev_pos_{request_id}_{source_id}",
            semantic_type="spatial_position",
            label="gps_position",
            confidence=1.0,
            source_id=source_id,
            modality=ModalityType.GPS,
            timestamp=captured_at,
            spatial=spatial_evidence,
            attributes=ev_attributes,
            provenance=provenance,
            correlation_id=correlation_id or request_id,
            causation_id=causation_id or input_id,
        )

        return pos_obs, [ev]
