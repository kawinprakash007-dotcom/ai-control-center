"""Phase 6.5c — Spatial & Telemetry Perception Test Suite.

Comprehensive tests validating:
1. Spatial domain models, bounds, validation, and immutability (A-I).
2. Spherical geodesy: Haversine distance, coincident points, bearing, relative positioning (J-O).
3. Geofencing: circular, bounding box, proximity buffers (P, BK).
4. Telemetry domain models, battery bounds, connectivity, health, units, quality, freshness (Q-Y, BJ).
5. Provider compliance, registry integration, reference/mock deterministic behavior (Z-AB).
6. Perception results for GPS, TELEMETRY, DEVICE_STATE modalities (AC-AE).
7. End-to-end normalization to MultimodalObservation & Gateway ingestion (AF-AK).
8. Multi-product edge support: Vision, Glass, Drone, Rover, and Digital Twin (AL-AP).
9. Robustness, error handling, bounded allocations, determinism, replay, and redaction (AQ-AW, BI).
10. Architectural boundary invariants: zero WorldState/Goal/Mission/Tool/DeviceGateway bypasses,
    zero physical hardware drivers, zero subprocess/shell execution (AX-BH).
"""

import copy
import datetime
import json
import math
import sys
import time
import uuid
from typing import Any, Dict, List, Optional, Sequence
from unittest.mock import MagicMock

# Fallbacks for lightweight test environments
if "chromadb" not in sys.modules:
    sys.modules["chromadb"] = MagicMock()
if "sentence_transformers" not in sys.modules:
    sys.modules["sentence_transformers"] = MagicMock()

import pytest

from core.models.device_contract import (
    ConnectivityStatus,
    DeviceHealthStatus,
    ProductType,
)
from core.models.orchestration import (
    GeoLocation,
    ModalityType,
    MultimodalObservation,
)
from core.models.perception import (
    PerceptionCapability,
    PerceptionError,
    PerceptionEvidence,
    PerceptionInput,
    PerceptionMetadata,
    PerceptionPrivacyClass,
    PerceptionRequest,
    PerceptionResult,
    PerceptionStatus,
    REQUIRED_PROVENANCE_KEYS,
    SpatialEvidence,
)
from core.models.spatial_telemetry import (
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
from core.models.world_state import WorldState
from orchestration.fusion_engine import SituationFusionConfig, SituationFusionEngine
from orchestration.input_gateway import CentralInputGateway
from perception.normalizer import PerceptionObservationNormalizer
from perception.registry import PerceptionProviderRegistry
from spatial_telemetry.provider import SpatialTelemetryPerceptionProvider
from spatial_telemetry.reference_provider import (
    MockSpatialTelemetryProvider,
    ReferenceSpatialTelemetryProvider,
)
from spatial_telemetry.spatial import (
    SpatialProcessor,
    calculate_bearing,
    calculate_distance,
    evaluate_relative_position,
    validate_coordinates,
)
from spatial_telemetry.telemetry import (
    TelemetryProcessor,
    evaluate_freshness,
    normalize_metric,
    process_telemetry,
)


def make_spatial_input(
    modality: ModalityType,
    payload: Dict[str, Any],
    source_id: Optional[str] = None,
    correlation_id: str = "",
    causation_id: Optional[str] = None,
) -> PerceptionInput:
    sid = source_id or payload.get("device_id", "test_source")
    return PerceptionInput(
        input_id=f"inp_{uuid.uuid4().hex[:8]}",
        modality=modality,
        payload_ref="inline://payload",
        captured_at=time.time(),
        source_id=sid,
        correlation_id=correlation_id,
        causation_id=causation_id,
        metadata=dict(payload),
    )


def make_spatial_request(
    modality: ModalityType,
    payload: Dict[str, Any],
    correlation_id: str = "",
    causation_id: Optional[str] = None,
    capabilities: Optional[Sequence[PerceptionCapability]] = None,
) -> PerceptionRequest:
    inp = make_spatial_input(
        modality=modality,
        payload=payload,
        correlation_id=correlation_id,
        causation_id=causation_id,
    )
    if capabilities is not None:
        caps = tuple(capabilities)
    elif modality == ModalityType.GPS:
        caps = (PerceptionCapability.SPATIAL_LOCALIZATION,)
    else:
        caps = (PerceptionCapability.TELEMETRY_NORMALIZATION,)

    return PerceptionRequest(
        request_id=f"req_{uuid.uuid4().hex[:8]}",
        input_data=inp,
        requested_capabilities=caps,
        correlation_id=correlation_id,
        causation_id=causation_id,
    )


# ==============================================================================
# SECTION A-D: Spatial & Telemetry Domain Models, Immutability & Serialization
# ==============================================================================

class TestDomainModelsAndSerialization:
    """Tests A, B, C, D: Models, immutability, and serialization."""

    def test_section_a_spatial_domain_models_instantiation(self):
        pos = PositionObservation(
            latitude=37.7749,
            longitude=-122.4194,
            altitude=15.0,
            accuracy=2.5,
            heading=90.0,
            speed=5.0,
        )
        assert pos.latitude == 37.7749
        assert pos.longitude == -122.4194
        assert pos.altitude == 15.0
        assert pos.accuracy == 2.5
        assert pos.heading == 90.0
        assert pos.speed == 5.0

        nav = NavigationObservation(
            current_position=pos,
            target_latitude=37.7849,
            target_longitude=-122.4094,
            distance_to_target=1420.5,
            bearing_to_target=45.0,
            speed=5.0,
            heading=90.0,
            movement_state=MovementState.MOVING,
        )
        assert nav.distance_to_target == 1420.5
        assert nav.movement_state == MovementState.MOVING

        fence = GeofenceArea(
            geofence_id="fence_hq",
            name="HQ Perimeter",
            center_latitude=37.7749,
            center_longitude=-122.4194,
            radius_meters=100.0,
        )
        assert fence.geofence_id == "fence_hq"
        assert fence.radius_meters == 100.0

        spatial_obs = SpatialObservation(
            position=pos,
            navigation=nav,
            geofences=[fence],
            spatial_relations=[],
        )
        assert spatial_obs.position == pos
        assert len(spatial_obs.geofences) == 1

    def test_section_b_telemetry_domain_models_instantiation(self):
        metric = TelemetryMetric(
            name="battery_voltage",
            value=12.6,
            unit="V",
            raw_name="batt_v",
        )
        assert metric.name == "battery_voltage"
        assert metric.value == 12.6
        assert metric.unit == "V"

        health = TelemetryHealth(
            health_status="HEALTHY",
            connectivity="ONLINE",
            battery_percent=88.5,
            cpu_usage_percent=24.0,
            temperature_celsius=38.2,
        )
        assert health.battery_percent == 88.5
        assert health.health_status == "HEALTHY"

        obs = TelemetryObservation(
            device_id="drone_alpha",
            product_type="drone",
            health=health,
            metrics=[metric],
            quality=TelemetryQuality.VALID,
            age_seconds=1.2,
            is_stale=False,
        )
        assert obs.device_id == "drone_alpha"
        assert obs.quality == TelemetryQuality.VALID
        assert not obs.is_stale

    def test_section_c_immutability_frozen_models(self):
        pos = PositionObservation(latitude=37.7749, longitude=-122.4194)
        with pytest.raises(Exception):
            pos.latitude = 40.0  # Should be frozen/immutable

        metric = TelemetryMetric(name="speed", value=12.0)
        with pytest.raises(Exception):
            metric.value = 15.0

        rel = RelativePosition(
            target_id="target_1",
            distance_meters=100.0,
            bearing_degrees=45.0,
            relation=SpatialRelationType.VICINITY,
        )
        with pytest.raises(Exception):
            rel.distance_meters = 200.0

    def test_section_d_serialization_and_deserialization(self):
        pos = PositionObservation(
            latitude=37.7749,
            longitude=-122.4194,
            altitude=10.0,
            accuracy=1.0,
            heading=180.0,
        )
        data = pos.model_dump()
        json_str = pos.model_dump_json()
        assert "latitude" in data
        assert json.loads(json_str)["latitude"] == 37.7749

        restored = PositionObservation.model_validate(data)
        assert restored == pos

        metric = TelemetryMetric(name="battery", value=95.0, unit="%")
        metric_data = metric.model_dump()
        restored_metric = TelemetryMetric.model_validate(metric_data)
        assert restored_metric == metric


# ==============================================================================
# SECTION E-I: WGS-84 Coordinates, Bounds, Rejection, Altitude & Accuracy
# ==============================================================================

class TestCoordinateValidation:
    """Tests E, F, G, H, I: Coordinate validation, bounds, and precision."""

    def test_section_e_valid_wgs84_coordinates(self):
        valid, err = validate_coordinates(37.7749, -122.4194)
        assert valid is True
        assert err is None

        valid, err = validate_coordinates(0.0, 0.0)
        assert valid is True
        assert err is None

    def test_section_f_latitude_bounds_enforcement(self):
        # Valid bounds [-90, +90]
        assert validate_coordinates(90.0, 0.0)[0] is True
        assert validate_coordinates(-90.0, 0.0)[0] is True

        # Invalid bounds
        valid, err = validate_coordinates(90.0001, 0.0)
        assert valid is False
        assert "Latitude out of bounds" in err

        valid, err = validate_coordinates(-95.0, 0.0)
        assert valid is False
        assert "Latitude out of bounds" in err

    def test_section_g_longitude_bounds_enforcement(self):
        # Valid bounds [-180, +180]
        assert validate_coordinates(0.0, 180.0)[0] is True
        assert validate_coordinates(0.0, -180.0)[0] is True

        # Invalid bounds
        valid, err = validate_coordinates(0.0, 180.001)
        assert valid is False
        assert "Longitude out of bounds" in err

        valid, err = validate_coordinates(0.0, -181.0)
        assert valid is False
        assert "Longitude out of bounds" in err

    def test_section_h_nan_inf_non_numeric_rejection(self):
        assert validate_coordinates(float("nan"), 0.0)[0] is False
        assert validate_coordinates(0.0, float("nan"))[0] is False
        assert validate_coordinates(float("inf"), 0.0)[0] is False
        assert validate_coordinates(0.0, float("-inf"))[0] is False
        assert validate_coordinates("invalid", 0.0)[0] is False
        assert validate_coordinates(0.0, None)[0] is False

    def test_section_i_altitude_and_accuracy_validation(self):
        proc = SpatialProcessor()
        # Valid altitude and accuracy
        pos = proc.create_position(
            lat=37.7749,
            lon=-122.4194,
            altitude=150.0,
            accuracy=2.0,
        )
        assert pos.altitude == 150.0
        assert pos.accuracy == 2.0

        # Out of bounds altitude (-1000m to 100000m)
        with pytest.raises(ValueError):
            proc.create_position(lat=0.0, lon=0.0, altitude=-2000.0)

        with pytest.raises(ValueError):
            proc.create_position(lat=0.0, lon=0.0, altitude=200000.0)

        # Negative accuracy rejected
        with pytest.raises(ValueError):
            proc.create_position(lat=0.0, lon=0.0, accuracy=-1.0)


# ==============================================================================
# SECTION J-O: Haversine Distance, Bearing, Relative Positioning & Relations
# ==============================================================================

class TestGeodesyAndRelativePositioning:
    """Tests J, K, L, M, N, O: Distance, bearing, and spatial relations."""

    def test_section_j_haversine_distance_accuracy(self):
        # Paris (48.8566, 2.3522) to London (51.5074, -0.1278) ≈ 343 km (343,000 m)
        dist = calculate_distance(48.8566, 2.3522, 51.5074, -0.1278)
        assert 340000 < dist < 346000

        # Equator 1 degree longitude ≈ 111.32 km
        dist_1deg = calculate_distance(0.0, 0.0, 0.0, 1.0)
        assert 111000 < dist_1deg < 112000

    def test_section_k_coincident_coordinates_zero_distance(self):
        dist = calculate_distance(37.7749, -122.4194, 37.7749, -122.4194)
        assert dist == 0.0

    def test_section_l_initial_bearing_calculation(self):
        # Directly North: 0.0 degrees
        bearing_n = calculate_bearing(0.0, 0.0, 10.0, 0.0)
        assert math.isclose(bearing_n, 0.0, abs_tol=1e-3)

        # Directly East along equator: 90.0 degrees
        bearing_e = calculate_bearing(0.0, 0.0, 0.0, 10.0)
        assert math.isclose(bearing_e, 90.0, abs_tol=1e-3)

        # Directly South: 180.0 degrees
        bearing_s = calculate_bearing(10.0, 0.0, 0.0, 0.0)
        assert math.isclose(bearing_s, 180.0, abs_tol=1e-3)

        # Directly West along equator: 270.0 degrees
        bearing_w = calculate_bearing(0.0, 10.0, 0.0, 0.0)
        assert math.isclose(bearing_w, 270.0, abs_tol=1e-3)

        # Coincident: 0.0 degrees
        assert calculate_bearing(37.0, -122.0, 37.0, -122.0) == 0.0

    def test_section_m_bearing_normalization_bounds(self):
        # All bearings must be strictly in [0.0, 360.0)
        for lat in [-45.0, 0.0, 45.0]:
            for lon in [-120.0, 0.0, 120.0]:
                b = calculate_bearing(lat, lon, lat + 1.0, lon + 1.0)
                assert 0.0 <= b < 360.0

    def test_section_n_relative_position_evaluation(self):
        rel = evaluate_relative_position(
            source_lat=37.7749,
            source_lon=-122.4194,
            target_lat=37.7750,
            target_lon=-122.4194,
            target_id="waypoint_alpha",
        )
        assert rel.target_id == "waypoint_alpha"
        assert rel.distance_meters > 0.0
        assert 0.0 <= rel.bearing_degrees < 360.0
        assert rel.relation in SpatialRelationType

    def test_section_o_spatial_relation_categorization(self):
        origin_lat, origin_lon = 37.7749, -122.4194

        # SAME_LOCATION: < 5m
        rel_same = evaluate_relative_position(origin_lat, origin_lon, origin_lat + 0.00001, origin_lon, "t1")
        assert rel_same.distance_meters < 5.0
        assert rel_same.relation == SpatialRelationType.SAME_LOCATION

        # NEARBY: 5m to 50m
        rel_near = evaluate_relative_position(origin_lat, origin_lon, origin_lat + 0.0002, origin_lon, "t2")
        assert 5.0 <= rel_near.distance_meters < 50.0
        assert rel_near.relation == SpatialRelationType.NEARBY

        # VICINITY: 50m to 500m
        rel_vic = evaluate_relative_position(origin_lat, origin_lon, origin_lat + 0.002, origin_lon, "t3")
        assert 50.0 <= rel_vic.distance_meters < 500.0
        assert rel_vic.relation == SpatialRelationType.VICINITY

        # DISTANT: 500m to 5000m
        rel_dist = evaluate_relative_position(origin_lat, origin_lon, origin_lat + 0.02, origin_lon, "t4")
        assert 500.0 <= rel_dist.distance_meters < 5000.0
        assert rel_dist.relation == SpatialRelationType.DISTANT

        # REMOTE: >= 5000m
        rel_remote = evaluate_relative_position(origin_lat, origin_lon, origin_lat + 0.1, origin_lon, "t5")
        assert rel_remote.distance_meters >= 5000.0
        assert rel_remote.relation == SpatialRelationType.REMOTE


# ==============================================================================
# SECTION P & BK: Geofencing (Circular, Bounding Box, Multi-fence)
# ==============================================================================

class TestGeofenceEvaluation:
    """Tests P, BK: Geofence evaluation and relations."""

    def test_section_p_circular_geofence(self):
        proc = SpatialProcessor()
        fence = GeofenceArea(
            geofence_id="circle_1",
            name="Zone 1",
            center_latitude=37.7749,
            center_longitude=-122.4194,
            radius_meters=100.0,
        )

        # Dead center -> INSIDE
        rel, d = proc.evaluate_geofence(37.7749, -122.4194, fence)
        assert rel == GeofenceRelationType.INSIDE
        assert d < 1.0

        # Far outside -> OUTSIDE
        rel_out, d_out = proc.evaluate_geofence(37.7849, -122.4194, fence)
        assert rel_out == GeofenceRelationType.OUTSIDE
        assert d_out > 1000.0

    def test_section_p_bounding_box_geofence(self):
        proc = SpatialProcessor()
        box_fence = GeofenceArea(
            geofence_id="box_1",
            name="Warehouse Box",
            min_latitude=37.7700,
            max_latitude=37.7800,
            min_longitude=-122.4200,
            max_longitude=-122.4100,
        )

        # Center of box -> INSIDE
        rel, _ = proc.evaluate_geofence(37.7750, -122.4150, box_fence)
        assert rel == GeofenceRelationType.INSIDE

        # Outside box -> OUTSIDE
        rel_out, _ = proc.evaluate_geofence(37.7600, -122.4150, box_fence)
        assert rel_out == GeofenceRelationType.OUTSIDE

    def test_section_bk_multiple_geofence_evaluation(self):
        proc = SpatialProcessor()
        fences = [
            GeofenceArea(
                geofence_id="inner_ring",
                name="Inner Ring",
                center_latitude=37.7749,
                center_longitude=-122.4194,
                radius_meters=50.0,
            ),
            GeofenceArea(
                geofence_id="outer_ring",
                name="Outer Ring",
                center_latitude=37.7749,
                center_longitude=-122.4194,
                radius_meters=200.0,
            ),
        ]

        # At 30m distance: inside both inner and outer
        results = proc.evaluate_geofences(37.7749 + 0.00027, -122.4194, fences)
        assert len(results) == 2
        assert results["inner_ring"] == GeofenceRelationType.INSIDE
        assert results["outer_ring"] == GeofenceRelationType.INSIDE

        # At 100m distance: outside inner, inside outer
        results_100 = proc.evaluate_geofences(37.7749 + 0.0009, -122.4194, fences)
        assert results_100["inner_ring"] in (GeofenceRelationType.OUTSIDE, GeofenceRelationType.APPROACHING)
        assert results_100["outer_ring"] == GeofenceRelationType.INSIDE


# ==============================================================================
# SECTION Q-Y & BJ: Telemetry Bounds, Quality, Freshness, Units & Metrics
# ==============================================================================

class TestTelemetryProcessing:
    """Tests Q-Y, BJ: Telemetry validation, normalization, and quality."""

    def test_section_q_battery_percentage_bounds(self):
        proc = TelemetryProcessor()
        # Normal bounds [0, 100]
        h1 = proc.normalize_health({"battery_percent": 75.5})
        assert h1.battery_percent == 75.5

        # Edge values
        assert proc.normalize_health({"battery_percent": 0.0}).battery_percent == 0.0
        assert proc.normalize_health({"battery_percent": 100.0}).battery_percent == 100.0

        # Clamp extreme values
        h_high = proc.normalize_health({"battery_percent": 105.0})
        assert h_high.battery_percent == 100.0

        h_low = proc.normalize_health({"battery_percent": -5.0})
        assert h_low.battery_percent == 0.0

    def test_section_r_connectivity_status_normalization(self):
        proc = TelemetryProcessor()
        # Standard valid statuses
        for status in ["ONLINE", "OFFLINE", "DEGRADED", "CONNECTING", "DISCONNECTED"]:
            h = proc.normalize_health({"connectivity": status})
            assert h.connectivity == status

        # Case normalization
        h_lower = proc.normalize_health({"connectivity": "online"})
        assert h_lower.connectivity == "ONLINE"

        # Unknown fallback
        h_unk = proc.normalize_health({"connectivity": "weird_mesh_status"})
        assert h_unk.connectivity == "UNKNOWN"

    def test_section_s_health_status_normalization(self):
        proc = TelemetryProcessor()
        for status in ["HEALTHY", "DEGRADED", "UNHEALTHY", "UNKNOWN"]:
            h = proc.normalize_health({"health_status": status})
            assert h.health_status == status

        h_case = proc.normalize_health({"health_status": "healthy"})
        assert h_case.health_status == "HEALTHY"

    def test_section_t_and_u_metric_normalization_and_units(self):
        # Valid standard metrics
        m1 = normalize_metric("speed_mps", 12.5, "m/s")
        assert m1.name == "speed_mps"
        assert m1.value == 12.5
        assert m1.unit == "m/s"

        m2 = normalize_metric("core_temp", 45.2, "C")
        assert m2.name == "core_temp"
        assert m2.value == 45.2
        assert m2.unit == "C"

        # Metric with unit in raw name or default
        m3 = normalize_metric("voltage", 12.4)
        assert m3.name == "voltage"
        assert m3.value == 12.4

    def test_section_v_w_x_and_bj_telemetry_quality_and_freshness(self):
        # Fresh telemetry (e.g. age = 2.0s <= 10.0s max_age)
        quality, stale = evaluate_freshness(age_seconds=2.0, max_age_seconds=10.0)
        assert quality == TelemetryQuality.VALID
        assert stale is False

        # Stale telemetry (Section X & BJ transition)
        quality_stale, stale_flag = evaluate_freshness(age_seconds=15.0, max_age_seconds=10.0)
        assert quality_stale == TelemetryQuality.STALE
        assert stale_flag is True

        proc = TelemetryProcessor(limits=SpatialTelemetryLimits(max_staleness_seconds=10.0))

        # Process fresh telemetry payload
        obs_fresh = proc.process(
            payload={"health": {"battery_percent": 85.0}},
            timestamp=time.time() - 2.0,
        )
        assert obs_fresh.quality == TelemetryQuality.VALID
        assert not obs_fresh.is_stale

        # Process stale payload (50 seconds old > 10s max_staleness)
        obs_old = proc.process(
            payload={"health": {"battery_percent": 85.0}},
            timestamp=time.time() - 50.0,
        )
        assert obs_old.quality == TelemetryQuality.STALE
        assert obs_old.is_stale is True

    def test_section_y_confidence_score_bounds(self):
        provider = SpatialTelemetryPerceptionProvider()
        req = make_spatial_request(
            modality=ModalityType.TELEMETRY,
            payload={"device_id": "test_device", "battery_percent": 90.0},
        )
        result = provider.perceive(req)
        assert result.status == PerceptionStatus.SUCCESS
        for ev in result.evidence:
            assert 0.0 <= ev.confidence <= 1.0


# ==============================================================================
# SECTION Z-AB: Provider Compliance, Registry Integration, Mock Provider
# ==============================================================================

class TestProviderAndRegistry:
    """Tests Z, AA, AB: Provider interface, registry, and mock provider."""

    def test_section_z_provider_interface_compliance(self):
        provider = SpatialTelemetryPerceptionProvider()
        # Modalities
        mods = provider.supported_modalities()
        assert ModalityType.GPS in mods
        assert ModalityType.TELEMETRY in mods
        assert ModalityType.DEVICE_STATE in mods

        # Capabilities
        caps = provider.supported_capabilities()
        assert PerceptionCapability.SPATIAL_LOCALIZATION in caps
        assert PerceptionCapability.TELEMETRY_NORMALIZATION in caps

        # Health check
        assert provider.health_check().status == "HEALTHY"

    def test_section_aa_provider_registration_in_registry(self):
        registry = PerceptionProviderRegistry()
        provider = SpatialTelemetryPerceptionProvider()
        registry.register_provider(provider)

        # Query by GPS modality and capability
        gps_prov = registry.select_provider(PerceptionCapability.SPATIAL_LOCALIZATION, ModalityType.GPS)
        assert gps_prov is provider

        # Query by TELEMETRY modality and capability
        telem_prov = registry.select_provider(PerceptionCapability.TELEMETRY_NORMALIZATION, ModalityType.TELEMETRY)
        assert telem_prov is provider

        # Retrieve by ID
        assert registry.get_provider(provider.provider_id) is provider

    def test_section_ab_mock_provider_deterministic_behavior(self):
        mock_prov = MockSpatialTelemetryProvider()
        assert mock_prov.health_check().status == "HEALTHY"

        req = make_spatial_request(
            modality=ModalityType.GPS,
            payload={"latitude": 37.7749, "longitude": -122.4194},
        )
        res1 = mock_prov.perceive(req)
        res2 = mock_prov.perceive(req)

        assert res1.status == PerceptionStatus.SUCCESS
        assert res2.status == PerceptionStatus.SUCCESS
        assert len(res1.evidence) == len(res2.evidence)
        assert res1.evidence[0].spatial.location.latitude == res2.evidence[0].spatial.location.latitude
        assert res1.evidence[0].confidence == res2.evidence[0].confidence


# ==============================================================================
# SECTION AC-AE: GPS, Telemetry & Device State Perception Results
# ==============================================================================

class TestPerceptionResults:
    """Tests AC, AD, AE: Modality-specific perception results."""

    def test_section_ac_gps_modality_perception(self):
        provider = SpatialTelemetryPerceptionProvider()
        req = make_spatial_request(
            modality=ModalityType.GPS,
            payload={
                "latitude": 37.7749,
                "longitude": -122.4194,
                "altitude": 20.0,
                "accuracy": 3.0,
                "heading": 180.0,
                "speed": 8.5,
                "device_id": "drone_01",
            },
        )
        result = provider.perceive(req)
        assert result.status == PerceptionStatus.SUCCESS
        assert len(result.evidence) == 1

        ev = result.evidence[0]
        assert ev.modality == ModalityType.GPS
        assert ev.spatial is not None
        assert ev.spatial.location is not None
        assert ev.spatial.location.latitude == 37.7749
        assert ev.spatial.location.longitude == -122.4194
        assert ev.attributes["speed"] == 8.5
        assert ev.attributes["heading"] == 180.0

    def test_section_ad_telemetry_modality_perception(self):
        provider = SpatialTelemetryPerceptionProvider()
        req = make_spatial_request(
            modality=ModalityType.TELEMETRY,
            payload={
                "device_id": "rover_01",
                "battery_percent": 92.0,
                "connectivity": "ONLINE",
                "health_status": "HEALTHY",
                "metrics": [
                    {"name": "motor_temp", "value": 42.1, "unit": "C"},
                    {"name": "odometry_m", "value": 530.0, "unit": "m"},
                ],
            },
        )
        result = provider.perceive(req)
        assert result.status == PerceptionStatus.SUCCESS
        assert len(result.evidence) == 1

        ev = result.evidence[0]
        assert ev.modality == ModalityType.TELEMETRY
        assert ev.attributes["battery_percent"] == 92.0
        assert ev.attributes["connectivity"] == "ONLINE"
        metrics = ev.attributes["metrics"]
        assert len(metrics) == 2
        assert metrics[0]["name"] == "motor_temp"

    def test_section_ae_device_state_modality_perception(self):
        provider = SpatialTelemetryPerceptionProvider()
        req = make_spatial_request(
            modality=ModalityType.DEVICE_STATE,
            payload={
                "device_id": "glass_01",
                "battery_percent": 45.0,
                "connectivity": "DEGRADED",
                "health_status": "DEGRADED",
                "display_active": True,
            },
        )
        result = provider.perceive(req)
        assert result.status == PerceptionStatus.SUCCESS
        assert len(result.evidence) == 1

        ev = result.evidence[0]
        assert ev.modality == ModalityType.DEVICE_STATE
        assert ev.attributes["health_status"] == "DEGRADED"
        assert ev.attributes["display_active"] is True


# ==============================================================================
# SECTION AF-AK: Normalization, MultimodalObservation & Provenance
# ==============================================================================

class TestNormalizationAndProvenance:
    """Tests AF-AK: Normalizer integration, MultimodalObservation, and provenance."""

    def test_section_af_ag_normalization_and_spatial_metadata(self):
        normalizer = PerceptionObservationNormalizer()
        provider = SpatialTelemetryPerceptionProvider()

        req = make_spatial_request(
            modality=ModalityType.GPS,
            payload={"latitude": 37.7749, "longitude": -122.4194, "device_id": "drone_01"},
            correlation_id="corr_test_123",
        )
        result = provider.perceive(req)
        obs = normalizer.normalize(result)

        assert isinstance(obs, MultimodalObservation)
        assert obs.modality == ModalityType.GPS

        # Section AG: Spatial metadata preserved in location
        assert obs.location is not None
        assert obs.location.latitude == 37.7749
        assert obs.location.longitude == -122.4194

    def test_section_ah_provenance_eight_mandatory_keys(self):
        normalizer = PerceptionObservationNormalizer()
        provider = SpatialTelemetryPerceptionProvider()

        req = make_spatial_request(
            modality=ModalityType.TELEMETRY,
            payload={"device_id": "rover_02", "battery_percent": 80.0},
            correlation_id="corr_prov_001",
            causation_id="caus_prov_001",
        )
        result = provider.perceive(req)
        obs = normalizer.normalize(result)

        prov = obs.metadata.get("provenance", {})
        # Check all 8 mandatory provenance keys in provenance dictionary
        for key in REQUIRED_PROVENANCE_KEYS:
            assert key in prov, f"Missing required provenance key: {key}"

        assert prov["provider_id"] == "atlas_spatial_telemetry_provider"
        assert obs.modality == ModalityType.TELEMETRY
        assert obs.confidence == 1.0

    def test_section_ai_aj_ak_timestamp_correlation_causation(self):
        normalizer = PerceptionObservationNormalizer()
        provider = SpatialTelemetryPerceptionProvider()

        req = make_spatial_request(
            modality=ModalityType.DEVICE_STATE,
            payload={"device_id": "glass_02", "battery_percent": 60.0},
            correlation_id="corr_chain_999",
            causation_id="caus_chain_888",
        )
        result = provider.perceive(req)
        obs = normalizer.normalize(result)

        # Section AI: Timestamp matches
        assert obs.timestamp > 0.0

        # Section AJ: Correlation ID preserved
        assert obs.correlation_id == "corr_chain_999"

        # Section AK: Causation ID preserved
        assert obs.causation_id == "caus_chain_888"


# ==============================================================================
# SECTION AL-AP: Multi-Product Edge Support & Digital Twin Simulation
# ==============================================================================

class TestMultiProductAndSimulation:
    """Tests AL, AM, AN, AO, AP: Product telemetry & Digital Twin simulation."""

    def test_section_al_vision_edge_product_telemetry(self):
        provider = SpatialTelemetryPerceptionProvider()
        req = make_spatial_request(
            modality=ModalityType.TELEMETRY,
            payload={
                "product_type": "vision",
                "device_id": "vision_cam_front",
                "health_status": "HEALTHY",
                "connectivity": "ONLINE",
                "metrics": [
                    {"name": "fps", "value": 30.0, "unit": "fps"},
                    {"name": "sensor_temp", "value": 41.5, "unit": "C"},
                ],
            },
        )
        res = provider.perceive(req)
        assert res.status == PerceptionStatus.SUCCESS
        assert res.evidence[0].provenance["product_type"] == "vision"
        assert res.evidence[0].attributes["product_type"] == "vision"

    def test_section_am_glass_edge_product_telemetry(self):
        provider = SpatialTelemetryPerceptionProvider()
        req = make_spatial_request(
            modality=ModalityType.TELEMETRY,
            payload={
                "product_type": "glass",
                "device_id": "glass_wearable_1",
                "battery_percent": 74.0,
                "connectivity": "ONLINE",
                "metrics": [
                    {"name": "display_brightness", "value": 80.0, "unit": "%"},
                    {"name": "head_pitch", "value": -5.2, "unit": "deg"},
                ],
            },
        )
        res = provider.perceive(req)
        assert res.status == PerceptionStatus.SUCCESS
        assert res.evidence[0].attributes["battery_percent"] == 74.0

    def test_section_an_drone_edge_product_spatial_and_telemetry(self):
        provider = SpatialTelemetryPerceptionProvider()
        req = make_spatial_request(
            modality=ModalityType.GPS,
            payload={
                "product_type": "drone",
                "device_id": "drone_scout",
                "latitude": 37.7749,
                "longitude": -122.4194,
                "altitude": 120.0,
                "heading": 270.0,
                "speed": 14.2,
            },
        )
        res = provider.perceive(req)
        assert res.status == PerceptionStatus.SUCCESS
        ev = res.evidence[0]
        assert ev.spatial.location.latitude == 37.7749
        assert ev.attributes["altitude"] == 120.0
        assert ev.attributes["heading"] == 270.0

    def test_section_ao_rover_edge_product_telemetry(self):
        provider = SpatialTelemetryPerceptionProvider()
        req = make_spatial_request(
            modality=ModalityType.GPS,
            payload={
                "product_type": "rover",
                "device_id": "rover_ground",
                "latitude": 37.7755,
                "longitude": -122.4185,
                "speed": 2.1,
                "heading": 45.0,
            },
        )
        res = provider.perceive(req)
        assert res.status == PerceptionStatus.SUCCESS
        assert res.evidence[0].attributes["speed"] == 2.1

    def test_section_ap_digital_twin_simulation_state_ingestion(self):
        # Digital Twin telemetry packet simulated from VirtualDevice/SimulationEngine
        sim_payload = {
            "device_id": "sim_drone_virtual_01",
            "is_simulated": True,
            "simulation_tick": 1420,
            "latitude": 37.7760,
            "longitude": -122.4170,
            "altitude": 55.0,
            "battery_percent": 91.5,
            "connectivity": "ONLINE",
            "health_status": "HEALTHY",
        }
        provider = SpatialTelemetryPerceptionProvider()
        req = make_spatial_request(
            modality=ModalityType.GPS,
            payload=sim_payload,
        )
        res = provider.perceive(req)
        assert res.status == PerceptionStatus.SUCCESS
        assert res.evidence[0].spatial.location.latitude == 37.7760


# ==============================================================================
# SECTION AQ-AW & BI: Robustness, Limits, Determinism, Replay & Redaction
# ==============================================================================

class TestRobustnessLimitsAndSecurity:
    """Tests AQ-AW, BI: Errors, limits, determinism, replay, and redaction."""

    def test_section_aq_malformed_spatial_input(self):
        provider = SpatialTelemetryPerceptionProvider()
        # Invalid coordinates (lat=120 outside [-90, 90])
        req = make_spatial_request(
            modality=ModalityType.GPS,
            payload={"latitude": 120.0, "longitude": -122.4194},
        )
        res = provider.perceive(req)
        # Should not crash; marks invalid input
        assert res.status in (PerceptionStatus.INVALID_INPUT, PerceptionStatus.FAILED, PerceptionStatus.NO_DETECTION)

    def test_section_ar_malformed_telemetry_input(self):
        provider = SpatialTelemetryPerceptionProvider()
        # Corrupted payload without numbers
        req = make_spatial_request(
            modality=ModalityType.TELEMETRY,
            payload={"corrupt_junk": "none", "battery_percent": "not_a_number"},
        )
        res = provider.perceive(req)
        # Handled gracefully without unhandled crash
        assert res.status in (PerceptionStatus.SUCCESS, PerceptionStatus.PARTIAL, PerceptionStatus.FAILED, PerceptionStatus.NO_DETECTION)

    def test_section_as_unsupported_modality_handling(self):
        provider = SpatialTelemetryPerceptionProvider()
        # ModalityType.IMAGE is not supported by SpatialTelemetry provider
        inp = PerceptionInput(
            input_id="bad_mod_in",
            modality=ModalityType.IMAGE,
            payload_ref="inline://bad",
            captured_at=time.time(),
            source_id="test_source",
            metadata={"pixels": [1, 2, 3]},
        )
        req = PerceptionRequest(
            request_id="req_bad_mod",
            input_data=inp,
        )
        res = provider.perceive(req)
        assert res.status in (PerceptionStatus.UNSUPPORTED_MODALITY, PerceptionStatus.FAILED)
        assert len(res.errors) > 0
        assert "not supported" in res.errors[0].message.lower()

    def test_section_at_bounded_metrics_count(self):
        proc = TelemetryProcessor(max_metrics=5)
        # Supply 20 metrics
        payload = {
            "metrics": [{"name": f"metric_{i}", "value": float(i)} for i in range(20)]
        }
        obs = proc.process(payload)
        assert len(obs.metrics) <= 5

    def test_section_au_bounded_relations_count(self):
        proc = SpatialProcessor(max_relations=3)
        # Supply 10 targets
        targets = [{"target_id": f"t_{i}", "latitude": 37.77 + i * 0.001, "longitude": -122.41} for i in range(10)]
        relations = proc.evaluate_multiple_relative_positions(37.7749, -122.4194, targets)
        assert len(relations) <= 3

    def test_section_av_deterministic_output_identical_inputs(self):
        provider = SpatialTelemetryPerceptionProvider()
        payload = {
            "latitude": 37.7749,
            "longitude": -122.4194,
            "altitude": 10.0,
            "heading": 90.0,
            "speed": 5.0,
        }
        req1 = make_spatial_request(modality=ModalityType.GPS, payload=payload)
        req2 = make_spatial_request(modality=ModalityType.GPS, payload=payload)
        res1 = provider.perceive(req1)
        res2 = provider.perceive(req2)
        assert res1.evidence[0].spatial.location.latitude == res2.evidence[0].spatial.location.latitude
        assert res1.evidence[0].spatial.location.longitude == res2.evidence[0].spatial.location.longitude
        assert res1.evidence[0].attributes["heading"] == res2.evidence[0].attributes["heading"]

    def test_section_aw_replay_serialization_compatibility(self):
        provider = SpatialTelemetryPerceptionProvider()
        req = make_spatial_request(
            modality=ModalityType.GPS,
            payload={"latitude": 37.7749, "longitude": -122.4194},
        )
        res = provider.perceive(req)
        res_json = res.model_dump_json() if hasattr(res, "model_dump_json") else json.dumps(res.to_dict())
        assert "latitude" in res_json
        assert res.status == PerceptionStatus.SUCCESS

    def test_section_bi_zero_credential_leakage(self):
        provider = SpatialTelemetryPerceptionProvider()
        # Malicious payload with API keys and passwords
        payload = {
            "device_id": "sensor_01",
            "battery_percent": 95.0,
            "api_key": "SECRET_KEY_12345",
            "password": "SuperSecretPassword",
            "bearer_token": "Bearer eyJhbGciOi...",
            "metrics": [
                {"name": "voltage", "value": 12.0},
            ],
        }
        req = make_spatial_request(
            modality=ModalityType.TELEMETRY,
            payload=payload,
        )
        res = provider.perceive(req)
        assert res.status == PerceptionStatus.SUCCESS
        custom_data = res.evidence[0].attributes

        # Ensure credentials are not retained in attributes
        assert "api_key" not in custom_data
        assert "password" not in custom_data
        assert "bearer_token" not in custom_data
        # Ensure voltage is retained
        assert len(custom_data["metrics"]) == 1


# ==============================================================================
# SECTION AX-BH: Architectural Boundary Invariants
# ==============================================================================

class TestArchitecturalBoundaryInvariants:
    """Tests AX-BH: Invariants against decision leakage, mutation, and physical hardware."""

    def test_section_ax_zero_world_state_mutation(self):
        ws = WorldState(state_id="ws_01", version=1, timestamp=time.time())
        initial_state = copy.deepcopy(ws.to_dict())

        provider = SpatialTelemetryPerceptionProvider()
        req = make_spatial_request(
            modality=ModalityType.GPS,
            payload={"latitude": 37.7749, "longitude": -122.4194},
        )
        _ = provider.perceive(req)

        # WorldState must remain strictly unmutated
        assert ws.to_dict() == initial_state

    def test_section_ay_zero_goal_store_mutation(self):
        provider = SpatialTelemetryPerceptionProvider()
        # Check provider has no goal store or goal management attributes
        assert not hasattr(provider, "goal_store")
        assert not hasattr(provider, "goal_manager")
        assert not hasattr(provider, "goals")

    def test_section_az_ba_bb_bc_bd_be_zero_authority_bypasses(self):
        provider = SpatialTelemetryPerceptionProvider()
        # Verify provider has no runtime, orchestrator, gateway, or policy bypasses
        assert not hasattr(provider, "autonomous_goal_manager")
        assert not hasattr(provider, "cognitive_runtime")
        assert not hasattr(provider, "tool_orchestrator")
        assert not hasattr(provider, "device_gateway")
        assert not hasattr(provider, "policy_engine")
        assert not hasattr(provider, "mission_engine")

    def test_section_bf_bg_bh_zero_hardware_and_execution_imports(self):
        # Verify no physical hardware drivers, subprocesses, or eval/exec are used
        import backend.spatial_telemetry.provider as p_mod
        import backend.spatial_telemetry.reference_provider as ref_mod
        import backend.spatial_telemetry.spatial as s_mod
        import backend.spatial_telemetry.telemetry as t_mod

        forbidden_tokens = ["subprocess", "os.system", "eval(", "exec(", "rospy", "rclpy", "pymavlink", "RPi.GPIO", "serial."]
        modules = [p_mod, ref_mod, s_mod, t_mod]

        for mod in modules:
            source = open(mod.__file__, "r", encoding="utf-8").read()
            for token in forbidden_tokens:
                assert token not in source, f"Forbidden token '{token}' found in {mod.__file__}"
