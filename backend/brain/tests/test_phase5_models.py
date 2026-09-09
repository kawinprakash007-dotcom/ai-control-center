import dataclasses
import json
import time
import pytest

from core.models.orchestration import (
    ConnectivityStatus,
    DeviceCapabilityDescriptor,
    DeviceIdentity,
    DeviceType,
    GeoLocation,
    ModalityType,
    MultimodalObservation,
    Situation,
    SituationCategory,
    SituationEvidence,
    SituationSeverity,
    SituationStatus,
)
from core.models.autonomy import EventSource, EventPriority, EventCategory


# ============================================================================
# 1. MultimodalObservation Creation & Validation
# ============================================================================

def test_multimodal_observation_creation():
    obs = MultimodalObservation(
        observation_id="obs_001",
        source_id="cam_front",
        source_type="camera",
        modality=ModalityType.IMAGE,
        timestamp=1000.0,
        payload={"width": 1920, "height": 1080},
        confidence=0.95,
        correlation_id="corr_123",
        causation_id="cause_000",
        artifact_reference="artifacts/img_001.jpg",
    )
    assert obs.observation_id == "obs_001"
    assert obs.source_id == "cam_front"
    assert obs.source_type == "camera"
    assert obs.modality == ModalityType.IMAGE
    assert obs.timestamp == 1000.0
    assert obs.confidence == 0.95
    assert obs.correlation_id == "corr_123"
    assert obs.causation_id == "cause_000"
    assert obs.artifact_reference == "artifacts/img_001.jpg"
    assert not obs.is_expired(now=1000.0)


def test_multimodal_observation_immutability():
    obs = MultimodalObservation(
        observation_id="obs_002",
        source_id="gps_01",
        source_type="gps_receiver",
        modality=ModalityType.GPS,
        timestamp=1000.0,
        payload={"lat": 37.7749, "lon": -122.4194},
        confidence=0.99,
    )
    with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
        obs.confidence = 0.5  # type: ignore

    with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
        obs.source_id = "new_source"  # type: ignore


def test_observation_modality_validation():
    # Valid string should be coerced to ModalityType
    obs = MultimodalObservation(
        observation_id="obs_003",
        source_id="mic_01",
        source_type="microphone",
        modality="AUDIO_EVENT",  # string coerced
        timestamp=1000.0,
        payload="glass_break",
    )
    assert obs.modality == ModalityType.AUDIO_EVENT

    # Unknown string safely falls back to ModalityType.UNKNOWN
    obs_unknown = MultimodalObservation(
        observation_id="obs_003_unk",
        source_id="future_sensor",
        source_type="quantum_sensor",
        modality="QUANTUM_TELEMETRY",
        timestamp=1000.0,
        payload="spin_up",
    )
    assert obs_unknown.modality == ModalityType.UNKNOWN


def test_observation_confidence_bounds():
    with pytest.raises(ValueError, match="confidence must be in"):
        MultimodalObservation(
            observation_id="obs_invalid_conf",
            source_id="s1",
            source_type="test",
            modality=ModalityType.TEXT,
            timestamp=1000.0,
            payload="text",
            confidence=1.5,
        )

    with pytest.raises(ValueError, match="confidence must be in"):
        MultimodalObservation(
            observation_id="obs_invalid_conf2",
            source_id="s1",
            source_type="test",
            modality=ModalityType.TEXT,
            timestamp=1000.0,
            payload="text",
            confidence=-0.1,
        )


def test_observation_timestamp_validation():
    with pytest.raises(ValueError, match="timestamp must be positive"):
        MultimodalObservation(
            observation_id="obs_invalid_ts",
            source_id="s1",
            source_type="test",
            modality=ModalityType.TEXT,
            timestamp=-5.0,
            payload="text",
        )


def test_freshness_and_expiration_semantics():
    # Observation with expiration
    obs = MultimodalObservation(
        observation_id="obs_exp",
        source_id="sensor",
        source_type="test",
        modality=ModalityType.TELEMETRY,
        timestamp=100.0,
        payload={"temp": 42.0},
        confidence=0.9,
        expires_at=150.0,
    )

    # Before expiration
    assert not obs.is_expired(now=120.0)
    assert obs.get_age(now=120.0) == 20.0

    # At or after expiration
    assert obs.is_expired(now=150.0)
    assert obs.is_expired(now=160.0)

    # Freshness decays over time, strictly distinct from confidence
    f_fresh = obs.freshness_score(now=100.0, half_life_seconds=60.0)
    assert pytest.approx(f_fresh, rel=1e-3) == 1.0

    f_half = obs.freshness_score(now=160.0, half_life_seconds=60.0)
    assert pytest.approx(f_half, rel=1e-3) == 0.5
    # Confidence remains invariant
    assert obs.confidence == 0.9

    # Expiration before timestamp is rejected
    with pytest.raises(ValueError, match="expires_at .* cannot be earlier than timestamp"):
        MultimodalObservation(
            observation_id="obs_bad_exp",
            source_id="s1",
            source_type="test",
            modality=ModalityType.TELEMETRY,
            timestamp=200.0,
            payload={},
            expires_at=150.0,
        )


# ============================================================================
# 2. GeoLocation Validation & Distance
# ============================================================================

def test_geolocation_validation():
    # Valid coordinates
    loc = GeoLocation(latitude=37.7749, longitude=-122.4194, altitude=15.0, accuracy=2.5)
    assert loc.latitude == 37.7749
    assert loc.longitude == -122.4194
    assert loc.altitude == 15.0
    assert loc.accuracy == 2.5

    # Latitude out of bounds
    with pytest.raises(ValueError, match="latitude must be in"):
        GeoLocation(latitude=95.0, longitude=0.0)

    with pytest.raises(ValueError, match="latitude must be in"):
        GeoLocation(latitude=-91.0, longitude=0.0)

    # Longitude out of bounds
    with pytest.raises(ValueError, match="longitude must be in"):
        GeoLocation(latitude=0.0, longitude=185.0)

    with pytest.raises(ValueError, match="longitude must be in"):
        GeoLocation(latitude=0.0, longitude=-181.0)

    # Negative accuracy
    with pytest.raises(ValueError, match="accuracy must be non-negative"):
        GeoLocation(latitude=0.0, longitude=0.0, accuracy=-1.0)


def test_geolocation_distance_calculation():
    # SF coordinates to Oakland coordinates (approx 12-14 km)
    sf = GeoLocation(latitude=37.7749, longitude=-122.4194)
    oakland = GeoLocation(latitude=37.8044, longitude=-122.2712)
    dist = sf.distance_to(oakland)
    assert 12000.0 < dist < 15000.0

    # Distance to self is 0
    assert sf.distance_to(sf) == 0.0


# ============================================================================
# 3. Situation & SituationEvidence
# ============================================================================

def test_situation_creation_and_fields():
    evidence = SituationEvidence(
        evidence_id="ev_001",
        observation_id="obs_001",
        source_id="cam_front",
        modality=ModalityType.IMAGE,
        evidence_weight=0.9,
        timestamp=1000.0,
        concise_summary="Obstacle detected in corridor at 1.5m",
    )

    sit = Situation(
        situation_id="sit_001",
        category=SituationCategory.NAVIGATIONAL,
        title="Path Obstructed",
        description="A stationary pallet is blocking the main navigation route.",
        severity=SituationSeverity.HIGH,
        confidence=0.88,
        status=SituationStatus.ACTIVE,
        involved_entities=("pallet_01", "rover_alpha"),
        supporting_evidence=(evidence,),
        created_at=1000.0,
        updated_at=1005.0,
        valid_until=1100.0,
        correlation_id="corr_path_1",
    )

    assert sit.situation_id == "sit_001"
    assert sit.category == SituationCategory.NAVIGATIONAL
    assert sit.severity == SituationSeverity.HIGH
    assert sit.confidence == 0.88
    assert sit.status == SituationStatus.ACTIVE
    assert sit.involved_entities == ("pallet_01", "rover_alpha")
    assert len(sit.supporting_evidence) == 1
    assert sit.is_valid(now=1050.0)
    assert not sit.is_valid(now=1150.0)


def test_situation_immutability():
    sit = Situation(
        situation_id="sit_002",
        category=SituationCategory.SYSTEM_HEALTH,
        title="Battery Low",
        description="Battery level dropped to 12%",
        severity=SituationSeverity.CRITICAL,
        confidence=0.99,
        status=SituationStatus.ACTIVE,
        involved_entities=("rover_alpha",),
        supporting_evidence=(),
        created_at=1000.0,
        updated_at=1000.0,
    )
    with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
        sit.status = SituationStatus.RESOLVED  # type: ignore

    with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
        sit.confidence = 0.5  # type: ignore


def test_situation_confidence_bounds():
    with pytest.raises(ValueError, match="confidence must be in"):
        Situation(
            situation_id="sit_bad_conf",
            category=SituationCategory.OPERATIONAL,
            title="Bad Confidence",
            description="desc",
            severity=SituationSeverity.LOW,
            confidence=1.2,
            status=SituationStatus.ACTIVE,
            involved_entities=(),
            supporting_evidence=(),
        )


def test_situation_temporal_validation():
    # updated_at cannot be before created_at
    with pytest.raises(ValueError, match="updated_at .* cannot be before created_at"):
        Situation(
            situation_id="sit_bad_times",
            category=SituationCategory.OPERATIONAL,
            title="Bad Times",
            description="desc",
            severity=SituationSeverity.LOW,
            confidence=0.8,
            status=SituationStatus.ACTIVE,
            involved_entities=(),
            supporting_evidence=(),
            created_at=1000.0,
            updated_at=900.0,
        )

    # valid_until cannot be before created_at
    with pytest.raises(ValueError, match="valid_until .* cannot be before created_at"):
        Situation(
            situation_id="sit_bad_valid",
            category=SituationCategory.OPERATIONAL,
            title="Bad Valid",
            description="desc",
            severity=SituationSeverity.LOW,
            confidence=0.8,
            status=SituationStatus.ACTIVE,
            involved_entities=(),
            supporting_evidence=(),
            created_at=1000.0,
            updated_at=1000.0,
            valid_until=500.0,
        )


def test_situation_evidence_no_large_payload_duplication():
    # Ensure SituationEvidence only stores reference metadata, not large raw payload
    evidence = SituationEvidence(
        evidence_id="ev_002",
        observation_id="obs_large_video_frame",
        source_id="drone_camera",
        modality=ModalityType.VIDEO_FRAME,
        evidence_weight=0.95,
        timestamp=1000.0,
        concise_summary="Thermal hotspot detected at pixel coordinates (420, 310)",
    )
    assert not hasattr(evidence, "payload")
    assert not hasattr(evidence, "raw_data")
    assert evidence.observation_id == "obs_large_video_frame"


def test_situation_to_autonomy_event_bridge():
    sit = Situation(
        situation_id="sit_bridge_01",
        category=SituationCategory.SECURITY,
        title="Unauthorized Intrusion",
        description="Movement detected in restricted zone",
        severity=SituationSeverity.CRITICAL,
        confidence=0.95,
        status=SituationStatus.ACTIVE,
        involved_entities=("zone_b",),
        supporting_evidence=(),
        created_at=1000.0,
        updated_at=1005.0,
        correlation_id="corr_sec_99",
        causation_id="obs_cam_99",
    )

    ev = sit.to_autonomy_event()
    assert ev.source == EventSource.SITUATION
    assert ev.priority == EventPriority.CRITICAL
    assert ev.event_type == "SITUATION_SECURITY"
    assert ev.correlation_id == "corr_sec_99"
    assert ev.provenance.source_id == "sit_bridge_01"
    assert ev.provenance.source_type == EventSource.SITUATION
    assert ev.payload["situation_id"] == "sit_bridge_01"
    assert ev.payload["severity"] == "CRITICAL"


def test_multimodal_observation_to_world_state_bridge():
    obs = MultimodalObservation(
        observation_id="obs_bridge_01",
        source_id="bms_subsystem",
        source_type="telemetry_sensor",
        modality=ModalityType.TELEMETRY,
        timestamp=1000.0,
        payload=78.5,
        confidence=0.99,
        device_id="rover_1",
        correlation_id="corr_bms_1",
    )
    ws_obs = obs.to_world_state_observation(entity_id="rover_1", property_name="battery_soc")
    assert ws_obs.observation_id == "obs_bridge_01"
    assert ws_obs.entity_id == "rover_1"
    assert ws_obs.property_name == "battery_soc"
    assert ws_obs.value == 78.5
    assert ws_obs.confidence == 0.99
    assert ws_obs.metadata["modality"] == "TELEMETRY"
    assert ws_obs.metadata["device_id"] == "rover_1"


# ============================================================================
# 4. Device Identity & Capability Descriptors
# ============================================================================

def test_device_capability_descriptor_validation():
    cap = DeviceCapabilityDescriptor(
        capability_name="camera_pan",
        action_name="pantilt_servo_move",
        parameters_schema={"type": "object", "properties": {"angle": {"type": "number"}}},
        is_reversible=True,
        requires_confirmation=False,
        rate_limit_hz=10.0,
    )
    assert cap.capability_name == "camera_pan"
    assert cap.action_name == "pantilt_servo_move"
    assert cap.is_reversible is True
    assert cap.requires_confirmation is False
    assert cap.rate_limit_hz == 10.0

    # Rate limit must be positive
    with pytest.raises(ValueError, match="rate_limit_hz must be positive"):
        DeviceCapabilityDescriptor(
            capability_name="cap_invalid",
            action_name="action",
            rate_limit_hz=-2.0,
        )


def test_device_identity_creation_and_immutability():
    cap = DeviceCapabilityDescriptor(
        capability_name="drive_forward",
        action_name="mobile_base_velocity",
        rate_limit_hz=20.0,
    )
    home = GeoLocation(latitude=37.7749, longitude=-122.4194)

    dev = DeviceIdentity(
        device_id="dev_rover_01",
        device_type=DeviceType.ROBOT_GROUND,
        display_name="Atlas Ground Scout Alpha",
        firmware_version="v2.1.0",
        capabilities=(cap,),
        home_location=home,
        is_simulation=True,
        connectivity_status=ConnectivityStatus.ONLINE,
    )

    assert dev.device_id == "dev_rover_01"
    assert dev.device_type == DeviceType.ROBOT_GROUND
    assert dev.display_name == "Atlas Ground Scout Alpha"
    assert dev.firmware_version == "v2.1.0"
    assert dev.is_simulation is True
    assert dev.is_online() is True
    assert dev.has_capability("drive_forward") is True
    assert dev.has_capability("fly") is False
    assert dev.get_capability("drive_forward") == cap

    # Immutability check
    with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
        dev.connectivity_status = ConnectivityStatus.OFFLINE  # type: ignore

    with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
        dev.device_id = "new_id"  # type: ignore


# ============================================================================
# 5. Deterministic Serialization & JSON Safety
# ============================================================================

def test_deterministic_serialization_multimodal_observation():
    obs = MultimodalObservation(
        observation_id="obs_ser_01",
        source_id="gps_node",
        source_type="gps",
        modality=ModalityType.GPS,
        timestamp=1000.0,
        payload={"fix_type": 3, "satellites": 12},
        confidence=0.98,
        location=GeoLocation(latitude=37.7749, longitude=-122.4194, altitude=12.0, accuracy=1.5),
        device_id="rover_1",
        correlation_id="corr_gps_01",
        expires_at=1060.0,
    )

    d = obs.to_dict()
    # Must be JSON serializable
    json_str = json.dumps(d, sort_keys=True)
    assert "obs_ser_01" in json_str

    # Roundtrip from_dict
    restored = MultimodalObservation.from_dict(json.loads(json_str))
    assert restored.observation_id == obs.observation_id
    assert restored.source_id == obs.source_id
    assert restored.modality == obs.modality
    assert restored.location is not None
    assert pytest.approx(restored.location.latitude) == 37.7749
    assert restored.expires_at == obs.expires_at


def test_deterministic_serialization_situation():
    ev = SituationEvidence(
        evidence_id="ev_ser_01",
        observation_id="obs_ser_01",
        source_id="gps_node",
        modality=ModalityType.GPS,
        evidence_weight=0.85,
        timestamp=1000.0,
        concise_summary="GPS coordinates show rover drift",
    )
    sit = Situation(
        situation_id="sit_ser_01",
        category=SituationCategory.NAVIGATIONAL,
        title="Position Drift",
        description="Vehicle position deviates from expected path",
        severity=SituationSeverity.MEDIUM,
        confidence=0.85,
        status=SituationStatus.ACTIVE,
        involved_entities=("rover_1",),
        supporting_evidence=(ev,),
        created_at=1000.0,
        updated_at=1010.0,
        correlation_id="corr_drift_01",
    )

    d = sit.to_dict()
    json_str = json.dumps(d, sort_keys=True)
    restored = Situation.from_dict(json.loads(json_str))
    assert restored.situation_id == sit.situation_id
    assert restored.category == sit.category
    assert restored.severity == sit.severity
    assert len(restored.supporting_evidence) == 1
    assert restored.supporting_evidence[0].evidence_id == "ev_ser_01"


def test_deterministic_serialization_device_identity():
    cap = DeviceCapabilityDescriptor(
        capability_name="sample_soil",
        action_name="actuate_drill",
        is_reversible=False,
        requires_confirmation=True,
    )
    dev = DeviceIdentity(
        device_id="drill_device",
        device_type=DeviceType.ACTUATOR,
        display_name="Core Soil Drill",
        capabilities=(cap,),
        is_simulation=False,
        connectivity_status=ConnectivityStatus.DEGRADED,
    )
    d = dev.to_dict()
    json_str = json.dumps(d, sort_keys=True)
    restored = DeviceIdentity.from_dict(json.loads(json_str))
    assert restored.device_id == dev.device_id
    assert restored.device_type == DeviceType.ACTUATOR
    assert restored.connectivity_status == ConnectivityStatus.DEGRADED
    assert restored.is_simulation is False
    assert len(restored.capabilities) == 1
    assert restored.capabilities[0].requires_confirmation is True


# ============================================================================
# 6. Unknown Enum & Value Safety
# ============================================================================

def test_unknown_enum_safety():
    assert ModalityType.from_str("NON_EXISTENT") == ModalityType.UNKNOWN
    assert SituationCategory.from_str("MAGIC_CATEGORY") == SituationCategory.UNKNOWN
    assert SituationSeverity.from_str("APOCALYPTIC") == SituationSeverity.INFO
    assert SituationStatus.from_str("TRANSCENDED") == SituationStatus.DETECTED
    assert DeviceType.from_str("FLYING_SAUCER") == DeviceType.UNKNOWN
    assert ConnectivityStatus.from_str("QUANTUM_ENTANGLED") == ConnectivityStatus.UNKNOWN
