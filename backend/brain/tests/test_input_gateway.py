import collections
import inspect
import sys
import threading
import time
from typing import Any, Dict, List, Optional
import pytest

from core.interfaces.orchestration_interface import CentralInputGatewayInterface
from core.interfaces.runtime_interface import CognitiveEventSinkInterface
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
    SituationSeverity,
    SituationStatus,
)
from core.models.runtime import CognitiveEvent, CognitiveEventType
from orchestration.fusion_engine import SituationFusionConfig, SituationFusionEngine
from orchestration.input_gateway import (
    AllowAllAuthValidator,
    AuthenticationValidatorInterface,
    BackpressurePolicy,
    CentralInputGateway,
    DuplicatePolicy,
    GatewayConfig,
    GatewayMetrics,
    IngressEnvelope,
    IngressRejectionReason,
    IngressResult,
    IngressStatus,
    TokenAuthValidator,
    normalize_modality,
)


class MockEventSink(CognitiveEventSinkInterface):
    """Deterministic event sink for testing observability."""
    def __init__(self):
        self.events: List[CognitiveEvent] = []

    def publish(self, event: CognitiveEvent) -> None:
        self.events.append(event)

    def record_event(self, event: CognitiveEvent) -> None:
        self.events.append(event)

    def get_events(self, turn_id: Optional[str] = None) -> List[CognitiveEvent]:
        if turn_id is None:
            return list(self.events)
        return [e for e in self.events if e.turn_id == turn_id]

    def clear(self) -> None:
        self.events.clear()


# ============================================================================
# 1-5: Valid Ingress Across Key Modalities
# ============================================================================

def test_01_valid_text_ingress():
    """Valid text message ingress generates canonical MultimodalObservation."""
    gateway = CentralInputGateway()
    now = 1000.0
    envelope = IngressEnvelope(
        message_id="msg_text_01",
        source_id="user_ui_01",
        source_type="USER_INTERFACE",
        modality="text",
        timestamp=now,
        payload="Turn off the living room lights",
        confidence=1.0,
    )
    result = gateway.receive_envelope(envelope, now=now)
    assert result.status == IngressStatus.ACCEPTED
    assert result.observation is not None
    assert result.observation.observation_id == "msg_text_01"
    assert result.observation.modality == ModalityType.TEXT
    assert result.observation.payload == "Turn off the living room lights"
    assert result.observation.confidence == 1.0


def test_02_valid_image_ingress():
    """Valid image message ingress handles artifact reference and image modality."""
    gateway = CentralInputGateway()
    now = 1000.0
    envelope = IngressEnvelope(
        message_id="msg_img_01",
        source_id="atlas_glass_01",
        source_type="ATLAS_GLASS",
        modality="image",
        timestamp=now,
        payload=None,
        artifact_reference="artifacts/glass/frame_001.jpg",
        confidence=0.95,
        metadata={"camera": "wide_angle"},
    )
    result = gateway.receive_envelope(envelope, now=now)
    assert result.status == IngressStatus.ACCEPTED
    assert result.observation.modality == ModalityType.IMAGE
    assert result.observation.artifact_reference == "artifacts/glass/frame_001.jpg"
    assert result.observation.metadata["camera"] == "wide_angle"


def test_03_valid_audio_event_ingress():
    """Valid audio event ingress normalized properly."""
    gateway = CentralInputGateway()
    now = 1000.0
    envelope = IngressEnvelope(
        message_id="msg_audio_01",
        source_id="glass_mic_01",
        source_type="ATLAS_GLASS",
        modality="audio_event",
        timestamp=now,
        payload={"event": "glass_break", "db_level": 82.5},
        confidence=0.88,
    )
    result = gateway.receive_envelope(envelope, now=now)
    assert result.status == IngressStatus.ACCEPTED
    assert result.observation.modality == ModalityType.AUDIO_EVENT
    assert result.observation.payload["event"] == "glass_break"


def test_04_valid_gps_ingress():
    """Valid GPS ingress normalizes coordinates into GeoLocation."""
    gateway = CentralInputGateway()
    now = 1000.0
    envelope = IngressEnvelope(
        message_id="msg_gps_01",
        source_id="drone_01",
        source_type="DRONE",
        modality="gps",
        timestamp=now,
        payload=None,
        location=GeoLocation(latitude=37.7749, longitude=-122.4194, altitude=120.0, accuracy=2.5),
        confidence=0.99,
    )
    result = gateway.receive_envelope(envelope, now=now)
    assert result.status == IngressStatus.ACCEPTED
    assert result.observation.modality == ModalityType.GPS
    assert result.observation.location is not None
    assert result.observation.location.latitude == 37.7749
    assert result.observation.location.longitude == -122.4194


def test_05_valid_telemetry_ingress():
    """Valid telemetry ingress from rover or sensor."""
    gateway = CentralInputGateway()
    now = 1000.0
    envelope = IngressEnvelope(
        message_id="msg_telem_01",
        source_id="rover_01",
        source_type="ROVER",
        modality="telemetry",
        timestamp=now,
        payload={"battery_pct": 84, "motor_temp_c": 42.1, "speed_mps": 1.2},
        confidence=1.0,
    )
    result = gateway.receive_envelope(envelope, now=now)
    assert result.status == IngressStatus.ACCEPTED
    assert result.observation.modality == ModalityType.TELEMETRY
    assert result.observation.payload["battery_pct"] == 84


# ============================================================================
# 6-9: Schema and Source Identity Validation
# ============================================================================

def test_06_schema_validation_valid():
    """Valid schema version 1.0 passes."""
    gateway = CentralInputGateway(config=GatewayConfig(supported_schema_versions=("1.0",)))
    envelope = IngressEnvelope(
        message_id="msg_schema_01",
        source_id="sensor_01",
        source_type="SENSOR",
        modality="telemetry",
        timestamp=1000.0,
        schema_version="1.0",
    )
    result = gateway.receive_envelope(envelope, now=1000.0)
    assert result.status == IngressStatus.ACCEPTED


def test_07_unsupported_schema():
    """Unsupported schema version is explicitly rejected."""
    gateway = CentralInputGateway(config=GatewayConfig(supported_schema_versions=("1.0",)))
    envelope = IngressEnvelope(
        message_id="msg_schema_bad",
        source_id="sensor_01",
        source_type="SENSOR",
        modality="telemetry",
        timestamp=1000.0,
        schema_version="2.0-experimental",
    )
    result = gateway.receive_envelope(envelope, now=1000.0)
    assert result.status == IngressStatus.REJECTED
    assert result.rejection_reason == IngressRejectionReason.INVALID_SCHEMA
    assert "Unsupported schema version" in result.error_message


def test_08_unknown_source_in_strict_mode():
    """In strict registration mode, unregistered sources are rejected."""
    gateway = CentralInputGateway(config=GatewayConfig(strict_source_registration=True))
    envelope = IngressEnvelope(
        message_id="msg_unregistered",
        source_id="unauthorized_drone",
        source_type="DRONE",
        modality="telemetry",
        timestamp=1000.0,
    )
    result = gateway.receive_envelope(envelope, now=1000.0)
    assert result.status == IngressStatus.REJECTED
    assert result.rejection_reason == IngressRejectionReason.UNKNOWN_SOURCE


def test_09_registered_source_in_strict_mode():
    """Registered source is accepted in strict registration mode."""
    gateway = CentralInputGateway(config=GatewayConfig(strict_source_registration=True))
    drone = DeviceIdentity(
        device_id="ATLAS_DRONE_01",
        device_type=DeviceType.DRONE_AERIAL,
        display_name="Atlas Drone Unit 1",
    )
    gateway.register_device(drone)
    assert gateway.is_source_registered("ATLAS_DRONE_01")

    envelope = IngressEnvelope(
        message_id="msg_reg_drone",
        source_id="ATLAS_DRONE_01",
        source_type="DRONE",
        modality="telemetry",
        timestamp=1000.0,
    )
    result = gateway.receive_envelope(envelope, now=1000.0)
    assert result.status == IngressStatus.ACCEPTED


def test_09b_invalid_source_and_message_identifiers():
    """Missing or empty message_id/source_id rejected deterministically."""
    gateway = CentralInputGateway()
    # Empty message ID
    env1 = IngressEnvelope(
        message_id="",
        source_id="source_01",
        source_type="SENSOR",
        modality="telemetry",
        timestamp=1000.0,
    )
    res1 = gateway.receive_envelope(env1, now=1000.0)
    assert res1.status == IngressStatus.REJECTED
    assert res1.rejection_reason == IngressRejectionReason.INVALID_SCHEMA

    # Empty source ID
    env2 = IngressEnvelope(
        message_id="msg_01",
        source_id="",
        source_type="SENSOR",
        modality="telemetry",
        timestamp=1000.0,
    )
    res2 = gateway.receive_envelope(env2, now=1000.0)
    assert res2.status == IngressStatus.REJECTED
    assert res2.rejection_reason == IngressRejectionReason.INVALID_SCHEMA


# ============================================================================
# 10-13: Timestamp, Freshness, and Clock Skew
# ============================================================================

def test_10_invalid_timestamp():
    """Negative, zero, or NaN timestamps rejected."""
    gateway = CentralInputGateway()
    envelope = IngressEnvelope(
        message_id="msg_bad_ts",
        source_id="sensor_01",
        source_type="SENSOR",
        modality="telemetry",
        timestamp=-5.0,
    )
    result = gateway.receive_envelope(envelope, now=1000.0)
    assert result.status == IngressStatus.REJECTED
    assert result.rejection_reason == IngressRejectionReason.INVALID_TIMESTAMP


def test_11_stale_message_rejected():
    """Message older than max_stale_seconds is rejected."""
    gateway = CentralInputGateway(config=GatewayConfig(max_stale_seconds=300.0))
    now = 1000.0
    stale_ts = now - 350.0  # 350s old
    envelope = IngressEnvelope(
        message_id="msg_stale",
        source_id="sensor_01",
        source_type="SENSOR",
        modality="telemetry",
        timestamp=stale_ts,
    )
    result = gateway.receive_envelope(envelope, now=now)
    assert result.status == IngressStatus.REJECTED
    assert result.rejection_reason == IngressRejectionReason.STALE
    assert gateway.get_metrics().stale_count == 1


def test_12_future_timestamp_beyond_skew_rejected():
    """Message with timestamp in future beyond clock skew tolerance is rejected."""
    gateway = CentralInputGateway(config=GatewayConfig(clock_skew_tolerance_seconds=5.0))
    now = 1000.0
    future_ts = now + 15.0  # 15s in future > 5s tolerance
    envelope = IngressEnvelope(
        message_id="msg_future",
        source_id="sensor_01",
        source_type="SENSOR",
        modality="telemetry",
        timestamp=future_ts,
    )
    result = gateway.receive_envelope(envelope, now=now)
    assert result.status == IngressStatus.REJECTED
    assert result.rejection_reason == IngressRejectionReason.FUTURE_TIMESTAMP


def test_13_clock_skew_tolerance_accepted():
    """Message slightly in future within skew tolerance is accepted and original timestamp preserved."""
    gateway = CentralInputGateway(config=GatewayConfig(clock_skew_tolerance_seconds=5.0))
    now = 1000.0
    skewed_ts = now + 3.0  # within 5.0s tolerance
    envelope = IngressEnvelope(
        message_id="msg_skewed",
        source_id="sensor_01",
        source_type="SENSOR",
        modality="telemetry",
        timestamp=skewed_ts,
    )
    result = gateway.receive_envelope(envelope, now=now)
    assert result.status == IngressStatus.ACCEPTED
    assert result.observation.timestamp == skewed_ts  # preserved!


# ============================================================================
# 14-15: Confidence and Location Validation
# ============================================================================

def test_14_confidence_validation():
    """Confidence out of [0.0, 1.0] bounds is rejected."""
    gateway = CentralInputGateway()
    now = 1000.0
    # > 1.0
    env_high = IngressEnvelope(
        message_id="msg_conf_high",
        source_id="sensor_01",
        source_type="SENSOR",
        modality="telemetry",
        timestamp=now,
        confidence=1.5,
    )
    res_high = gateway.receive_envelope(env_high, now=now)
    assert res_high.status == IngressStatus.REJECTED
    assert res_high.rejection_reason == IngressRejectionReason.INVALID_CONFIDENCE

    # < 0.0
    env_low = IngressEnvelope(
        message_id="msg_conf_low",
        source_id="sensor_01",
        source_type="SENSOR",
        modality="telemetry",
        timestamp=now,
        confidence=-0.1,
    )
    res_low = gateway.receive_envelope(env_low, now=now)
    assert res_low.status == IngressStatus.REJECTED
    assert res_low.rejection_reason == IngressRejectionReason.INVALID_CONFIDENCE


def test_15_location_validation():
    """Invalid latitude/longitude coordinates rejected."""
    gateway = CentralInputGateway()
    now = 1000.0
    # Invalid latitude > 90
    env_bad_lat = IngressEnvelope(
        message_id="msg_bad_lat",
        source_id="sensor_01",
        source_type="SENSOR",
        modality="gps",
        timestamp=now,
        location={"latitude": 95.0, "longitude": 0.0},
    )
    res_lat = gateway.receive_envelope(env_bad_lat, now=now)
    assert res_lat.status == IngressStatus.REJECTED
    assert res_lat.rejection_reason == IngressRejectionReason.INVALID_LOCATION

    # Invalid longitude > 180
    env_bad_lon = IngressEnvelope(
        message_id="msg_bad_lon",
        source_id="sensor_01",
        source_type="SENSOR",
        modality="gps",
        timestamp=now,
        location={"latitude": 10.0, "longitude": 185.0},
    )
    res_lon = gateway.receive_envelope(env_bad_lon, now=now)
    assert res_lon.status == IngressStatus.REJECTED
    assert res_lon.rejection_reason == IngressRejectionReason.INVALID_LOCATION


# ============================================================================
# 16-17: Duplicate Message & Observation Detection
# ============================================================================

def test_16_duplicate_message_suppression():
    """Identical message_id with SUPPRESS policy is suppressed deterministically."""
    gateway = CentralInputGateway(config=GatewayConfig(duplicate_policy=DuplicatePolicy.SUPPRESS))
    now = 1000.0
    env = IngressEnvelope(
        message_id="msg_dup_01",
        source_id="drone_01",
        source_type="DRONE",
        modality="telemetry",
        timestamp=now,
        payload={"altitude": 100},
    )
    res1 = gateway.receive_envelope(env, now=now)
    assert res1.status == IngressStatus.ACCEPTED

    res2 = gateway.receive_envelope(env, now=now + 1.0)
    assert res2.status == IngressStatus.DUPLICATE_SUPPRESSED
    assert res2.rejection_reason == IngressRejectionReason.DUPLICATE
    assert gateway.get_metrics().duplicate_count == 1


def test_17_duplicate_observation_accept_with_metadata():
    """Identical message with ACCEPT_WITH_METADATA policy is marked with metadata."""
    gateway = CentralInputGateway(config=GatewayConfig(duplicate_policy=DuplicatePolicy.ACCEPT_WITH_METADATA))
    now = 1000.0
    env = IngressEnvelope(
        message_id="msg_dup_meta_01",
        source_id="rover_01",
        source_type="ROVER",
        modality="telemetry",
        timestamp=now,
        payload={"speed": 5},
    )
    res1 = gateway.receive_envelope(env, now=now)
    assert res1.status == IngressStatus.ACCEPTED
    assert not res1.observation.metadata.get("is_duplicate", False)

    res2 = gateway.receive_envelope(env, now=now + 0.5)
    assert res2.status == IngressStatus.ACCEPTED_DUPLICATE
    assert res2.observation.metadata.get("is_duplicate") is True


# ============================================================================
# 18-19: Rate Limiting and Payload Size Limits
# ============================================================================

def test_18_rate_limiting_per_source():
    """Messages exceeding rate limit for a single source are rejected."""
    gateway = CentralInputGateway(
        config=GatewayConfig(
            max_messages_per_source_per_second=5,
            rate_limit_window_seconds=1.0,
        )
    )
    now = 1000.0
    # Send 5 messages within 1 second -> all accepted
    for i in range(5):
        env = IngressEnvelope(
            message_id=f"msg_rl_{i}",
            source_id="chatty_sensor",
            source_type="SENSOR",
            modality="telemetry",
            timestamp=now + (i * 0.05),
            payload={"seq": i},
        )
        res = gateway.receive_envelope(env, now=now + (i * 0.05))
        assert res.status == IngressStatus.ACCEPTED

    # 6th message within same 1 second window -> RATE_LIMITED
    env_overflow = IngressEnvelope(
        message_id="msg_rl_overflow",
        source_id="chatty_sensor",
        source_type="SENSOR",
        modality="telemetry",
        timestamp=now + 0.30,
        payload={"seq": 5},
    )
    res_overflow = gateway.receive_envelope(env_overflow, now=now + 0.30)
    assert res_overflow.status == IngressStatus.REJECTED
    assert res_overflow.rejection_reason == IngressRejectionReason.RATE_LIMITED
    assert gateway.get_metrics().rate_limited_count == 1

    # But a different source is NOT rate limited
    env_other = IngressEnvelope(
        message_id="msg_other_source",
        source_id="quiet_sensor",
        source_type="SENSOR",
        modality="telemetry",
        timestamp=now + 0.35,
        payload={"seq": 0},
    )
    assert gateway.receive_envelope(env_other, now=now + 0.35).status == IngressStatus.ACCEPTED


def test_19_payload_size_limit_rejection():
    """Inline payload exceeding max_payload_size_bytes is rejected."""
    gateway = CentralInputGateway(config=GatewayConfig(max_payload_size_bytes=1024))
    now = 1000.0
    oversized_payload = "A" * 2048  # 2048 bytes > 1024 bytes
    env = IngressEnvelope(
        message_id="msg_oversized",
        source_id="sensor_01",
        source_type="SENSOR",
        modality="text",
        timestamp=now,
        payload=oversized_payload,
    )
    res = gateway.receive_envelope(env, now=now)
    assert res.status == IngressStatus.REJECTED
    assert res.rejection_reason == IngressRejectionReason.PAYLOAD_TOO_LARGE


# ============================================================================
# 20-22: Batch Ingestion, Partial Rejection, and Backpressure
# ============================================================================

def test_20_batch_ingestion_success():
    """Batch ingestion of multiple valid envelopes."""
    gateway = CentralInputGateway()
    now = 1000.0
    batch = [
        IngressEnvelope(
            message_id=f"batch_msg_{i}",
            source_id=f"sensor_{i}",
            source_type="SENSOR",
            modality="telemetry",
            timestamp=now,
            payload={"index": i},
        )
        for i in range(10)
    ]
    results = gateway.ingest_batch(batch, now=now)
    assert len(results) == 10
    assert all(r.status == IngressStatus.ACCEPTED for r in results)
    assert gateway.get_metrics().accepted_count == 10


def test_21_partial_batch_rejection():
    """Malformed item in batch does not crash entire batch."""
    gateway = CentralInputGateway()
    now = 1000.0
    batch = [
        IngressEnvelope(
            message_id="valid_01",
            source_id="sensor_01",
            source_type="SENSOR",
            modality="telemetry",
            timestamp=now,
        ),
        IngressEnvelope(
            message_id="bad_02",
            source_id="sensor_02",
            source_type="SENSOR",
            modality="unsupported_modality_type_xyz",
            timestamp=now,
        ),
        IngressEnvelope(
            message_id="valid_03",
            source_id="sensor_03",
            source_type="SENSOR",
            modality="text",
            timestamp=now,
            payload="Hello",
        ),
    ]
    results = gateway.ingest_batch(batch, now=now)
    assert len(results) == 3
    assert results[0].status == IngressStatus.ACCEPTED
    assert results[1].status == IngressStatus.REJECTED
    assert results[1].rejection_reason == IngressRejectionReason.UNSUPPORTED_MODALITY
    assert results[2].status == IngressStatus.ACCEPTED


def test_22_backpressure_reject_policy():
    """Gateway rejects ingress when capacity is reached under REJECT policy."""
    gateway = CentralInputGateway(
        config=GatewayConfig(
            max_buffer_size=5,
            backpressure_policy=BackpressurePolicy.REJECT,
        )
    )
    now = 1000.0
    for i in range(5):
        env = IngressEnvelope(
            message_id=f"buf_msg_{i}",
            source_id=f"sensor_{i}",
            source_type="SENSOR",
            modality="telemetry",
            timestamp=now,
            payload={"idx": i},
        )
        assert gateway.receive_envelope(env, now=now).status == IngressStatus.ACCEPTED

    # 6th message should be rejected due to capacity exceeded
    env_cap = IngressEnvelope(
        message_id="buf_msg_overflow",
        source_id="sensor_overflow",
        source_type="SENSOR",
        modality="telemetry",
        timestamp=now,
        payload={"idx": 99},
    )
    res_cap = gateway.receive_envelope(env_cap, now=now)
    assert res_cap.status == IngressStatus.REJECTED
    assert res_cap.rejection_reason == IngressRejectionReason.CAPACITY_EXCEEDED


def test_22b_backpressure_drop_lowest_priority_policy():
    """Gateway drops telemetry observation to make room under DROP_LOWEST_PRIORITY."""
    gateway = CentralInputGateway(
        config=GatewayConfig(
            max_buffer_size=3,
            backpressure_policy=BackpressurePolicy.DROP_LOWEST_PRIORITY,
        )
    )
    now = 1000.0
    # Ingest 3 telemetry messages
    for i in range(3):
        env = IngressEnvelope(
            message_id=f"telem_{i}",
            source_id=f"src_{i}",
            source_type="SENSOR",
            modality="telemetry",
            timestamp=now,
            payload={"idx": i},
        )
        gateway.receive_envelope(env, now=now)

    # Ingest a text observation -> drops one telemetry and accepts text
    env_text = IngressEnvelope(
        message_id="important_text",
        source_id="user_ui",
        source_type="USER_INTERFACE",
        modality="text",
        timestamp=now,
        payload="Command override",
    )
    res_text = gateway.receive_envelope(env_text, now=now)
    assert res_text.status == IngressStatus.ACCEPTED

    recent = gateway.get_recent_observations()
    assert len(recent) == 3
    assert any(obs.observation_id == "important_text" for obs in recent)


# ============================================================================
# 23-26: Correlation, Causation, Artifacts, and No Binary Duplication
# ============================================================================

def test_23_correlation_id_preservation():
    """Correlation ID from external message is strictly preserved."""
    gateway = CentralInputGateway()
    now = 1000.0
    env = IngressEnvelope(
        message_id="msg_corr_01",
        source_id="drone_01",
        source_type="DRONE",
        modality="telemetry",
        timestamp=now,
        correlation_id="corr_external_98765",
    )
    res = gateway.receive_envelope(env, now=now)
    assert res.status == IngressStatus.ACCEPTED
    assert res.observation.correlation_id == "corr_external_98765"


def test_24_causation_id_preservation():
    """Causation ID is strictly preserved."""
    gateway = CentralInputGateway()
    now = 1000.0
    env = IngressEnvelope(
        message_id="msg_caus_01",
        source_id="drone_01",
        source_type="DRONE",
        modality="telemetry",
        timestamp=now,
        causation_id="caus_trigger_4321",
    )
    res = gateway.receive_envelope(env, now=now)
    assert res.status == IngressStatus.ACCEPTED
    assert res.observation.causation_id == "caus_trigger_4321"


def test_25_artifact_reference_handling():
    """Artifact references pass through without reading disk or copying bytes."""
    gateway = CentralInputGateway()
    now = 1000.0
    env = IngressEnvelope(
        message_id="msg_art_01",
        source_id="camera_01",
        source_type="ATLAS_GLASS",
        modality="image",
        timestamp=now,
        artifact_reference="/data/storage/artifacts/cam_01/img_442.png",
    )
    res = gateway.receive_envelope(env, now=now)
    assert res.status == IngressStatus.ACCEPTED
    assert res.observation.artifact_reference == "/data/storage/artifacts/cam_01/img_442.png"
    assert res.observation.payload is None


def test_26_no_binary_payload_duplication():
    """Large binary payload is not duplicated into multiple places; artifact reference preferred."""
    gateway = CentralInputGateway(config=GatewayConfig(max_payload_size_bytes=512))
    # Large media passes via artifact reference
    env_art = IngressEnvelope(
        message_id="msg_big_art",
        source_id="camera_01",
        source_type="ATLAS_GLASS",
        modality="video_frame",
        timestamp=1000.0,
        payload=None,
        artifact_reference="s3://atlas-media/video/frame_9999.mp4",
    )
    res = gateway.receive_envelope(env_art, now=1000.0)
    assert res.status == IngressStatus.ACCEPTED
    assert res.observation.payload is None
    assert res.observation.artifact_reference == "s3://atlas-media/video/frame_9999.mp4"


# ============================================================================
# 27-30: Simulation, Observability, Metrics, Deterministic Replay
# ============================================================================

def test_27_simulation_source_produces_canonical_multimodal_observation():
    """Virtual/simulation devices produce identical canonical MultimodalObservation type."""
    gateway = CentralInputGateway(config=GatewayConfig(strict_source_registration=True, allow_simulation_sources=True))
    now = 1000.0
    env_sim = IngressEnvelope(
        message_id="sim_msg_01",
        source_id="virtual_drone_01",
        source_type="SIMULATION",
        modality="gps",
        timestamp=now,
        location=GeoLocation(latitude=37.77, longitude=-122.41),
    )
    res = gateway.receive_envelope(env_sim, now=now)
    assert res.status == IngressStatus.ACCEPTED
    assert isinstance(res.observation, MultimodalObservation)
    assert type(res.observation) is MultimodalObservation
    assert res.observation.source_type == "SIMULATION"


def test_28_observability_events_emitted_sanitized():
    """Gateway emits cognitive events on accept/reject, sanitizing credentials."""
    event_sink = MockEventSink()
    gateway = CentralInputGateway(event_sink=event_sink)
    now = 1000.0
    env = IngressEnvelope(
        message_id="msg_obs_01",
        source_id="rover_01",
        source_type="ROVER",
        modality="telemetry",
        timestamp=now,
        auth_token="SECRET_TOKEN_DO_NOT_LEAK",
    )
    res = gateway.receive_envelope(env, now=now)
    assert res.status == IngressStatus.ACCEPTED

    events = event_sink.get_events()
    assert len(events) == 1
    event = events[0]
    assert event.event_type == CognitiveEventType.INGRESS_OBSERVATION_ACCEPTED
    # CRITICAL: token must NOT appear in metadata
    meta_str = str(event.metadata)
    assert "SECRET_TOKEN_DO_NOT_LEAK" not in meta_str
    assert "secret" not in event.metadata


def test_29_gateway_metrics_tracking():
    """Operational counters track received, accepted, rejected, and active sources accurately."""
    gateway = CentralInputGateway(config=GatewayConfig(max_stale_seconds=60.0))
    now = 1000.0
    # 2 valid
    gateway.receive_envelope(
        IngressEnvelope("m1", "s1", "SENSOR", "telemetry", now, payload={"v": 1}),
        now=now,
    )
    gateway.receive_envelope(
        IngressEnvelope("m2", "s2", "SENSOR", "text", now, payload="hi"),
        now=now,
    )
    # 1 stale
    gateway.receive_envelope(
        IngressEnvelope("m3", "s1", "SENSOR", "telemetry", now - 100.0, payload={"v": 2}),
        now=now,
    )

    metrics = gateway.get_metrics()
    assert metrics.received_count == 3
    assert metrics.accepted_count == 2
    assert metrics.rejected_count == 1
    assert metrics.stale_count == 1
    assert metrics.active_sources == 2


def test_30_replay_determinism():
    """Identical replay sequence produces identical outcomes and observation IDs."""
    def run_simulation():
        gw = CentralInputGateway()
        outcomes = []
        clock_ticks = [100.0, 100.2, 100.4, 101.0]
        for i, tick in enumerate(clock_ticks):
            env = IngressEnvelope(
                message_id=f"replay_{i}",
                source_id="drone_alpha",
                source_type="DRONE",
                modality="telemetry",
                timestamp=tick,
                payload={"tick": i},
            )
            res = gw.receive_envelope(env, now=tick)
            outcomes.append((res.status, res.observation.observation_id, res.observation.timestamp))
        return outcomes

    run1 = run_simulation()
    run2 = run_simulation()
    assert run1 == run2


# ============================================================================
# 31-36: Architectural Boundary Rules
# ============================================================================

def test_31_no_world_state_mutation():
    """Gateway has zero WorldState dependencies and performs no world mutations."""
    gw_fields = set(CentralInputGateway.__dict__.keys())
    assert "world_state" not in gw_fields
    assert "mutate_world_state" not in gw_fields
    assert "apply_observation" not in gw_fields


def test_32_no_goal_store_mutation():
    """Gateway does not create goals or interact with GoalStore."""
    gw_fields = set(CentralInputGateway.__dict__.keys())
    assert "goal_store" not in gw_fields
    assert "create_goal" not in gw_fields
    assert "add_goal" not in gw_fields


def test_33_no_tool_call_execution():
    """Gateway does not call ToolOrchestrator or execute tool calls."""
    gw_fields = set(CentralInputGateway.__dict__.keys())
    assert "tool_orchestrator" not in gw_fields
    assert "execute_tool" not in gw_fields


def test_34_no_reasoning_engine_invocation():
    """Gateway contains no ReasoningEngine invocation."""
    gw_fields = set(CentralInputGateway.__dict__.keys())
    assert "reasoning_engine" not in gw_fields
    assert "reason" not in gw_fields


def test_35_no_model_router_invocation():
    """Gateway contains no ModelRouter invocation."""
    gw_fields = set(CentralInputGateway.__dict__.keys())
    assert "model_router" not in gw_fields
    assert "route" not in gw_fields


def test_36_no_hardware_or_network_imports():
    """Module source audit: Zero hardware or network socket imports."""
    import orchestration.input_gateway as gateway_module
    src = inspect.getsource(gateway_module)

    forbidden_patterns = [
        "import socket",
        "import serial",
        "import pyserial",
        "import RPi.GPIO",
        "import paho.mqtt",
        "import pymavlink",
        "import mavsdk",
        "import rclpy",
        "import requests",
        "import httpx",
        "import aiohttp",
        "import urllib.request",
        "import subprocess",
        "os.system",
    ]
    for pattern in forbidden_patterns:
        assert pattern not in src, f"Discovered forbidden pattern '{pattern}' in input_gateway.py"


# ============================================================================
# 37-38: Thread-Safety and End-to-End Situation Fusion Integration
# ============================================================================

def test_37_thread_safety_bounded_state():
    """Concurrent ingress from multiple threads remains safe and bounded."""
    gateway = CentralInputGateway(config=GatewayConfig(max_buffer_size=100))
    now = 1000.0

    def worker(worker_id: int):
        for i in range(20):
            env = IngressEnvelope(
                message_id=f"worker_{worker_id}_msg_{i}",
                source_id=f"worker_source_{worker_id}",
                source_type="SENSOR",
                modality="telemetry",
                timestamp=now + (i * 0.01),
                payload={"worker": worker_id, "step": i},
            )
            gateway.receive_envelope(env, now=now + (i * 0.01))

    threads = [threading.Thread(target=worker, args=(w,)) for w in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    metrics = gateway.get_metrics()
    assert metrics.received_count == 100
    assert metrics.accepted_count == 100
    recent = gateway.get_recent_observations(limit=200)
    assert len(recent) == 100


def test_38_end_to_end_gateway_to_situation_fusion():
    """End-to-end integration: Gateway receives messages -> MultimodalObservation -> SituationFusionEngine."""
    clock_time = 1000.0
    fusion_engine = SituationFusionEngine(clock=lambda: clock_time)
    gateway = CentralInputGateway(fusion_engine=fusion_engine, clock=lambda: clock_time)

    # Device 1: Drone camera spots smoke
    env_smoke = IngressEnvelope(
        message_id="obs_drone_smoke",
        source_id="ATLAS_DRONE_01",
        source_type="DRONE",
        modality="image",
        timestamp=1000.0,
        payload="Smoke plume detected at perimeter",
        location=GeoLocation(latitude=37.7749, longitude=-122.4194),
        confidence=0.85,
        correlation_id="INCIDENT_FIRE_01",
    )
    res_smoke = gateway.receive_envelope(env_smoke, now=1000.0)
    assert res_smoke.status == IngressStatus.ACCEPTED

    # Device 2: Rover thermal camera detects high heat at same incident location
    clock_time = 1002.0
    env_thermal = IngressEnvelope(
        message_id="obs_rover_heat",
        source_id="ATLAS_ROVER_01",
        source_type="ROVER",
        modality="telemetry",
        timestamp=1002.0,
        payload="Thermal spike: 180C",
        location=GeoLocation(latitude=37.7750, longitude=-122.4195),
        confidence=0.90,
        correlation_id="INCIDENT_FIRE_01",
    )
    res_thermal = gateway.receive_envelope(env_thermal, now=1002.0)
    assert res_thermal.status == IngressStatus.ACCEPTED

    # Verify SituationFusionEngine fused them into an ACTIVE situation
    situations = fusion_engine.get_active_situations(now=1002.0)
    assert len(situations) == 1
    sit = situations[0]
    assert len(sit.supporting_evidence) == 2
    assert "ATLAS_DRONE_01" in sit.involved_entities
    assert "ATLAS_ROVER_01" in sit.involved_entities
    assert sit.status == SituationStatus.ACTIVE


# ============================================================================
# Section 27: Security Tests
# ============================================================================

def test_sec_01_authentication_failure_handled_safely():
    """Token mismatch triggers AUTHENTICATION_FAILED without exceptions or leaks."""
    validator = TokenAuthValidator(valid_tokens=["VALID_TOKEN_SECRET_123"])
    gateway = CentralInputGateway(auth_validator=validator)
    now = 1000.0

    # Malicious/invalid token
    env_bad = IngressEnvelope(
        message_id="msg_sec_bad",
        source_id="attacker",
        source_type="DRONE",
        modality="telemetry",
        timestamp=now,
        auth_token="MALICIOUS_TOKEN_XYZ",
    )
    res_bad = gateway.receive_envelope(env_bad, now=now)
    assert res_bad.status == IngressStatus.REJECTED
    assert res_bad.rejection_reason == IngressRejectionReason.AUTHENTICATION_FAILED

    # Valid token succeeds
    env_good = IngressEnvelope(
        message_id="msg_sec_good",
        source_id="authorized_drone",
        source_type="DRONE",
        modality="telemetry",
        timestamp=now,
        auth_token="VALID_TOKEN_SECRET_123",
    )
    res_good = gateway.receive_envelope(env_good, now=now)
    assert res_good.status == IngressStatus.ACCEPTED


def test_sec_02_unknown_source_cannot_bypass_registration_policy():
    """Spoofed source_id is rejected when not registered under strict registration mode."""
    gateway = CentralInputGateway(
        config=GatewayConfig(
            strict_source_registration=True,
            allow_simulation_sources=False,
        )
    )
    now = 1000.0
    env_spoof = IngressEnvelope(
        message_id="msg_spoof",
        source_id="ATLAS_DRONE_999",  # not registered!
        source_type="DRONE",
        modality="telemetry",
        timestamp=now,
    )
    res = gateway.receive_envelope(env_spoof, now=now)
    assert res.status == IngressStatus.REJECTED
    assert res.rejection_reason == IngressRejectionReason.UNKNOWN_SOURCE


def test_sec_03_oversized_payload_rejected_bounds_memory():
    """Huge payload (>64KB default) is rejected deterministically."""
    gateway = CentralInputGateway()
    now = 1000.0
    huge_data = {"junk": "x" * 70000}
    env_huge = IngressEnvelope(
        message_id="msg_huge",
        source_id="sensor_01",
        source_type="SENSOR",
        modality="telemetry",
        timestamp=now,
        payload=huge_data,
    )
    res = gateway.receive_envelope(env_huge, now=now)
    assert res.status == IngressStatus.REJECTED
    assert res.rejection_reason == IngressRejectionReason.PAYLOAD_TOO_LARGE


def test_sec_04_tokens_never_appear_in_observation_metadata():
    """Authentication tokens never leak into MultimodalObservation metadata."""
    validator = TokenAuthValidator(valid_tokens=["SECRET_BEARER_KEY_777"])
    gateway = CentralInputGateway(auth_validator=validator)
    now = 1000.0

    env = IngressEnvelope(
        message_id="msg_auth_clean",
        source_id="secure_glass",
        source_type="ATLAS_GLASS",
        modality="text",
        timestamp=now,
        auth_token="SECRET_BEARER_KEY_777",
        auth_metadata={"token": "SECRET_BEARER_KEY_777", "algo": "HS256"},
    )
    res = gateway.receive_envelope(env, now=now)
    assert res.status == IngressStatus.ACCEPTED
    obs = res.observation
    assert "auth_token" not in obs.metadata
    assert "auth_metadata" not in obs.metadata
    assert "SECRET_BEARER_KEY_777" not in str(obs.to_dict())


# ============================================================================
# Section 28: Performance Tests
# ============================================================================

def test_perf_01_scaling_benchmarks_10_100_500_messages():
    """Benchmark gateway throughput and latency for 10, 100, and 500 messages."""
    gateway = CentralInputGateway(config=GatewayConfig(max_messages_per_source_per_second=10000))
    now = 1000.0

    for count in (10, 100, 500):
        base_time = now + (count * 1000.0)
        batch = [
            IngressEnvelope(
                message_id=f"perf_{count}_{i}",
                source_id=f"device_{i % 20}",
                source_type="SENSOR",
                modality="telemetry",
                timestamp=base_time + (i * 0.001),
                payload={"val": i, "batch": count},
            )
            for i in range(count)
        ]

        t0 = time.perf_counter()
        results = gateway.ingest_batch(batch, now=base_time + 1.0)
        t1 = time.perf_counter()
        elapsed_ms = (t1 - t0) * 1000.0

        assert len(results) == count
        assert all(r.status == IngressStatus.ACCEPTED for r in results)
        assert elapsed_ms < 500.0, f"{count} msgs took {elapsed_ms:.2f}ms"
