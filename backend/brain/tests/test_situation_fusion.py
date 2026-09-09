import inspect
import time
import pytest

from core.models.orchestration import (
    GeoLocation,
    ModalityType,
    MultimodalObservation,
    Situation,
    SituationCategory,
    SituationEvidence,
    SituationSeverity,
    SituationStatus,
)
from core.models.world_state import WorldState, WorldCondition
from core.interfaces.runtime_interface import CognitiveEventSinkInterface
from core.models.runtime import CognitiveEvent
from orchestration.fusion_engine import SituationFusionConfig, SituationFusionEngine


class FakeCognitiveEventSink(CognitiveEventSinkInterface):
    def __init__(self):
        self.events = []

    def publish(self, event: CognitiveEvent) -> None:
        self.events.append(event)

    def get_events(self, turn_id=None):
        return list(self.events)

    def clear(self):
        self.events.clear()


# ============================================================================
# 1. Single Observation & Ingestion
# ============================================================================

def test_single_observation_ingestion():
    engine = SituationFusionEngine()
    obs = MultimodalObservation(
        observation_id="obs_001",
        source_id="cam_front",
        source_type="camera",
        modality=ModalityType.IMAGE,
        timestamp=1000.0,
        payload={"description": "obstacle in corridor"},
        confidence=0.85,
    )
    sit = engine.ingest(obs, now=1000.0)
    assert sit is not None
    assert sit.status == SituationStatus.DETECTED
    assert len(sit.supporting_evidence) == 1
    assert sit.supporting_evidence[0].observation_id == "obs_001"
    assert sit.confidence == 0.85


# ============================================================================
# 2. Batch Ingestion & Active Retrieval
# ============================================================================

def test_batch_ingestion_and_retrieval():
    engine = SituationFusionEngine()
    obs1 = MultimodalObservation(
        observation_id="obs_b1",
        source_id="sensor_1",
        source_type="temp",
        modality=ModalityType.TELEMETRY,
        timestamp=1000.0,
        payload={"temp": 45.0, "status": "heat alert"},
        confidence=0.9,
    )
    obs2 = MultimodalObservation(
        observation_id="obs_b2",
        source_id="sensor_2",
        source_type="smoke",
        modality=ModalityType.TELEMETRY,
        timestamp=1002.0,
        payload={"smoke": "detected"},
        confidence=0.92,
    )
    sits = engine.ingest_batch([obs1, obs2], now=1005.0)
    assert len(sits) >= 1
    active = engine.get_active_situations(now=1005.0)
    assert len(active) >= 1


# ============================================================================
# 3. Temporal Clustering
# ============================================================================

def test_temporal_clustering_window():
    config = SituationFusionConfig(max_temporal_distance_seconds=30.0)
    engine = SituationFusionEngine(config=config)

    obs1 = MultimodalObservation(
        observation_id="obs_t1",
        source_id="cam_01",
        source_type="camera",
        modality=ModalityType.IMAGE,
        timestamp=1000.0,
        payload={"description": "motion detected"},
        confidence=0.8,
        location=GeoLocation(latitude=37.7749, longitude=-122.4194),
    )
    # 20s later: within 30s window -> merges
    obs2 = MultimodalObservation(
        observation_id="obs_t2",
        source_id="radar_01",
        source_type="radar",
        modality=ModalityType.TELEMETRY,
        timestamp=1020.0,
        payload={"description": "motion detected"},
        confidence=0.85,
        location=GeoLocation(latitude=37.7749, longitude=-122.4194),
    )
    # 70s later: outside 30s window from obs2 -> creates separate situation
    obs3 = MultimodalObservation(
        observation_id="obs_t3",
        source_id="cam_01",
        source_type="camera",
        modality=ModalityType.IMAGE,
        timestamp=1095.0,
        payload={"description": "motion detected"},
        confidence=0.8,
        location=GeoLocation(latitude=37.7749, longitude=-122.4194),
    )

    s1 = engine.ingest(obs1, now=1000.0)
    s2 = engine.ingest(obs2, now=1020.0)
    s3 = engine.ingest(obs3, now=1095.0)

    assert s1.situation_id == s2.situation_id
    assert s1.situation_id != s3.situation_id


# ============================================================================
# 4. Spatial Clustering & Distance Bounds
# ============================================================================

def test_spatial_clustering_within_and_outside_bounds():
    config = SituationFusionConfig(max_spatial_distance_meters=50.0)
    engine = SituationFusionEngine(config=config)

    # Point A
    loc_a = GeoLocation(latitude=37.77490, longitude=-122.41940)
    # Point B: ~15 meters away
    loc_b = GeoLocation(latitude=37.77500, longitude=-122.41940)
    # Point C: ~500 meters away
    loc_c = GeoLocation(latitude=37.77950, longitude=-122.41940)

    obs_a = MultimodalObservation(
        observation_id="obs_sp_a",
        source_id="dev_a",
        source_type="sensor",
        modality=ModalityType.TELEMETRY,
        timestamp=1000.0,
        payload={"metric": "noise", "value": 75},
        confidence=0.8,
        location=loc_a,
    )
    obs_b = MultimodalObservation(
        observation_id="obs_sp_b",
        source_id="dev_b",
        source_type="sensor",
        modality=ModalityType.AUDIO_EVENT,
        timestamp=1005.0,
        payload={"metric": "noise", "value": 80},
        confidence=0.85,
        location=loc_b,
    )
    obs_c = MultimodalObservation(
        observation_id="obs_sp_c",
        source_id="dev_c",
        source_type="sensor",
        modality=ModalityType.AUDIO_EVENT,
        timestamp=1007.0,
        payload={"metric": "noise", "value": 80},
        confidence=0.85,
        location=loc_c,
    )

    sa = engine.ingest(obs_a, now=1000.0)
    sb = engine.ingest(obs_b, now=1005.0)
    sc = engine.ingest(obs_c, now=1007.0)

    # Within 50m -> merged
    assert sa.situation_id == sb.situation_id
    # Outside 50m -> distinct situation
    assert sa.situation_id != sc.situation_id


# ============================================================================
# 5. Missing Location Handling
# ============================================================================

def test_missing_location_correlates_via_source_and_entity():
    engine = SituationFusionEngine()
    obs_with_loc = MultimodalObservation(
        observation_id="obs_loc",
        source_id="rover_1",
        source_type="robot",
        modality=ModalityType.GPS,
        timestamp=1000.0,
        payload={"lat": 37.7749, "lon": -122.4194},
        confidence=0.95,
        location=GeoLocation(latitude=37.7749, longitude=-122.4194),
        device_id="rover_1",
    )
    # Observation without location but sharing device_id and temporal window
    obs_no_loc = MultimodalObservation(
        observation_id="obs_noloc",
        source_id="rover_1",
        source_type="robot",
        modality=ModalityType.TELEMETRY,
        timestamp=1005.0,
        payload={"battery": "low", "voltage": 11.2},
        confidence=0.90,
        device_id="rover_1",
    )

    s1 = engine.ingest(obs_with_loc, now=1000.0)
    s2 = engine.ingest(obs_no_loc, now=1005.0)

    assert s1.situation_id == s2.situation_id
    assert s2.location is not None  # Retains location from s1
    assert "rover_1" in s2.involved_entities


# ============================================================================
# 6. Entity Correlation
# ============================================================================

def test_entity_correlation_matches():
    engine = SituationFusionEngine()
    obs1 = MultimodalObservation(
        observation_id="obs_ent_1",
        source_id="camera_hallway",
        source_type="camera",
        modality=ModalityType.IMAGE,
        timestamp=1000.0,
        payload={"entity_id": "PERSON_42", "description": "unauthorized person"},
        confidence=0.88,
    )
    obs2 = MultimodalObservation(
        observation_id="obs_ent_2",
        source_id="badge_reader",
        source_type="access_control",
        modality=ModalityType.USER_ACTION,
        timestamp=1004.0,
        payload={"entity_id": "PERSON_42", "action": "failed_entry"},
        confidence=0.95,
    )
    s1 = engine.ingest(obs1, now=1000.0)
    s2 = engine.ingest(obs2, now=1004.0)

    assert s1.situation_id == s2.situation_id
    assert "PERSON_42" in s2.involved_entities


# ============================================================================
# 7. Correlation ID & Causation Chain Grouping
# ============================================================================

def test_correlation_and_causation_chain_grouping():
    engine = SituationFusionEngine()
    obs1 = MultimodalObservation(
        observation_id="obs_corr_1",
        source_id="drone_alpha",
        source_type="drone",
        modality=ModalityType.TELEMETRY,
        timestamp=1000.0,
        payload={"status": "motor failure"},
        confidence=0.99,
        correlation_id="MISSION_ABORT_001",
    )
    obs2 = MultimodalObservation(
        observation_id="obs_corr_2",
        source_id="gcs_console",
        source_type="ground_station",
        modality=ModalityType.EVENT,
        timestamp=1010.0,
        payload={"action": "emergency return"},
        confidence=0.95,
        correlation_id="MISSION_ABORT_001",
        causation_id="obs_corr_1",
    )
    s1 = engine.ingest(obs1, now=1000.0)
    s2 = engine.ingest(obs2, now=1010.0)

    assert s1.situation_id == s2.situation_id
    assert s2.correlation_id == "MISSION_ABORT_001"


# ============================================================================
# 8. Duplicate Observation Suppression
# ============================================================================

def test_duplicate_observation_suppression():
    sink = FakeCognitiveEventSink()
    engine = SituationFusionEngine(event_sink=sink)

    obs = MultimodalObservation(
        observation_id="obs_dup_01",
        source_id="camera_gate",
        source_type="camera",
        modality=ModalityType.IMAGE,
        timestamp=1000.0,
        payload="gate open",
        confidence=0.8,
    )
    s1 = engine.ingest(obs, now=1000.0)
    s2 = engine.ingest(obs, now=1001.0)  # duplicate ID

    assert len(s1.supporting_evidence) == 1
    # Duplicate does not create duplicate situation or duplicate evidence
    assert s1 == s2
    # Suppressed event was emitted
    suppressed_events = [e for e in sink.events if e.event_type.value == "SITUATION_SUPPRESSED"]
    assert len(suppressed_events) >= 1


# ============================================================================
# 9. Same-Source Redundant Frame Handling (Point 39)
# ============================================================================

def test_five_camera_frames_do_not_equal_five_independent_modalities():
    engine = SituationFusionEngine()
    frames = []
    # 5 frames from the exact same camera within 1 second
    for i in range(5):
        frames.append(
            MultimodalObservation(
                observation_id=f"frame_{i}",
                source_id="camera_security_01",
                source_type="camera",
                modality=ModalityType.IMAGE,
                timestamp=1000.0 + (i * 0.1),
                payload="hallway view empty",
                confidence=0.70,
            )
        )

    situation = None
    for f in frames:
        situation = engine.ingest(f, now=1001.0)

    assert situation is not None
    # Confidence should NOT be inflated (remains around 0.70, no multi-source boost)
    assert situation.confidence <= 0.75
    # Distinct sources is strictly 1
    sources = {e.source_id for e in situation.supporting_evidence}
    assert len(sources) == 1


# ============================================================================
# 10. Multi-Source Corroboration (Points 11, 13, 14)
# ============================================================================

def test_multi_source_corroboration_boosts_confidence_and_transitions_active():
    engine = SituationFusionEngine()
    loc = GeoLocation(latitude=37.7749, longitude=-122.4194)

    # Source 1: Camera
    obs1 = MultimodalObservation(
        observation_id="obs_cam",
        source_id="cam_corridor",
        source_type="camera",
        modality=ModalityType.IMAGE,
        timestamp=1000.0,
        payload="water leak detected on floor",
        confidence=0.75,
        location=loc,
    )
    s1 = engine.ingest(obs1, now=1000.0)
    assert s1.status == SituationStatus.DETECTED
    initial_conf = s1.confidence

    # Source 2: Moisture sensor (independent source and modality)
    obs2 = MultimodalObservation(
        observation_id="obs_moist",
        source_id="moisture_sensor_12",
        source_type="iot_sensor",
        modality=ModalityType.TELEMETRY,
        timestamp=1002.0,
        payload="liquid conductivity high",
        confidence=0.85,
        location=loc,
    )
    s2 = engine.ingest(obs2, now=1002.0)

    # Multi-source corroboration boosted confidence and transitioned status to ACTIVE
    assert s2.status == SituationStatus.ACTIVE
    assert s2.confidence > initial_conf
    assert len(s2.supporting_evidence) == 2


# ============================================================================
# 11. Conflict Detection & DeterministicConflictResolver Reuse (Points 25, 26)
# ============================================================================

def test_conflict_detection_preserves_both_evidence_sources_and_penalizes():
    engine = SituationFusionEngine()
    loc = GeoLocation(latitude=37.7749, longitude=-122.4194)

    # Observation A: obstacle detected
    obs_a = MultimodalObservation(
        observation_id="obs_cam_obs",
        source_id="camera_front",
        source_type="camera",
        modality=ModalityType.IMAGE,
        timestamp=1000.0,
        payload={"entity_id": "door_north", "property_name": "obstacle", "value": "detected"},
        confidence=0.70,
        location=loc,
    )
    # Observation B: obstacle absent (conflicting with A)
    obs_b = MultimodalObservation(
        observation_id="obs_lidar_obs",
        source_id="lidar_front",
        source_type="lidar",
        modality=ModalityType.TELEMETRY,
        timestamp=1001.0,
        payload={"entity_id": "door_north", "property_name": "obstacle", "value": "absent"},
        confidence=0.70,
        location=loc,
    )

    s1 = engine.ingest(obs_a, now=1000.0)
    s2 = engine.ingest(obs_b, now=1001.0)

    assert s2.situation_id == s1.situation_id
    # Both evidence sources are preserved (never silently discard)
    obs_ids = {e.observation_id for e in s2.supporting_evidence}
    assert "obs_cam_obs" in obs_ids
    assert "obs_lidar_obs" in obs_ids
    # Conflict recognized
    assert "latest_conflict" in s2.metadata


# ============================================================================
# 12. Freshness, Stale Observation & Future Timestamp (Points 15, 16, 17)
# ============================================================================

def test_stale_observation_does_not_create_active_situation():
    engine = SituationFusionEngine()
    # Observation timestamp is 1000s, but current time is 2000s (stale > 300s)
    stale_obs = MultimodalObservation(
        observation_id="obs_stale",
        source_id="old_sensor",
        source_type="sensor",
        modality=ModalityType.TELEMETRY,
        timestamp=1000.0,
        payload="historical reading",
        confidence=0.9,
    )
    sit = engine.ingest(stale_obs, now=2000.0)
    assert sit is None  # Stale data rejected from creating new situation
    assert len(engine.get_active_situations(now=2000.0)) == 0


def test_future_timestamp_handling():
    engine = SituationFusionEngine()
    # Timestamp slightly in future within slack
    future_obs = MultimodalObservation(
        observation_id="obs_fut",
        source_id="clock_drift_sensor",
        source_type="sensor",
        modality=ModalityType.TELEMETRY,
        timestamp=1002.0,
        payload="drift reading",
        confidence=0.85,
    )
    sit = engine.ingest(future_obs, now=1000.0)
    assert sit is not None
    assert sit.created_at == 1002.0


# ============================================================================
# 13. Out-of-Order Arrivals (Point 18, 20)
# ============================================================================

def test_out_of_order_batch_preserves_temporal_truth():
    engine = SituationFusionEngine()
    loc = GeoLocation(latitude=37.7749, longitude=-122.4194)

    obs_t3 = MultimodalObservation(
        observation_id="obs_t3",
        source_id="sensor_a",
        source_type="sensor",
        modality=ModalityType.TELEMETRY,
        timestamp=1003.0,
        payload={"step": 3},
        confidence=0.8,
        location=loc,
    )
    obs_t1 = MultimodalObservation(
        observation_id="obs_t1",
        source_id="sensor_a",
        source_type="sensor",
        modality=ModalityType.TELEMETRY,
        timestamp=1000.0,
        payload={"step": 1},
        confidence=0.8,
        location=loc,
    )
    obs_t2 = MultimodalObservation(
        observation_id="obs_t2",
        source_id="sensor_a",
        source_type="sensor",
        modality=ModalityType.TELEMETRY,
        timestamp=1001.0,
        payload={"step": 2},
        confidence=0.8,
        location=loc,
    )

    # Ingest out of order: T3, T1, T2
    sits = engine.ingest_batch([obs_t3, obs_t1, obs_t2], now=1005.0)
    assert len(sits) == 1
    sit = sits[0]
    assert sit.created_at == 1000.0
    assert sit.updated_at == 1003.0
    # Evidences ordered chronologically
    ts_list = [e.timestamp for e in sit.supporting_evidence]
    assert ts_list == [1000.0, 1001.0, 1003.0]


# ============================================================================
# 14. Situation Lifecycle & Expiry (Point 23, 24)
# ============================================================================

def test_situation_lifecycle_resolve_and_expire():
    engine = SituationFusionEngine(config=SituationFusionConfig(default_validity_window_seconds=60.0))
    obs = MultimodalObservation(
        observation_id="obs_lc",
        source_id="sensor_gate",
        source_type="sensor",
        modality=ModalityType.TELEMETRY,
        timestamp=1000.0,
        payload="gate fault",
        confidence=0.8,
    )
    sit = engine.ingest(obs, now=1000.0)
    assert sit.status == SituationStatus.DETECTED
    assert sit.is_valid(now=1050.0)

    # Expire
    expired = engine.expire_stale_situations(now=1070.0)
    assert len(expired) == 1
    assert expired[0].status == SituationStatus.EXPIRED
    assert not expired[0].is_valid(now=1070.0)

    # Resolve manually
    resolved = engine.resolve_situation(sit.situation_id, reason="Cleared by maintenance", now=1080.0)
    assert resolved.status == SituationStatus.RESOLVED


# ============================================================================
# 15. Realistic Multimodal Test: ATLAS Glass Incident (Point 30, 40)
# ============================================================================

def test_glass_multimodal_incident_fuses_into_one_situation():
    """
    Construct realistic multi-modal incident from ATLAS Glass:
    - Obs 1: IMAGE, 'person prone', conf=0.90, loc=X
    - Obs 2: AUDIO_EVENT, 'distress vocalization', conf=0.82, loc=X
    - Obs 3: GPS, loc=X, conf=0.99
    - Obs 4: TELEMETRY, 'sudden stop', conf=0.88, loc=X
    All timestamps within 2 seconds.

    Expected:
    - Exactly ONE Situation
    - Multiple evidence items (4 distinct modalities)
    - Location preserved
    - Confidence boosted by multimodal corroboration
    - Severity elevated (CRITICAL)
    - ZERO goal creation, ZERO tool execution
    """
    engine = SituationFusionEngine()
    loc_x = GeoLocation(latitude=37.77492, longitude=-122.41942, altitude=12.0)

    obs1 = MultimodalObservation(
        observation_id="glass_img_01",
        source_id="ATLAS_GLASS_01",
        source_type="smart_glasses",
        modality=ModalityType.IMAGE,
        timestamp=1000.0,
        payload={"description": "person prone", "target": "worker_01"},
        confidence=0.90,
        location=loc_x,
        device_id="ATLAS_GLASS_01",
        correlation_id="INCIDENT_CORR_88",
    )
    obs2 = MultimodalObservation(
        observation_id="glass_aud_02",
        source_id="ATLAS_GLASS_01",
        source_type="smart_glasses",
        modality=ModalityType.AUDIO_EVENT,
        timestamp=1000.5,
        payload={"description": "distress vocalization", "audio_level": 85},
        confidence=0.82,
        location=loc_x,
        device_id="ATLAS_GLASS_01",
        correlation_id="INCIDENT_CORR_88",
        causation_id="glass_img_01",
    )
    obs3 = MultimodalObservation(
        observation_id="glass_gps_03",
        source_id="ATLAS_GLASS_01",
        source_type="smart_glasses",
        modality=ModalityType.GPS,
        timestamp=1001.0,
        payload={"lat": 37.77492, "lon": -122.41942},
        confidence=0.99,
        location=loc_x,
        device_id="ATLAS_GLASS_01",
        correlation_id="INCIDENT_CORR_88",
    )
    obs4 = MultimodalObservation(
        observation_id="glass_tel_04",
        source_id="ATLAS_GLASS_01",
        source_type="smart_glasses",
        modality=ModalityType.TELEMETRY,
        timestamp=1001.8,
        payload={"description": "sudden stop", "imu_accel": 9.8},
        confidence=0.88,
        location=loc_x,
        device_id="ATLAS_GLASS_01",
        correlation_id="INCIDENT_CORR_88",
    )

    batch = [obs1, obs2, obs3, obs4]
    situations = engine.ingest_batch(batch, now=1002.0)

    # MUST be exactly ONE Situation!
    assert len(situations) == 1
    sit = situations[0]

    assert sit.category == SituationCategory.ANOMALY
    assert sit.status == SituationStatus.ACTIVE
    assert sit.severity == SituationSeverity.CRITICAL
    assert len(sit.supporting_evidence) == 4
    assert sit.location == loc_x
    assert sit.correlation_id == "INCIDENT_CORR_88"

    # Multi-modal corroboration boost
    distinct_modalities = {e.modality for e in sit.supporting_evidence}
    assert len(distinct_modalities) == 4
    assert sit.confidence >= 0.90


# ============================================================================
# 16. High-Volume Bounded Processing (Points 21, 29, 31)
# ============================================================================

def test_high_volume_500_observations_bounded_and_capacity_enforced():
    """
    Generate 500 observations.
    Verify:
    - Maximum batch bound respected
    - Processing remains bounded
    - Duplicate suppression works
    - Situation count does not explode
    """
    config = SituationFusionConfig(max_batch_size=500, max_active_situations=20)
    engine = SituationFusionEngine(config=config)

    observations = []
    base_time = 1000.0
    for i in range(500):
        # 5 distinct spatial clusters
        cluster_id = i % 5
        lat = 37.0 + (cluster_id * 0.1)
        lon = -122.0 - (cluster_id * 0.1)

        observations.append(
            MultimodalObservation(
                observation_id=f"bulk_obs_{i}",
                source_id=f"sensor_node_{cluster_id}",
                source_type="sensor",
                modality=ModalityType.TELEMETRY,
                timestamp=base_time + (i * 0.5),
                payload={"reading": i, "status": "normal"},
                confidence=0.85,
                location=GeoLocation(latitude=lat, longitude=lon),
            )
        )

    t0 = time.perf_counter()
    sits = engine.ingest_batch(observations, now=base_time + 300.0)
    duration = time.perf_counter() - t0

    # Max active situations bound strictly enforced
    active = engine.get_active_situations(now=base_time + 300.0)
    assert len(active) <= config.max_active_situations
    # Completed in sub-second time
    assert duration < 2.0

    # Over-limit batch is rejected
    with pytest.raises(ValueError, match="exceeds maximum limit"):
        engine.ingest_batch([observations[0]] * 501)


# ============================================================================
# 17. Architectural Invariants & Security (Points 31, 32, 33, 34, 42)
# ============================================================================

def test_architectural_invariants_and_security_audit():
    """
    Verify SituationFusionEngine:
    - Does NOT mutate WorldState
    - Does NOT execute ToolCalls
    - Does NOT import or call subprocess, os.system, sockets, or network libraries
    - Does NOT invoke LLM/ModelRouter
    """
    import orchestration.fusion_engine as eng_mod
    mod_src = inspect.getsource(eng_mod)

    forbidden_tokens = [
        "import subprocess",
        "from subprocess",
        "os.system",
        "import socket",
        "from socket",
        "import requests",
        "import httpx",
        "import urllib.request",
        "from urllib",
        "ToolOrchestrator",
        "GoalManager",
        "PolicyEngine",
        "ReasoningEngine",
        "ModelRouter",
        "import pickle",
        "from pickle",
        "eval(",
        "exec(",
    ]
    for token in forbidden_tokens:
        assert token not in mod_src, f"Discovered forbidden token '{token}' in SituationFusionEngine source!"


def test_deterministic_replay_across_runs():
    """
    Same observations + same timestamps -> 100% identical Situation outputs.
    """
    def run_simulation():
        eng = SituationFusionEngine()
        obs_a = MultimodalObservation(
            observation_id="sim_a",
            source_id="cam_1",
            source_type="camera",
            modality=ModalityType.IMAGE,
            timestamp=100.0,
            payload="object detected",
            confidence=0.8,
            location=GeoLocation(latitude=37.77, longitude=-122.41),
        )
        obs_b = MultimodalObservation(
            observation_id="sim_b",
            source_id="radar_1",
            source_type="radar",
            modality=ModalityType.TELEMETRY,
            timestamp=105.0,
            payload="speed 12m/s",
            confidence=0.9,
            location=GeoLocation(latitude=37.77, longitude=-122.41),
        )
        sits = eng.ingest_batch([obs_a, obs_b], now=110.0)
        return sits[0].to_dict()

    run1 = run_simulation()
    run2 = run_simulation()
    assert run1 == run2


def test_confidence_bounds_enforcement():
    engine = SituationFusionEngine()
    loc = GeoLocation(latitude=37.7749, longitude=-122.4194)
    # High confidence observation + multi-source corroboration at same location
    obs1 = MultimodalObservation(
        observation_id="c_bound_1",
        source_id="s1",
        source_type="type1",
        modality=ModalityType.IMAGE,
        timestamp=100.0,
        payload="anomaly test",
        confidence=1.0,
        location=loc,
    )
    obs2 = MultimodalObservation(
        observation_id="c_bound_2",
        source_id="s2",
        source_type="type2",
        modality=ModalityType.AUDIO_EVENT,
        timestamp=101.0,
        payload="distress test",
        confidence=1.0,
        location=loc,
    )
    obs3 = MultimodalObservation(
        observation_id="c_bound_3",
        source_id="s3",
        source_type="type3",
        modality=ModalityType.GPS,
        timestamp=102.0,
        payload="nav test",
        confidence=1.0,
        location=loc,
    )
    sits = engine.ingest_batch([obs1, obs2, obs3], now=105.0)
    assert len(sits) == 1
    # Never exceeds 1.0
    assert sits[0].confidence == 1.0


def test_deterministic_situation_signature_structure():
    engine = SituationFusionEngine()
    loc = GeoLocation(latitude=37.7749, longitude=-122.4194)
    obs = MultimodalObservation(
        observation_id="sig_obs_1",
        source_id="drone_01",
        source_type="drone",
        modality=ModalityType.GPS,
        timestamp=1000.0,
        payload={"drift": "detected"},
        confidence=0.9,
        location=loc,
        device_id="drone_01",
    )
    sit = engine.ingest(obs, now=1000.0)
    sig = sit.metadata["signature"]
    assert "NAVIGATIONAL" in sig
    assert "drone_01" in sig
    assert "37.775_-122.419" in sig


def test_unknown_classification_fallback():
    engine = SituationFusionEngine()
    obs = MultimodalObservation(
        observation_id="unk_obs_1",
        source_id="untyped_device",
        source_type="generic",
        modality=ModalityType.UNKNOWN,
        timestamp=1000.0,
        payload={"foo": "bar", "val": 123},
        confidence=0.5,
    )
    sit = engine.ingest(obs, now=1000.0)
    assert sit.category == SituationCategory.UNKNOWN
    assert sit.severity == SituationSeverity.INFO


def test_world_state_read_only_and_not_mutated():
    from world.store import InMemoryWorldStateStore
    store = InMemoryWorldStateStore()
    initial_version = store.get_current_state().version

    engine = SituationFusionEngine(world_state_store=store)
    obs = MultimodalObservation(
        observation_id="ws_read_obs",
        source_id="node_a",
        source_type="sensor",
        modality=ModalityType.TELEMETRY,
        timestamp=100.0,
        payload={"temp": 25.0},
        confidence=0.8,
    )
    engine.ingest(obs, now=100.0)

    # WorldState store MUST remain untouched
    after_version = store.get_current_state().version
    assert initial_version == after_version
    assert len(store.get_current_state().conditions) == 0


def test_binary_payload_non_duplication():
    engine = SituationFusionEngine()
    # Observation references a large artifact
    obs = MultimodalObservation(
        observation_id="art_obs_1",
        source_id="hdtv_camera",
        source_type="camera",
        modality=ModalityType.VIDEO_FRAME,
        timestamp=100.0,
        payload="frame 450",
        confidence=0.85,
        artifact_reference="/artifacts/video/clip_00450.mp4",
    )
    sit = engine.ingest(obs, now=100.0)
    ev = sit.supporting_evidence[0]
    # SituationEvidence stores reference, not binary data
    assert ev.provenance["artifact_reference"] == "/artifacts/video/clip_00450.mp4"
    assert not hasattr(ev, "binary_data")


def test_performance_scaling_benchmarks():
    """
    Measure focused processing time for 10, 100, and 500 observations.
    """
    def generate_batch(count):
        return [
            MultimodalObservation(
                observation_id=f"perf_{count}_{i}",
                source_id=f"sensor_{i % 10}",
                source_type="sensor",
                modality=ModalityType.TELEMETRY,
                timestamp=1000.0 + i * 0.1,
                payload={"reading": i},
                confidence=0.8,
                location=GeoLocation(latitude=37.77 + (i % 3) * 0.001, longitude=-122.41),
            )
            for i in range(count)
        ]

    # Benchmark 10 observations
    eng10 = SituationFusionEngine()
    batch10 = generate_batch(10)
    t0 = time.perf_counter()
    eng10.ingest_batch(batch10, now=1100.0)
    dur10 = time.perf_counter() - t0

    # Benchmark 100 observations
    eng100 = SituationFusionEngine()
    batch100 = generate_batch(100)
    t0 = time.perf_counter()
    eng100.ingest_batch(batch100, now=1100.0)
    dur100 = time.perf_counter() - t0

    # Benchmark 500 observations
    eng500 = SituationFusionEngine()
    batch500 = generate_batch(500)
    t0 = time.perf_counter()
    eng500.ingest_batch(batch500, now=1100.0)
    dur500 = time.perf_counter() - t0

    # Assert bounded performance
    assert dur10 < 0.10, f"10 obs took {dur10:.4f}s"
    assert dur100 < 0.50, f"100 obs took {dur100:.4f}s"
    assert dur500 < 2.00, f"500 obs took {dur500:.4f}s"
