import time
import pytest
from fastapi.testclient import TestClient

from config.settings import AtlasSettings
from main import create_app
from core.models.orchestration import ModalityType, GeoLocation
from orchestration.input_gateway import (
    CentralInputGateway,
    GatewayConfig,
    IngressEnvelope,
    IngressRejectionReason,
    IngressStatus,
    normalize_modality,
)


@pytest.fixture
def test_client():
    """Create a FastAPI TestClient configured for simulation testing."""
    settings = AtlasSettings(
        app_env="test",
        simulation_mode=True,
    )
    app = create_app(settings=settings, in_memory_stores=True)
    with TestClient(app) as client:
        yield client, settings


# ============================================================================
# 1. REGRESSION: REPRODUCE PREVIOUS 422 ON DEFECTIVE PAYLOAD
# ============================================================================

def test_regression_previous_faulty_payload_returns_422(test_client):
    """
    Requirement 16: Reproduce the exact previous frontend payload that produced
    HTTP 422 Unprocessable Content due to missing 'observation_id' and 'source_id'.
    """
    client, settings = test_client

    # The exact payload previously sent by simulation.component.ts:
    faulty_payload = {
        "device_id": "ATLAS_VISION_01",
        "source_type": "SIMULATION_TWIN",
        "timestamp": time.time(),
        "modality": "VISUAL",
        "payload": {
            "scenario_id": "SCN-01",
            "deterministic_hash": "0x8f2a101b4e9c",
            "title": "Multi-Agent Perimeter Breach",
        },
    }

    resp = client.post(
        "/api/v1/ingress/observation",
        json=faulty_payload,
        headers={"Authorization": f"Bearer {settings.api_auth_token}"},
    )

    assert resp.status_code == 422, f"Expected 422 Unprocessable Content, got {resp.status_code}: {resp.text}"
    detail = resp.json().get("detail", [])
    missing_fields = {tuple(d.get("loc", [])) for d in detail}
    # Both observation_id and source_id must be flagged as missing
    assert any("observation_id" in loc for loc in missing_fields)
    assert any("source_id" in loc for loc in missing_fields)


# ============================================================================
# 2. CANONICAL SCN-01, SCN-02, SCN-03 INJECTIONS SUCCEED
# ============================================================================

def test_canonical_scn_01_injection_succeeds(test_client):
    """SCN-01 Multi-Agent Perimeter Breach (IMAGE) executes successfully."""
    client, settings = test_client
    now = time.time()
    payload = {
        "observation_id": f"obs_sim_scn_01_{int(now * 1000)}",
        "source_id": "ATLAS_VISION_01",
        "device_id": "ATLAS_VISION_01",
        "source_type": "SIMULATION_TWIN",
        "modality": "IMAGE",
        "timestamp": now,
        "confidence": 0.98,
        "location": {"latitude": 37.7750, "longitude": -122.4192, "altitude": 15.0, "accuracy": 1.0},
        "correlation_id": f"corr_sim_scn_01_{int(now * 1000)}",
        "causation_id": "cause_catalog_scn_01",
        "payload": {
            "scenario_id": "SCN-01",
            "title": "Multi-Agent Perimeter Breach",
            "detections": [
                {"label": "person_intruder", "confidence": 0.98, "bounding_box": [0.15, 0.22, 0.45, 0.65], "sector": 4}
            ],
        },
        "metadata": {"scenario_id": "SCN-01", "deterministic_hash": "0x8f2a101b4e9c"},
    }

    resp = client.post(
        "/api/v1/ingress/observation",
        json=payload,
        headers={"Authorization": f"Bearer {settings.api_auth_token}"},
    )

    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    data = resp.json()
    assert data.get("success") is True
    assert data.get("observation_id") == payload["observation_id"]
    assert data.get("correlation_id") == payload["correlation_id"]
    assert data.get("observations_ingested_count", 0) >= 1
    assert isinstance(data.get("causal_trace"), list)
    assert isinstance(data.get("situations"), list)
    assert isinstance(data.get("world_transitions"), list)
    assert isinstance(data.get("goals"), list)
    assert isinstance(data.get("tool_results"), list)



def test_canonical_scn_02_injection_succeeds(test_client):
    """SCN-02 Dense Urban GPS Denial Navigation (GPS) executes successfully."""
    client, settings = test_client
    now = time.time()
    payload = {
        "observation_id": f"obs_sim_scn_02_{int(now * 1000)}",
        "source_id": "ATLAS_DRONE_01",
        "device_id": "ATLAS_DRONE_01",
        "source_type": "SIMULATION_TWIN",
        "modality": "GPS",
        "timestamp": now,
        "confidence": 0.95,
        "location": {"latitude": 37.7749, "longitude": -122.4194, "altitude": 28.5, "accuracy": 5.0},
        "correlation_id": f"corr_sim_scn_02_{int(now * 1000)}",
        "causation_id": "cause_catalog_scn_02",
        "payload": {
            "scenario_id": "SCN-02",
            "navigation_state": "DEAD_RECKONING",
            "gps_lock": False,
            "vio_odometry": {"vx": 1.2, "vy": 0.0, "vz": -0.4, "drift_m": 0.12},
        },
        "metadata": {"scenario_id": "SCN-02", "deterministic_hash": "0x3c9e472a11bf"},
    }

    resp = client.post(
        "/api/v1/ingress/observation",
        json=payload,
        headers={"Authorization": f"Bearer {settings.api_auth_token}"},
    )

    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    data = resp.json()
    assert data.get("success") is True
    assert data.get("observations_ingested_count", 0) >= 1


def test_canonical_scn_03_injection_succeeds(test_client):
    """SCN-03 Thermal Hotspot Search & Rescue (TELEMETRY) executes successfully."""
    client, settings = test_client
    now = time.time()
    payload = {
        "observation_id": f"obs_sim_scn_03_{int(now * 1000)}",
        "source_id": "ATLAS_ROVER_01",
        "device_id": "ATLAS_ROVER_01",
        "source_type": "SIMULATION_TWIN",
        "modality": "TELEMETRY",
        "timestamp": now,
        "confidence": 0.99,
        "location": {"latitude": 37.7747, "longitude": -122.4193, "altitude": 0.5, "accuracy": 1.0},
        "correlation_id": f"corr_sim_scn_03_{int(now * 1000)}",
        "causation_id": "cause_catalog_scn_03",
        "payload": {
            "scenario_id": "SCN-03",
            "thermal_anomaly": True,
            "metrics": {"heat_signature_c": 37.4, "ambient_c": 18.2, "classification": "HUMAN"},
        },
        "metadata": {"scenario_id": "SCN-03", "deterministic_hash": "0x91d582fa038c"},
    }

    resp = client.post(
        "/api/v1/ingress/observation",
        json=payload,
        headers={"Authorization": f"Bearer {settings.api_auth_token}"},
    )

    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    data = resp.json()
    assert data.get("success") is True
    assert data.get("observations_ingested_count", 0) >= 1


# ============================================================================
# 3. MAJOR MODALITIES COVERAGE
# ============================================================================

@pytest.mark.parametrize("modality,source_id,payload_data", [
    ("IMAGE", "ATLAS_VISION_01", {"image_hash": "sha256:abc", "width": 1920, "height": 1080}),
    ("GPS", "ATLAS_DRONE_01", {"fix": "3D", "satellites": 12}),
    ("TELEMETRY", "ATLAS_ROVER_01", {"battery": 92.5, "motor_temp_c": 41.2}),
    ("EVENT", "ATLAS_GLASS_01", {"event_name": "button_tap", "component": "hud"}),
    ("VOICE_TRANSCRIPT", "ATLAS_GLASS_01", {"text": "scan perimeter sector four", "speaker": "operator"}),
])
def test_all_major_modalities_succeed(test_client, modality, source_id, payload_data):
    """Verify ingress accepts all major canonical modalities."""
    client, settings = test_client
    now = time.time()
    payload = {
        "observation_id": f"obs_mod_{modality.lower()}_{int(now * 1000)}",
        "source_id": source_id,
        "source_type": "SIMULATION_TWIN",
        "modality": modality,
        "timestamp": now,
        "confidence": 0.95,
        "payload": payload_data,
        "correlation_id": f"corr_{modality.lower()}",
    }

    resp = client.post(
        "/api/v1/ingress/observation",
        json=payload,
        headers={"Authorization": f"Bearer {settings.api_auth_token}"},
    )

    assert resp.status_code == 200, f"Failed for modality {modality}: {resp.text}"
    assert resp.json().get("success") is True


# ============================================================================
# 4. ERROR & REJECTION BOUNDARIES
# ============================================================================

def test_stale_timestamp_rejected():
    """Verify CentralInputGateway rejects stale timestamps beyond max_stale_seconds."""
    gw = CentralInputGateway(config=GatewayConfig(max_stale_seconds=60.0))
    now = time.time()
    env = IngressEnvelope(
        message_id="msg_stale",
        source_id="ATLAS_DRONE_01",
        source_type="SIMULATION_TWIN",
        modality=ModalityType.GPS,
        timestamp=now - 120.0,  # 120s old > 60s max_stale
        payload={"data": "test"},
    )
    res = gw.receive_envelope(env, now=now)
    assert res.status == IngressStatus.REJECTED
    assert res.rejection_reason == IngressRejectionReason.STALE


def test_future_timestamp_rejected():
    """Verify CentralInputGateway rejects future timestamps beyond clock skew tolerance."""
    gw = CentralInputGateway(config=GatewayConfig(clock_skew_tolerance_seconds=5.0))
    now = time.time()
    env = IngressEnvelope(
        message_id="msg_future",
        source_id="ATLAS_DRONE_01",
        source_type="SIMULATION_TWIN",
        modality=ModalityType.GPS,
        timestamp=now + 60.0,  # 60s into future > 5s tolerance
        payload={"data": "test"},
    )
    res = gw.receive_envelope(env, now=now)
    assert res.status == IngressStatus.REJECTED
    assert res.rejection_reason == IngressRejectionReason.FUTURE_TIMESTAMP


def test_invalid_confidence_rejected():
    """Verify CentralInputGateway rejects confidence outside [0.0, 1.0]."""
    gw = CentralInputGateway()
    now = time.time()
    env = IngressEnvelope(
        message_id="msg_bad_conf",
        source_id="ATLAS_VISION_01",
        source_type="SIMULATION_TWIN",
        modality=ModalityType.IMAGE,
        timestamp=now,
        confidence=1.5,  # Invalid confidence
        payload={"data": "test"},
    )
    res = gw.receive_envelope(env, now=now)
    assert res.status == IngressStatus.REJECTED
    assert res.rejection_reason == IngressRejectionReason.INVALID_CONFIDENCE


def test_unknown_source_rejected_under_strict_registration():
    """Verify CentralInputGateway rejects unknown non-simulation source under strict policy."""
    gw = CentralInputGateway(config=GatewayConfig(strict_source_registration=True, allow_simulation_sources=False))
    now = time.time()
    env = IngressEnvelope(
        message_id="msg_unregistered",
        source_id="ROGUE_SENSOR_99",
        source_type="UNREGISTERED_HARDWARE",
        modality=ModalityType.TELEMETRY,
        timestamp=now,
        payload={"data": "test"},
    )
    res = gw.receive_envelope(env, now=now)
    assert res.status == IngressStatus.REJECTED
    assert res.rejection_reason == IngressRejectionReason.UNKNOWN_SOURCE


def test_modality_normalization_aliases():
    """Verify defensive normalization in CentralInputGateway maps visual/spatial/temporal cleanly."""
    assert normalize_modality("visual") == ModalityType.IMAGE
    assert normalize_modality("spatial") == ModalityType.GPS
    assert normalize_modality("temporal") == ModalityType.TELEMETRY
    assert normalize_modality("IMAGE") == ModalityType.IMAGE
    assert normalize_modality("GPS") == ModalityType.GPS
    assert normalize_modality("TELEMETRY") == ModalityType.TELEMETRY
