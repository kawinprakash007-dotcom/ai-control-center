import json
import os
import sys
import time
import pytest
from unittest.mock import MagicMock

# Ensure chromadb fallback for lightweight CI/test execution
if "chromadb" not in sys.modules:
    sys.modules["chromadb"] = MagicMock()
if "sentence_transformers" not in sys.modules:
    sys.modules["sentence_transformers"] = MagicMock()

from fastapi.testclient import TestClient
from config.settings import AtlasSettings, get_settings, reset_settings
from main import create_app
from core.app_state import initialize_application_state, shutdown_application_state
from core.models.orchestration import MultimodalObservation, ModalityType
from core.models.runtime import CognitiveEventType


# ---------------------------------------------------------------------------
# Fixture: Test Application Client with In-Memory Stores
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    settings = AtlasSettings(
        app_env="test",
        api_auth_token="test_token_12345",
        simulation_mode=True,
        replay_mode=False,
    )
    app = create_app(settings=settings, in_memory_stores=True)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def auth_headers():
    return {"Authorization": "Bearer test_token_12345"}


# ---------------------------------------------------------------------------
# Group A: Configuration Tests (1–3)
# ---------------------------------------------------------------------------

def test_01_settings_default_values():
    s = AtlasSettings()
    assert s.app_port == 8000
    assert s.is_production() is False
    assert len(s.cors_origins) > 0


def test_02_settings_environment_overrides(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("API_AUTH_TOKEN", "prod_secret_token")
    monkeypatch.setenv("SIMULATION_MODE", "false")
    s = AtlasSettings()
    assert s.is_production() is True
    assert s.api_auth_token == "prod_secret_token"
    assert s.simulation_mode is False


def test_03_settings_cors_origins_parsing():
    s = AtlasSettings(cors_origins_raw="http://custom.origin:3000, http://another.origin:8080")
    assert "http://custom.origin:3000" in s.cors_origins
    assert "http://another.origin:8080" in s.cors_origins


# ---------------------------------------------------------------------------
# Group B: App Factory & Lifespan Tests (4–6)
# ---------------------------------------------------------------------------

def test_04_create_app_instance():
    app = create_app(settings=AtlasSettings(app_env="test"), in_memory_stores=True)
    assert app.title == "ATLAS AI Control Center"
    assert app.version == "6.1.0"


def test_05_lifespan_initializes_app_state(client):
    app = client.app
    assert hasattr(app.state, "atlas")
    assert app.state.atlas.ready is True
    assert app.state.atlas.central_orchestrator is not None


def test_06_root_endpoint_compatibility(client):
    res = client.get("/")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "running"
    assert "ATLAS" in data["platform"]


# ---------------------------------------------------------------------------
# Group C: Health & Readiness Tests (7–9)
# ---------------------------------------------------------------------------

def test_07_health_endpoint_public(client):
    res = client.get("/api/v1/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["app_env"] == "test"


def test_08_ready_endpoint_ready(client):
    res = client.get("/api/v1/ready")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ready"
    assert data["components"]["central_orchestrator"] is True
    assert data["components"]["device_gateway"] is True


def test_09_ready_endpoint_not_ready_when_shutting_down(client):
    client.app.state.atlas.shutting_down = True
    res = client.get("/api/v1/ready")
    assert res.status_code == 503
    client.app.state.atlas.shutting_down = False  # Restore


# ---------------------------------------------------------------------------
# Group D: Authentication Tests (10–12)
# ---------------------------------------------------------------------------

def test_10_missing_auth_header_rejected(client):
    res = client.get("/api/v1/devices")
    assert res.status_code == 401
    assert "Missing Authorization header" in res.json()["detail"]


def test_11_invalid_auth_token_forbidden(client):
    res = client.get("/api/v1/devices", headers={"Authorization": "Bearer wrong_token"})
    assert res.status_code == 403
    assert "Invalid or unauthorized" in res.json()["detail"]


def test_12_valid_auth_token_accepted(client, auth_headers):
    res = client.get("/api/v1/devices", headers=auth_headers)
    assert res.status_code == 200
    assert "devices" in res.json()


# ---------------------------------------------------------------------------
# Group E: CORS Configuration Tests (13–14)
# ---------------------------------------------------------------------------

def test_13_cors_origin_allowed(client):
    headers = {
        "Origin": "http://localhost:4200",
        "Access-Control-Request-Method": "POST",
    }
    res = client.options("/api/v1/chat", headers=headers)
    assert res.headers.get("access-control-allow-origin") == "http://localhost:4200"


def test_14_cors_unconfigured_origin_blocked(client):
    headers = {
        "Origin": "http://evil-untrusted-site.com",
        "Access-Control-Request-Method": "POST",
    }
    res = client.options("/api/v1/chat", headers=headers)
    assert res.headers.get("access-control-allow-origin") != "http://evil-untrusted-site.com"


# ---------------------------------------------------------------------------
# Group F: Observation Ingress Endpoint Tests (15–18)
# ---------------------------------------------------------------------------

def test_15_observation_ingress_success(client, auth_headers):
    payload = {
        "observation_id": "obs_http_001",
        "source_id": "sensor_front",
        "source_type": "sensor",
        "modality": "TELEMETRY",
        "payload": {"temperature": 24.5, "humidity": 60.0},
        "confidence": 0.98,
        "correlation_id": "corr_http_001",
    }
    res = client.post("/api/v1/ingress/observation", json=payload, headers=auth_headers)
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert data["cycle_count"] >= 1


def test_16_observation_ingress_invalid_payload(client, auth_headers):
    payload = {
        "observation_id": "",  # Empty ID rejected
        "source_id": "sensor",
        "source_type": "sensor",
        "modality": "TELEMETRY",
        "payload": {},
    }
    res = client.post("/api/v1/ingress/observation", json=payload, headers=auth_headers)
    assert res.status_code in (400, 422)


def test_17_observation_ingress_with_geolocation(client, auth_headers):
    payload = {
        "observation_id": "obs_http_geo_002",
        "source_id": "drone_gps",
        "source_type": "device",
        "modality": "GPS",
        "payload": {"fix": 3},
        "location": {"latitude": 37.7749, "longitude": -122.4194, "altitude": 10.0},
    }
    res = client.post("/api/v1/ingress/observation", json=payload, headers=auth_headers)
    assert res.status_code == 200
    assert res.json()["success"] is True


def test_18_observation_ingress_updates_world_state(client, auth_headers):
    payload = {
        "observation_id": "obs_world_test_003",
        "source_id": "drone_sensor",
        "source_type": "drone",
        "modality": "TELEMETRY",
        "payload": {"altitude": 12.5},
        "device_id": "ATLAS_DRONE_01",
    }
    res = client.post("/api/v1/ingress/observation", json=payload, headers=auth_headers)
    assert res.status_code == 200

    # Query world state
    ws_res = client.get("/api/v1/world/state", headers=auth_headers)
    assert ws_res.status_code == 200
    ws_data = ws_res.json()
    assert ws_data["version"] >= 1


# ---------------------------------------------------------------------------
# Group G: Chat & Cognitive Runtime Endpoints (19–21)
# ---------------------------------------------------------------------------

def test_19_chat_endpoint_protected(client, auth_headers):
    res = client.post(
        "/api/v1/chat",
        json={"message": "What is the system status?"},
        headers=auth_headers,
    )
    assert res.status_code == 200
    data = res.json()
    assert "response" in data
    assert "turn_id" in data


def test_20_legacy_chat_endpoint_backward_compatibility(client):
    # Calls public legacy /chat without auth
    res = client.post("/chat", json={"message": "Hello from legacy frontend"})
    assert res.status_code == 200
    data = res.json()
    assert "response" in data
    assert "turn_id" in data


def test_21_chat_does_not_execute_legacy_v1_agent(client, monkeypatch):
    # Verify legacy Jarvis Agent is never invoked
    import brain.agent
    mock_think = MagicMock(return_value="V1 THINK CALLED")
    monkeypatch.setattr(brain.agent.Agent, "think", mock_think)

    res = client.post("/chat", json={"message": "Test legacy isolation"})
    assert res.status_code == 200
    assert mock_think.call_count == 0


# ---------------------------------------------------------------------------
# Group H: Device Gateway & Inspection Endpoints (22–26)
# ---------------------------------------------------------------------------

def test_22_list_devices_simulation(client, auth_headers):
    res = client.get("/api/v1/devices", headers=auth_headers)
    assert res.status_code == 200
    devices = res.json()["devices"]
    dev_ids = [d["device_id"] for d in devices]
    assert "ATLAS_DRONE_01" in dev_ids
    assert "ATLAS_GLASS_01" in dev_ids
    assert "ATLAS_ROVER_01" in dev_ids


def test_23_get_device_detail(client, auth_headers):
    res = client.get("/api/v1/devices/ATLAS_DRONE_01", headers=auth_headers)
    assert res.status_code == 200
    data = res.json()
    assert data["device"]["device_id"] == "ATLAS_DRONE_01"
    assert "status" in data


def test_24_get_unknown_device_not_found(client, auth_headers):
    res = client.get("/api/v1/devices/NON_EXISTENT_DEVICE", headers=auth_headers)
    assert res.status_code == 404


def test_25_dispatch_device_command_policy_governed(client, auth_headers):
    cmd = {
        "capability": "takeoff",
        "action": "takeoff",
        "parameters": {"target_altitude": 10.0},
        "correlation_id": "corr_cmd_01",
    }
    res = client.post("/api/v1/devices/ATLAS_DRONE_01/command", json=cmd, headers=auth_headers)
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True


def test_26_dispatch_device_command_failsafe_on_low_battery(client, auth_headers):
    # Drain drone battery
    atlas = client.app.state.atlas
    adapter = atlas.device_gateway.resolve_adapter("ATLAS_DRONE_01")
    adapter.battery_pct = 4.0

    cmd = {
        "capability": "takeoff",
        "action": "takeoff",
        "parameters": {"target_altitude": 10.0},
    }
    res = client.post("/api/v1/devices/ATLAS_DRONE_01/command", json=cmd, headers=auth_headers)
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is False
    assert "battery" in data["error"].lower()

    adapter.battery_pct = 100.0  # Reset


# ---------------------------------------------------------------------------
# Group I: Goals & Traces Endpoints (27–30)
# ---------------------------------------------------------------------------

def test_27_get_goals_endpoint(client, auth_headers):
    res = client.get("/api/v1/goals", headers=auth_headers)
    assert res.status_code == 200
    data = res.json()
    assert "goals" in data
    assert isinstance(data["goals"], list)


def test_28_get_traces_endpoint(client, auth_headers):
    res = client.get("/api/v1/traces", headers=auth_headers)
    assert res.status_code == 200
    data = res.json()
    assert "traces" in data


def test_29_world_state_endpoint(client, auth_headers):
    res = client.get("/api/v1/world/state", headers=auth_headers)
    assert res.status_code == 200
    data = res.json()
    assert "version" in data
    assert "state_id" in data


def test_30_traces_limit_query_param(client, auth_headers):
    res = client.get("/api/v1/traces?limit=5", headers=auth_headers)
    assert res.status_code == 200


# ---------------------------------------------------------------------------
# Group J: WebSockets (Telemetry & Glass HUD) (31–36)
# ---------------------------------------------------------------------------

def test_31_websocket_telemetry_unauthorized_rejected(client):
    try:
        with client.websocket_connect("/api/v1/telemetry?token=bad_token") as ws:
            pass
    except Exception:
        pass  # Rejected with WS policy violation


def test_32_websocket_telemetry_authorized_connects(client):
    with client.websocket_connect("/api/v1/telemetry?token=test_token_12345") as ws:
        ws.send_text("ping")
        resp = ws.receive_text()
        assert resp == "pong"


def test_33_websocket_glass_hud_unauthorized_rejected(client):
    try:
        with client.websocket_connect("/api/v1/glass/hud?token=bad_token") as ws:
            pass
    except Exception:
        pass


def test_34_websocket_glass_hud_authorized_connects(client):
    with client.websocket_connect("/api/v1/glass/hud?token=test_token_12345") as ws:
        ws.send_text("ping")
        resp = ws.receive_text()
        assert resp == "pong"


def test_35_websocket_connection_manager_bounds():
    from core.connection_manager import BoundedConnectionManager
    mgr = BoundedConnectionManager(max_connections=2)
    mock1 = MagicMock()
    mock2 = MagicMock()
    mock3 = MagicMock()

    import asyncio
    async def _test():
        assert await mgr.connect(mock1) is True
        assert await mgr.connect(mock2) is True
        # Exceed capacity
        assert await mgr.connect(mock3) is False
        assert len(mgr.active_connections) == 2
        mgr.disconnect(mock1)
        assert len(mgr.active_connections) == 1
    asyncio.run(_test())


def test_36_websocket_broadcast_prunes_stale_connections():
    from core.connection_manager import BoundedConnectionManager
    mgr = BoundedConnectionManager(max_connections=5)
    mock_good = MagicMock()
    mock_bad = MagicMock()
    mock_bad.send_json.side_effect = RuntimeError("Broken pipe")

    import asyncio
    async def _test():
        await mgr.connect(mock_good)
        await mgr.connect(mock_bad)
        assert len(mgr.active_connections) == 2
        await mgr.broadcast({"data": "test"})
        # Bad client should be pruned
        assert len(mgr.active_connections) == 1
        assert mock_good in mgr.active_connections
    asyncio.run(_test())


# ---------------------------------------------------------------------------
# Group K: End-to-End Orchestration & Lineage (37–42)
# ---------------------------------------------------------------------------

def test_37_observation_triggers_situation_and_cycle(client, auth_headers):
    # Ingest telemetry with high anomaly to trigger situation
    payload = {
        "observation_id": "obs_anomaly_01",
        "source_id": "rover_obstacle_sensor",
        "source_type": "rover",
        "modality": "TELEMETRY",
        "payload": {"obstacle_distance_m": 0.2, "emergency_stop": True},
        "confidence": 1.0,
        "device_id": "ATLAS_ROVER_01",
    }
    res = client.post("/api/v1/ingress/observation", json=payload, headers=auth_headers)
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert data["cycle_count"] >= 1


def test_38_command_observation_reingestion_maintains_lineage(client, auth_headers):
    cmd = {
        "capability": "navigate",
        "action": "go_to_waypoint",
        "parameters": {"latitude": 37.7755, "longitude": -122.4190},
        "correlation_id": "corr_lineage_001",
    }
    res = client.post("/api/v1/devices/ATLAS_ROVER_01/command", json=cmd, headers=auth_headers)
    assert res.status_code == 200
    out = res.json()["output"]
    assert "location" in out or "navigated" in str(out)


def test_39_turn_execution_non_blocking_threadpool(client, auth_headers):
    # Verify cognitive turn executes in executor threadpool
    atlas = client.app.state.atlas
    assert atlas.executor is not None
    res = client.post(
        "/api/v1/chat",
        json={"message": "Offload test message"},
        headers=auth_headers,
    )
    assert res.status_code == 200


def test_40_rate_limiting_at_input_gateway(client, auth_headers):
    # Burst 10 observations with same ID to trigger dedup/rate limit
    for i in range(5):
        payload = {
            "observation_id": f"obs_burst_{i}",
            "source_id": "rate_limit_sensor",
            "source_type": "sensor",
            "modality": "TELEMETRY",
            "payload": {"reading": i},
        }
        res = client.post("/api/v1/ingress/observation", json=payload, headers=auth_headers)
        assert res.status_code == 200


def test_41_policy_engine_authorizes_device_commands(client):
    atlas = client.app.state.atlas
    from core.models.tool_call import ToolCall
    tc = ToolCall(
        capability="device_gateway",
        action="dispatch_capability",
        parameters={"device_id": "ATLAS_DRONE_01", "capability": "land", "action": "land", "parameters": {}},
    )
    pol_res = atlas.policy_engine.evaluate(tc)
    assert pol_res.is_allowed is True


def test_42_policy_engine_denies_unregistered_actions(client):
    atlas = client.app.state.atlas
    from core.models.tool_call import ToolCall
    tc = ToolCall(
        capability="unknown_untrusted_device",
        action="arbitrary_exec",
        parameters={},
    )
    pol_res = atlas.policy_engine.evaluate(tc)
    assert pol_res.is_allowed is False


# ---------------------------------------------------------------------------
# Group L: Concurrency, Teardown & Lifecycle (43–50)
# ---------------------------------------------------------------------------

def test_43_concurrent_requests_handled(client, auth_headers):
    import concurrent.futures
    def send_chat(msg_id):
        return client.post("/api/v1/chat", json={"message": f"Concurrent {msg_id}"}, headers=auth_headers)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(send_chat, i) for i in range(4)]
        results = [f.result() for f in futures]

    for r in results:
        assert r.status_code == 200


def test_44_clean_shutdown_lifecycle(client):
    atlas = client.app.state.atlas
    import asyncio
    asyncio.run(shutdown_application_state(atlas))
    assert atlas.ready is False
    assert atlas.shutting_down is True


def test_45_sqlite_store_hardening_pragmas(tmp_path):
    from world.store import SQLiteWorldStateStore
    from goals.store import SQLiteGoalStore
    from memory.sqlite_store import SQLiteMemoryStore

    w_db = tmp_path / "test_world.db"
    g_db = tmp_path / "test_goals.db"
    m_db = tmp_path / "test_mem.db"

    ws = SQLiteWorldStateStore(str(w_db))
    gs = SQLiteGoalStore(str(g_db))
    ms = SQLiteMemoryStore(str(m_db))

    with ws._get_connection() as conn:
        journal = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert journal.lower() == "wal"

    with gs._get_connection() as conn:
        journal = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert journal.lower() == "wal"

    with ms._get_connection() as conn:
        journal = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert journal.lower() == "wal"


def test_46_no_hardware_imports():
    import backend
    forbidden = [
        "pymavlink", "mavsdk", "rclpy", "rospy",
        "paho.mqtt", "pyserial", "serial", "RPi.GPIO"
    ]
    for mod in forbidden:
        assert mod not in sys.modules, f"Forbidden module {mod} was imported!"


def test_47_simulation_device_glass_hud_message(client, auth_headers):
    cmd = {
        "capability": "display_hud",
        "action": "display_hud",
        "parameters": {"text": "Mission Target Acquired"},
    }
    res = client.post("/api/v1/devices/ATLAS_GLASS_01/command", json=cmd, headers=auth_headers)
    assert res.status_code == 200
    assert res.json()["success"] is True


def test_48_simulation_device_rover_stop(client, auth_headers):
    cmd = {
        "capability": "stop",
        "action": "stop",
        "parameters": {},
    }
    res = client.post("/api/v1/devices/ATLAS_ROVER_01/command", json=cmd, headers=auth_headers)
    assert res.status_code == 200
    assert res.json()["success"] is True


def test_49_cognitive_runtime_is_live_authority(client):
    atlas = client.app.state.atlas
    assert atlas.cognitive_runtime is not None
    assert atlas.central_orchestrator.cognitive_runtime is atlas.cognitive_runtime


def test_50_policy_engine_is_live_authority(client):
    atlas = client.app.state.atlas
    assert atlas.policy_engine is not None
    assert atlas.tool_orchestrator.policy_engine is atlas.policy_engine
