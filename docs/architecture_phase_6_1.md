# ATLAS Phase 6.1 — Production Central Runtime & Real-Time Ingress Architecture

## 1. Executive Summary

ATLAS Phase 6.1 transforms the Phase 5 Central Orchestration foundation into the operational, running FastAPI application (`backend/main.py`). The legacy monolithic `/chat` runtime and its silent fallback to the legacy V1 assistant have been fully retired from default live execution. In their place, a clean Application Composition Root with a FastAPI `lifespan` manager coordinates the complete authoritative stack:

```
FastAPI Application Lifecycle
  ↓
AtlasApplicationState (Composition Root / Dependency Container)
  ↓
CentralOrchestrator (Phase 5.0 Cross-Layer Coordinator)
  ↓
Authoritative Engines:
  - CentralInputGateway (Validation, Normalization, Rate-Limiting)
  - SituationFusionEngine (Temporal & Spatial Aggregation)
  - WorldState / WorldStateStore (Believed Reality)
  - EventAutonomy / Anticipation (Rules, Reactive Triggering)
  - AutonomousGoalManager (Goal Lifecycle & Execution)
  - CognitiveRuntime (Understanding, Planning, Reasoning)
  - PolicyEngine (Security & Safety Authorization Boundary)
  - ToolOrchestrator (Execution Boundary)
  - DeviceGateway (Device Identity & Routing Boundary)
  ↓
Device Adapters (VirtualGlass, VirtualDrone, VirtualRover)
```

---

## 2. Runtime Composition & Application State

The application state container (`backend/core/app_state.py`) instantiates and manages single-authority components without global singleton sprawl:

- **CognitiveRuntime**: Sole cognitive authority (reasoning, turn execution, trace capture).
- **PolicyEngine**: Sole authorization authority (policy rules, failsafe checks).
- **ToolOrchestrator**: Sole execution boundary (dispatches to registered tools and capabilities).
- **DeviceGateway**: Sole device identity and routing boundary (registry, status tracking, capability lookup).
- **DeviceAdapter**: Protocol translation boundary (VirtualGlass, VirtualDrone, VirtualRover).
- **AutonomousGoalManager**: Sole goal lifecycle authority (creation, state progression, cancellation).
- **WorldState**: Sole believed reality authority (state transitions, entity observations).
- **Memory / Knowledge**: Historical and informational context.
- **CentralOrchestrator**: Sole integration and closed-loop coordinator.

No endpoint instantiates engines or mutating stores directly; all endpoints access authoritative instances via `request.app.state.atlas`.

---

## 3. Application Lifecycle

The application lifecycle is managed strictly through the FastAPI `lifespan` context manager:

### Startup Phase:
1. Load typed `AtlasSettings` from environment or configuration profiles.
2. Initialize SQLite persistence stores with production-hardened pragmas (`WAL`, `busy_timeout=5000`, `foreign_keys=ON`).
3. Instantiate memory, knowledge RAG, world store, goal store, and event sinks.
4. Construct the `CentralInputGateway` with rate-limiting and deduplication caches.
5. Initialize `SituationFusionEngine`, `EventAutonomyEngine`, `DeterministicAnticipatoryAnalyzer`, `GoalExecutionEngine`, and `AutonomousGoalManager`.
6. Construct `CognitiveRuntime` with tool orchestrator and event sink.
7. Construct `DeviceGateway` and register virtual simulation adapters (`VirtualDroneAdapter`, `VirtualRoverAdapter`, `VirtualGlassAdapter`) when `SIMULATION_MODE=true`.
8. Wire `CentralOrchestrator` to tie ingress, fusion, world state, autonomy, goals, runtime, and devices together.
9. Initialize threadpool executor (`concurrent.futures.ThreadPoolExecutor`) for non-blocking cognitive execution.
10. Mark `AtlasApplicationState.ready = True` and emit `APPLICATION_STARTED` & `APPLICATION_READY` cognitive events.

### Shutdown Phase:
1. Mark `AtlasApplicationState.ready = False` to refuse new ingress and commands.
2. Emit `APPLICATION_SHUTTING_DOWN` event.
3. Terminate active WebSockets gracefully via `BoundedConnectionManager.disconnect_all()`.
4. Shutdown threadpool executor (`wait=True`), completing any in-flight cognitive turns.
5. Flush event sinks and close persistence resources.
6. Emit `APPLICATION_STOPPED` event.

---

## 4. Configuration Architecture

Typed settings (`backend/config/settings.py`) are backed by Pydantic:

- `APP_ENV`: Environment profile (`development`, `simulation`, `production`, `replay`).
- `APP_HOST` / `APP_PORT`: Bind address and port (default `127.0.0.1:8000`).
- `DEBUG`: Diagnostic flag (default `False`).
- `API_AUTH_TOKEN`: Secret token for bearer authentication (default empty in dev, enforced in production).
- `CORS_ORIGINS`: Comma-delimited list of trusted origins (no unrestricted wildcard `*` combined with credentials).
- `SIMULATION_MODE`: Enables virtual hardware adapters (default `True`).
- `REPLAY_MODE`: Enables deterministic replay isolation (default `False`).
- `DATABASE_DIR` / `TRACE_DIR`: Configurable paths for persistence storage.

---

## 5. Security & Authentication Boundary

Authentication and authorization are separated into distinct boundaries:

1. **Authentication Boundary (API Level)**:
   - Protected endpoints require `Authorization: Bearer <API_AUTH_TOKEN>`.
   - Token validation uses `secrets.compare_digest` for constant-time comparison to prevent timing attacks.
   - WebSocket handshakes authenticate via `?token=<API_AUTH_TOKEN>`.
   - Missing or mismatched tokens produce immediate 401/403 responses before reaching application logic.
2. **Authorization Boundary (PolicyEngine)**:
   - Device commands and tool executions are strictly submitted as `ToolCall` objects to `PolicyEngine`.
   - PolicyEngine validates permissions, failsafes (e.g. minimum battery thresholds, critical safety rules), and capability validity.
   - Handlers never bypass PolicyEngine or invoke device adapters directly.

---

## 6. REST API Surface (`/api/v1/*`)

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| `GET` | `/api/v1/health` | Public | Liveness probe (process alive, uptime, timestamp) |
| `GET` | `/api/v1/ready` | Public | Readiness probe (validates all required engines are initialized) |
| `POST` | `/api/v1/chat` | Protected | Modern cognitive turn execution (non-blocking threadpool) |
| `POST` | `/api/v1/ingress/observation` | Protected | Edge observation ingress routed through CentralInputGateway & CentralOrchestrator |
| `GET` | `/api/v1/devices` | Protected | Lists registered devices from DeviceGateway |
| `GET` | `/api/v1/devices/{device_id}` | Protected | Detailed device status and capabilities |
| `POST` | `/api/v1/devices/{device_id}/command` | Protected | Policy-governed device command dispatch |
| `GET` | `/api/v1/world/state` | Protected | Current believed world state snapshot |
| `GET` | `/api/v1/goals` | Protected | Active, pending, and completed autonomous goals |
| `GET` | `/api/v1/traces` | Protected | Cognitive trace logs with limit query param |

### Backward Compatibility
- `POST /chat`: Compatibility wrapper routing to the modern `CognitiveRuntime` turn execution, preserving existing frontend integrations while retiring legacy V1 code execution.

---

## 7. Real-Time WebSocket Architecture

Two dedicated WebSocket routes manage outbound streaming with bounded resource guarantees:

1. `WS /api/v1/telemetry`:
   - Streams outbound device telemetry, orchestration events, and cognitive lifecycle updates.
   - Enforces query param authentication (`?token=...`).
2. `WS /api/v1/glass/hud`:
   - Dedicated semantic HUD message channel for smart glasses.
   - Streams formatted messages (`{"type": "hud_update", "text": "...", "severity": "...", "timestamp": ...}`).

### BoundedConnectionManager
- Caps active connections (`max_connections=100`) to prevent resource exhaustion.
- Enforces non-blocking broadcast with queue pruning for stale/disconnected clients.
- Clean shutdown sweeps all active connections.

---

## 8. Runtime Non-Blocking Safety

Cognitive turn execution (which involves LLM inference, planning, and verification) is synchronous and can block for hundreds of milliseconds or seconds. To prevent event-loop starvation:
- Cognitive turns are offloaded to `AtlasApplicationState.executor` (a bounded `ThreadPoolExecutor(max_workers=8)`).
- FastAPI's async event loop remains immediately responsive to incoming health checks, telemetry WebSockets, and observation ingress.

---

## 9. SQLite Persistence Hardening

All persistent SQLite stores (`GoalStore`, `WorldStateStore`, `SQLiteMemoryStore`) apply production-grade configuration pragmas upon connection opening:
- `PRAGMA journal_mode = WAL;` (Write-Ahead Logging for concurrent read/write access)
- `PRAGMA busy_timeout = 5000;` (5-second wait before throwing database locked errors)
- `PRAGMA foreign_keys = ON;` (Referential integrity enforcement)

---

## 10. Device Simulation & Closed-Loop Lineage

In `SIMULATION_MODE=true`:
- `VirtualDroneAdapter` (`ATLAS_DRONE_01`), `VirtualRoverAdapter` (`ATLAS_ROVER_01`), and `VirtualGlassAdapter` (`ATLAS_GLASS_01`) are registered with `DeviceGateway`.
- When an actuation command is executed, the adapter generates a deterministic `MultimodalObservation` retaining `correlation_id` and setting `causation_id` to the command's `dispatch_id`.
- The observation is reingested through `CentralInputGateway` into `SituationFusionEngine` and `WorldState`, completing the closed loop.

---

## 11. Scope Boundaries: Central vs. Edge

### Handled in Phase 6.1 (Production Central Runtime):
- Live FastAPI composition root, lifespan, configuration, and security.
- Versioned REST API and bounded WebSockets.
- Central orchestration loop from ingress to cognitive execution and simulated actuation.
- Threadpool isolation for non-blocking cognitive operations.

### Reserved for Phase 6.2+ (Physical Edge & Device Transports):
- Hardware transport drivers (MAVLink, MAVSDK, ROS 2, MQTT, Serial, GPIO, WebRTC).
- Physical edge nodes, companion computers, and microcontrollers.
- Physical sensor serialisation and video streaming codecs.
