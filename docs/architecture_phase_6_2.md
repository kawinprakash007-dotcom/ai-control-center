# ATLAS Phase 6.2 — Unified Edge/Product Contract Layer Architecture

## Executive Summary

Phase 6.2 establishes the unified, transport-neutral edge and product contract layer for ATLAS Central. It formalizes contracts, lifecycle state machines, telemetry models, health monitoring, error taxonomies, and simulation adapters for all four first-class ATLAS edge products:
- **ATLAS Vision** (Persistent Environmental Sensing / Surveillance)
- **ATLAS Glass** (Wearable Perception & HUD Interaction)
- **ATLAS Drone** (Aerial Sensing & Physical Intervention)
- **ATLAS Rover** (Ground Sensing & Physical Intervention)

> [!IMPORTANT]
> **PHYSICAL TRANSPORTS ARE NOT IMPLEMENTED IN PHASE 6.2.**
> This phase defines pure protocol contracts, canonical data structures, gateway routing, and virtual simulation adapters. No physical hardware drivers (MAVLink, ROS2, MQTT, BLE, Serial, GPIO, WebRTC) are introduced or executed.

---

## 1. Four First-Class ATLAS Products

All four products are first-class participants in the ATLAS ecosystem with dedicated product types, roles, and capability descriptors:

| Product | ProductType | Primary ProductRole | Primary Description | Default Capabilities |
| :--- | :--- | :--- | :--- | :--- |
| **ATLAS Vision** | `VISION` | `OBSERVATION_SOURCE` | Stationary/wall-mounted or PTZ smart camera hub for continuous situational awareness | `detect_motion`, `detect_person`, `detect_anomaly`, `capture_image`, `capture_video`, `get_telemetry`, `emit_event` |
| **ATLAS Glass** | `GLASS` | `HYBRID` | Wearable smart glasses providing egocentric vision, spatial audio, and HUD display | `display_hud`, `capture_image`, `capture_audio`, `get_location`, `get_telemetry`, `send_notification` |
| **ATLAS Drone** | `DRONE` | `HYBRID` | Autonomous aerial system for rapid inspection and aerial intervention | `takeoff`, `land`, `navigate_waypoint`, `hover`, `return_to_base`, `capture_image`, `capture_video`, `get_telemetry` |
| **ATLAS Rover** | `ROVER` | `HYBRID` | Ground robotic platform for persistent surface inspection and intervention | `navigate_to`, `patrol_zone`, `dock`, `stop`, `capture_image`, `get_telemetry` |

---

## 2. Device Contract Models

All contract models are defined in `backend/core/models/device_contract.py` using Python standard library dataclasses (`frozen=True` where immutable) and enums:

### Enums
- **`ProductType`**: `VISION`, `GLASS`, `DRONE`, `ROVER`, `UNKNOWN`
- **`ProductRole`**: `OBSERVATION_SOURCE`, `ACTUATOR`, `HYBRID`
- **`CommandState`**: `PENDING`, `DISPATCHED`, `ACKNOWLEDGED`, `IN_PROGRESS`, `COMPLETED`, `FAILED`, `CANCELLED`, `TIMEOUT`, `REJECTED`, `UNKNOWN`
- **`AcknowledgementStatus`**: `ACCEPTED`, `REJECTED`, `DEFERRED`, `BUSY`, `UNSUPPORTED`, `PREEMPTED`, `UNKNOWN`
- **`DeviceHealthStatus`**: `HEALTHY`, `DEGRADED`, `UNHEALTHY`, `UNKNOWN`
- **`DeviceErrorCode`**: 13 canonical error codes (`DEVICE_OFFLINE`, `COMMAND_TIMEOUT`, `CAPABILITY_UNSUPPORTED`, `AUTHENTICATION_FAILED`, `PERMISSION_DENIED`, `HARDWARE_FAULT`, `PARAMETER_INVALID`, `RESOURCE_EXHAUSTED`, `SAFETY_INTERVENTION`, `STATE_CONFLICT`, `COMMUNICATION_ERROR`, `BATTERY_CRITICAL`, `UNKNOWN_ERROR`)

### Dataclass Models
- **`DeviceError`**: Structured, immutable error representation with code, message, severity, retryability, timestamp, and sanitized details.
- **`DeviceHeartbeat`**: Periodic status beacon containing device ID, status, battery, sequence number, timestamp, and optional metrics.
- **`DeviceHealth`**: Aggregated health report containing overall status, last heartbeat timestamp, battery level, error counts, active alerts, and diagnostic metrics.
- **`DeviceCommandRequest`**: Full command lifecycle request with unique ID, correlation/trace IDs, target device, capability, action, parameters, timeout, priority, and submission timestamp.
- **`DeviceCommandAcknowledgement`**: Explicit early handshake from device indicating command acceptance, rejection reason, or estimated execution duration.
- **`DeviceCommandResult`**: Final execution result with terminal command state, output payload, error details, start/end timestamps, and duration.
- **`DeviceTelemetry`**: Standardized multi-sensor telemetry frame with battery, location, orientation, velocity, operational mode, and sensor payloads.

---

## 3. Product Roles and Architectural Authority

Products operate under strict architectural boundaries governed by `ProductRole`:
1. **`OBSERVATION_SOURCE`** (e.g., ATLAS Vision):
   - Primarily ingests multimodal observations into the central nervous system.
   - Its role is capability metadata and **never** bypasses `PolicyEngine` or `ToolOrchestrator`.
   - Cannot directly mutate `WorldState`; all data flows through `CentralInputGateway` -> `SituationFusionEngine` -> `WorldStateUpdater`.
2. **`ACTUATOR`**:
   - Executes authorized physical actions dispatched by `ToolOrchestrator`.
3. **`HYBRID`** (e.g., Glass, Drone, Rover):
   - Emits observations and telemetry while accepting governed actuator commands.

---

## 4. Device Identity and Security Scrubbing

`DeviceIdentity` uniquely specifies an edge device:
- Fields: `device_id`, `product_type`, `product_role`, `vendor`, `model`, `hardware_version`, `firmware_version`, `contract_version`, `capabilities`, `metadata`.
- **Security Invariant**: Sensitive credentials (passwords, secrets, tokens, API keys, private keys) are scrubbed at serialization via `sanitize_contract_metadata()`. They are never logged, persisted in traces, or returned over public REST/WebSocket endpoints.

---

## 5. Capability Contract & Schema Enforcement

`DeviceCapabilityDescriptor` specifies capability interfaces:
- `capability_id`: Canonical capability name (e.g., `camera`, `navigation`, `hud`).
- `description`: Human and LLM-readable description.
- `supported_actions`: Tuple of permitted action verbs (e.g., `("takeoff", "land", "hover")`).
- `schema_version`: Contract schema version (default `"1.0"`).
- `parameter_schema`: JSON-schema style validation dictionary defining `required`, `type`, `minimum`, `maximum`, and `enum` bounds.
- `expected_result_category`: Category of output (e.g., `telemetry`, `image`, `state_change`).

Validation is enforced deterministically before any command is sent to an adapter.

---

## 6. Command Lifecycle State Machine

Commands progress through a deterministic, linear state machine:
```
PENDING -> DISPATCHED -> ACKNOWLEDGED -> IN_PROGRESS -> COMPLETED
   |            |             |              |
   v            v             v              v
REJECTED     TIMEOUT       REJECTED       FAILED / CANCELLED / TIMEOUT
```
- A command is uniquely tracked by `command_id` and correlated across central pipelines by `correlation_id` and `trace_id`.
- Duplicate command submissions are detected and rejected deterministically via gateway LRU caches.

---

## 7. Acknowledgement Contract

The acknowledgement handshake (`DeviceCommandAcknowledgement`) is explicitly decoupled from command completion:
- **`ACCEPTED`**: Device validated parameters, resources, and queued/started execution.
- **`REJECTED`**: Pre-execution rejection (e.g., low battery, state conflict).
- **`BUSY` / `DEFERRED`**: Device is processing prior work.
- **`UNSUPPORTED`**: Device does not support the action.
- Completion occurs asynchronously and is reported via `DeviceCommandResult`.

---

## 8. Telemetry Contract & Multimodal Ingress

`DeviceTelemetry` captures device state and environmental observations:
- Native conversion to `MultimodalObservation` via `to_multimodal_observation()`.
- Observations enter `CentralInputGateway`, are correlated by `SituationFusionEngine`, and update `WorldState` without direct store mutations.

---

## 9. Heartbeat, Health, and Failsafe Monitoring

- `DeviceHeartbeat` provides high-frequency liveness beacons with monotonically increasing sequence numbers.
- `DeviceHealth` computes overall status (`HEALTHY`, `DEGRADED`, `UNHEALTHY`) based on heartbeat age, battery thresholds, and error rates.
- On-demand evaluation in `DeviceGateway` ensures zero daemon background threads, preserving test reproducibility and runtime determinism.

---

## 10. Canonical Error Semantics

Device errors are mapped directly to `DeviceErrorCode`:
- Transient vs permanent errors are distinguished via `is_retryable`.
- Adapters catch protocol-specific exceptions and map them to canonical `DeviceError` structures embedded in `DeviceCommandResult(success=False, error=...)`.

---

## 11. DeviceAdapter Interface Boundary

The `DeviceAdapterInterface` represents the protocol boundary between ATLAS Central and edge devices:
- Default implementations for `connect()`, `disconnect()`, `get_health()`, `get_status()`, `get_capabilities()`, and `heartbeat()`.
- Protocol naming is strictly transport-neutral (no "mavlink_adapter" or "ros_adapter" hardcoded in core logic).

---

## 12. ATLAS Vision Central Integration

ATLAS Vision acts as a continuous observation feeder:
```
ATLAS Vision Adapter (VirtualVisionAdapter)
    |
    v (MultimodalObservation / Raw dict)
CentralInputGateway.ingest_observation()
    |
    v
SituationFusionEngine.process_observation()
    |
    v (Fused Situations)
WorldStateUpdater.apply_situation()
    |
    v
WorldState (Believed Reality)
    |
    v
AutonomousGoalManager / CognitiveRuntime
```
ATLAS Central maintains absolute authority: Vision cannot create goals or dispatch other devices directly.

---

## 13. Virtual Simulation Architecture

Each product is provided with a rich virtual adapter for deterministic testing and simulation:
- `VirtualVisionAdapter`: Simulates motion, person, and anomaly detection with configurable events.
- `VirtualGlassAdapter`: Simulates HUD message display, egocentric image/audio capture, and notifications.
- `VirtualDroneAdapter`: Simulates multi-step flight states (grounded, climbing, cruising, descending), battery decay, and low-battery failsafes.
- `VirtualRoverAdapter`: Simulates waypoint navigation, obstacle detection, and emergency stops.

Simulation adapters support fault-injection flags (`simulate_obstacle`, `simulate_timeout`, `simulate_failure`) for testing robustness.

---

## 14. Replay and Determinism

All command dispatches, acknowledgements, results, and observations retain causal lineage:
- Invariant: Given identical sequences of inputs and random seeds, the Central Orchestrator and Device Gateway produce identical command dispatch sequences and state transitions.

---

## 15. Versioning and Forward Compatibility

- `contract_version`: Declared on `DeviceIdentity` (e.g., `"2.0"`).
- `schema_version`: Declared on `DeviceCapabilityDescriptor` (e.g., `"1.0"`).
- Forward compatibility: Unknown future attributes in telemetry or metadata dictionaries are safely preserved without schema corruption or crashes.

---

## 16. Future Physical Adapters (Transport Neutrality)

When physical edge devices are deployed in future phases:
1. Physical drivers will run either out-of-process or as pluggable adapters implementing `DeviceAdapterInterface`.
2. Core ATLAS services (`CognitiveRuntime`, `PolicyEngine`, `ToolOrchestrator`, `WorldState`, `DeviceGateway`) will remain 100% unchanged.
3. No hardware-specific headers or transport dependencies will ever leak into `backend/core`.

---

## 17. Phase 6.2 Compliance Verification

- **Total Phase 6.2 Tests**: 70 passed (100%).
- **Phase 6.1 Production Runtime Tests**: 50 passed (100%).
- **Phase 5.0 Gateway & Orchestration Tests**: 174 passed (100%).
- **Phase 4.0-4.6 World State & Autonomy Tests**: 96 passed (100%).
- **Total Passing Regression Tests**: 390 passed.
- **Hardware/Transport Audit**: 0 prohibited imports, 0 raw sockets, 0 hardware drivers.
