# ATLAS Phase 6.5c: Spatial & Telemetry Perception Architecture

## 1. Executive Summary

ATLAS Phase 6.5c introduces the **Spatial and Telemetry Perception Layer**, formalizing the semantic perception, normalization, and quality validation of geographic positions, spatial relationships, geofencing, kinematic navigation, and multi-product device telemetry across the ATLAS ecosystem.

In conformance with the core ATLAS architectural invariant:
> **The perception layer produces descriptive semantic evidence. It NEVER makes operational decisions, evaluates missions, triggers goal replanning, mutates Central World State, or commands actuators.**

Telemetry metrics (e.g., `battery_percent = 12.5%`, `connectivity = DEGRADED`) and spatial observations (e.g., `distance_to_fence = 1.2m`) are purely factual evidence emitted as typed `PerceptionEvidence` and normalized to `MultimodalObservation` for standard ingestion via `CentralInputGateway` into `SituationFusionEngine`.

---

## 2. Architectural Context & Placement

```
                                  Edge Devices & Sensors
                  (Drone, Rover, Glass, Vision, Simulation Twin)
                                            │
                                            ▼
                           PerceptionProviderRegistry
                                            │
                                            ▼
                        SpatialTelemetryPerceptionProvider
                     ┌──────────────────────┴──────────────────────┐
                     ▼                                             ▼
              SpatialProcessor                             TelemetryProcessor
       (WGS-84, Haversine, Bearing,                (Units, Freshness, Health,
         Relative Pos, Geofencing)                    Quality, Sanitization)
                     └──────────────────────┬──────────────────────┘
                                            │ PerceptionResult (Typed Evidence)
                                            ▼
                            PerceptionObservationNormalizer
                                            │ Canonical MultimodalObservation
                                            ▼
                                   CentralInputGateway
                                            │
                                            ▼
                                   SituationFusionEngine
                                            │
                                            ▼
                                     WorldState / Events
                                            │
                                            ▼
                         Multi-Product Situation & Mission Intelligence
                                            │
                                            ▼
                                   AutonomousGoalManager
                                            │
                                            ▼
                                     CognitiveRuntime
                                            │
                                            ▼
                                       PolicyEngine
                                            │
                                            ▼
                                     ToolOrchestrator
                                            │
                                            ▼
                                      DeviceGateway
                                            │
                                            ▼
                                      Edge Products
```

---

## 3. Architectural Flow & Chain of Custody

Every spatial coordinate and telemetry reading travels through a strict, deterministic, and auditable chain of custody:

1. **Raw Ingestion**: Raw measurements from physical sensors or digital twin simulation engines arrive wrapped in `PerceptionInput` with unique `input_id`, `captured_at` timestamp, and `source_id`.
2. **Provider Evaluation**: `SpatialTelemetryPerceptionProvider` dispatches input to `SpatialProcessor` and `TelemetryProcessor`.
3. **Validation & Normalization**: Coordinate bounds, altitude precision, battery limits, SI units, and connectivity statuses are verified and normalized.
4. **Quality & Freshness Assessment**: Telemetry age is evaluated against `max_staleness_seconds`. Quality is categorized as `VALID`, `STALE`, `PARTIAL`, or `INVALID`.
5. **Provenance Assembly**: All 8 mandatory provenance keys (`source_id`, `provider_id`, `provider_version`, `input_id`, `request_id`, `observation_timestamp`, `correlation_id`, `causation_id`) are attached to every evidence record.
6. **Canonical Ingestion**: Evidence is translated by `PerceptionObservationNormalizer` into a `MultimodalObservation` and submitted to `CentralInputGateway`.
7. **Semantic Fusion**: `SituationFusionEngine` correlates spatial telemetry evidence with temporal context to update `WorldState`.

---

## 4. Modality Definitions

Phase 6.5c standardizes three fundamental modalities:

| Modality Type | Value | Scope & Purpose |
| :--- | :--- | :--- |
| `ModalityType.GPS` | `"gps"` | Spatial coordinates (WGS-84 lat/lon), altitude, accuracy, speed, heading, geofences, and relative bearings. |
| `ModalityType.TELEMETRY` | `"telemetry"` | Time-series metrics: battery, signal quality, core temperature, CPU usage, voltage, velocity, and custom sensor readings. |
| `ModalityType.DEVICE_STATE` | `"device_state"` | Aggregate device operational health, online/offline connectivity status, and peripheral readiness indicators. |

---

## 5. Spatial Domain Models

Located in `backend/core/models/spatial_telemetry.py`:

- `PositionObservation`: Immutable geographic point with `latitude`, `longitude`, `altitude`, `accuracy`, `heading`, `speed`, and provenance.
- `RelativePosition`: Spatial relationship between reference entity and target entity (`distance_meters`, `bearing_degrees`, `relative_altitude`, `relation`).
- `NavigationObservation`: Entity kinematic trajectory (`position`, `movement_state`, `heading`, `speed`, `route_reference`).
- `GeofenceArea`: Boundary definition supporting circular radii and axis-aligned bounding boxes with tolerance buffers.
- `SpatialObservation`: Comprehensive container combining position, navigation, geofence evaluations, and relative positions.

---

## 6. Telemetry Domain Models

- `TelemetryMetric`: Individual measurement (`name`, `value`, `unit`, `raw_name`, `quality`, `confidence`, `timestamp`).
- `TelemetryHealth`: Summarized device vitals (`health_status`, `connectivity`, `battery_percent`, `temperature_celsius`, `signal_quality`, `cpu_usage_percent`).
- `TelemetryObservation`: Structured collection of metrics, health state, age, and staleness evaluation.
- `TelemetryQuality`: Quality classification enum (`VALID`, `STALE`, `PARTIAL`, `INVALID`).
- `MovementState`: Kinematic state enum (`STATIONARY`, `MOVING`, `ACCELERATING`, `DECELERATING`, `UNKNOWN`).

---

## 7. Coordinate Reference System (WGS-84) & Bounds

All spatial coordinates conform strictly to WGS-84 geographic coordinates:
- **Latitude**: Must be finite float in $[-90.0, +90.0]$ degrees.
- **Longitude**: Must be finite float in $[-180.0, +180.0]$ degrees.
- **Altitude**: Bounded in $[-1000.0, +100000.0]$ meters relative to sea level.
- **Accuracy**: Non-negative finite float representing horizontal error in meters.
- **Rejection**: Any NaN, Infinite, string, or out-of-bound coordinates are rejected with descriptive errors.

---

## 8. Geodesy Algorithms

Mathematical algorithms in `backend/spatial_telemetry/spatial.py`:

### Haversine Great-Circle Distance
$$a = \sin^2\left(\frac{\Delta\phi}{2}\right) + \cos(\phi_1)\cos(\phi_2)\sin^2\left(\frac{\Delta\lambda}{2}\right)$$
$$c = 2 \cdot \operatorname{atan2}\left(\sqrt{a}, \sqrt{1 - a}\right)$$
$$d = R \cdot c \quad (\text{where } R = 6{,}371{,}000\text{ m})$$
- Coincident coordinates ($p_1 == p_2$) return $0.0\text{ m}$ deterministically.

### Initial Forward Azimuth (Bearing)
$$y = \sin(\Delta\lambda)\cos(\phi_2)$$
$$x = \cos(\phi_1)\sin(\phi_2) - \sin(\phi_1)\cos(\phi_2)\cos(\Delta\lambda)$$
$$\theta = \operatorname{atan2}(y, x)$$
$$\text{bearing} = (\operatorname{degrees}(\theta) + 360.0) \pmod{360.0}$$
- Coincident points return $0.0^\circ$ deterministically.
- All returned bearings are normalized strictly to $[0.0, 360.0)^\circ$.

---

## 9. Relative Positioning & Categorization

Distance and bearing between entities are classified into semantic spatial relations:

| Relation | Range Threshold | Description |
| :--- | :--- | :--- |
| `SAME_LOCATION` | $< 5.0\text{ m}$ | Co-located entities within GPS tolerance |
| `NEARBY` | $5.0\text{ m} \le d < 50.0\text{ m}$ | Immediate operating proximity |
| `VICINITY` | $50.0\text{ m} \le d < 500.0\text{ m}$ | Local operational area |
| `DISTANT` | $500.0\text{ m} \le d < 5000.0\text{ m}$ | Extended sensor/visual range |
| `REMOTE` | $\ge 5000.0\text{ m}$ | Distant edge or station horizon |

---

## 10. Lightweight Geofencing

Geofence evaluation supports two geometric modes without heavyweight GIS dependencies:

1. **Circular Geofence**: Centered at $(lat_c, lon_c)$ with radius $R$ and tolerance buffer $\tau$:
   - `INSIDE`: $d < R - \tau$
   - `BOUNDARY`: $|d - R| \le \tau$
   - `APPROACHING`: $R + \tau < d \le 1.2 \cdot R$
   - `OUTSIDE`: $d > 1.2 \cdot R$

2. **Bounding Box Geofence**: Defined by $[lat_{min}, lat_{max}] \times [lon_{min}, lon_{max}]$:
   - `INSIDE`: $lat_{min} \le lat \le lat_{max}$ and $lon_{min} \le lon \le lon_{max}$
   - `OUTSIDE`: Coordinate outside bounding box

---

## 11. Telemetry Quality Assessment

Independent of model confidence, telemetry data is classified by operational validity:

- `VALID`: Fresh, structurally intact, within nominal engineering tolerances.
- `STALE`: Structure is valid, but observation age exceeds `max_staleness_seconds`.
- `PARTIAL`: Incomplete payload with missing metrics or partial readouts.
- `INVALID`: Unparseable numeric values, out-of-bounds metrics, or corrupted payload.

---

## 12. Freshness & Staleness Rules

Freshness is evaluated dynamically using:
$$\text{age} = \max(0.0, t_{\text{current}} - t_{\text{captured}})$$
- Default threshold: `max_staleness_seconds = 300.0s`.
- When $\text{age} > \text{threshold}$, `TelemetryObservation.is_stale = True` and quality transitions to `TelemetryQuality.STALE`.

---

## 13. Metric & Unit Normalization

Measurements are normalized to standard SI and engineering units:
- Temperature: `Celsius` / `C`
- Battery: `percent` / `%` bounded in $[0.0, 100.0]$
- Speed: `m/s`
- Heading / Azimuth: `deg` in $[0.0, 360.0)$
- Voltage: `V`
- Current: `A`
- Signal Quality: `percent` / `%` or `dBm`
- Frequency / Frame Rate: `fps`

---

## 14. Health & Connectivity State Normalization

Device status fields are canonicalized against Phase 6.2 contracts:
- `ConnectivityStatus`: `ONLINE`, `OFFLINE`, `DEGRADED`, `CONNECTING`, `DISCONNECTED`, `UNKNOWN`.
- `DeviceHealthStatus`: `HEALTHY`, `DEGRADED`, `UNHEALTHY`, `UNKNOWN`.

---

## 15. Concrete Provider: SpatialTelemetryPerceptionProvider

Located in `backend/spatial_telemetry/provider.py`:
- Implements `PerceptionProviderInterface`.
- Supported Modalities: `ModalityType.GPS`, `ModalityType.TELEMETRY`, `ModalityType.DEVICE_STATE`.
- Supported Capabilities: `PerceptionCapability.SPATIAL_LOCALIZATION`, `PerceptionCapability.TELEMETRY_NORMALIZATION`.
- Deterministic error handling with typed `PerceptionError` codes (`PROVIDER_UNAVAILABLE`, `UNSUPPORTED_MODALITY`, `INVALID_INPUT`, `PROCESSING_FAILURE`).

---

## 16. Reference Provider: MockSpatialTelemetryProvider

Located in `backend/spatial_telemetry/reference_provider.py`:
- Provides deterministic synthetic telemetry and GPS tracks for unit tests, offline replays, and CI pipelines without network dependencies.
- Supports pre-canned edge product profiles: Aerial Drone, Ground Rover, Smart Glass, and Static Vision Camera.

---

## 17. Registry Integration & Provider Discovery

- Registers with `PerceptionProviderRegistry`.
- Discovered deterministically via `select_provider(capability, modality)` or `get_provider(provider_id)`.
- Dual callable/set interface enables standard set queries (`mod in provider.supported_modalities`) and functional invocations (`provider.supported_modalities()`).

---

## 18. Observation Normalizer Integration & Gateway Ingestion

- Normalized by `PerceptionObservationNormalizer` into canonical `MultimodalObservation`.
- Automatically populates `MultimodalObservation.location` from `SpatialEvidence.location`.
- Preserves complete 8-key provenance in `metadata["provenance"]`.
- Ingested directly into `CentralInputGateway` without bypassing central routing.

---

## 19. Edge Product Support

Standardized payloads for all Phase 6 edge products:
1. **Vision Product**: Sensor temperature, framerate (`fps`), streaming connectivity, camera status.
2. **Glass Product**: Wearable battery, pitch/yaw head orientation, display brightness, degraded BLE link.
3. **Drone Product**: 3D coordinates, altitude, vertical velocity, ground speed, compass heading, circular geofences.
4. **Rover Product**: 2D ground coordinates, odometer distance, motor temperature, surface speed, perimeter box fences.

---

## 20. Digital Twin Simulation Integration

- Ingests simulation state packets produced by Phase 6.3 `VirtualDevice` and `SimulationEngine`.
- Flags simulated observations with `is_simulated = True` in provenance.
- Evaluates simulated telemetry with the exact same semantic rigor applied to physical hardware.

---

## 21. Security, Credential Redaction & Privacy

- Strict sanitization: `api_key`, `password`, `bearer_token`, `secret`, `token` are stripped before metadata creation.
- Adheres to `PerceptionPrivacyClass` classification (`INTERNAL`, `CONFIDENTIAL`, `RESTRICTED`).
- Prevents leaking device credentials into Central World State or long-term cognitive memory.

---

## 22. Resource Limits & Bounded Buffers

Configured via `SpatialTelemetryLimits`:
- `max_metrics_per_result`: 50
- `max_attributes_per_metric`: 20
- `max_spatial_relations`: 20
- `max_geofences`: 20
- `max_batch_size`: 50
- `max_string_length`: 128
- `max_staleness_seconds`: 300.0

---

## 23. Architectural Boundary Invariants

Perception components strictly obey the following prohibitions:
1. **Zero World State Mutation**: Cannot mutate `WorldState` instances.
2. **Zero Goal Store Mutation**: Cannot create, update, or cancel goals.
3. **Zero Autonomous Goal Manager Calls**: Cannot trigger goal replanning.
4. **Zero Cognitive Runtime Calls**: Cannot execute cognitive loops.
5. **Zero Tool Orchestrator Calls**: Cannot execute tools or functions.
6. **Zero Device Gateway Actuation**: Cannot invoke device commands.
7. **Zero Decision Leakage**: Evidence reports factual data only (`battery = 10%`), never commands (`abort mission`).
8. **Zero Physical Drivers**: No imports of `ros2`, `rospy`, `rclpy`, `pymavlink`, `mavsdk`, `serial`, or `RPi.GPIO`.
9. **Zero Subprocess / Eval**: No `subprocess`, `os.system`, `eval()`, or `exec()`.

---

## 24. Verification & Regression Matrix

| Test Suite | File Path | Tests | Result |
| :--- | :--- | :--- | :--- |
| **Phase 6.5c Spatial & Telemetry** | `backend/brain/tests/test_spatial_telemetry_perception_phase6_5c.py` | 50 | **PASSED (100%)** |
| **Phase 6.5b Visual Perception** | `backend/brain/tests/test_visual_perception_phase6_5b.py` | 45 | **PASSED (100%)** |
| **Phase 6.5a Perception Contracts** | `backend/brain/tests/test_perception_contracts_phase6_5a.py` | 46 | **PASSED (100%)** |
| **Phase 6.4 Mission Intelligence** | `backend/brain/tests/test_mission_intelligence_phase6_4.py` | 51 | **PASSED (100%)** |
| **Phase 6.3 Simulation Twin** | `backend/brain/tests/test_simulation_phase6_3.py` | 76 | **PASSED (100%)** |
| **Phase 6.2 Device Contracts** | `backend/brain/tests/test_device_contract_phase6_2.py` | 70 | **PASSED (100%)** |
| **Phase 6.1 Production Runtime** | `backend/brain/tests/test_production_runtime.py` | 50 | **PASSED (100%)** |
| **Phase 5.0 Central Orchestration** | `backend/brain/tests/test_central_orchestration.py` | 174 | **PASSED (100%)** |
| **Phase 4.x Brain Capabilities** | `backend/brain/tests/test_system_coherence.py` etc. | 96 | **PASSED (100%)** |
