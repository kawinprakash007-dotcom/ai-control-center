# ATLAS Phase 6.3 — Digital Twin & Simulation Framework Architecture

## 1. Executive Summary & Purpose of Phase 6.3

Phase 6.3 establishes the high-fidelity, deterministic **Digital Twin & Simulation Subsystem** for ATLAS Central. It introduces full digital twin implementations for all four edge products (ATLAS Vision, ATLAS Glass, ATLAS Drone, and ATLAS Rover), a monotonic deterministic simulation clock, an environmental and spatial entity model, controlled multi-modal fault injection, transparent adapter equivalence with future physical hardware, and an automated scenario scripting and assertion engine.

> [!IMPORTANT]
> **SIMULATION ONLY — ZERO PHYSICAL DRIVERS OR OS SUBPROCESSES.**
> This phase does NOT integrate physical hardware drivers (`pymavlink`, `mavsdk`, `rclpy`, `rospy`, `paho-mqtt`, `serial`, `RPi.GPIO`). All simulation runs deterministically in-process using explicit state stepping, zero OS subprocesses, zero browser automation, zero daemon threads, and zero `time.sleep()` calls.

### Core Architectural Guarantee
ATLAS Central remains the singular semantic, reasoning, and policy brain. The Digital Twin layer acts strictly as a peripheral simulation boundary. Digital twins produce canonical `MultimodalObservation` objects and consume canonical `DeviceCommand` instances. They **never** bypass `CentralInputGateway`, `SituationFusion`, `WorldState`, `CognitiveRuntime`, `PolicyEngine`, or `DeviceGateway`.

---

## 2. Digital Twin Architecture & Structural Relationship to Central

The digital twin subsystem connects to ATLAS Central through the exact same peripheral boundary used by physical devices:

```
+-----------------------------------------------------------------------------------+
|                            ATLAS CENTRAL BRAIN                                    |
|                                                                                   |
|   +-----------------------+                    +------------------------------+   |
|   |  CentralInputGateway  |                    |        DeviceGateway         |   |
|   +-----------+-----------+                    +--------------+---------------+   |
|               |                                               ^                   |
|               v                                               |                   |
|   +-----------+-----------+                    +--------------+---------------+   |
|   | SituationFusionEngine |                    |       ToolOrchestrator       |   |
|   +-----------+-----------+                    +--------------+---------------+   |
|               |                                               ^                   |
|               v                                               |                   |
|   +-----------+-----------+                    +--------------+---------------+   |
|   |      WorldState       |                    |         PolicyEngine         |   |
|   +-----------+-----------+                    +--------------+---------------+   |
|               |                                               ^                   |
|               v                                               |                   |
|   +-----------+-----------------------------------------------+-----------+   |   |
|   |                CognitiveRuntime / Goal Management                     |   |   |
|   +-----------------------------------------------------------------------+   |   |
+-----------------------------------------------------------------------------------+
                                ^                               |
            Observations        |                               | Dispatched Commands
          & Telemetry Ingress   |                               | via DeviceAdapter
                                |                               v
+-----------------------------------------------------------------------------------+
|                        DIGITAL TWIN SIMULATION SUBSYSTEM                          |
|                                                                                   |
|  +------------------------+  +------------------------+  +---------------------+  |
|  |    SimulationWorld     |  |    SimulationClock     |  | FaultInjectionMgr   |  |
|  +------------------------+  +------------------------+  +---------------------+  |
|                                                                                   |
|  +---------------------+  +---------------------+  +------------------------+     |
|  |  VisionDigitalTwin  |  |  GlassDigitalTwin   |  |   DroneDigitalTwin     |     |
|  | (OBSERVATION_SOURCE)|  |       (HYBRID)      |  |        (HYBRID)        |     |
|  +---------------------+  +---------------------+  +------------------------+     |
|                                                                                   |
|  +---------------------+  +---------------------+  +------------------------+     |
|  |   RoverDigitalTwin  |  | SimulatedEnvironment|  |     ScenarioRunner     |     |
|  |       (HYBRID)      |  |  (Entities/Hazards) |  |   (Script & Assert)    |     |
|  +---------------------+  +---------------------+  +------------------------+     |
+-----------------------------------------------------------------------------------+
```

---

## 3. Product Digital Twins: ATLAS Vision, Glass, Drone, Rover

All four edge products are modeled with complete, deterministic behavioral state machines:

### 3.1 ATLAS Vision (`VisionDigitalTwin`)
- **Product Role**: `ProductRole.OBSERVATION_SOURCE`
- **Capabilities**: `detect_motion`, `detect_person`, `detect_anomaly`, `capture_image`, `capture_video`, `get_telemetry`
- **Behavioral Model**: Simulates stationary or PTZ optical perception. Detects simulated entities and hazards within its sensory FOV. Synthesizes canonical `MultimodalObservation` events (e.g., `PERSON_DETECTED`, `MOTION_DETECTED`, `ANOMALY_DETECTED`) with bounding boxes and confidence scores.
- **Actuation Boundary**: Rejects actuator motion commands; acts strictly as an observational sensory source.

### 3.2 ATLAS Glass (`GlassDigitalTwin`)
- **Product Role**: `ProductRole.HYBRID`
- **Capabilities**: `display_hud`, `capture_image`, `capture_audio`, `send_notification`, `get_position`, `get_telemetry`
- **Behavioral Model**: Simulates wearable AR glasses. Maintains an internal HUD message and alert buffer with priority levels (`LOW`, `NORMAL`, `HIGH`, `CRITICAL`), simulated battery drain, audio waveform capture metadata, and user point-of-view imagery.

### 3.3 ATLAS Drone (`DroneDigitalTwin`)
- **Product Role**: `ProductRole.HYBRID`
- **Flight State Machine**: `GROUNDED` -> `HOVERING` <-> `NAVIGATING` -> `LANDED` (plus `EMERGENCY_LAND`)
- **Capabilities**: `takeoff`, `land`, `goto_location`, `navigate_waypoint`, `hover`, `return_to_base`, `capture_image`, `capture_video`, `get_position`, `get_telemetry`
- **Behavioral Model**: 3D spatial dynamics with realistic waypoint interpolation, altitude constraints, battery consumption curves (accelerated during ascent and high-speed transit), and automatic return-to-base / emergency landing triggers when battery falls below 15.0%.

### 3.4 ATLAS Rover (`RoverDigitalTwin`)
- **Product Role**: `ProductRole.HYBRID`
- **Drive State Machine**: `STOPPED` <-> `MOVING` <-> `PATROLLING` <-> `DOCKED`
- **Capabilities**: `move`, `navigate_to`, `goto_location`, `stop`, `patrol_zone`, `dock`, `capture_image`, `capture_video`, `get_position`, `get_telemetry`
- **Behavioral Model**: Surface navigation across 2D ground coordinates. Includes obstacle proximity detection that triggers deterministic safety stops (`SAFETY_REJECTION`), zone patrols, docking station docking, and battery recharging (+0.2% battery/sec when docked).

---

## 4. SimulationClock & Deterministic Monotonic Time Progression

The `SimulationClock` is the single source of time truth for all simulated components:
- **Monotonicity**: Simulation time cannot move backwards. `advance(seconds)` and `set_time(timestamp)` reject negative time deltas or historical timestamps with `ValueError`.
- **Zero Drift / Determinism**: Eliminates all wall-clock dependency (`time.sleep()`, OS thread scheduling). Stepping 10.0 seconds always executes precisely 10.0 seconds of simulation logic regardless of CPU speed.
- **Listener Callbacks**: Thread-safe notification system allows twins, world environments, and logging hooks to synchronize state whenever time advances. Faults inside listener callbacks are isolated and cannot crash clock progression.
- **Replay Compatibility**: Exact timestamp sequences can be recorded and re-stepped deterministically to reproduce complex edge phenomena.

---

## 5. Spatial Environment & Geographic Simulation (Entities & Hazards)

The spatial environment is managed by `SimulatedEnvironment` in `backend/simulation/environment.py`:
- **`SimulatedEntity`**: Represents physical or semantic objects in the world:
  - Fields: `entity_id`, `entity_type` (`PERSON`, `VEHICLE`, `OBSTACLE`, `ANOMALY`), `position` (`TwinPosition`), `velocity`, `heading`, `attributes`.
- **`SimulatedHazard`**: Represents localized environmental conditions or danger zones:
  - Fields: `hazard_id`, `hazard_type` (`MOTION`, `SMOKE`, `HEAT_ANOMALY`, `RESTRICTED_ZONE`), `location` (`TwinPosition`), `radius_meters`, `severity`, `is_active`.
- **Spatial Queries**:
  - `get_entities_near(position, radius_meters)`: Identifies all entities within sensory reach.
  - `get_hazards_near(position, radius_meters)`: Detects active hazards intersecting a device's sensory boundary using 3D Euclidean distance approximation.
- **Capacity Limits**: Bounded collections prevent unbounded memory growth in long-running scenarios.

---

## 6. State Immutability & Thread Safety Model

To prevent concurrency bugs and simulation contamination:
- **`TwinState`**: Fully immutable (`frozen=True`) dataclass snapshot representing the complete operational condition of a twin at a specific instant (connectivity, health, battery, position, active capabilities, active faults, and internal telemetry metrics).
- **`TwinPosition`**: Frozen 3D coordinate model (`latitude`, `longitude`, `altitude`, `heading`, `speed`) with distance computation.
- **Reentrant Locks (`threading.RLock`)**: All mutable simulation state containers (`BaseDigitalTwin`, `SimulationClock`, `SimulatedEnvironment`, `FaultInjectionManager`, `SimulationWorld`) guard internal mutations with dedicated reentrant locks.
- **Defensive Copies**: Methods returning collections return immutable tuples or deep dictionary copies (`dict(self._internal_state)`), ensuring external callers cannot mutate internal twin state.

---

## 7. Fault Injection Subsystem & Failure Mode Catalog

The `FaultInjectionManager` provides deterministic, scriptable fault injection across all digital twins.

### Supported Fault Taxonomy (`TwinFaultType`)
| Fault Type | Description | Behavioral Impact on Twin |
| :--- | :--- | :--- |
| `OFFLINE` | Hardware or link disconnection | State reports `DISCONNECTED`/`UNHEALTHY`; commands rejected with `OFFLINE_DEVICE`. |
| `LOW_BATTERY` | Rapid battery depletion | Clamps battery to specified critical percentage (e.g. 8.0%); triggers failsafes. |
| `GPS_LOSS` | Satellite signal disruption | Strips GPS coordinate data from state and telemetry; sets health to `DEGRADED`. |
| `SENSOR_FAILURE` | Optical/LiDAR/audio sensor fault | Sensor actions fail with `DEVICE_ERROR`; observations suppressed. |
| `TELEMETRY_STALE` | Frozen telemetry generator | Telemetry frames repeat frozen historical timestamp without updating. |
| `COMMAND_TIMEOUT` | Unresponsive peripheral | Command execution simulates dropped response, returning `COMMAND_TIMEOUT`. |
| `COMMAND_FAILURE` | Actuator hardware rejection | Command execution fails with configured `DeviceErrorCode` and simulated error message. |
| `DUPLICATE_OBSERVATION` | Transmission echo | Ingress generates identical observation IDs to test Central deduplication. |
| `CONFLICTING_OBSERVATION` | Multi-sensor inconsistency | Produces divergent sensory payloads for the same spatial target. |

### Auto-Expiry
Faults support optional `duration_seconds`. During clock ticks, expired faults are automatically culled, restoring standard device behavior deterministically.

---

## 8. Device Contract Equivalence & Transparent Adapter Boundary

A foundational architectural requirement of Phase 6.3 is **100% contract equivalence**:
- `DigitalTwinAdapter` implements the canonical `DeviceAdapterInterface`.
- Future physical adapters (e.g., MAVLink, ROS2, or WebRTC adapters) will implement the **exact same** `DeviceAdapterInterface`.
- `create_digital_twin_device(twin)` produces a standard `DeviceIdentity` and `DigitalTwinAdapter` registered into `DeviceGateway` without special hooks or simulation flags.
- `DeviceGateway`, `PolicyEngine`, and `ToolOrchestrator` treat digital twins identically to real physical edge hardware:
  ```python
  ident, adapter = create_digital_twin_device(drone_twin)
  gw.register_device(ident)
  gw.register_adapter(adapter, ident.device_id)
  res = gw.dispatch_to_device(ident.device_id, "flight", "takeoff", parameters={"altitude": 10.0})
  ```

---

## 9. Ingress Routing & Observation Pipeline Preservation

Digital twins NEVER write directly to `WorldState`. All sensory events are emitted as canonical `MultimodalObservation` objects and follow the full central pipeline:
1. `DigitalTwin.generate_observation()` or `SimulationWorld.step()` produces `MultimodalObservation`.
2. `SimulationWorld.flush_observations()` yields pending observations.
3. `CentralInputGateway.ingest_observation()` validates metadata, assigns trace IDs, and rejects malformed/duplicate payloads.
4. `SituationFusionEngine` correlates multi-device observations and resolves conflicts.
5. `WorldStateUpdater` applies verified spatial updates to `WorldState`.

---

## 10. Autonomy Governance & Execution Authority Preservation

Digital twins have zero reasoning capabilities:
- Twins contain no LLM invocations, no planners, no goal stores, and no autonomous decision authority.
- Twins cannot dispatch commands to other twins; they only accept commands originating from Central's `DeviceGateway`.
- All commands undergo central validation:
  - `PolicyEngine` enforces safety constraints, authorization, and operational boundaries.
  - `DeviceGateway` enforces capability schema checks and duplicate suppression.

---

## 11. Deterministic Scenario Authoring & Multi-Agent Scripting

Scenarios are authored using a clean, declarative builder pattern (`ScenarioBuilder`):
```python
scenario = (
    ScenarioBuilder(scenario_id="perimeter_breach_01", name="Perimeter Breach & Drone Response")
    .with_initial_time(1000000.0)
    .add_twin_config(TwinConfiguration("VISION_NORTH", ProductType.VISION, ProductRole.OBSERVATION_SOURCE))
    .add_twin_config(TwinConfiguration("DRONE_ALPHA", ProductType.DRONE, ProductRole.HYBRID))
    .add_step("s1", 5.0, "EMIT_OBSERVATION", "VISION_NORTH", {"type": "MOTION_DETECTED", "zone": "perimeter_north"})
    .add_step("s2", 10.0, "DISPATCH_COMMAND", "DRONE_ALPHA", {"capability": "flight", "action": "takeoff", "parameters": {"altitude": 15.0}})
    .add_assertion("a1", "TWIN_STATE", "DRONE_ALPHA", "flight_state", "HOVERING")
    .build()
)
```

Supported step action types:
- `DISPATCH_COMMAND`: Dispatches a `DeviceCommand` to a digital twin.
- `EMIT_OBSERVATION`: Forces a twin to synthesize an observation.
- `INJECT_FAULT`: Injects a `TwinFault` into a target twin.
- `REMOVE_FAULT`: Clears a fault.
- `ADVANCE_TIME`: Explicitly steps simulation time forward.
- `SPAWN_ENTITY`: Adds a spatial entity to the simulated environment.
- `ADD_HAZARD`: Adds an environmental hazard.

---

## 12. Scenario Assertions & Post-Condition Verification Engine

The `ScenarioRunner` evaluates post-condition assertions (`ScenarioAssertion`) at the conclusion of scenario execution:
- **Target Types**:
  - `TWIN_STATE`: Verifies fields on `TwinState` (e.g., `battery`, `flight_state`, `connectivity`, `position`).
  - `COMMAND_RESULT`: Verifies success, output, or error codes of dispatched commands.
  - `OBSERVATION_COUNT`: Verifies total observations generated and ingested into Central.
  - `ACTIVE_FAULT`: Verifies the presence or absence of active faults.
- **Operators**: `EQUALS`, `CONTAINS`, `GREATER_THAN`, `LESS_THAN`, `IS_NONE`, `NOT_NONE`.
- **Result Summaries**: If any assertion fails, `ScenarioResult.success` is `False`, and detailed failure records (`assertion_id`, `field`, `expected`, `actual`) are captured in `assertion_failures`.

---

## 13. Deterministic Replay & Flight Recorder Compatibility

Because all state transitions are strictly governed by `SimulationClock` and deterministic arithmetic without wall-clock races:
- Re-running the identical `Scenario` produces an identical execution trace, identical observation sequence, and identical end states.
- Traces capture step execution IDs, timestamps, dispatched commands, returned results, and observation counts.
- Replays are 100% compatible with Phase 6.1 production flight recorders and audit logs.

---

## 14. Capacity Limits, Bound Enforcement & Resource Safety

To prevent memory leaks and resource exhaustion in long-running tests:
- `SimulationLimits` enforces strict defaults:
  - `max_twins`: 100
  - `max_entities`: 500
  - `max_pending_observations`: 1,000
  - `max_telemetry_records`: 100 per twin
  - `max_fault_records`: 200
  - `max_scenario_steps`: 1,000
- Deque buffers (`_telemetry_history`, `_pending_observations`, `_recent_snapshots`) use bounded `maxlen`.
- Over-capacity entity and twin additions raise explicit capacity errors.

---

## 15. Transport Neutrality & Hardware Abstraction

The simulation subsystem maintains absolute transport neutrality:
- Protocol names are reported purely as `"digital_twin"`.
- Zero MAVLink message IDs, ROS topic names, MQTT topics, or AT serial commands are used.
- Commands use semantic actions (`takeoff`, `land`, `move`, `display_hud`, `detect_person`) rather than protocol-level byte arrays or hardware packets.

---

## 16. Zero Prohibited Imports & Security Compliance

A dedicated static security test (`test_AA01_zero_hardware_or_system_imports`) scans all Phase 6.3 files (`clock.py`, `environment.py`, `fault_injection.py`, `twin.py`, `adapters.py`, `world.py`, `scenario.py`, `runner.py`) ensuring:
- **Zero Hardware Driver Imports**: No `pymavlink`, `mavsdk`, `rclpy`, `rospy`, `paho`, `serial`, `RPi.GPIO`.
- **Zero Shell / Subprocess Code**: No `subprocess`, `os.system`, `eval`, `exec`, `shutil.rmtree`.
- **Zero Browser Automation**: No `playwright`, `selenium`.
- **Zero Daemon Threads / Sleep**: No `time.sleep()`, no background thread workers.
- **Credential Scrubbing**: Telemetry and configuration dictionaries scrub tokens and API keys via `sanitize_contract_metadata()`.

---

## 17. Boundary Defense: No Direct WorldState, Goal, or Tool Bypasses

Rigorous unit tests enforce architectural isolation:
- `test_AB01`: Confirms simulation world mutations do not touch `WorldState`.
- `test_AC01`: Confirms policy rejections prevent execution on twins.
- `test_AD01`: Confirms digital twins possess no `create_goal()`, `plan()`, or `reason()` methods.

---

## 18. Test Coverage Matrix across Sections A through AF

The Phase 6.3 test suite in `backend/brain/tests/test_simulation_phase6_3.py` contains **76 tests** covering all 32 requirement sections:

| Section | Topic | Tests | Status |
| :--- | :--- | :--- | :--- |
| **A** | Domain validation (positions, configurations, state immutability) | `test_A01`, `test_A02`, `test_A03`, `test_A04`, `test_A05` | PASSED |
| **B** | Serialization & credential scrubbing | `test_B01`, `test_B02`, `test_B03`, `test_B04` | PASSED |
| **C** | Simulation clock (monotonicity, advance, set_time, listeners) | `test_C01`, `test_C02`, `test_C03`, `test_C04`, `test_C05` | PASSED |
| **D** | Simulation world & environment (entities, hazards, snapshots) | `test_D01`, `test_D02`, `test_D03`, `test_D04`, `test_D05` | PASSED |
| **E** | Twin registration & retrieval | `test_E01`, `test_E02`, `test_E03` | PASSED |
| **F** | Twin lifecycle & status transitions | `test_F01`, `test_F02`, `test_F03` | PASSED |
| **G** | Vision simulation (motion, person, anomaly, media capture) | `test_G01`, `test_G02`, `test_G03`, `test_G04` | PASSED |
| **H** | Glass simulation (HUD display, image/audio capture, position) | `test_H01`, `test_H02`, `test_H03`, `test_H04` | PASSED |
| **I** | Drone simulation (takeoff, hover, waypoints, RTB, battery failsafe) | `test_I01`, `test_I02`, `test_I03`, `test_I04`, `test_I05` | PASSED |
| **J** | Rover simulation (move, obstacle safety stop, dock, recharge) | `test_J01`, `test_J02`, `test_J03`, `test_J04` | PASSED |
| **K** | Telemetry (generation, observation conversion, history bounds) | `test_K01`, `test_K02`, `test_K03`, `test_K04`, `test_K05` | PASSED |
| **L** | Observation generation (canonical schema compliance) | `test_L01`, `test_L02` | PASSED |
| **M** | DeviceGateway integration & adapter equivalence | `test_M01`, `test_M02`, `test_M03` | PASSED |
| **N** | CentralInputGateway ingress integration | `test_N01` | PASSED |
| **O** | Fault injection (inject, auto-expiry, clearing) | `test_O01`, `test_O02`, `test_O03` | PASSED |
| **P** | OFFLINE fault handling | `test_P01` | PASSED |
| **Q** | LOW_BATTERY fault handling | `test_Q01` | PASSED |
| **R** | GPS_LOSS fault handling | `test_R01` | PASSED |
| **S** | COMMAND_FAILURE & COMMAND_TIMEOUT fault handling | `test_S01`, `test_S02` | PASSED |
| **T** | TELEMETRY_STALE fault handling | `test_T01` | PASSED |
| **U** | DUPLICATE_OBSERVATION fault handling | `test_U01` | PASSED |
| **V** | CONFLICTING_OBSERVATION fault handling | `test_V01` | PASSED |
| **W** | Scenario execution & assertion evaluation | `test_W01`, `test_W02` | PASSED |
| **X** | Scenario assertion operators | `test_X01` | PASSED |
| **Y** | Deterministic replay verification | `test_Y01` | PASSED |
| **Z** | Capacity bounds enforcement | `test_Z01`, `test_Z02` | PASSED |
| **AA** | Security & prohibited imports audit | `test_AA01` | PASSED |
| **AB** | WorldState boundary defense | `test_AB01` | PASSED |
| **AC** | PolicyEngine boundary defense | `test_AC01` | PASSED |
| **AD** | CognitiveRuntime / Reasoning boundary defense | `test_AD01` | PASSED |
| **AE** | Transport neutrality verification | `test_AE01` | PASSED |
| **AF** | Adapter substitutability verification | `test_AF01` | PASSED |

---

## 19. Production Deployment Considerations & Hardware Handover Roadmap

When physical hardware is introduced in future phases (e.g. Phase 7.0+):
1. **Drop-in Physical Adapters**: Implement physical adapters (`MavlinkDroneAdapter`, `Ros2RoverAdapter`, `WebRtcVisionAdapter`) satisfying `DeviceAdapterInterface`.
2. **Zero Central Brain Modifications**: Neither `CentralInputGateway`, `DeviceGateway`, `PolicyEngine`, nor `CognitiveRuntime` will require code changes.
3. **Hybrid Deployments**: Central can orchestrate a mixed fleet containing both physical hardware and digital twins simultaneously.
4. **Hardware-in-the-Loop (HIL) Testing**: Digital twins can act as shadow twins running in parallel with physical devices to detect anomalies or sensor calibration drift.

---

## 20. Architectural Conclusions & Phase 6.4 Readiness

ATLAS Phase 6.3 achieves complete digital twin and deterministic simulation capabilities for the entire ATLAS product ecosystem:
- **100% Deterministic**: Monotonic simulation clock controls all time progression.
- **100% Boundary Preservation**: Central Brain remains the singular cognitive, reasoning, and policy authority.
- **100% Adapter Equivalence**: Seamless drop-in replacement between digital twins and future physical adapters.
- **Comprehensive Verification**: 76 Phase 6.3 tests passing alongside 390 regression tests from Phase 4.x, 5.0, 6.1, and 6.2 (466 tests total).

ATLAS is fully prepared for Phase 6.4 (Multi-Modal Event Ingestion & Fusion Enhancements).
