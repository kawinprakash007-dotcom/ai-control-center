# ATLAS Phase 6.5e: Multimodal Digital-Twin Scenarios Architecture

## 1. Executive Summary

ATLAS Phase 6.5e introduces the **Multimodal Digital-Twin Scenario Framework**, establishing a deterministic, bounded simulation scenario runner and catalog that exercises the complete ATLAS ecosystem using Digital Twins (Vision, Glass, Drone, Rover) and synthetic multimodal perception inputs.

In strict conformance with ATLAS architectural invariants:
> **The Scenario Framework is an observer and simulation controller. It coordinates simulation time, virtual environment, and digital twins. It NEVER acts as an autonomous brain, reasoning engine, mission engine, or goal engine.**
>
> **Phase 6.5e strictly observes and coordinates simulation. It NEVER mutates central WorldState directly, NEVER mutates GoalStore, NEVER bypasses Mission coordinators, NEVER executes tools or bypasses DeviceGateway/PolicyEngine, and NEVER imports physical hardware drivers.**

---

## 2. Authoritative Architectural Placement & Chain of Custody

```
                       ┌──────────────────────────────────────────────┐
                       │    ATLAS Phase 6.5e: Scenario Framework      │
                       │                                              │
                       │  ┌────────────────────┐   ┌───────────────┐  │
                       │  │ ScenarioCatalog    │   │ ScenarioRunner│  │
                       │  │ (10 Canonicals)    │   │ (Step & Trace)│  │
                       │  └─────────┬──────────┘   └───────┬───────┘  │
                       │            └───────────┬──────────┘          │
                       │                        ▼                     │
                       │  ┌────────────────────────────────────────┐  │
                       │  │ SimulationWorld & SimulationClock      │  │
                       │  │ (Bounded Environment, Entities, Faults)│  │
                       │  └─────────────────────┬──────────────────┘  │
                       └────────────────────────┼─────────────────────┘
                                                │
                 ┌──────────────────────────────┼──────────────────────────────┐
                 ▼                              ▼                              ▼
        ┌─────────────────┐            ┌─────────────────┐            ┌─────────────────┐
        │ VisionDigitalTwin│           │ GlassDigitalTwin│            │ DroneDigitalTwin│
        │ (Camera / Vid)  │            │ (HUD / GPS / LKL)│           │ (Flight / Nav)  │
        └────────┬────────┘            └────────┬────────┘            └────────┬────────┘
                 │                              │                              │
                 └──────────────────────┬───────┴──────────────────────────────┘
                                        ▼ Synthetic Multimodal Observations
                       ┌─────────────────────────────────┐
                       │ TemporalCrossModalFusionEngine   │ (Phase 6.5d Fusion)
                       └────────────────┬────────────────┘
                                        │
                                        ▼ Ingress Envelope (now = simulation_time)
                       ┌─────────────────────────────────┐
                       │ CentralInputGateway             │ (Phase 5.0 Input)
                       └────────────────┬────────────────┘
                                        │
                                        ▼ Observation Batch
                       ┌─────────────────────────────────┐
                       │ SituationFusionEngine           │ (Phase 5.0 Situation Creation)
                       └────────────────┬────────────────┘
                                        │
                                        ▼ Multi-Product Situation Intelligence
                       ┌─────────────────────────────────┐
                       │ AutonomousGoalManager / Planner │ (Phase 6.4 Coordination)
                       └────────────────┬────────────────┘
                                        │
                                        ▼ Authorized Tool Dispatches
                       ┌─────────────────────────────────┐
                       │ DeviceGateway / Adapters        │ (Transparent Device Abstraction)
                       └────────────────┬────────────────┘
                                        │
                                        ▼ Device Command
                       ┌─────────────────────────────────┐
                       │ DigitalTwinAdapter -> Twin      │ (100% Contract Equivalent)
                       └─────────────────────────────────┘
```

---

## 3. Boundary Distinctions & Strict Invariants

| Layer | Responsibility | Prohibitions |
| :--- | :--- | :--- |
| **ScenarioRunner** | Advances `SimulationClock`, ticks `SimulationWorld`, injects step observations and faults, evaluates assertions, computes deterministic result hash. | NEVER writes to production `WorldState`, NEVER mutates `GoalStore`, NEVER uses `time.sleep()`, NEVER spawns subprocesses. |
| **ScenarioCatalog** | Thread-safe bounded registry maintaining canonical multi-product scenarios. | Rejects duplicate IDs, enforces strict max scenario limits, zero credentials/secrets. |
| **DigitalTwinAdapter** | Maps digital twin commands to `DeviceGateway` and vice versa with 100% physical protocol equivalence. | Operates purely as a standard adapter implementing `DeviceAdapterInterface`. |
| **Perception / Fusion** | Analyzes synthetic observations across time, space, and modality (`TemporalCrossModalFusionEngine`). | NEVER creates Situations or Missions; produces purely descriptive evidence clusters. |
| **Central Nervous System** | Evaluates situations, decomposes missions, assigns goals, enforces policies, and coordinates actions. | Production authorities remain single sources of truth; zero bypasses allowed. |

---

## 4. The 10 Canonical Scenarios

The framework implements 10 canonical scenarios covering key multi-product workflows:

1. **`SCENARIO_01_HOME_INTRUSION`**: Perimeter motion detection via Vision, wearer Glass HUD notification, Rover ground dispatch, and Drone aerial tracking.
2. **`SCENARIO_02_LOST_PERSON`**: Multi-product search correlating Glass last known location (LKL), Drone aerial sighting, and Rover thermal corroboration.
3. **`SCENARIO_03_DRONE_FAILURE`**: Injected flight command failure on Drone, health degradation observation, and seamless Rover ground alternate verification.
4. **`SCENARIO_04_CONFLICTING_SENSOR_REPORTS`**: Opposing reports from Vision and Rover (e.g. motion vs clear path), exercising contradiction detection without system crash.
5. **`SCENARIO_05_STALE_TELEMETRY`**: Time advancement beyond freshness threshold validating stale telemetry detection and degraded health status.
6. **`SCENARIO_06_DEVICE_OFFLINE`**: Injected offline communication fault on Drone and verification of disconnected twin state and offline error responses.
7. **`SCENARIO_07_MOVING_TARGET`**: Dynamic spatial updates tracking a simulated moving vehicle across multiple sensor vantage points.
8. **`SCENARIO_08_MULTI_SOURCE_CORROBORATION`**: Independent multi-modality evidence from Vision, Glass, and Rover confirming an event with cross-modal confidence.
9. **`SCENARIO_09_MISSION_RESOLUTION`**: Multi-step waypoint navigation, sensor verification, and mission objective completion.
10. **`SCENARIO_10_FULL_ECOSYSTEM`**: Complete end-to-end multi-agent execution exercising Vision, Glass, Drone, Rover, spatial tracking, fusion, input gateway, and central orchestration.

---

## 5. Determinism & SHA-256 Replay Guarantee

All scenarios operate exclusively on `SimulationClock`:
- Zero reliance on wall-clock time (`time.sleep()`, `datetime.now()`, `time.time()`).
- Step ordering is deterministically normalized by `(time_offset, step_id)`.
- Replaying any scenario from the catalog produces identical step traces, identical assertion evaluations, and an identical SHA-256 hash digest.
