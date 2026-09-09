# ATLAS Phase 6.5d: Temporal & Cross-Modal Fusion Architecture

## 1. Executive Summary

ATLAS Phase 6.5d introduces the **Temporal & Cross-Modal Fusion Layer**, establishing a bounded, deterministic, and model-neutral semantic layer that correlates multimodal perception evidence across:

- **Time** (interval relations, windowing, ordering, sequence analysis)
- **Modality** (cross-modal compatibility, complementary corroboration)
- **Source** (source diversity scoring, independent verification)
- **Product** (Drone, Rover, Glass, Vision, Digital Twin edge fusion)
- **Spatial Proximity** (geodesic distance, bearing, relative position)
- **Entity Reference** (shared candidate target tracking)
- **Correlation ID & Causation ID** (traceable causal provenance)

In strict conformance with ATLAS architectural invariants:
> **The Temporal & Cross-Modal Fusion Layer organizes pre-situation evidence. It answers: *"When are observations related?"* and *"Which observations probably describe the same underlying event or entity?"* It NEVER answers: *"What should ATLAS do?"***
>
> **Phase 6.5d strictly produces descriptive, clustered evidence (`FusionResult`). It NEVER creates Situations, NEVER creates Missions, NEVER creates Goals, NEVER mutates Central World State, and NEVER actuates devices or tools.**

---

## 2. Authoritative Architectural Placement & Chain of Custody

```
                                  Edge Devices & Sensors
                   (Drone, Rover, Glass, Vision, Simulation Twin)
                                             │
                                             ▼
                             Perception Provider Registry
                                (6.5a / 6.5b / 6.5c)
                                             │
                                             ▼
                               Multimodal Perception Streams
                        (Images, Audio, GPS, Telemetry, State)
                                             │
                                             ▼
                      ┌─────────────────────────────────────────────┐
                      │    ATLAS Phase 6.5d: Temporal &             │
                      │    Cross-Modal Fusion Engine                │
                      │                                             │
                      │  ┌─────────────────┐   ┌─────────────────┐  │
                      │  │ Temporal Subsys │   │ Spatial Subsys  │  │
                      │  │ (Windowing/Seq) │   │ (6.5c Geodesy)  │  │
                      │  └────────┬────────┘   └────────┬────────┘  │
                      │           └──────────┬──────────┘           │
                      │                      ▼                      │
                      │  ┌───────────────────────────────────────┐  │
                      │  │ Compatibility & Diversity Subsystem   │  │
                      │  │ (Compatibility, Source Diversity,     │  │
                      │  │  Contradiction Detection)             │  │
                      │  └───────────────────┬───────────────────┘  │
                      │                      ▼                      │
                      │  ┌───────────────────────────────────────┐  │
                      │  │ Bounded Fusion Clustering Engine      │  │
                      │  │ (Deterministic IDs, Evidence Links)   │  │
                      │  └───────────────────────────────────────┘  │
                      └──────────────────────┬──────────────────────┘
                                             │ FusionResult (Pre-Situation Evidence)
                                             ▼
                                    CentralInputGateway
                                             │
                                             ▼
                                   SituationFusionEngine
                             (Situation Creation & Resolution)
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

## 3. Boundary Distinction: Phase 6.5d vs. SituationFusionEngine

Phase 6.5d and downstream `SituationFusionEngine` have distinctly separated responsibilities:

| Property | Phase 6.5d (Temporal & Cross-Modal Fusion) | SituationFusionEngine (Phase 5.0) |
| :--- | :--- | :--- |
| **Architectural Role** | Pre-situation evidence organization & correlation | Situational awareness & world-state lifecycle |
| **Output Type** | `FusionResult` (containing `FusionCluster`, `EvidenceRelationship`) | `Situation` (containing `SituationStatus`, `SituationSeverity`) |
| **WorldState Interaction** | **Zero mutation**. Purely read-only / stateless | Evaluates conflicts against WorldState, emits StateConflicts |
| **Contradiction Handling** | Flags pairs as `EvidenceRelationType.CONTRADICTS` | Resolves conflicting assertions via `DeterministicConflictResolver` |
| **Temporal Focus** | Pairwise temporal calculus (BEFORE, AFTER, OVERLAPS, SEQUENCE) | Temporal validity windows & situation expiration |
| **Clustering Basis** | Graph connected components across spatiotemporal links | Semantic clustering across multi-observation signatures |

---

## 4. Domain Models & Enumerations

Located in `backend/core/models/multimodal_fusion.py`:

### 4.1 Enumerations
- **`TemporalRelation`**: `BEFORE`, `AFTER`, `OVERLAPS`, `COINCIDENT`, `WITHIN_WINDOW`, `SEQUENCE`.
- **`EvidenceRelationType`**: `SUPPORTS`, `CORROBORATES`, `CONTRADICTS`, `PRECEDES`, `FOLLOWS`, `OVERLAPS`, `COLOCATED_WITH`, `SAME_ENTITY_CANDIDATE`, `DUPLICATES`.
- **`ModalityCompatibilityLevel`**: `COMPLEMENTARY`, `COMPATIBLE`, `INDEPENDENT`, `CONFLICTING`.
- **`FusionFreshnessStatus`**: `VALID`, `STALE`, `EXPIRED`.

### 4.2 Core Models
- **`TemporalWindow`**: Immutable interval abstraction strictly validating `start_time <= end_time`, non-negative bounds, and positive duration bounded by `max_duration`. Decoupled from wall-clock time.
- **`SpatialEvidenceLink`**: Immutable record of geodesic distance, bearing, relative position, and confidence between two spatial observations (reusing Phase 6.5c geodesy).
- **`EvidenceRelationship`**: Pairwise typed relationship (`relation_type`, `source_evidence_id`, `target_evidence_id`, `confidence`, `reason`, `temporal_relation`, `spatial_link`).
- **`CrossModalCorrelation`**: Aggregate correlation summary over a set of evidence items and modalities.
- **`FusionCluster`**: Bounded cluster of connected observations with deterministic ID (`fcluster_<sha256(ev_ids|src_ids|start|end)[:16]>`), temporal bounding window, and source diversity score.
- **`FusionLimits`**: Strict capacity bounds (`max_input_observations`, `max_evidence_relationships`, `max_clusters`, `max_duplicate_cache`, etc.).
- **`FusionResult`**: Immutable aggregate result with deterministic SHA-256 semantic content hash and dual serialization (`to_dict`/`from_dict`, `model_dump`/`model_validate`/`model_dump_json`).

---

## 5. Subsystem Architecture

### 5.1 Temporal Alignment Subsystem (`multimodal_fusion.temporal`)
- Evaluates temporal intervals or point timestamps with configurable coincidence and sequence tolerances.
- Dynamically constructs bounded `TemporalWindow` instances with span clipping to prevent runaway time intervals.
- Provides strict interval filtering without expanding windows.

### 5.2 Spatial Correlation Subsystem (`multimodal_fusion.spatial`)
- Reuses Phase 6.5c Haversine spherical geodesy (`spatial_telemetry.spatial.calculate_distance`), initial forward bearing (`calculate_bearing`), and relative positioning (`evaluate_relative_position`).
- Evaluates colocation and proximity without duplicating geodesy math.

### 5.3 Modality Compatibility & Contradiction Subsystem (`multimodal_fusion.compatibility`)
- Implements semantic compatibility rules:
  - Complementary: Visual + Acoustic, Visual + Spatial, Visual + Telemetry, GPS + Telemetry, Acoustic + Linguistic.
- Calculates source diversity: $\frac{\text{unique\_sources} - 1}{\text{total\_observations} - 1}$ for multi-item sets, ensuring multiple frames from a single camera do not overwhelm cross-product corroboration.
- Identifies direct negations, status contradictions (e.g., `clear` vs `blocked`, `normal` vs `critical`), boolean contradictions, and kinematic conflicts (e.g., `speed = 10.0` vs `speed = 0.0`).

### 5.4 Deduplication & Freshness Lifecycle (`multimodal_fusion.engine`)
- Maintains a bounded LRU duplicate cache (`max_duplicate_cache`).
- Ingested duplicates are flagged as `EvidenceRelationType.DUPLICATES` without crashing or reprocessing.
- Evaluates freshness against configurable thresholds (`freshness_threshold_seconds`, `expiration_threshold_seconds`).

### 5.5 Deterministic Fusion Engine (`multimodal_fusion.engine`)
- Sorts inputs deterministically by `(timestamp, source_id, id)`.
- Builds an adjacency graph across non-contradictory pairwise relationships.
- Partitions evidence into connected component clusters.
- Generates 100% deterministic cluster and result IDs using SHA-256 semantic hashing.
- Sanitizes all attributes and metadata to prevent credential leakage.

---

## 6. Multi-Product Edge Support

Phase 6.5d natively correlates evidence across all ATLAS product edges:
1. **Vision Edge**: Object detection bounding boxes, visual classification, and motion tracking.
2. **Glass Edge**: Head-mounted camera imagery, audio event detection, and speech transcription.
3. **Drone Edge**: Aerial imagery, GPS positioning, altitude AGL, and battery telemetry.
4. **Rover Edge**: Ground lidar/radar detection, wheel velocity, obstacle proximity, and terrain telemetry.
5. **Digital Twin Edge**: Simulated entity states, digital twin telemetry, and virtual spatial coordinates.

---

## 7. Security, Invariant & AST Audit Guarantees

Phase 6.5d satisfies the following architectural and security guarantees:
- **Zero Subprocess / Shell Execution**: No `subprocess`, `os.system`, `eval()`, `exec()`, or `shell=True`.
- **Zero Physical Hardware Drivers**: No `rospy`, `rclpy`, `pymavlink`, `serial`, or `RPi.GPIO`.
- **Model Neutrality**: No neural network weights, PyTorch/TensorFlow runtime, or hardcoded inference pipelines.
- **Zero Decision / Action Leakage**: No Goal, Mission, Task, or Tool generation.
- **Zero WorldState Mutation**: Central WorldState is never mutated.
- **Zero Credential Leakage**: Sensitive metadata (API keys, passwords, bearer tokens) are automatically redacted.
- **Deterministic Hashing**: Zero random UUIDs used in clusters or result IDs.
