# ATLAS Phase 6.5a: Multimodal Perception Contracts

## 1. Architectural Overview & Placement

ATLAS Phase 6.5a introduces the contract-first **Multimodal Perception Layer** into the ATLAS ecosystem.
It establishes formal, model-neutral, transport-neutral, immutable schemas and interfaces for multi-product edge and simulation perception.

### Authoritative Architecture Flow

```
+-----------------------------------------------------------------------------------------------+
|                                      PERCEPTION INGRESS                                       |
|  [ATLAS Vision]         [ATLAS Glass]          [ATLAS Drone]          [ATLAS Rover]           |
|  (RTSP/WebRTC)          (HUD/Egocentric)       (Down-Cam/Gimbal)      (LiDAR/Stereo)          |
+-----------------------------------------------------------------------------------------------+
                                               |
                                               v
+-----------------------------------------------------------------------------------------------+
|                       PHASE 6.5a: MULTIMODAL PERCEPTION CONTRACT LAYER                        |
|                                                                                               |
|  PerceptionRequest (Request ID, Modality, Requested Capabilities, Bounds, Privacy Class)      |
|                                              |                                                |
|                                              v                                                |
|  PerceptionProviderRegistry (Thread-safe Registration, Capability Match, Provider Selection)   |
|                                              |                                                |
|                                              v                                                |
|  PerceptionProviderInterface (Pure Python Provider: Mock, OCR, CV, Audio, Speech, Telemetry)   |
|                                              |                                                |
|                                              v                                                |
|  PerceptionResult (Status, Provenance, Spatial Evidence, Diagnostic Errors, Bounded Payload)  |
|                                              |                                                |
|                                              v                                                |
|  PerceptionObservationNormalizer (Deterministic Translation to Canonical Observations)        |
+-----------------------------------------------------------------------------------------------+
                                               |
                                               v
                                    CentralInputGateway
                                               |
                                               v
                                        SituationFusion
                                               |
                                               v
                                   WorldState / Event Autonomy
                                               |
                                               v
                              Multi-Product Situation Intelligence
                                               |
                                               v
                                      Mission Intelligence
                                               |
                                               v
                                     AutonomousGoalManager
                                               |
                                               v
                                        CognitiveRuntime
                                               |
                                               v
                                          PolicyEngine
                                               |
                                               v
                                        ToolOrchestrator
                                               |
                                               v
                                         DeviceGateway
                                               |
                                               v
                                      Products / Twins
```

---

## 2. Core Architectural Invariants

1. **Strict Contract-First Design**:
   - Zero real model dependencies (no PyTorch, YOLO, Whisper, OpenCV pipelines).
   - Zero physical hardware or driver libraries (no ROS2, MAVLink, GPIO, serial).
   - Pure Python, thread-safe, model-neutral contracts.
2. **No Second Brain / Ingress Authority**:
   - Perception providers extract semantic facts and structural evidence; they **NEVER** reason, plan missions, create goals, mutate WorldState, execute tools, or dispatch directly to DeviceGateway.
   - All perception outputs must pass through the `PerceptionObservationNormalizer` and enter Central through `CentralInputGateway.ingest_observation()`.
3. **Data Immutability & Strict Bounds**:
   - All contract entities (`PerceptionInput`, `PerceptionEvidence`, `PerceptionRequest`, `PerceptionResult`, `PerceptionMetadata`, `SpatialEvidence`, `PerceptionError`) are frozen dataclasses.
   - Confidence is strictly finite in $[0.0, 1.0]$ (rejecting NaN and $\pm\infty$).
   - Strict bounded collections: `max_evidence_per_result = 100`, `max_attributes = 50`, `max_provider_count = 50`.
4. **Credential & Secret Protection**:
   - All metadata serialization strictly executes `sanitize_contract_metadata()`. Keys such as `api_key`, `auth_token`, `password`, `secret` are automatically masked to `[REDACTED]`.
5. **Full Backward Compatibility**:
   - Preserves 100% compatibility with Phase 4.4 visual UI grounding models (`VisualElement`, `BoundingBox`, `VisualScene`, `ElementType`, `SpatialRelation`, `GroundingRequest`, `GroundingResult`).
   - `PerceptionSource` maintains dual identity: structured Phase 6.5a provenance dataclass while supporting legacy class-level attributes (`PerceptionSource.OCR`, `PerceptionSource.CV`, `PerceptionSource.ACCESSIBILITY`, `PerceptionSource.MANUAL`).

---

## 3. Contract Models

### Enums
- **`PerceptionStatus`**: `SUCCESS`, `PARTIAL`, `NO_DETECTION`, `INVALID_INPUT`, `UNSUPPORTED`, `UNSUPPORTED_MODALITY`, `TIMEOUT`, `FAILED`, `UNKNOWN`.
- **`PerceptionSourceType`**: `DEVICE`, `SIMULATION`, `LOCAL_MODEL`, `REMOTE_MODEL`, `RULE_BASED`, `HUMAN`, `SYSTEM`, `SENSOR`, `UNKNOWN`.
- **`PerceptionPrivacyClass`**: `PUBLIC`, `INTERNAL`, `CONFIDENTIAL`, `SENSITIVE`, `RESTRICTED`.
- **`PerceptionCapability`**: `VISION_OBJECT_DETECTION`, `VISION_MOTION_DETECTION`, `VISION_SCENE_CLASSIFICATION`, `VISION_OCR`, `AUDIO_EVENT_CLASSIFICATION`, `SPEECH_TRANSCRIPTION`, `SPATIAL_LOCALIZATION`, `TELEMETRY_NORMALIZATION`, `MULTIMODAL_CORRELATION`.

### Core Dataclasses
- **`SpatialEvidence`**: Reuses existing `BoundingBox` and `GeoLocation` models without duplicating geometry primitives; captures `point`, `region_label`, `relative_position`, and `spatial_confidence`.
- **`PerceptionMetadata`**: Captures provider identification, model versioning, execution timing, privacy classification, and provenance.
- **`PerceptionInput`**: Transport-neutral reference to raw input modality buffers (`payload_ref`), timestamp, source ID, correlation ID, and causation ID.
- **`PerceptionEvidence`**: Fine-grained semantic entity detected by a provider. Carries mandatory provenance keys:
  `source_id`, `provider_id`, `provider_version`, `input_id`, `request_id`, `observation_timestamp`, `correlation_id`, `causation_id`.
- **`PerceptionRequest`**: Formal query dispatched to a provider specifying requested capabilities, input data reference, confidence thresholds, and deadlines.
- **`PerceptionResult`**: Envelope encapsulating execution status, evidence collection, processing metadata, and diagnostic errors.

---

## 4. Components

### `PerceptionProviderRegistry`
- **Thread Safety**: Backed by `threading.RLock()`.
- **Capacity Limits**: Enforces `max_provider_count = 50`.
- **Duplicate Prevention**: Rejects duplicate `provider_id` registrations with `ValueError`.
- **Discovery**: Exposes `list_capabilities()`, `find_providers_for_capability()`.
- **Deterministic Selection**: `select_provider(capability, modality)` filters available providers matching criteria and returns the provider sorted deterministically by `provider_id`.

### `PerceptionObservationNormalizer`
- Bridges perception results to canonical `MultimodalObservation` instances ready for `CentralInputGateway`.
- **Deterministic Mapping**:
  - `observation_id = f"obs_{result.request_id}_{evidence.evidence_id}"`
  - `source_id = evidence.source_id`
  - `source_type = "perception_provider"`
  - `modality = evidence.modality`
  - `timestamp = evidence.timestamp`
  - `confidence = evidence.confidence`
  - `location = evidence.spatial.location`
  - `payload = {"semantic_type": evidence.semantic_type, "label": evidence.label, "attributes": evidence.attributes, "spatial": ...}`
  - `metadata = {"provider_id": ..., "privacy_class": ..., "provenance": ...}`

### `MockPerceptionProvider`
- Dual-contract provider implementing both Phase 6.5a `PerceptionProviderInterface` and Phase 4.4 `VisualPerceptionProvider`.
- Supports configurable capabilities, modalities, simulated errors, unsupported modalities, and deterministic synthetic evidence.

---

## 5. Verification & Test Matrix

- **Unit & Contract Tests**: `backend/brain/tests/test_perception_contracts_phase6_5a.py` (46 tests passing).
- **Mission Intelligence (Phase 6.4)**: `backend/brain/tests/test_mission_intelligence_phase6_4.py` (51 tests passing).
- **Simulation & Twin (Phase 6.3)**: `backend/brain/tests/test_simulation_phase6_3.py` (76 tests passing).
- **Edge Product Contracts (Phase 6.2)**: `backend/brain/tests/test_device_contract_phase6_2.py` (70 tests passing).
- **Production Runtime (Phase 6.1)**: `backend/brain/tests/test_production_runtime.py` (50 tests passing).
- **Central Orchestration & Gateways (Phase 5.0)**: 202 tests passing.
- **Visual Perception & Core Autonomy (Phase 4.x)**: 128 tests passing.
- **Total Test Suite**: 623+ tests passing with 0 regressions.
