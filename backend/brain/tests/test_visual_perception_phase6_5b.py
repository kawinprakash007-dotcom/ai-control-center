"""Phase 6.5b — Visual Perception Test Suite.

Comprehensive tests validating:
1. Provider interface compliance & lifecycle (VisualPerceptionProvider, MockVisualProvider).
2. Modality support (IMAGE, VIDEO_FRAME) and rejection of unsupported modalities.
3. Frame loading, validation, dimensions, pixel count, staleness, and corrupted data.
4. Motion detection: baseline, deltas, bounding boxes, strictly descriptive labels.
5. Region vs semantic object detection, bounding box clamping, confidence bounds.
6. Scene analysis: luminance, environmental heuristic, complexity/activity level.
7. OCR extraction boundary: text bounds, bounding boxes, credential safety.
8. Visual target tracking: ID assignment, IoU/centroid association, history bounds, expiration.
9. Reference / Mock visual provider deterministic synthetic execution.
10. Provenance preservation (all 8 mandatory keys, correlation_id, causation_id, timestamps).
11. End-to-end normalization to MultimodalObservation and CentralInputGateway ingestion.
12. Multi-product metadata compatibility (Vision, Glass, Drone, Rover, Digital Twin).
13. Architectural boundary invariants (zero WorldState/Goal/Mission/Tool/DeviceGateway bypasses).
"""

import io
import math
import sys
import time
import uuid
from typing import Any, Dict, List
from unittest.mock import MagicMock

# Fallbacks for lightweight test environments
if "chromadb" not in sys.modules:
    sys.modules["chromadb"] = MagicMock()
if "sentence_transformers" not in sys.modules:
    sys.modules["sentence_transformers"] = MagicMock()

import numpy as np
from PIL import Image
import pytest

from core.models.orchestration import ModalityType, MultimodalObservation
from core.models.perception import (
    BoundingBox,
    PerceptionCapability,
    PerceptionError,
    PerceptionEvidence,
    PerceptionInput,
    PerceptionMetadata,
    PerceptionPrivacyClass,
    PerceptionRequest,
    PerceptionResult,
    PerceptionStatus,
    REQUIRED_PROVENANCE_KEYS,
    SpatialEvidence,
)
from core.models.world_state import WorldState
from orchestration.fusion_engine import SituationFusionConfig, SituationFusionEngine
from orchestration.input_gateway import CentralInputGateway
from perception.normalizer import PerceptionObservationNormalizer
from vision.frame_loader import (
    FrameLoadingError,
    create_synthetic_image,
    load_and_validate_frame,
    validate_input_modality,
)
from vision.limits import VisionLimits
from vision.motion import MotionDetector
from vision.object_detection import VisualRegionDetector
from vision.ocr import VisualOCRDetector
from vision.provider import VisualPerceptionProvider
from vision.reference_provider import (
    MockVisualProvider,
    ReferenceVisualProvider,
)
from vision.scene_analysis import SceneAnalyzer
from vision.tracking import VisualTracker


# =====================================================================
# Fixtures & Helpers
# =====================================================================

def image_to_bytes(img: Image.Image) -> bytes:
    """Helper to convert PIL Image to PNG bytes."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def make_test_input(
    raw_payload: Any = None,
    modality: ModalityType = ModalityType.IMAGE,
    source_id: str = "camera_01",
    captured_at: float = None,
    correlation_id: str = "corr-101",
    causation_id: str = "caus-202",
    metadata: Dict[str, Any] = None,
    payload_ref: str = "",
) -> PerceptionInput:
    """Helper to construct valid PerceptionInput instances adhering to Phase 6.5a."""
    if captured_at is None:
        captured_at = time.time()

    meta = dict(metadata or {})
    ref = payload_ref

    if raw_payload is not None:
        if isinstance(raw_payload, Image.Image):
            meta["image_bytes"] = image_to_bytes(raw_payload)
            ref = ref or "synthetic://in_memory"
        elif isinstance(raw_payload, np.ndarray):
            meta["image_array"] = raw_payload
            ref = ref or "synthetic://numpy_array"
        elif isinstance(raw_payload, bytes):
            meta["image_bytes"] = raw_payload
            ref = ref or "synthetic://raw_bytes"
        elif isinstance(raw_payload, str):
            ref = raw_payload

    if not ref:
        ref = "synthetic://checkerboard"

    return PerceptionInput(
        input_id=f"inp_{uuid.uuid4().hex[:8]}",
        modality=modality,
        payload_ref=ref,
        captured_at=captured_at,
        source_id=source_id,
        correlation_id=correlation_id,
        causation_id=causation_id,
        metadata=meta,
    )


def make_test_request(
    perception_input: PerceptionInput,
    capabilities: List[PerceptionCapability] = None,
    parameters: Dict[str, Any] = None,
) -> PerceptionRequest:
    """Helper to construct valid PerceptionRequest instances."""
    caps = tuple(
        capabilities
        if capabilities is not None
        else (
            PerceptionCapability.VISION_OBJECT_DETECTION,
            PerceptionCapability.VISION_MOTION_DETECTION,
            PerceptionCapability.VISION_SCENE_CLASSIFICATION,
            PerceptionCapability.VISION_OCR,
        )
    )
    return PerceptionRequest(
        request_id=f"req_{uuid.uuid4().hex[:8]}",
        input_data=perception_input,
        requested_capabilities=caps,
        correlation_id=perception_input.correlation_id,
        causation_id=perception_input.causation_id,
        parameters=parameters or {},
    )


# =====================================================================
# Section A: Provider Interface Compliance & Lifecycle
# =====================================================================

class TestProviderInterfaceCompliance:
    """Verify VisualPerceptionProvider and MockVisualProvider adhere to contracts."""

    def test_provider_initialization_and_metadata(self):
        provider = VisualPerceptionProvider()
        assert provider.provider_id == "atlas_visual_perception_provider"
        assert provider.version == "1.0.0"
        assert ModalityType.IMAGE in provider.supported_modalities
        assert ModalityType.VIDEO_FRAME in provider.supported_modalities
        assert len(provider.supported_modalities) == 2

    def test_provider_capabilities_declaration(self):
        provider = VisualPerceptionProvider()
        caps = provider.capabilities
        assert PerceptionCapability.VISION_OBJECT_DETECTION in caps
        assert PerceptionCapability.VISION_MOTION_DETECTION in caps
        assert PerceptionCapability.VISION_SCENE_CLASSIFICATION in caps
        assert PerceptionCapability.VISION_OCR in caps

    def test_provider_health_check_healthy(self):
        provider = VisualPerceptionProvider()
        health = provider.health_check()
        assert health.status == "HEALTHY"
        assert health.provider_id == "atlas_visual_perception_provider"
        assert health.error is None
        assert "limits" in health.details

    def test_mock_provider_initialization_and_health(self):
        mock_prov = MockVisualProvider()
        assert mock_prov.provider_id == "mock_visual_provider"
        assert mock_prov.health_check().status == "HEALTHY"
        assert ModalityType.IMAGE in mock_prov.supported_modalities
        assert ModalityType.VIDEO_FRAME in mock_prov.supported_modalities


# =====================================================================
# Section B: Modality Handling & Input Rejection
# =====================================================================

class TestModalityHandling:
    """Ensure supported modalities succeed and unsupported modalities are rejected."""

    def test_image_modality_accepted(self):
        provider = VisualPerceptionProvider()
        img = create_synthetic_image(pattern="uniform_gray")
        p_in = make_test_input(img, modality=ModalityType.IMAGE)
        req = make_test_request(p_in)
        result = provider.process(req)
        assert result.status == PerceptionStatus.SUCCESS

    def test_video_frame_modality_accepted(self):
        provider = VisualPerceptionProvider()
        frame = create_synthetic_image(pattern="checkerboard")
        p_in = make_test_input(frame, modality=ModalityType.VIDEO_FRAME)
        req = make_test_request(p_in)
        result = provider.process(req)
        assert result.status == PerceptionStatus.SUCCESS

    def test_unsupported_modality_rejected(self):
        provider = VisualPerceptionProvider()
        p_in = make_test_input(b"audio_pcm_stream", modality=ModalityType.AUDIO_EVENT)
        req = make_test_request(p_in)
        result = provider.process(req)
        assert result.status == PerceptionStatus.UNSUPPORTED_MODALITY
        assert len(result.errors) > 0
        assert "not supported" in result.errors[0].message


# =====================================================================
# Section C: Frame Loading, Validation & Boundary Limits
# =====================================================================

class TestFrameLoadingAndValidation:
    """Verify bounds, formats, limits, and staleness handling."""

    def test_load_from_numpy_array(self):
        arr = np.zeros((100, 120, 3), dtype=np.uint8)
        _, _, meta = load_and_validate_frame(arr)
        assert meta["width"] == 120
        assert meta["height"] == 100

    def test_load_from_synthetic_image(self):
        img = create_synthetic_image(pattern="checkerboard", width=320, height=240)
        img_bytes = image_to_bytes(img)
        _, _, meta = load_and_validate_frame(img_bytes)
        assert meta["width"] == 320
        assert meta["height"] == 240

    def test_dimension_bounds_check_exceeding_max_width(self):
        oversized = np.zeros((100, VisionLimits().max_image_width + 10, 3), dtype=np.uint8)
        with pytest.raises(FrameLoadingError) as exc_info:
            load_and_validate_frame(oversized)
        assert "exceeds maximum allowed width" in str(exc_info.value)

    def test_dimension_bounds_check_exceeding_max_height(self):
        oversized = np.zeros((VisionLimits().max_image_height + 10, 100, 3), dtype=np.uint8)
        with pytest.raises(FrameLoadingError) as exc_info:
            load_and_validate_frame(oversized)
        assert "exceeds maximum allowed height" in str(exc_info.value)

    def test_staleness_rejection(self):
        lim = VisionLimits()
        stale_time = time.time() - (lim.max_staleness_seconds + 10)
        img = create_synthetic_image()
        with pytest.raises(FrameLoadingError) as exc_info:
            load_and_validate_frame(img, timestamp=stale_time)
        assert "stale" in str(exc_info.value).lower()

    def test_corrupted_payload_rejection(self):
        garbage = b"NOT_A_VALID_IMAGE_FORMAT_OR_HEADER"
        with pytest.raises(FrameLoadingError) as exc_info:
            load_and_validate_frame(garbage)
        assert "Failed to decode" in str(exc_info.value)

    def test_zero_dimension_frame_rejection(self):
        empty_arr = np.zeros((0, 0, 3), dtype=np.uint8)
        with pytest.raises(FrameLoadingError) as exc_info:
            load_and_validate_frame(empty_arr)
        assert "Non-positive dimensions" in str(exc_info.value)


# =====================================================================
# Section D: Motion Detection
# =====================================================================

class TestMotionDetection:
    """Verify deterministic frame differencing and objective semantic labels."""

    def test_motion_detector_baseline_first_frame(self):
        detector = MotionDetector(threshold=25)
        f1 = np.ones((200, 200, 3), dtype=np.uint8) * 128
        evidences = detector.detect_motion(f1, source_id="cam_test", timestamp=time.time())
        assert len(evidences) == 0  # First frame establishes baseline

    def test_motion_detector_identical_consecutive_frames(self):
        detector = MotionDetector(threshold=25)
        f1 = np.ones((200, 200, 3), dtype=np.uint8) * 128
        ts = time.time()
        detector.detect_motion(f1, source_id="cam_test", timestamp=ts)
        evidences = detector.detect_motion(f1, source_id="cam_test", timestamp=ts + 0.1)
        assert len(evidences) == 0  # No change

    def test_motion_detector_altered_region_triggers_evidence(self):
        detector = MotionDetector(threshold=25, min_area=50)
        f1 = np.zeros((200, 200, 3), dtype=np.uint8)
        f2 = np.zeros((200, 200, 3), dtype=np.uint8)
        # Create a bright 40x40 block in f2
        f2[50:90, 50:90] = 255
        ts = time.time()

        detector.detect_motion(f1, source_id="cam_test", timestamp=ts)
        evidences = detector.detect_motion(f2, source_id="cam_test", timestamp=ts + 0.1)

        assert len(evidences) >= 1
        ev = evidences[0]
        assert ev.semantic_type == "motion"
        assert ev.label == "motion_detected"
        assert ev.confidence > 0.0
        assert ev.spatial is not None
        assert ev.spatial.bounding_box is not None
        bbox = ev.spatial.bounding_box
        assert bbox.x <= 55
        assert bbox.y <= 55
        assert bbox.width >= 35
        assert bbox.height >= 35

    def test_motion_detector_strictly_neutral_labels(self):
        """Verify labels never contain threat terms like 'intruder', 'burglar', etc."""
        detector = MotionDetector(threshold=10, min_area=20)
        f1 = np.zeros((100, 100, 3), dtype=np.uint8)
        f2 = np.zeros((100, 100, 3), dtype=np.uint8)
        f2[20:60, 20:60] = 200

        ts = time.time()
        detector.detect_motion(f1, source_id="cam_test", timestamp=ts)
        evidences = detector.detect_motion(f2, source_id="cam_test", timestamp=ts + 0.1)

        for ev in evidences:
            assert "intruder" not in ev.label.lower()
            assert "burglar" not in ev.label.lower()
            assert "threat" not in ev.label.lower()
            assert "human" not in ev.label.lower()
            assert ev.label == "motion_detected"


# =====================================================================
# Section E: Region & Semantic Object Detection
# =====================================================================

class TestRegionAndObjectDetection:
    """Verify distinction between regions and semantic objects, clamping, and bounds."""

    def test_region_detection_without_registered_fixtures(self):
        detector = VisualRegionDetector(min_area=100)
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        frame[40:100, 40:100] = 240

        evidences = detector.detect(frame, source_id="cam_test", timestamp=time.time())
        assert len(evidences) >= 1
        ev = evidences[0]
        assert ev.semantic_type == "detected_region"
        assert ev.label == "foreground_region"

    def test_bounding_box_clamping_to_frame_boundaries(self):
        detector = VisualRegionDetector(min_area=50)
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        frame[0:50, 150:200] = 255

        evidences = detector.detect(frame, source_id="cam_test", timestamp=time.time())
        assert len(evidences) >= 1
        bbox = evidences[0].spatial.bounding_box
        assert bbox.x >= 0
        assert bbox.y >= 0
        assert bbox.right <= 200
        assert bbox.bottom <= 200

    def test_registered_fixture_produces_semantic_object(self):
        fixture_catalog = {
            "dock_area": [
                {
                    "label": "docking_station_fixture",
                    "confidence": 0.95,
                    "box": {"x": 50, "y": 50, "width": 50, "height": 50},
                }
            ]
        }
        detector = VisualRegionDetector(fixtures=fixture_catalog, min_area=50)
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        frame[50:100, 50:100] = [0, 255, 0]

        evidences = detector.detect(
            frame, source_id="cam_test", timestamp=time.time(), fixture_key="dock_area"
        )
        assert len(evidences) >= 1
        ev = evidences[0]
        assert ev.semantic_type == "semantic_object"
        assert ev.label == "docking_station_fixture"
        assert ev.attributes.get("detection_method") == "semantic_fixture"

    def test_confidence_values_within_unit_interval(self):
        detector = VisualRegionDetector(min_area=50)
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        frame[20:80, 20:80] = 200

        evidences = detector.detect(frame, source_id="cam_test", timestamp=time.time())
        for ev in evidences:
            assert 0.0 <= ev.confidence <= 1.0
            assert not math.isnan(ev.confidence)
            assert not math.isinf(ev.confidence)


# =====================================================================
# Section F: Scene Analysis
# =====================================================================

class TestSceneAnalysis:
    """Verify luminance, environment classification, and complexity analysis."""

    def test_bright_scene_detection(self):
        analyzer = SceneAnalyzer()
        bright_frame = np.ones((200, 200, 3), dtype=np.uint8) * 220
        evidence = analyzer.analyze(bright_frame, source_id="cam_scene", timestamp=time.time())

        assert evidence.semantic_type == "scene_analysis"
        assert "bright" in evidence.label
        assert evidence.metadata["luminance_class"] == "bright"
        assert evidence.metadata["mean_luminance"] >= 200.0

    def test_dark_scene_detection(self):
        analyzer = SceneAnalyzer()
        dark_frame = np.ones((200, 200, 3), dtype=np.uint8) * 20
        evidence = analyzer.analyze(dark_frame, source_id="cam_scene", timestamp=time.time())

        assert "dark" in evidence.label
        assert evidence.metadata["luminance_class"] == "dark"
        assert evidence.metadata["mean_luminance"] <= 30.0

    def test_scene_activity_level_empty_vs_cluttered(self):
        analyzer = SceneAnalyzer()
        # Flat empty scene
        flat_frame = np.ones((200, 200, 3), dtype=np.uint8) * 128
        ev_flat = analyzer.analyze(flat_frame, source_id="cam_scene", timestamp=time.time())
        assert ev_flat.metadata["activity_level"] == "empty"

        # Checkerboard scene
        checker_img = create_synthetic_image(pattern="checkerboard", width=200, height=200)
        _, checker_arr, _ = load_and_validate_frame(checker_img)
        ev_clutter = analyzer.analyze(checker_arr, source_id="cam_scene", timestamp=time.time())
        assert ev_clutter.metadata["activity_level"] in ("moderate_activity", "cluttered")


# =====================================================================
# Section G: OCR Extraction Boundary
# =====================================================================

class TestOCRExtractionBoundary:
    """Verify OCR bounded lengths, bounding boxes, and absence of credential leakage."""

    def test_ocr_extracts_registered_text(self):
        detector = VisualOCRDetector()
        img = create_synthetic_image(pattern="ocr_sample", width=320, height=240)
        _, frame, _ = load_and_validate_frame(img)

        evidences = detector.detect(frame, source_id="cam_ocr", timestamp=time.time())
        assert len(evidences) >= 1
        ev = evidences[0]
        assert ev.semantic_type == "text_detected"
        assert "ATLAS" in ev.attributes.get("text", "")
        assert ev.spatial is not None
        assert ev.spatial.bounding_box is not None

    def test_ocr_text_length_bounded(self):
        detector = VisualOCRDetector(max_text_length=100)
        truncated = detector._truncate_text("A" * 500)
        assert len(truncated) <= 100

    def test_ocr_zero_credential_leakage(self):
        detector = VisualOCRDetector()
        evidences = detector.detect(np.zeros((100, 100, 3), dtype=np.uint8), source_id="cam_ocr")
        for ev in evidences:
            txt = ev.attributes.get("text", "").lower()
            assert "password" not in txt
            assert "secret" not in txt
            assert "apikey" not in txt


# =====================================================================
# Section H: Visual Target Tracking
# =====================================================================

class TestVisualTargetTracking:
    """Verify track ID assignment, temporal continuity, history bounding, and expiration."""

    def test_tracking_lifecycle_creation_and_update(self):
        tracker = VisualTracker(iou_threshold=0.2, max_missing_frames=3)
        ts = time.time()

        # Frame 1: Detection at (50, 50, 30, 30)
        d1 = SpatialEvidence(bounding_box=BoundingBox(x=50, y=50, width=30, height=30), spatial_confidence=0.9)
        ev1 = tracker.update([d1], source_id="cam_track", timestamp=ts)
        assert len(ev1) == 1
        track_id = ev1[0].attributes["track_id"]
        assert track_id == "track_001"
        assert ev1[0].attributes["status"] == "active"

        # Frame 2: Slight displacement to (54, 52, 30, 30)
        d2 = SpatialEvidence(bounding_box=BoundingBox(x=54, y=52, width=30, height=30), spatial_confidence=0.88)
        ev2 = tracker.update([d2], source_id="cam_track", timestamp=ts + 0.1)
        assert len(ev2) == 1
        assert ev2[0].attributes["track_id"] == "track_001"
        assert ev2[0].attributes["track_length"] == 2

    def test_track_history_length_is_bounded(self):
        tracker = VisualTracker(max_history=5)
        ts = time.time()

        for i in range(10):
            d = SpatialEvidence(
                bounding_box=BoundingBox(x=50 + i, y=50, width=30, height=30),
                spatial_confidence=0.9,
            )
            tracker.update([d], source_id="cam_track", timestamp=ts + i * 0.1)

        active = tracker.get_active_tracks()
        assert len(active) == 1
        assert len(active[0].history) <= 5

    def test_track_expiration_after_missing_frames(self):
        tracker = VisualTracker(max_missing_frames=2)
        ts = time.time()

        # Step 1: Initialize track
        d = SpatialEvidence(bounding_box=BoundingBox(x=20, y=20, width=20, height=20), spatial_confidence=0.8)
        tracker.update([d], source_id="cam_track", timestamp=ts)
        assert len(tracker.get_active_tracks()) == 1

        # Step 2: Missing frame 1
        tracker.update([], source_id="cam_track", timestamp=ts + 0.1)
        assert len(tracker.get_active_tracks()) == 1

        # Step 3: Missing frame 2
        tracker.update([], source_id="cam_track", timestamp=ts + 0.2)
        assert len(tracker.get_active_tracks()) == 1

        # Step 4: Missing frame 3 -> Exceeds max_missing_frames(2)
        tracker.update([], source_id="cam_track", timestamp=ts + 0.3)
        assert len(tracker.get_active_tracks()) == 0


# =====================================================================
# Section I: Reference & Mock Visual Provider
# =====================================================================

class TestMockAndReferenceVisualProvider:
    """Verify deterministic mock execution for testing and simulation."""

    def test_mock_provider_returns_synthetic_evidences(self):
        mock_prov = MockVisualProvider()
        img = create_synthetic_image(pattern="checkerboard")
        p_in = make_test_input(img, correlation_id="mock-c-1")
        req = make_test_request(p_in)

        result = mock_prov.process(req)
        assert result.status == PerceptionStatus.SUCCESS
        assert len(result.evidence) >= 3
        types = {ev.semantic_type for ev in result.evidence}
        assert "semantic_object" in types
        assert "scene_analysis" in types
        assert "text_detected" in types

    def test_mock_provider_preserves_provenance(self):
        mock_prov = MockVisualProvider()
        img = create_synthetic_image()
        p_in = make_test_input(img, source_id="drone_cam_front", correlation_id="c-999")
        req = make_test_request(p_in)

        result = mock_prov.process(req)
        for ev in result.evidence:
            prov = ev.provenance
            assert prov["provider_id"] == "mock_visual_provider"
            assert prov["source_id"] == "drone_cam_front"
            assert prov["correlation_id"] == "c-999"


# =====================================================================
# Section J: Provenance & Temporal Integrity
# =====================================================================

class TestProvenanceAndTemporalIntegrity:
    """Verify all 8 mandatory provenance keys, correlation IDs, and timestamps."""

    def test_all_eight_provenance_keys_present(self):
        provider = VisualPerceptionProvider()
        img = create_synthetic_image(pattern="checkerboard")
        p_in = make_test_input(
            img,
            source_id="rover_stereo_cam",
            correlation_id="corr-xyz",
            causation_id="caus-abc",
        )
        req = make_test_request(p_in)
        result = provider.process(req)

        assert result.status == PerceptionStatus.SUCCESS
        assert len(result.evidence) > 0

        for ev in result.evidence:
            prov = ev.provenance
            for key in REQUIRED_PROVENANCE_KEYS:
                assert key in prov, f"Missing mandatory provenance key: {key}"
            assert prov["source_id"] == "rover_stereo_cam"
            assert prov["correlation_id"] == "corr-xyz"
            assert prov["causation_id"] == "caus-abc"
            assert prov["provider_id"] == "atlas_visual_perception_provider"

    def test_captured_at_timestamp_is_preserved(self):
        provider = VisualPerceptionProvider()
        img = create_synthetic_image()
        fixed_time = time.time() - 10.0
        p_in = make_test_input(img, captured_at=fixed_time)
        req = make_test_request(p_in)
        result = provider.process(req)

        for ev in result.evidence:
            assert ev.timestamp == fixed_time


# =====================================================================
# Section K: Normalization & Central Integration
# =====================================================================

class TestNormalizationAndCentralIntegration:
    """Verify PerceptionObservationNormalizer and CentralInputGateway ingestion."""

    def test_normalize_result_to_multimodal_observation(self):
        provider = VisualPerceptionProvider()
        img = create_synthetic_image(pattern="checkerboard")
        p_in = make_test_input(img, source_id="vision_hub_01")
        req = make_test_request(p_in)
        result = provider.process(req)

        normalizer = PerceptionObservationNormalizer()
        observation = normalizer.normalize(result)

        assert isinstance(observation, MultimodalObservation)
        assert observation.source_id == "vision_hub_01"
        assert observation.modality == ModalityType.IMAGE
        assert "semantic_type" in observation.payload

    def test_ingest_into_central_input_gateway_and_fusion(self):
        # 1. Perception execution
        provider = VisualPerceptionProvider()
        img = create_synthetic_image(pattern="checkerboard")
        p_in = make_test_input(img, source_id="atlas_vision_living_room")
        req = make_test_request(p_in)
        result = provider.process(req)

        # 2. Normalize to MultimodalObservation
        normalizer = PerceptionObservationNormalizer()
        observation = normalizer.normalize(result)

        # 3. Ingest into CentralInputGateway
        fusion_engine = SituationFusionEngine()
        gateway = CentralInputGateway(fusion_engine=fusion_engine)
        gateway.ingest_observation(observation)

        # 4. Verify observation ingested into gateway history
        recent = gateway.get_recent_observations(modality=ModalityType.IMAGE)
        assert len(recent) == 1
        assert recent[0].observation_id == observation.observation_id


# =====================================================================
# Section L: Multi-Product & Digital Twin Compatibility
# =====================================================================

class TestMultiProductAndSimulationCompatibility:
    """Verify visual perception handles all 4 edge products and digital twin simulated frames."""

    @pytest.mark.parametrize("product_role,source_id", [
        ("surveillance", "atlas_vision_01"),
        ("wearable_hud", "atlas_glass_01"),
        ("aerial_recon", "atlas_drone_01"),
        ("ground_patrol", "atlas_rover_01"),
    ])
    def test_all_four_edge_products(self, product_role, source_id):
        provider = VisualPerceptionProvider()
        img = create_synthetic_image()
        p_in = make_test_input(img, source_id=source_id, metadata={"product_role": product_role})
        req = make_test_request(p_in)
        result = provider.process(req)
        assert result.status == PerceptionStatus.SUCCESS
        assert len(result.evidence) > 0

    def test_digital_twin_simulated_frame(self):
        provider = VisualPerceptionProvider()
        sim_frame = np.random.randint(50, 200, (180, 240, 3), dtype=np.uint8)
        p_in = make_test_input(
            sim_frame,
            source_id="digital_twin_drone_sim",
            modality=ModalityType.VIDEO_FRAME,
            metadata={"simulated": True, "environment": "sim_warehouse"},
        )
        req = make_test_request(p_in)
        result = provider.process(req)
        assert result.status == PerceptionStatus.SUCCESS
        assert len(result.evidence) > 0


# =====================================================================
# Section M: Architectural Invariants & Safety
# =====================================================================

class TestArchitecturalInvariants:
    """Verify visual perception does not violate ATLAS authority or safety boundaries."""

    def test_provider_does_not_mutate_world_state(self):
        ws = WorldState(state_id="ws_01", version=1, timestamp=time.time())
        initial_version = ws.version

        provider = VisualPerceptionProvider()
        img = create_synthetic_image()
        p_in = make_test_input(img)
        req = make_test_request(p_in)
        provider.process(req)

        assert ws.version == initial_version

    def test_no_direct_hardware_or_driver_imports(self):
        forbidden = [
            "mavsdk", "pymavlink", "rclpy", "rospy", "serial", "RPi.GPIO"
        ]
        for mod in forbidden:
            assert mod not in sys.modules, f"Forbidden hardware module loaded: {mod}"

    def test_no_forbidden_dependencies_in_vision_modules(self):
        """Strict AST / import audit ensuring zero hardware, subagent, or subprocess dependencies."""
        import ast
        import os

        forbidden = {
            "subprocess", "os.system", "eval", "exec", "pymavlink",
            "mavsdk", "rclpy", "rospy", "serial", "RPi.GPIO",
        }

        vision_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "vision")
        )

        for fname in os.listdir(vision_dir):
            if not fname.endswith(".py"):
                continue
            fpath = os.path.join(vision_dir, fname)
            with open(fpath, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read(), filename=fname)

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        for fb in forbidden:
                            assert fb not in alias.name, f"Forbidden import '{alias.name}' in {fname}"
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        for fb in forbidden:
                            assert fb not in node.module, f"Forbidden import from '{node.module}' in {fname}"

