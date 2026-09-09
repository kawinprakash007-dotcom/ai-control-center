"""
Test Suite for ATLAS Phase 6.5a — Multimodal Perception Contracts

Verifies:
1. Enum completeness and string conversions (PerceptionStatus, PerceptionSourceType, PerceptionPrivacyClass, PerceptionCapability).
2. Domain model integrity, validation constraints, NaN/Inf rejection, and immutability (PerceptionLimits, PerceptionError, SpatialEvidence, PerceptionSource, PerceptionMetadata, PerceptionInput, PerceptionEvidence, PerceptionRequest, PerceptionResult).
3. Sanitization of sensitive keys (passwords, tokens, api_keys) in contract serialization.
4. Backward compatibility of PerceptionSource and VisualPerceptionProvider.
5. PerceptionProviderRegistry registration, deduplication, capability discovery, modality filtering, deterministic selection, capacity limits, and thread safety.
6. MockPerceptionProvider implementation of PerceptionProviderInterface and VisualPerceptionProvider, health reporting, unsupported modalities/capabilities, and simulated errors.
7. PerceptionObservationNormalizer translation of PerceptionResult and PerceptionEvidence into canonical MultimodalObservation instances.
8. End-to-end flow: PerceptionRequest -> Provider -> PerceptionResult -> Normalizer -> MultimodalObservation -> CentralInputGateway.
9. Architectural invariants: No direct execution/dispatch from perception layer, immutability, zero forbidden dependencies.
"""

import math
import sys
import threading
import time
from typing import Any, Dict, List, Set
from unittest.mock import MagicMock

# Ensure chromadb and sentence_transformers fallback for lightweight CI/test execution
if "chromadb" not in sys.modules:
    sys.modules["chromadb"] = MagicMock()
if "sentence_transformers" not in sys.modules:
    sys.modules["sentence_transformers"] = MagicMock()

import pytest

from core.interfaces.perception_interface import (
    PerceptionObservationNormalizerInterface,
    PerceptionProviderInterface,
    PerceptionProviderRegistryInterface,
    VisualPerceptionProvider,
)
from core.models.computer import ComputerObservation, ScreenDimensions
from core.models.orchestration import GeoLocation, ModalityType, MultimodalObservation
from core.models.perception import (
    BoundingBox,
    ElementType,
    PerceptionCapability,
    PerceptionError,
    PerceptionEvidence,
    PerceptionInput,
    PerceptionLimits,
    PerceptionMetadata,
    PerceptionPrivacyClass,
    PerceptionRequest,
    PerceptionResult,
    PerceptionSource,
    PerceptionSourceType,
    PerceptionStatus,
    REQUIRED_PROVENANCE_KEYS,
    SpatialEvidence,
    VisualElement,
)
from orchestration.input_gateway import CentralInputGateway
from perception.mock_provider import MockPerceptionProvider
from perception.normalizer import PerceptionObservationNormalizer
from perception.registry import PerceptionProviderRegistry


def make_valid_provenance(
    source_id: str = "src_01",
    provider_id: str = "prov_01",
    provider_version: str = "1.0.0",
    input_id: str = "in_01",
    request_id: str = "req_01",
    observation_timestamp: float = 100.0,
    correlation_id: str = "corr_01",
    causation_id: str = "cause_01",
    **extra: Any,
) -> Dict[str, Any]:
    prov = {
        "source_id": source_id,
        "provider_id": provider_id,
        "provider_version": provider_version,
        "input_id": input_id,
        "request_id": request_id,
        "observation_timestamp": observation_timestamp,
        "correlation_id": correlation_id,
        "causation_id": causation_id,
    }
    prov.update(extra)
    return prov


# =====================================================================
# 1. ENUM INTEGRITY & CONVERSIONS
# =====================================================================

class TestPerceptionEnums:
    """Validates perception enums, string conversions, and casing."""

    def test_perception_status_values_and_from_str(self):
        for status in PerceptionStatus:
            assert PerceptionStatus.from_str(status.value) == status
            assert PerceptionStatus.from_str(status.name) == status
        assert PerceptionStatus.from_str("UNKNOWN_STATUS") == PerceptionStatus.UNKNOWN

    def test_perception_source_type_values_and_from_str(self):
        for st in PerceptionSourceType:
            assert PerceptionSourceType.from_str(st.value) == st
            assert PerceptionSourceType.from_str(st.name) == st
        assert PerceptionSourceType.from_str("nonexistent") == PerceptionSourceType.UNKNOWN

    def test_perception_privacy_class_values_and_from_str(self):
        for pc in PerceptionPrivacyClass:
            assert PerceptionPrivacyClass.from_str(pc.value) == pc
            assert PerceptionPrivacyClass.from_str(pc.name) == pc
        assert PerceptionPrivacyClass.from_str("nonexistent") == PerceptionPrivacyClass.CONFIDENTIAL

    def test_perception_capability_values_and_from_str(self):
        for cap in PerceptionCapability:
            assert PerceptionCapability.from_str(cap.value) == cap
            assert PerceptionCapability.from_str(cap.name) == cap
        with pytest.raises(ValueError, match="Unknown PerceptionCapability"):
            PerceptionCapability.from_str("nonexistent")


# =====================================================================
# 2. DOMAIN MODEL INTEGRITY, VALIDATION & BOUNDS
# =====================================================================

class TestPerceptionModels:
    """Validates dataclass construction, validation bounds, and immutability."""

    def test_perception_limits_constants(self):
        limits = PerceptionLimits()
        assert limits.max_evidence_per_result == 100
        assert limits.max_attributes == 50
        assert limits.max_provider_count == 50
        assert limits.max_payload_ref_length == 256

    def test_perception_error_construction_and_serialization(self):
        err = PerceptionError(
            code="TIMEOUT",
            message="Perception inference timed out",
            provider_id="prov_01",
            request_id="req_01",
            input_id="in_01",
            recoverable=True,
            details={"retry_after_ms": 500},
        )
        data = err.to_dict()
        assert data["code"] == "TIMEOUT"
        assert data["recoverable"] is True
        assert data["details"]["retry_after_ms"] == 500

        roundtrip = PerceptionError.from_dict(data)
        assert roundtrip.code == err.code
        assert roundtrip.message == err.message
        assert roundtrip.recoverable == err.recoverable

    def test_perception_error_validation_rejects_empty(self):
        with pytest.raises(ValueError, match="code must be a non-empty string"):
            PerceptionError(code="", message="Something happened", provider_id="p", request_id="r", input_id="i")

        with pytest.raises(ValueError, match="message must be a non-empty string"):
            PerceptionError(code="ERR", message="", provider_id="p", request_id="r", input_id="i")

    def test_spatial_evidence_valid_and_bounds(self):
        bbox = BoundingBox(x=10, y=20, width=100, height=80)
        loc = GeoLocation(latitude=37.7749, longitude=-122.4194, altitude=15.0)
        spatial = SpatialEvidence(
            bounding_box=bbox,
            location=loc,
            region_label="quadrant_1",
            relative_position="top_left",
            spatial_confidence=0.95,
        )
        assert spatial.region_label == "quadrant_1"
        assert spatial.relative_position == "top_left"
        assert spatial.bounding_box.width == 100

        data = spatial.to_dict()
        assert data["region_label"] == "quadrant_1"
        assert data["bounding_box"]["x"] == 10
        assert data["location"]["latitude"] == 37.7749

        roundtrip = SpatialEvidence.from_dict(data)
        assert roundtrip.region_label == "quadrant_1"
        assert roundtrip.bounding_box.height == 80

    def test_spatial_evidence_rejects_invalid_confidence(self):
        with pytest.raises(ValueError, match="spatial_confidence must be in range"):
            SpatialEvidence(spatial_confidence=1.5)

    def test_perception_source_validation_and_backward_compatibility(self):
        # Phase 6.5a dataclass instantiation
        src = PerceptionSource(
            source_id="cam_front_01",
            source_type=PerceptionSourceType.SENSOR,
            product_id="atlas_drone_01",
            device_id="dev_drone_01",
            provider_id="yolo_detector",
            provider_version="2.1.0",
        )
        assert src.source_id == "cam_front_01"
        assert src.source_type == PerceptionSourceType.SENSOR
        assert src.product_id == "atlas_drone_01"

        data = src.to_dict()
        roundtrip = PerceptionSource.from_dict(data)
        assert roundtrip.source_id == "cam_front_01"
        assert roundtrip.provider_version == "2.1.0"

        # Legacy Phase 4.4 attribute compatibility
        assert PerceptionSource.OCR.value == "ocr"
        assert PerceptionSource.CV.value == "cv"
        assert PerceptionSource.ACCESSIBILITY.value == "accessibility"
        assert PerceptionSource.MANUAL.value == "manual"
        assert PerceptionSource.FUTURE_VISION_MODEL.value == "future_vision_model"

    def test_perception_source_rejects_empty_ids(self):
        with pytest.raises(ValueError, match="source_id must be a non-empty string"):
            PerceptionSource(source_id="", source_type=PerceptionSourceType.SENSOR)

    def test_perception_metadata_sanitizes_credentials(self):
        meta = PerceptionMetadata(
            provider_id="test_provider",
            provider_version="1.0.0",
            model_id="test_model",
            privacy_classification=PerceptionPrivacyClass.RESTRICTED,
            extra={
                "api_key": "sk-secret-12345",
                "auth_token": "bearer-abc",
                "password": "my_password",
                "public_meta": "allowed_value",
            },
        )
        data = meta.to_dict()
        assert data["extra"]["api_key"] == "[REDACTED]"
        assert data["extra"]["auth_token"] == "[REDACTED]"
        assert data["extra"]["password"] == "[REDACTED]"
        assert data["extra"]["public_meta"] == "allowed_value"

    def test_perception_input_validation(self):
        inp = PerceptionInput(
            input_id="in_001",
            modality=ModalityType.IMAGE,
            payload_ref="storage/frames/frame_001.jpg",
            captured_at=100.0,
            source_id="cam_01",
        )
        assert inp.input_id == "in_001"
        assert inp.modality == ModalityType.IMAGE

        data = inp.to_dict()
        roundtrip = PerceptionInput.from_dict(data)
        assert roundtrip.input_id == "in_001"
        assert roundtrip.modality == ModalityType.IMAGE
        assert roundtrip.payload_ref == "storage/frames/frame_001.jpg"

    def test_perception_input_rejects_non_positive_captured_at(self):
        with pytest.raises(ValueError, match="captured_at must be positive"):
            PerceptionInput(
                input_id="in_001",
                modality=ModalityType.IMAGE,
                payload_ref="storage/ref.jpg",
                captured_at=-5.0,
                source_id="cam_01",
            )

    def test_perception_evidence_validation_and_bounds(self):
        prov = make_valid_provenance(source_id="drone_cam_01")
        ev = PerceptionEvidence(
            evidence_id="ev_001",
            semantic_type="detected_person",
            label="person",
            confidence=0.88,
            source_id="drone_cam_01",
            modality=ModalityType.IMAGE,
            timestamp=100.0,
            attributes={"color": "blue", "is_moving": True},
            provenance=prov,
        )
        assert ev.evidence_id == "ev_001"
        assert ev.confidence == 0.88
        assert ev.attributes["color"] == "blue"

        data = ev.to_dict()
        roundtrip = PerceptionEvidence.from_dict(data)
        assert roundtrip.evidence_id == "ev_001"
        assert roundtrip.confidence == 0.88
        assert roundtrip.label == "person"

    def test_perception_evidence_rejects_invalid_confidence(self):
        prov = make_valid_provenance()
        with pytest.raises(ValueError, match="confidence must be a finite number"):
            PerceptionEvidence(
                evidence_id="ev_001",
                semantic_type="test",
                label="test_label",
                confidence=1.5,
                source_id="src_01",
                modality=ModalityType.IMAGE,
                timestamp=10.0,
                provenance=prov,
            )

        with pytest.raises(ValueError, match="confidence must be a finite number"):
            PerceptionEvidence(
                evidence_id="ev_001",
                semantic_type="test",
                label="test_label",
                confidence=float("nan"),
                source_id="src_01",
                modality=ModalityType.IMAGE,
                timestamp=10.0,
                provenance=prov,
            )

    def test_perception_evidence_rejects_missing_provenance(self):
        # Missing required provenance key
        invalid_prov = {"source_id": "s1"}
        with pytest.raises(ValueError, match="provenance missing required keys"):
            PerceptionEvidence(
                evidence_id="ev_001",
                semantic_type="test",
                label="test",
                confidence=0.9,
                source_id="src_01",
                modality=ModalityType.IMAGE,
                timestamp=10.0,
                provenance=invalid_prov,
            )

    def test_perception_evidence_rejects_excessive_attributes(self):
        prov = make_valid_provenance()
        excessive = {f"k_{i}": f"v_{i}" for i in range(PerceptionLimits.max_attributes + 1)}
        with pytest.raises(ValueError, match="attributes exceed maximum allowed count"):
            PerceptionEvidence(
                evidence_id="ev_001",
                semantic_type="test",
                label="test",
                confidence=0.9,
                source_id="src_01",
                modality=ModalityType.IMAGE,
                timestamp=10.0,
                provenance=prov,
                attributes=excessive,
            )

    def test_perception_request_and_result_roundtrip(self):
        inp = PerceptionInput(
            input_id="in_req_01",
            modality=ModalityType.IMAGE,
            payload_ref="storage/frame.jpg",
            captured_at=100.0,
            source_id="cam_01",
        )
        req = PerceptionRequest(
            request_id="req_001",
            input_data=inp,
            requested_capabilities=(PerceptionCapability.VISION_OBJECT_DETECTION,),
        )
        assert req.request_id == "req_001"
        assert req.requested_capabilities == (PerceptionCapability.VISION_OBJECT_DETECTION,)

        prov = make_valid_provenance()
        ev = PerceptionEvidence(
            evidence_id="ev_res_01",
            semantic_type="detected_car",
            label="car",
            confidence=0.94,
            source_id="cam_01",
            modality=ModalityType.IMAGE,
            timestamp=100.0,
            provenance=prov,
        )
        meta = PerceptionMetadata(
            provider_id="yolo_provider",
            provider_version="1.0.0",
            model_id="yolo_v8",
        )
        res = PerceptionResult(
            request_id=req.request_id,
            input_id=inp.input_id,
            status=PerceptionStatus.SUCCESS,
            evidence=(ev,),
            processing_metadata=meta,
        )
        assert res.is_success() is True
        assert len(res.evidence) == 1

        data = res.to_dict()
        assert data["request_id"] == "req_001"
        assert data["status"] == "SUCCESS"
        assert data["evidence"][0]["label"] == "car"

        roundtrip = PerceptionResult.from_dict(data)
        assert roundtrip.request_id == res.request_id
        assert roundtrip.status == PerceptionStatus.SUCCESS
        assert len(roundtrip.evidence) == 1
        assert roundtrip.evidence[0].confidence == 0.94

    def test_perception_result_rejects_excessive_evidence(self):
        prov = make_valid_provenance()
        ev_list = [
            PerceptionEvidence(
                evidence_id=f"ev_{i}",
                semantic_type="item",
                label=f"item_{i}",
                confidence=0.8,
                source_id="src_01",
                modality=ModalityType.IMAGE,
                timestamp=10.0,
                provenance=prov,
            )
            for i in range(PerceptionLimits.max_evidence_per_result + 1)
        ]
        with pytest.raises(ValueError, match="evidence count exceeds maximum allowed"):
            PerceptionResult(
                request_id="req_001",
                input_id="in_001",
                status=PerceptionStatus.SUCCESS,
                evidence=tuple(ev_list),
            )

    def test_perception_models_are_frozen(self):
        prov = make_valid_provenance()
        ev = PerceptionEvidence(
            evidence_id="ev_001",
            semantic_type="item",
            label="item",
            confidence=0.8,
            source_id="src_01",
            modality=ModalityType.IMAGE,
            timestamp=10.0,
            provenance=prov,
        )
        with pytest.raises((TypeError, AttributeError)):
            ev.confidence = 0.9  # type: ignore


# =====================================================================
# 3. REGISTRY BEHAVIOR & DISCOVERY
# =====================================================================

class TestPerceptionProviderRegistry:
    """Validates provider registration, filtering, selection, and concurrency."""

    def test_register_and_get_provider(self):
        registry = PerceptionProviderRegistry()
        mock_p = MockPerceptionProvider(provider_id="mock_vision_01")
        registry.register_provider(mock_p)

        retrieved = registry.get_provider("mock_vision_01")
        assert retrieved is not None
        assert retrieved.provider_id == "mock_vision_01"
        assert len(registry.list_providers()) == 1

    def test_register_duplicate_provider_raises_error(self):
        registry = PerceptionProviderRegistry()
        p1 = MockPerceptionProvider(provider_id="prov_dup")
        p2 = MockPerceptionProvider(provider_id="prov_dup")

        registry.register_provider(p1)
        with pytest.raises(ValueError, match="already registered"):
            registry.register_provider(p2)

    def test_register_invalid_provider_raises_error(self):
        registry = PerceptionProviderRegistry()
        with pytest.raises(ValueError, match="Invalid provider instance"):
            registry.register_provider(None)  # type: ignore

        # Provider missing capabilities
        p_no_caps = MockPerceptionProvider(
            provider_id="no_caps",
            supported_capabilities=set(),
        )
        with pytest.raises(ValueError, match="at least one supported capability"):
            registry.register_provider(p_no_caps)

        # Provider missing modalities
        p_no_mods = MockPerceptionProvider(
            provider_id="no_mods",
            supported_modalities=set(),
        )
        with pytest.raises(ValueError, match="at least one supported modality"):
            registry.register_provider(p_no_mods)

    def test_unregister_provider(self):
        registry = PerceptionProviderRegistry()
        p = MockPerceptionProvider(provider_id="prov_removable")
        registry.register_provider(p)
        assert registry.unregister_provider("prov_removable") is True
        assert registry.unregister_provider("prov_removable") is False
        assert registry.get_provider("prov_removable") is None

    def test_capacity_limit_enforced(self):
        registry = PerceptionProviderRegistry(max_providers=2)
        p1 = MockPerceptionProvider(provider_id="p1")
        p2 = MockPerceptionProvider(provider_id="p2")
        p3 = MockPerceptionProvider(provider_id="p3")

        registry.register_provider(p1)
        registry.register_provider(p2)
        with pytest.raises(RuntimeError, match="provider limit.*exceeded"):
            registry.register_provider(p3)

    def test_capability_discovery_and_filtering(self):
        registry = PerceptionProviderRegistry()
        p_ocr = MockPerceptionProvider(
            provider_id="ocr_prov",
            supported_capabilities={PerceptionCapability.VISION_OCR},
            supported_modalities={ModalityType.IMAGE},
        )
        p_audio = MockPerceptionProvider(
            provider_id="audio_prov",
            supported_capabilities={PerceptionCapability.SPEECH_TRANSCRIPTION},
            supported_modalities={ModalityType.VOICE_TRANSCRIPT},
        )
        registry.register_provider(p_ocr)
        registry.register_provider(p_audio)

        all_caps = registry.list_capabilities()
        assert PerceptionCapability.VISION_OCR in all_caps
        assert PerceptionCapability.SPEECH_TRANSCRIPTION in all_caps

        text_providers = registry.find_providers_for_capability(PerceptionCapability.VISION_OCR)
        assert len(text_providers) == 1
        assert text_providers[0].provider_id == "ocr_prov"

        audio_providers = registry.find_providers_for_capability(PerceptionCapability.SPEECH_TRANSCRIPTION)
        assert len(audio_providers) == 1
        assert audio_providers[0].provider_id == "audio_prov"

        scene_providers = registry.find_providers_for_capability(PerceptionCapability.VISION_SCENE_CLASSIFICATION)
        assert len(scene_providers) == 0

    def test_deterministic_provider_selection(self):
        registry = PerceptionProviderRegistry()
        p_b = MockPerceptionProvider(
            provider_id="prov_b",
            supported_capabilities={PerceptionCapability.VISION_OBJECT_DETECTION},
            supported_modalities={ModalityType.IMAGE},
        )
        p_a = MockPerceptionProvider(
            provider_id="prov_a",
            supported_capabilities={PerceptionCapability.VISION_OBJECT_DETECTION},
            supported_modalities={ModalityType.IMAGE},
        )
        registry.register_provider(p_b)
        registry.register_provider(p_a)

        # Deterministic sorting by provider_id chooses prov_a
        selected = registry.select_provider(
            capability=PerceptionCapability.VISION_OBJECT_DETECTION,
            modality=ModalityType.IMAGE,
        )
        assert selected is not None
        assert selected.provider_id == "prov_a"

    def test_select_provider_skips_unavailable(self):
        registry = PerceptionProviderRegistry()
        p_unavailable = MockPerceptionProvider(
            provider_id="prov_a_down",
            supported_capabilities={PerceptionCapability.VISION_OBJECT_DETECTION},
            available=False,
        )
        p_available = MockPerceptionProvider(
            provider_id="prov_b_up",
            supported_capabilities={PerceptionCapability.VISION_OBJECT_DETECTION},
            available=True,
        )
        registry.register_provider(p_unavailable)
        registry.register_provider(p_available)

        selected = registry.select_provider(PerceptionCapability.VISION_OBJECT_DETECTION)
        assert selected is not None
        assert selected.provider_id == "prov_b_up"

    def test_select_provider_returns_none_when_unmatched(self):
        registry = PerceptionProviderRegistry()
        p = MockPerceptionProvider(
            provider_id="prov_1",
            supported_capabilities={PerceptionCapability.VISION_OBJECT_DETECTION},
            supported_modalities={ModalityType.IMAGE},
        )
        registry.register_provider(p)

        assert registry.select_provider(PerceptionCapability.SPATIAL_LOCALIZATION) is None
        assert registry.select_provider(
            PerceptionCapability.VISION_OBJECT_DETECTION, modality=ModalityType.AUDIO_EVENT
        ) is None

    def test_thread_safe_concurrent_operations(self):
        registry = PerceptionProviderRegistry(max_providers=64)
        errors = []

        def worker(idx: int):
            try:
                p = MockPerceptionProvider(provider_id=f"worker_prov_{idx}")
                registry.register_provider(p)
                assert registry.get_provider(f"worker_prov_{idx}") is not None
                registry.list_providers()
                registry.list_capabilities()
                registry.select_provider(PerceptionCapability.VISION_OBJECT_DETECTION)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(16)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(registry.list_providers()) == 16


# =====================================================================
# 4. MOCK PERCEPTION PROVIDER BEHAVIOR
# =====================================================================

class TestMockPerceptionProvider:
    """Validates MockPerceptionProvider multimodal & visual behavior."""

    def test_mock_provider_perceive_success(self):
        provider = MockPerceptionProvider(provider_id="mock_test_01")
        req = PerceptionRequest(
            request_id="req_test_01",
            input_data=PerceptionInput(
                input_id="in_01",
                modality=ModalityType.IMAGE,
                payload_ref="storage/in_01.jpg",
                captured_at=10.0,
                source_id="cam_01",
            ),
            requested_capabilities=(PerceptionCapability.VISION_OBJECT_DETECTION,),
        )
        result = provider.perceive(req)
        assert isinstance(result, PerceptionResult)
        assert result.is_success() is True
        assert len(result.evidence) == 1
        ev = result.evidence[0]
        assert ev.modality == ModalityType.IMAGE
        assert ev.confidence == 0.95
        assert ev.spatial is not None
        assert ev.spatial.bounding_box is not None
        assert provider.call_count == 1

    def test_mock_provider_unsupported_modality(self):
        provider = MockPerceptionProvider(
            provider_id="mock_test_02",
            supported_modalities={ModalityType.IMAGE},
        )
        req = PerceptionRequest(
            request_id="req_test_02",
            input_data=PerceptionInput(
                input_id="in_02",
                modality=ModalityType.VOICE_TRANSCRIPT,
                payload_ref="storage/audio.wav",
                captured_at=10.0,
                source_id="mic_01",
            ),
            requested_capabilities=(PerceptionCapability.SPEECH_TRANSCRIPTION,),
        )
        result = provider.perceive(req)
        assert isinstance(result, PerceptionResult)
        assert result.status == PerceptionStatus.UNSUPPORTED_MODALITY
        assert len(result.errors) > 0
        assert "not supported" in result.errors[0].message

    def test_mock_provider_unsupported_capability(self):
        provider = MockPerceptionProvider(
            provider_id="mock_test_03",
            supported_capabilities={PerceptionCapability.VISION_OBJECT_DETECTION},
            supported_modalities={ModalityType.IMAGE},
        )
        req = PerceptionRequest(
            request_id="req_test_03",
            input_data=PerceptionInput(
                input_id="in_03",
                modality=ModalityType.IMAGE,
                payload_ref="storage/img.jpg",
                captured_at=10.0,
                source_id="cam_01",
            ),
            requested_capabilities=(PerceptionCapability.VISION_OCR,),
        )
        result = provider.perceive(req)
        assert isinstance(result, PerceptionResult)
        assert result.status == PerceptionStatus.FAILED
        assert len(result.errors) > 0
        assert "Capability" in result.errors[0].message

    def test_mock_provider_simulated_status_and_error(self):
        provider = MockPerceptionProvider(
            provider_id="mock_test_04",
            simulate_status=PerceptionStatus.TIMEOUT,
            simulate_error=PerceptionError(
                code="INFERENCE_TIMEOUT",
                message="Simulated inference timeout",
                provider_id="mock_test_04",
                request_id="req_test_04",
                input_id="in_04",
                recoverable=True,
            ),
        )
        req = PerceptionRequest(
            request_id="req_test_04",
            input_data=PerceptionInput(
                input_id="in_04",
                modality=ModalityType.IMAGE,
                payload_ref="storage/img.jpg",
                captured_at=10.0,
                source_id="cam_01",
            ),
            requested_capabilities=(PerceptionCapability.VISION_OBJECT_DETECTION,),
        )
        result = provider.perceive(req)
        assert isinstance(result, PerceptionResult)
        assert result.status == PerceptionStatus.TIMEOUT
        assert result.errors[0].code == "INFERENCE_TIMEOUT"
        assert result.is_success() is False

    def test_mock_provider_health_check(self):
        provider = MockPerceptionProvider(provider_id="health_prov")
        health = provider.get_health()
        assert health["status"] == "HEALTHY"
        assert health["available"] is True
        assert health["provider_id"] == "health_prov"
        assert health["capabilities_count"] > 0
        assert health["modalities_count"] > 0

    def test_mock_provider_backward_compatibility_with_computer_observation(self):
        provider = MockPerceptionProvider()
        assert isinstance(provider, VisualPerceptionProvider)
        assert provider.source == PerceptionSource.MANUAL

        provider.add_element(
            element_id="btn_submit",
            element_type=ElementType.BUTTON,
            x=50,
            y=100,
            width=80,
            height=30,
            text="Submit",
        )
        obs = ComputerObservation(
            timestamp=1.0,
            screen_dimensions=ScreenDimensions(1920, 1080),
        )
        elements = provider.perceive(obs)
        assert isinstance(elements, list)
        assert len(elements) == 1
        assert elements[0].text == "Submit"
        assert elements[0].bounding_box.x == 50


# =====================================================================
# 5. OBSERVATION NORMALIZER BEHAVIOR
# =====================================================================

class TestPerceptionObservationNormalizer:
    """Validates conversion of PerceptionResult into canonical MultimodalObservation."""

    def test_normalize_single_evidence(self):
        normalizer = PerceptionObservationNormalizer()
        bbox = BoundingBox(x=100, y=200, width=50, height=60)
        loc = GeoLocation(latitude=37.77, longitude=-122.42)
        spatial = SpatialEvidence(bounding_box=bbox, location=loc)

        prov = make_valid_provenance(
            source_id="drone_cam_front",
            source_type="edge_perception",
            device_id="ATLAS_DRONE_01",
            correlation_id="corr_999",
            causation_id="cause_888",
        )

        evidence = PerceptionEvidence(
            evidence_id="ev_001",
            semantic_type="drone_target",
            label="target_quad",
            confidence=0.92,
            source_id="drone_cam_front",
            modality=ModalityType.IMAGE,
            timestamp=50.0,
            spatial=spatial,
            attributes={"color": "red", "altitude": 12.5},
            provenance=prov,
            correlation_id="corr_999",
            causation_id="cause_888",
        )

        metadata = PerceptionMetadata(
            provider_id="test_vision_prov",
            provider_version="2.0.0",
            model_id="yolov8_custom",
            privacy_classification=PerceptionPrivacyClass.INTERNAL,
            processing_time_ms=5.4,
        )

        result = PerceptionResult(
            request_id="req_norm_01",
            input_id="in_norm_01",
            status=PerceptionStatus.SUCCESS,
            evidence=(evidence,),
            processing_metadata=metadata,
        )

        observation = normalizer.normalize(result, evidence_index=0)
        assert isinstance(observation, MultimodalObservation)
        assert observation.observation_id == "obs_req_norm_01_ev_001"
        assert observation.source_id == "drone_cam_front"
        assert observation.source_type == "edge_perception"
        assert observation.modality == ModalityType.IMAGE
        assert observation.confidence == 0.92
        assert observation.timestamp == 50.0
        assert observation.device_id == "ATLAS_DRONE_01"
        assert observation.correlation_id == "corr_999"
        assert observation.causation_id == "cause_888"

        # Payload structure
        payload = observation.payload
        assert payload["semantic_type"] == "drone_target"
        assert payload["label"] == "target_quad"
        assert payload["attributes"]["color"] == "red"
        assert payload["spatial"]["bounding_box"]["x"] == 100

        # Location preservation
        assert observation.location is not None
        assert observation.location.latitude == 37.77

        # Metadata preservation
        assert observation.metadata["provider_id"] == "test_vision_prov"
        assert observation.metadata["privacy_class"] == "INTERNAL"

    def test_normalize_all_multiple_evidences(self):
        normalizer = PerceptionObservationNormalizer()
        prov1 = make_valid_provenance(request_id="req_multi", input_id="in_multi")
        prov2 = make_valid_provenance(request_id="req_multi", input_id="in_multi")
        ev1 = PerceptionEvidence(
            evidence_id="ev_1",
            semantic_type="person",
            label="person_a",
            confidence=0.85,
            source_id="cam_01",
            modality=ModalityType.IMAGE,
            timestamp=10.0,
            provenance=prov1,
        )
        ev2 = PerceptionEvidence(
            evidence_id="ev_2",
            semantic_type="vehicle",
            label="car_b",
            confidence=0.90,
            source_id="cam_01",
            modality=ModalityType.IMAGE,
            timestamp=10.0,
            provenance=prov2,
        )
        result = PerceptionResult(
            request_id="req_multi",
            input_id="in_multi",
            status=PerceptionStatus.SUCCESS,
            evidence=(ev1, ev2),
            processing_metadata=PerceptionMetadata(provider_id="prov_1", provider_version="1.0"),
        )

        observations = normalizer.normalize_all(result)
        assert len(observations) == 2
        assert observations[0].observation_id == "obs_req_multi_ev_1"
        assert observations[1].observation_id == "obs_req_multi_ev_2"
        assert observations[0].payload["label"] == "person_a"
        assert observations[1].payload["label"] == "car_b"

    def test_normalize_empty_evidence_behavior(self):
        normalizer = PerceptionObservationNormalizer()
        result = PerceptionResult(
            request_id="req_empty",
            input_id="in_empty",
            status=PerceptionStatus.SUCCESS,
            evidence=(),
            processing_metadata=PerceptionMetadata(provider_id="prov_1", provider_version="1.0"),
        )
        # normalize_all on empty evidence returns empty list
        assert list(normalizer.normalize_all(result)) == []

        # normalize with index 0 on empty evidence raises IndexError
        with pytest.raises(IndexError, match="No evidence in perception result"):
            normalizer.normalize(result, 0)

    def test_normalize_out_of_range_index_raises_index_error(self):
        normalizer = PerceptionObservationNormalizer()
        prov = make_valid_provenance()
        ev = PerceptionEvidence(
            evidence_id="ev_1",
            semantic_type="item",
            label="item",
            confidence=0.8,
            source_id="cam_01",
            modality=ModalityType.IMAGE,
            timestamp=10.0,
            provenance=prov,
        )
        result = PerceptionResult(
            request_id="req_idx",
            input_id="in_idx",
            status=PerceptionStatus.SUCCESS,
            evidence=(ev,),
            processing_metadata=PerceptionMetadata(provider_id="prov_1", provider_version="1.0"),
        )
        with pytest.raises(IndexError, match="out of range"):
            normalizer.normalize(result, 5)

    def test_normalize_unsuccessful_result_raises_value_error(self):
        normalizer = PerceptionObservationNormalizer()
        err = PerceptionError(code="ERR", message="Inference crashed", provider_id="p", request_id="r", input_id="i")
        result = PerceptionResult(
            request_id="req_fail",
            input_id="in_fail",
            status=PerceptionStatus.FAILED,
            evidence=(),
            processing_metadata=PerceptionMetadata(provider_id="prov_1", provider_version="1.0"),
            errors=(err,),
        )
        with pytest.raises(ValueError, match="Cannot normalize unsuccessful perception result"):
            normalizer.normalize_all(result)


# =====================================================================
# 6. END-TO-END FLOW & ARCHITECTURAL INVARIANTS
# =====================================================================

class TestPerceptionToCentralFlowAndInvariants:
    """Validates complete perception -> normalizer -> gateway intake pipeline and safety boundaries."""

    def test_perception_to_central_gateway_pipeline(self):
        # 1. Setup registry & normalizer
        registry = PerceptionProviderRegistry()
        normalizer = PerceptionObservationNormalizer()
        provider = MockPerceptionProvider(provider_id="cam_provider_01")
        registry.register_provider(provider)

        # 2. Select provider for capability
        selected = registry.select_provider(
            capability=PerceptionCapability.VISION_OBJECT_DETECTION,
            modality=ModalityType.IMAGE,
        )
        assert selected is not None

        # 3. Create perception request
        now_ts = time.time()
        req = PerceptionRequest(
            request_id="req_e2e_01",
            input_data=PerceptionInput(
                input_id="in_e2e_01",
                modality=ModalityType.IMAGE,
                payload_ref="storage/frame.jpg",
                captured_at=now_ts,
                source_id="cam_01",
            ),
            requested_capabilities=(PerceptionCapability.VISION_OBJECT_DETECTION,),
            privacy_constraints=PerceptionPrivacyClass.PUBLIC,
        )

        # 4. Perceive evidence
        result = selected.perceive(req)
        assert result.is_success() is True

        # 5. Normalize into canonical observations
        observations = normalizer.normalize_all(result)
        assert len(observations) == 1
        obs = observations[0]

        # 6. Ingest into CentralInputGateway
        gateway = CentralInputGateway()
        gateway.ingest_observation(obs)

        # 7. Gateway history holds the observation
        recent = gateway.get_recent_observations(modality=ModalityType.IMAGE)
        assert len(recent) == 1
        assert recent[0].observation_id == obs.observation_id

    def test_perception_layer_has_no_side_effects(self):
        """PerceptionResult and PerceptionEvidence do not alter gateway or world state on their own."""
        gateway = CentralInputGateway()
        prov = make_valid_provenance()
        ev = PerceptionEvidence(
            evidence_id="ev_isolated",
            semantic_type="intrusion",
            label="intruder",
            confidence=0.99,
            source_id="isolated_cam",
            modality=ModalityType.IMAGE,
            timestamp=20.0,
            provenance=prov,
        )
        res = PerceptionResult(
            request_id="req_iso",
            input_id="in_iso",
            status=PerceptionStatus.SUCCESS,
            evidence=(ev,),
            processing_metadata=PerceptionMetadata(provider_id="p", provider_version="1"),
        )
        # Verify no direct dispatch method exists
        assert not hasattr(res, "submit")
        assert not hasattr(res, "dispatch")
        assert not hasattr(res, "execute")
        assert len(gateway.get_recent_observations()) == 0

    def test_app_state_contains_perception_subsystems(self):
        from config.settings import AtlasSettings
        from core.app_state import initialize_application_state

        cfg = AtlasSettings(app_env="test", simulation_mode=False)
        app_state = initialize_application_state(settings=cfg, in_memory_stores=True)

        assert app_state.perception_registry is not None
        assert isinstance(app_state.perception_registry, PerceptionProviderRegistry)
        assert app_state.perception_normalizer is not None
        assert isinstance(app_state.perception_normalizer, PerceptionObservationNormalizer)

    def test_no_forbidden_dependencies_in_perception_modules(self):
        """Strict AST / import audit ensuring zero hardware, subagent, or subprocess dependencies."""
        import ast
        import os

        forbidden = {
            "subprocess", "os.system", "eval", "exec", "pymavlink",
            "mavsdk", "rclpy", "rospy", "serial", "RPi.GPIO",
        }

        perception_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "perception")
        )

        for fname in os.listdir(perception_dir):
            if not fname.endswith(".py"):
                continue
            fpath = os.path.join(perception_dir, fname)
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
