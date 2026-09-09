"""
Visual Perception Provider (Phase 6.5b)

Authoritative implementation of the PerceptionProviderInterface for visual modalities
(IMAGE and VIDEO_FRAME). Composes modular motion, region/object, scene, OCR,
and tracking sub-detectors into a unified, model-neutral provider.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from core.interfaces.perception_interface import (
    PerceptionProviderInterface,
    VisualPerceptionProvider as LegacyVisualPerceptionProvider,
)
from core.models.computer import ComputerObservation
from core.models.orchestration import ModalityType
from core.models.perception import (
    ElementType,
    PerceptionCapability,
    PerceptionError,
    PerceptionEvidence,
    PerceptionMetadata,
    PerceptionRequest,
    PerceptionResult,
    PerceptionSource,
    PerceptionStatus,
    VisualElement,
)
from vision.frame_loader import FrameLoadingError, load_and_validate_frame
from vision.limits import VisionLimits
from vision.motion import MotionDetector
from vision.object_detection import VisualRegionDetector
from vision.ocr import VisualOCRDetector
from vision.scene_analysis import SceneAnalyzer
from vision.tracking import VisualTracker

logger = logging.getLogger(__name__)


class VisualPerceptionProvider(LegacyVisualPerceptionProvider, PerceptionProviderInterface):
    """
    Authoritative Visual Perception Provider for ATLAS Phase 6.5b.
    Implements PerceptionProviderInterface for multimodal requests and maintains
    backward-compatibility with LegacyVisualPerceptionProvider for desktop observations.
    """

    source = PerceptionSource.CV

    def __init__(
        self,
        provider_id: str = "atlas_visual_perception_provider",
        provider_version: str = "1.0.0",
        limits: Optional[VisionLimits] = None,
        motion_detector: Optional[MotionDetector] = None,
        region_detector: Optional[VisualRegionDetector] = None,
        scene_analyzer: Optional[SceneAnalyzer] = None,
        tracker: Optional[VisualTracker] = None,
        ocr_detector: Optional[VisualOCRDetector] = None,
        available: bool = True,
    ) -> None:
        self._provider_id = provider_id
        self._provider_version = provider_version
        self.limits = limits or VisionLimits()
        self._available = available
        self.call_count = 0

        # Compose modular sub-detectors
        self.motion_detector = motion_detector or MotionDetector(limits=self.limits)
        self.region_detector = region_detector or VisualRegionDetector(limits=self.limits)
        self.scene_analyzer = scene_analyzer or SceneAnalyzer(limits=self.limits)
        self.tracker = tracker or VisualTracker(limits=self.limits)
        self.ocr_detector = ocr_detector or VisualOCRDetector(limits=self.limits)

        self._supported_capabilities: Set[PerceptionCapability] = {
            PerceptionCapability.VISION_MOTION_DETECTION,
            PerceptionCapability.VISION_OBJECT_DETECTION,
            PerceptionCapability.VISION_SCENE_CLASSIFICATION,
            PerceptionCapability.VISION_OCR,
            PerceptionCapability.SPATIAL_LOCALIZATION,
        }

        self._supported_modalities: Set[ModalityType] = {
            ModalityType.IMAGE,
            ModalityType.VIDEO_FRAME,
        }

    @property
    def provider_id(self) -> str:
        return self._provider_id

    @property
    def provider_version(self) -> str:
        return self._provider_version

    @property
    def version(self) -> str:
        return self._provider_version

    @property
    def capabilities(self) -> Set[PerceptionCapability]:
        return set(self._supported_capabilities)

    @property
    def supported_capabilities(self) -> Set[PerceptionCapability]:
        return set(self._supported_capabilities)

    @property
    def supported_modalities(self) -> Set[ModalityType]:
        return set(self._supported_modalities)

    def is_available(self) -> bool:
        return self._available

    def set_available(self, available: bool) -> None:
        self._available = available

    def get_health(self) -> Dict[str, Any]:
        return {
            "provider_id": self._provider_id,
            "provider_version": self._provider_version,
            "status": "HEALTHY" if self._available else "UNAVAILABLE",
            "available": self._available,
            "call_count": self.call_count,
            "supported_capabilities": [c.value for c in self._supported_capabilities],
            "supported_modalities": [m.value for m in self._supported_modalities],
            "limits": self.limits.to_dict() if hasattr(self.limits, "to_dict") else {},
        }

    def health_check(self) -> Any:
        class HealthResult:
            def __init__(self, d: Dict[str, Any]):
                self.status = d["status"]
                self.provider_id = d["provider_id"]
                self.error = None if d["available"] else "Provider unavailable"
                self.details = d
            def __getitem__(self, item):
                return self.details[item]
        return HealthResult(self.get_health())

    def process(self, request_or_observation: Union[PerceptionRequest, ComputerObservation]) -> Union[PerceptionResult, List[VisualElement]]:
        """Convenience alias for perceive."""
        return self.perceive(request_or_observation)

    def perceive(
        self,
        request_or_observation: Union[PerceptionRequest, ComputerObservation],
    ) -> Union[PerceptionResult, List[VisualElement]]:
        """
        Process a visual perception request or desktop computer observation.
        """
        self.call_count += 1

        # Phase 4.4 Desktop Visual Perception Mode
        if isinstance(request_or_observation, ComputerObservation):
            # Compatibility fallback returning empty visual element list
            return []

        if not isinstance(request_or_observation, PerceptionRequest):
            raise TypeError(
                f"Unsupported request type for VisualPerceptionProvider: {type(request_or_observation)}"
            )

        request = request_or_observation
        start_time = time.perf_counter()
        req_id = request.request_id
        inp_id = request.input_data.input_id
        source_id = request.input_data.source_id

        meta = PerceptionMetadata(
            provider_id=self._provider_id,
            provider_version=self._provider_version,
            privacy_classification=request.privacy_constraints,
        )

        # 1. Provider availability check
        if not self._available:
            err = PerceptionError(
                code="PROVIDER_UNAVAILABLE",
                message=f"Visual perception provider '{self._provider_id}' is currently unavailable",
                provider_id=self._provider_id,
                request_id=req_id,
                input_id=inp_id,
                recoverable=True,
            )
            return PerceptionResult(
                request_id=req_id,
                input_id=inp_id,
                status=PerceptionStatus.FAILED,
                processing_metadata=meta,
                errors=(err,),
            )

        # 2. Modality support check
        if request.input_data.modality not in self._supported_modalities:
            err = PerceptionError(
                code="UNSUPPORTED_MODALITY",
                message=f"Modality '{request.input_data.modality.value}' is not supported by visual provider",
                provider_id=self._provider_id,
                request_id=req_id,
                input_id=inp_id,
                recoverable=False,
            )
            return PerceptionResult(
                request_id=req_id,
                input_id=inp_id,
                status=PerceptionStatus.UNSUPPORTED_MODALITY,
                processing_metadata=meta,
                errors=(err,),
            )

        # 3. Decode & validate frame
        allow_stale = bool(request.parameters.get("allow_stale", False))
        try:
            _, bgr_arr, frame_meta = load_and_validate_frame(
                request.input_data,
                limits=self.limits,
                allow_stale=allow_stale,
            )
        except FrameLoadingError as e:
            status = PerceptionStatus.INVALID_INPUT
            if e.code == "STALE_FRAME":
                status = PerceptionStatus.FAILED
            elif e.code == "UNSUPPORTED_MODALITY":
                status = PerceptionStatus.UNSUPPORTED_MODALITY

            err = PerceptionError(
                code=e.code,
                message=e.message,
                provider_id=self._provider_id,
                request_id=req_id,
                input_id=inp_id,
                recoverable=False,
            )
            return PerceptionResult(
                request_id=req_id,
                input_id=inp_id,
                status=status,
                processing_metadata=meta,
                errors=(err,),
            )
        except Exception as e:
            err = PerceptionError(
                code="INVALID_FRAME",
                message=f"Failed to process frame: {e}",
                provider_id=self._provider_id,
                request_id=req_id,
                input_id=inp_id,
                recoverable=False,
            )
            return PerceptionResult(
                request_id=req_id,
                input_id=inp_id,
                status=PerceptionStatus.INVALID_INPUT,
                processing_metadata=meta,
                errors=(err,),
            )

        # Build baseline provenance
        provenance = {
            "source_id": source_id,
            "provider_id": self._provider_id,
            "provider_version": self._provider_version,
            "input_id": inp_id,
            "request_id": req_id,
            "observation_timestamp": request.input_data.captured_at,
            "correlation_id": request.correlation_id or req_id,
            "causation_id": request.causation_id or inp_id,
            "product_type": request.input_data.metadata.get("product_type", "UNKNOWN"),
            "device_id": request.input_data.metadata.get("device_id", source_id),
        }

        # 4. Capability dispatch
        requested_caps = (
            set(request.requested_capabilities)
            if request.requested_capabilities
            else self._supported_capabilities
        )

        all_evidence: List[PerceptionEvidence] = []
        errors: List[PerceptionError] = []

        try:
            # A. Motion detection
            if PerceptionCapability.VISION_MOTION_DETECTION in requested_caps:
                prev_frame_bgr = request.parameters.get("previous_frame_bgr")
                motion_ev = self.motion_detector.detect(
                    current_bgr=bgr_arr,
                    source_id=source_id,
                    timestamp=request.input_data.captured_at,
                    provenance=provenance,
                    request_id=req_id,
                    input_id=inp_id,
                    previous_bgr=prev_frame_bgr,
                )
                all_evidence.extend(motion_ev)

            # B. Object / Region detection
            detected_objects: List[PerceptionEvidence] = []
            if PerceptionCapability.VISION_OBJECT_DETECTION in requested_caps:
                fixture_key = request.parameters.get("fixture_key")
                detected_objects = self.region_detector.detect(
                    frame_bgr=bgr_arr,
                    source_id=source_id,
                    timestamp=request.input_data.captured_at,
                    provenance=provenance,
                    request_id=req_id,
                    input_id=inp_id,
                    fixture_key=fixture_key,
                    parameters=request.parameters,
                )
                all_evidence.extend(detected_objects)

            # C. Scene analysis
            if PerceptionCapability.VISION_SCENE_CLASSIFICATION in requested_caps:
                scene_ev = self.scene_analyzer.analyze(
                    frame_bgr=bgr_arr,
                    source_id=source_id,
                    timestamp=request.input_data.captured_at,
                    provenance=provenance,
                    request_id=req_id,
                    input_id=inp_id,
                )
                all_evidence.extend(scene_ev)

            # D. OCR
            if PerceptionCapability.VISION_OCR in requested_caps:
                ocr_fixture = request.parameters.get("ocr_fixture") or request.parameters.get("fixture_key")
                ocr_ev = self.ocr_detector.extract_text(
                    frame_bgr=bgr_arr,
                    source_id=source_id,
                    timestamp=request.input_data.captured_at,
                    provenance=provenance,
                    request_id=req_id,
                    input_id=inp_id,
                    fixture_key=ocr_fixture,
                    parameters=request.parameters,
                )
                all_evidence.extend(ocr_ev)

            # E. Tracking (SPATIAL_LOCALIZATION or explicit tracking parameter)
            if (
                PerceptionCapability.SPATIAL_LOCALIZATION in requested_caps
                or bool(request.parameters.get("enable_tracking", False))
            ):
                tracks_ev = self.tracker.update(
                    detections=detected_objects or all_evidence,
                    timestamp=request.input_data.captured_at,
                    source_id=source_id,
                    provenance=provenance,
                    request_id=req_id,
                    input_id=inp_id,
                )
                all_evidence.extend(tracks_ev)

        except Exception as e:
            logger.exception("Error during visual perception execution: %s", e)
            errors.append(
                PerceptionError(
                    code="PROCESSING_FAILURE",
                    message=f"Visual detection failed: {e}",
                    provider_id=self._provider_id,
                    request_id=req_id,
                    input_id=inp_id,
                    recoverable=True,
                )
            )

        # 5. Cap evidence collection
        bounded_evidence = tuple(all_evidence[: self.limits.max_evidence_items])

        # 6. Determine final status
        if errors:
            status = PerceptionStatus.FAILED
        elif not bounded_evidence:
            status = PerceptionStatus.NO_DETECTION
        else:
            status = PerceptionStatus.SUCCESS

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
        final_meta = PerceptionMetadata(
            provider_id=self._provider_id,
            provider_version=self._provider_version,
            privacy_classification=request.privacy_constraints,
            processing_time_ms=elapsed_ms,
        )

        return PerceptionResult(
            request_id=req_id,
            input_id=inp_id,
            status=status,
            evidence=bounded_evidence,
            processing_metadata=final_meta,
            errors=tuple(errors),
            created_at=time.time(),
        )
