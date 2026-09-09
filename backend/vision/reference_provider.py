"""
Reference / Mock Visual Provider (Phase 6.5b)

Deterministic reference implementation of PerceptionProviderInterface for visual modalities.
Produces predictable synthetic outputs for motion, semantic objects, OCR, scene analysis,
and tracking without relying on real camera hardware or deep learning models.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from core.interfaces.perception_interface import (
    PerceptionProviderInterface,
    VisualPerceptionProvider as LegacyVisualPerceptionProvider,
)
from core.models.computer import ComputerObservation
from core.models.orchestration import ModalityType
from core.models.perception import (
    BoundingBox,
    PerceptionCapability,
    PerceptionError,
    PerceptionEvidence,
    PerceptionMetadata,
    PerceptionRequest,
    PerceptionResult,
    PerceptionSource,
    PerceptionStatus,
    SpatialEvidence,
    VisualElement,
)
from vision.limits import VisionLimits


class MockVisualProvider(LegacyVisualPerceptionProvider, PerceptionProviderInterface):
    """
    Deterministic mock visual provider for multi-product edge and simulation testing.
    Outputs fully predictable, repeatable evidence for all visual capabilities.
    """

    source = PerceptionSource.MANUAL

    def __init__(
        self,
        provider_id: str = "mock_visual_provider",
        provider_version: str = "1.0.0",
        limits: Optional[VisionLimits] = None,
        available: bool = True,
        simulate_status: Optional[PerceptionStatus] = None,
        simulate_error: Optional[PerceptionError] = None,
        simulated_evidence: Optional[Union[List[PerceptionEvidence], Tuple[PerceptionEvidence, ...]]] = None,
    ) -> None:
        self._provider_id = provider_id
        self._provider_version = provider_version
        self.limits = limits or VisionLimits()
        self._available = available
        self.simulate_status = simulate_status
        self.simulate_error = simulate_error
        self._simulated_evidence = tuple(simulated_evidence) if simulated_evidence is not None else None
        self.call_count = 0

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

    def set_simulated_evidence(
        self, evidence: Optional[Union[List[PerceptionEvidence], Tuple[PerceptionEvidence, ...]]]
    ) -> None:
        self._simulated_evidence = tuple(evidence) if evidence is not None else None

    def get_health(self) -> Dict[str, Any]:
        return {
            "provider_id": self._provider_id,
            "provider_version": self._provider_version,
            "status": "HEALTHY" if self._available else "UNAVAILABLE",
            "available": self._available,
            "call_count": self.call_count,
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
        return self.perceive(request_or_observation)

    def perceive(
        self,
        request_or_observation: Union[PerceptionRequest, ComputerObservation],
    ) -> Union[PerceptionResult, List[VisualElement]]:
        self.call_count += 1

        if isinstance(request_or_observation, ComputerObservation):
            return []

        if not isinstance(request_or_observation, PerceptionRequest):
            raise TypeError(
                f"Unsupported request type for MockVisualProvider: {type(request_or_observation)}"
            )

        request = request_or_observation
        req_id = request.request_id
        inp_id = request.input_data.input_id
        source_id = request.input_data.source_id
        timestamp = request.input_data.captured_at

        meta = PerceptionMetadata(
            provider_id=self._provider_id,
            provider_version=self._provider_version,
            privacy_classification=request.privacy_constraints,
            processing_time_ms=1.0,
        )

        # 1. Simulated status / error override
        if self.simulate_status is not None:
            err = self.simulate_error or PerceptionError(
                code="SIMULATED_FAILURE",
                message="Mock simulated visual error",
                provider_id=self._provider_id,
                request_id=req_id,
                input_id=inp_id,
                recoverable=True,
            )
            return PerceptionResult(
                request_id=req_id,
                input_id=inp_id,
                status=self.simulate_status,
                processing_metadata=meta,
                errors=(err,),
            )

        # 2. Availability check
        if not self._available:
            err = PerceptionError(
                code="PROVIDER_UNAVAILABLE",
                message="Mock visual provider is unavailable",
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

        # 3. Modality check
        if request.input_data.modality not in self._supported_modalities:
            err = PerceptionError(
                code="UNSUPPORTED_MODALITY",
                message=f"Modality '{request.input_data.modality.value}' not supported",
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

        # 4. Custom simulated evidence override
        if self._simulated_evidence is not None:
            return PerceptionResult(
                request_id=req_id,
                input_id=inp_id,
                status=PerceptionStatus.SUCCESS,
                evidence=self._simulated_evidence,
                processing_metadata=meta,
            )

        # 5. Deterministic synthetic evidence generation based on requested capabilities
        requested_caps = (
            set(request.requested_capabilities)
            if request.requested_capabilities
            else self._supported_capabilities
        )

        provenance = {
            "source_id": source_id,
            "provider_id": self._provider_id,
            "provider_version": self._provider_version,
            "input_id": inp_id,
            "request_id": req_id,
            "observation_timestamp": timestamp,
            "correlation_id": request.correlation_id or req_id,
            "causation_id": request.causation_id or inp_id,
            "product_type": request.input_data.metadata.get("product_type", "UNKNOWN"),
            "device_id": request.input_data.metadata.get("device_id", source_id),
        }

        evidence_items: List[PerceptionEvidence] = []

        # A. Motion
        if PerceptionCapability.VISION_MOTION_DETECTION in requested_caps:
            bbox = BoundingBox(x=50, y=50, width=120, height=80)
            spatial = SpatialEvidence(bounding_box=bbox, region_label="motion_0", spatial_confidence=0.92)
            evidence_items.append(
                PerceptionEvidence(
                    evidence_id=f"ev_mot_{req_id}_0",
                    semantic_type="motion",
                    label="motion_detected",
                    confidence=0.92,
                    source_id=source_id,
                    modality=request.input_data.modality,
                    timestamp=timestamp,
                    spatial=spatial,
                    attributes={"motion_area": 9600, "simulated": True},
                    provenance=provenance,
                    correlation_id=request.correlation_id or req_id,
                    causation_id=request.causation_id or inp_id,
                )
            )

        # B. Object Detection
        if PerceptionCapability.VISION_OBJECT_DETECTION in requested_caps:
            bbox = BoundingBox(x=200, y=150, width=80, height=160)
            spatial = SpatialEvidence(bounding_box=bbox, region_label="obj_0", spatial_confidence=0.95)
            evidence_items.append(
                PerceptionEvidence(
                    evidence_id=f"ev_obj_{req_id}_0",
                    semantic_type="semantic_object",
                    label="person",
                    confidence=0.95,
                    source_id=source_id,
                    modality=request.input_data.modality,
                    timestamp=timestamp,
                    spatial=spatial,
                    attributes={"class": "person", "simulated": True},
                    provenance=provenance,
                    correlation_id=request.correlation_id or req_id,
                    causation_id=request.causation_id or inp_id,
                )
            )

        # C. Scene Analysis
        if PerceptionCapability.VISION_SCENE_CLASSIFICATION in requested_caps:
            evidence_items.append(
                PerceptionEvidence(
                    evidence_id=f"ev_scn_{req_id}_0",
                    semantic_type="scene_analysis",
                    label="bright_indoor_like",
                    confidence=0.90,
                    source_id=source_id,
                    modality=request.input_data.modality,
                    timestamp=timestamp,
                    attributes={
                        "brightness": "bright",
                        "environment": "indoor_like",
                        "simulated": True,
                    },
                    provenance=provenance,
                    correlation_id=request.correlation_id or req_id,
                    causation_id=request.causation_id or inp_id,
                )
            )

        # D. OCR
        if PerceptionCapability.VISION_OCR in requested_caps:
            bbox = BoundingBox(x=10, y=10, width=150, height=30)
            spatial = SpatialEvidence(bounding_box=bbox, region_label="ocr_0", spatial_confidence=0.94)
            evidence_items.append(
                PerceptionEvidence(
                    evidence_id=f"ev_ocr_{req_id}_0",
                    semantic_type="text_detected",
                    label="ocr_text",
                    confidence=0.94,
                    source_id=source_id,
                    modality=request.input_data.modality,
                    timestamp=timestamp,
                    spatial=spatial,
                    attributes={"text": "ATLAS_ZONE_A", "text_length": 12, "simulated": True},
                    provenance=provenance,
                    correlation_id=request.correlation_id or req_id,
                    causation_id=request.causation_id or inp_id,
                )
            )

        # E. Tracking
        if PerceptionCapability.SPATIAL_LOCALIZATION in requested_caps:
            bbox = BoundingBox(x=200, y=150, width=80, height=160)
            spatial = SpatialEvidence(bounding_box=bbox, region_label="track_001", spatial_confidence=0.95)
            evidence_items.append(
                PerceptionEvidence(
                    evidence_id=f"ev_trk_{req_id}_0",
                    semantic_type="visual_track",
                    label="person_track_001",
                    confidence=0.95,
                    source_id=source_id,
                    modality=request.input_data.modality,
                    timestamp=timestamp,
                    spatial=spatial,
                    attributes={"track_id": "track_001", "status": "active", "simulated": True},
                    provenance=provenance,
                    correlation_id=request.correlation_id or req_id,
                    causation_id=request.causation_id or inp_id,
                )
            )

        status = PerceptionStatus.SUCCESS if evidence_items else PerceptionStatus.NO_DETECTION
        return PerceptionResult(
            request_id=req_id,
            input_id=inp_id,
            status=status,
            evidence=tuple(evidence_items),
            processing_metadata=meta,
        )


ReferenceVisualProvider = MockVisualProvider
