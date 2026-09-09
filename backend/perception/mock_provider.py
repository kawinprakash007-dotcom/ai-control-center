"""
Mock Perception Provider (Phase 6.5a & Backward-Compatible with Phase 4.4)

Deterministic in-memory mock provider implementing both PerceptionProviderInterface
and VisualPerceptionProvider with desktop/hardware isolation.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from core.interfaces.perception_interface import (
    PerceptionProviderInterface,
    VisualPerceptionProvider,
)
from core.models.computer import ComputerObservation
from core.models.orchestration import ModalityType
from core.models.perception import (
    BoundingBox,
    ElementType,
    PerceptionCapability,
    PerceptionError,
    PerceptionEvidence,
    PerceptionMetadata,
    PerceptionPrivacyClass,
    PerceptionRequest,
    PerceptionResult,
    PerceptionSource,
    PerceptionStatus,
    SpatialEvidence,
    VisualElement,
)


class MockPerceptionProvider(VisualPerceptionProvider, PerceptionProviderInterface):
    """
    Deterministic in-memory mock perception provider.
    Fulfills Phase 6.5a PerceptionProviderInterface for multimodal testing
    and Phase 4.4 VisualPerceptionProvider for desktop UI perception tests.
    """

    # Legacy Phase 4.4 source attribute
    source = PerceptionSource.MANUAL

    def __init__(
        self,
        provider_id: str = "mock_perception_provider",
        provider_version: str = "1.0.0",
        supported_capabilities: Optional[Set[PerceptionCapability]] = None,
        supported_modalities: Optional[Set[ModalityType]] = None,
        elements: Optional[List[VisualElement]] = None,
        available: bool = True,
        should_fail: bool = False,
        failure_message: str = "Mock perception failure",
        simulate_status: Optional[PerceptionStatus] = None,
        simulate_error: Optional[PerceptionError] = None,
        simulated_evidence: Optional[Union[List[PerceptionEvidence], Tuple[PerceptionEvidence, ...]]] = None,
        health_status: str = "HEALTHY",
    ) -> None:
        self._provider_id = provider_id
        self._provider_version = provider_version
        self._supported_capabilities = (
            set(supported_capabilities)
            if supported_capabilities is not None
            else {
                PerceptionCapability.VISION_OBJECT_DETECTION,
                PerceptionCapability.VISION_OCR,
                PerceptionCapability.VISION_SCENE_CLASSIFICATION,
                PerceptionCapability.SPEECH_TRANSCRIPTION,
                PerceptionCapability.AUDIO_EVENT_CLASSIFICATION,
                PerceptionCapability.TELEMETRY_NORMALIZATION,
                PerceptionCapability.MULTIMODAL_CORRELATION,
            }
        )
        self._supported_modalities = (
            set(supported_modalities)
            if supported_modalities is not None
            else {
                ModalityType.IMAGE,
                ModalityType.VIDEO_FRAME,
                ModalityType.VOICE_TRANSCRIPT,
                ModalityType.AUDIO_EVENT,
                ModalityType.TELEMETRY,
                ModalityType.TEXT,
                ModalityType.DEVICE_STATE,
            }
        )
        self._elements: List[VisualElement] = list(elements) if elements else []
        self._available = available
        self.should_fail = should_fail
        self.failure_message = failure_message
        self.simulate_status = simulate_status
        self.simulate_error = simulate_error
        self._simulated_evidence = tuple(simulated_evidence) if simulated_evidence is not None else None
        self._health_status = health_status
        self.call_count = 0

    @property
    def provider_id(self) -> str:
        return self._provider_id

    @property
    def provider_version(self) -> str:
        return self._provider_version

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

    def set_elements(self, elements: List[VisualElement]) -> None:
        self._elements = list(elements)

    def set_simulated_evidence(
        self, evidence: Optional[Union[List[PerceptionEvidence], Tuple[PerceptionEvidence, ...]]]
    ) -> None:
        self._simulated_evidence = tuple(evidence) if evidence is not None else None

    def add_element(
        self,
        element_id: str,
        element_type: ElementType,
        x: int,
        y: int,
        width: int,
        height: int,
        text: Optional[str] = None,
        confidence: float = 1.0,
        source: PerceptionSource = PerceptionSource.MANUAL,
    ) -> VisualElement:
        box = BoundingBox(x=x, y=y, width=width, height=height)
        el = VisualElement(
            element_id=element_id,
            element_type=element_type,
            bounding_box=box,
            text=text,
            confidence=confidence,
            source=source,
        )
        self._elements.append(el)
        return el

    def get_health(self) -> Dict[str, Any]:
        return {
            "status": self._health_status,
            "available": self._available,
            "call_count": self.call_count,
            "provider_id": self._provider_id,
            "provider_version": self._provider_version,
            "capabilities_count": len(self._supported_capabilities),
            "modalities_count": len(self._supported_modalities),
        }

    def perceive(
        self,
        request_or_observation: Union[PerceptionRequest, ComputerObservation],
    ) -> Union[PerceptionResult, List[VisualElement]]:
        self.call_count += 1

        # Phase 4.4 Desktop Visual Perception Mode
        if isinstance(request_or_observation, ComputerObservation):
            if self.should_fail:
                raise RuntimeError(self.failure_message)
            return list(self._elements)

        # Phase 6.5a Multimodal Perception Request Mode
        if isinstance(request_or_observation, PerceptionRequest):
            request = request_or_observation

            if self.should_fail:
                raise RuntimeError(self.failure_message)

            metadata = PerceptionMetadata(
                provider_id=self.provider_id,
                provider_version=self.provider_version,
                privacy_classification=request.privacy_constraints,
                processing_time_ms=1.5,
            )

            if self.simulate_status is not None:
                err = self.simulate_error or PerceptionError(
                    code="SIMULATED_FAILURE",
                    message=self.failure_message,
                    provider_id=self.provider_id,
                    request_id=request.request_id,
                    input_id=request.input_data.input_id,
                    recoverable=True,
                )
                return PerceptionResult(
                    request_id=request.request_id,
                    input_id=request.input_data.input_id,
                    status=self.simulate_status,
                    processing_metadata=metadata,
                    errors=(err,),
                )

            # Check modality support
            if request.input_data.modality not in self._supported_modalities:
                err = PerceptionError(
                    code="UNSUPPORTED_MODALITY",
                    message=f"Modality '{request.input_data.modality.value}' is not supported by provider '{self.provider_id}'",
                    provider_id=self.provider_id,
                    request_id=request.request_id,
                    input_id=request.input_data.input_id,
                    recoverable=False,
                )
                return PerceptionResult(
                    request_id=request.request_id,
                    input_id=request.input_data.input_id,
                    status=PerceptionStatus.UNSUPPORTED_MODALITY,
                    processing_metadata=metadata,
                    errors=(err,),
                )

            # Check capability support
            for cap in request.requested_capabilities:
                if cap not in self._supported_capabilities:
                    err = PerceptionError(
                        code="UNSUPPORTED_CAPABILITY",
                        message=f"Capability '{cap.value}' is not supported by provider '{self.provider_id}'",
                        provider_id=self.provider_id,
                        request_id=request.request_id,
                        input_id=request.input_data.input_id,
                        recoverable=False,
                    )
                    return PerceptionResult(
                        request_id=request.request_id,
                        input_id=request.input_data.input_id,
                        status=PerceptionStatus.FAILED,
                        processing_metadata=metadata,
                        errors=(err,),
                    )

            # Return simulated evidence if provided
            if self._simulated_evidence is not None:
                evidence_items = tuple(self._simulated_evidence)
            else:
                # Deterministic synthetic evidence based on requested capabilities
                spatial = None
                if request.input_data.modality in (ModalityType.IMAGE, ModalityType.VIDEO_FRAME):
                    spatial = SpatialEvidence(
                        bounding_box=BoundingBox(x=10, y=20, width=100, height=80),
                        relative_position="center",
                    )

                primary_cap = (
                    request.requested_capabilities[0].value
                    if request.requested_capabilities
                    else "entity"
                )
                provenance = {
                    "source_id": request.input_data.source_id,
                    "provider_id": self.provider_id,
                    "provider_version": self.provider_version,
                    "input_id": request.input_data.input_id,
                    "request_id": request.request_id,
                    "observation_timestamp": request.input_data.captured_at,
                    "correlation_id": request.correlation_id,
                    "causation_id": request.causation_id or "",
                }
                evidence_items = (
                    PerceptionEvidence(
                        evidence_id=f"ev_{request.request_id}_0",
                        semantic_type="detected_entity",
                        label=f"mock_{primary_cap}",
                        confidence=0.95,
                        source_id=self.provider_id,
                        modality=request.input_data.modality,
                        timestamp=request.input_data.captured_at,
                        spatial=spatial,
                        attributes={"capability": primary_cap, "mock": True},
                        provenance=provenance,
                        correlation_id=request.correlation_id,
                        causation_id=request.causation_id,
                    ),
                )

            return PerceptionResult(
                request_id=request.request_id,
                input_id=request.input_data.input_id,
                status=PerceptionStatus.SUCCESS,
                evidence=evidence_items,
                processing_metadata=metadata,
                created_at=time.time(),
            )

        raise TypeError(
            f"Unsupported observation/request type for MockPerceptionProvider: {type(request_or_observation)}"
        )
