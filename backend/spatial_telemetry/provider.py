"""ATLAS Phase 6.5c — Spatial & Telemetry Perception Provider.

Authoritative implementation of PerceptionProviderInterface for:
- ModalityType.GPS
- ModalityType.TELEMETRY
- ModalityType.DEVICE_STATE

Composes SpatialProcessor and TelemetryProcessor into a unified, model-neutral provider.
Strictly isolated from decision-making, WorldState mutation, and device actuation.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from core.interfaces.perception_interface import PerceptionProviderInterface
from core.models.computer import ComputerObservation
from core.models.orchestration import ModalityType
from core.models.perception import (
    PerceptionCapability,
    PerceptionError,
    PerceptionEvidence,
    PerceptionMetadata,
    PerceptionPrivacyClass,
    PerceptionRequest,
    PerceptionResult,
    PerceptionStatus,
    REQUIRED_PROVENANCE_KEYS,
)
from core.models.spatial_telemetry import SpatialTelemetryLimits
from spatial_telemetry.spatial import SpatialProcessor
from spatial_telemetry.telemetry import TelemetryProcessor

logger = logging.getLogger(__name__)


class CallableSet(set):
    def __call__(self) -> "CallableSet":
        return self


class SpatialTelemetryPerceptionProvider(PerceptionProviderInterface):
    """
    Authoritative Spatial and Telemetry Perception Provider for ATLAS Phase 6.5c.
    Observes and normalizes spatial coordinates, relative navigation, and device telemetry.
    """

    def __init__(
        self,
        provider_id: str = "atlas_spatial_telemetry_provider",
        provider_version: str = "1.0.0",
        limits: Optional[SpatialTelemetryLimits] = None,
        spatial_processor: Optional[SpatialProcessor] = None,
        telemetry_processor: Optional[TelemetryProcessor] = None,
        available: bool = True,
    ) -> None:
        self._provider_id = provider_id
        self._provider_version = provider_version
        self.limits = limits or SpatialTelemetryLimits()
        self.spatial_processor = spatial_processor or SpatialProcessor(limits=self.limits)
        self.telemetry_processor = telemetry_processor or TelemetryProcessor(limits=self.limits)
        self._available = available
        self.call_count = 0

        self._supported_capabilities: Set[PerceptionCapability] = {
            PerceptionCapability.SPATIAL_LOCALIZATION,
            PerceptionCapability.TELEMETRY_NORMALIZATION,
        }

        self._supported_modalities: Set[ModalityType] = {
            ModalityType.GPS,
            ModalityType.TELEMETRY,
            ModalityType.DEVICE_STATE,
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
    def capabilities(self) -> CallableSet:
        return CallableSet(self._supported_capabilities)

    @property
    def supported_capabilities(self) -> CallableSet:
        return CallableSet(self._supported_capabilities)

    @property
    def supported_modalities(self) -> CallableSet:
        return CallableSet(self._supported_modalities)

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
            "limits": self.limits.to_dict(),
        }

    def health_check(self) -> Any:
        class HealthResult:
            def __init__(self, d: Dict[str, Any]):
                self.status = d["status"]
                self.provider_id = d["provider_id"]
                self.error = None if d["available"] else "Provider unavailable"
                self.details = d

            def __bool__(self) -> bool:
                return self.status == "HEALTHY"

            def __eq__(self, other: Any) -> bool:
                if isinstance(other, bool):
                    return (self.status == "HEALTHY") == other
                return super().__eq__(other)

            def __getitem__(self, item: str) -> Any:
                return self.details[item]

        return HealthResult(self.get_health())

    def process(
        self, request_or_observation: Union[PerceptionRequest, ComputerObservation]
    ) -> Union[PerceptionResult, List[Any]]:
        """Convenience alias for perceive."""
        return self.perceive(request_or_observation)

    def perceive(
        self,
        request_or_observation: Union[PerceptionRequest, ComputerObservation],
    ) -> Union[PerceptionResult, List[Any]]:
        """
        Process a spatial or telemetry perception request.
        """
        self.call_count += 1

        if isinstance(request_or_observation, ComputerObservation):
            return []

        if not isinstance(request_or_observation, PerceptionRequest):
            raise TypeError(
                f"Unsupported request type for SpatialTelemetryPerceptionProvider: {type(request_or_observation)}"
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
                message=f"Spatial telemetry provider '{self._provider_id}' is currently unavailable",
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
                message=f"Modality '{request.input_data.modality.value}' is not supported by spatial telemetry provider",
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

        # 3. Resolve payload dictionary from metadata or payload_ref
        raw_payload = self._resolve_payload(request)
        if not isinstance(raw_payload, dict):
            err = PerceptionError(
                code="INVALID_PAYLOAD",
                message="Expected dictionary payload for spatial/telemetry perception",
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

        # 4. Mandatory Provenance Construction
        provenance = {
            "source_id": source_id,
            "provider_id": self._provider_id,
            "provider_version": self._provider_version,
            "input_id": inp_id,
            "request_id": req_id,
            "observation_timestamp": request.input_data.captured_at,
            "correlation_id": request.correlation_id or req_id,
            "causation_id": request.causation_id or inp_id,
            "device_id": request.input_data.metadata.get("device_id", source_id),
            "product_type": request.input_data.metadata.get("product_type", "UNKNOWN"),
        }

        all_evidence: List[PerceptionEvidence] = []
        errors: List[PerceptionError] = []
        ref_time = request.parameters.get("reference_time")

        try:
            # A. Process GPS / Position modality
            if request.input_data.modality == ModalityType.GPS or "latitude" in raw_payload:
                _, spatial_evs = self.spatial_processor.process_position(
                    source_id=source_id,
                    raw_payload=raw_payload,
                    captured_at=request.input_data.captured_at,
                    provenance=provenance,
                    request_id=req_id,
                    input_id=inp_id,
                    correlation_id=request.correlation_id,
                    causation_id=request.causation_id,
                )
                all_evidence.extend(spatial_evs)

            # B. Process Telemetry & Device State modality
            if request.input_data.modality in (ModalityType.TELEMETRY, ModalityType.DEVICE_STATE) or (
                "battery" in raw_payload
                or "battery_pct" in raw_payload
                or "metrics" in raw_payload
                or "connectivity" in raw_payload
            ):
                _, telem_evs = self.telemetry_processor.process_telemetry(
                    source_id=source_id,
                    raw_payload=raw_payload,
                    captured_at=request.input_data.captured_at,
                    provenance=provenance,
                    request_id=req_id,
                    input_id=inp_id,
                    reference_time=ref_time,
                    correlation_id=request.correlation_id,
                    causation_id=request.causation_id,
                )
                # If specifically requested DEVICE_STATE or TELEMETRY, filter accordingly
                if request.input_data.modality == ModalityType.DEVICE_STATE:
                    telem_evs = [e for e in telem_evs if e.modality == ModalityType.DEVICE_STATE] or telem_evs
                elif request.input_data.modality == ModalityType.TELEMETRY:
                    telem_evs = [e for e in telem_evs if e.modality == ModalityType.TELEMETRY] or telem_evs
                all_evidence.extend(telem_evs)

        except ValueError as ve:
            errors.append(
                PerceptionError(
                    code="INVALID_DATA",
                    message=str(ve),
                    provider_id=self._provider_id,
                    request_id=req_id,
                    input_id=inp_id,
                    recoverable=False,
                )
            )
        except Exception as ex:
            logger.exception("Unexpected error in spatial telemetry perception: %s", ex)
            errors.append(
                PerceptionError(
                    code="PROCESSING_FAILURE",
                    message=f"Processing failed: {ex}",
                    provider_id=self._provider_id,
                    request_id=req_id,
                    input_id=inp_id,
                    recoverable=True,
                )
            )

        # 5. Cap evidence and evaluate status
        bounded_evidence = tuple(all_evidence[: self.limits.max_batch_size])

        if errors:
            status = PerceptionStatus.INVALID_INPUT if any(e.code == "INVALID_DATA" for e in errors) else PerceptionStatus.FAILED
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

    def _resolve_payload(self, request: PerceptionRequest) -> Optional[Dict[str, Any]]:
        """Resolve raw payload from metadata or payload_ref."""
        meta = request.input_data.metadata or {}
        if "payload" in meta and isinstance(meta["payload"], dict):
            return dict(meta["payload"])
        if "telemetry" in meta and isinstance(meta["telemetry"], dict):
            return dict(meta["telemetry"])
        if "position" in meta and isinstance(meta["position"], dict):
            return dict(meta["position"])

        ref = request.input_data.payload_ref
        if ref and ref.startswith("{") and ref.endswith("}"):
            try:
                return json.loads(ref)
            except Exception:
                pass

        if meta:
            return dict(meta)

        if request.parameters:
            return dict(request.parameters)

        return {}
