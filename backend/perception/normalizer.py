"""
Perception Observation Normalizer (Phase 6.5a)

Bridges perception outputs to canonical MultimodalObservation instances
for ingestion into CentralInputGateway without violating architectural layers.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Sequence

from core.interfaces.perception_interface import (
    PerceptionObservationNormalizerInterface,
)
from core.models.orchestration import MultimodalObservation
from core.models.perception import PerceptionResult

logger = logging.getLogger(__name__)


class PerceptionObservationNormalizer(PerceptionObservationNormalizerInterface):
    """
    Normalizes structured PerceptionResult and PerceptionEvidence into canonical
    MultimodalObservation objects ready for CentralInputGateway intake.
    """

    def normalize(
        self,
        result: PerceptionResult,
        evidence_index: int = 0,
    ) -> MultimodalObservation:
        """
        Normalize a specific evidence item within a perception result to a canonical
        MultimodalObservation.

        Raises:
            ValueError: If result is invalid or unsuccessful.
            IndexError: If evidence_index is out of range.
        """
        if not isinstance(result, PerceptionResult):
            raise ValueError(f"Expected PerceptionResult, got {type(result)}")

        if not result.is_success():
            err_msg = result.errors[0].message if result.errors else "Unknown error"
            raise ValueError(
                f"Cannot normalize unsuccessful perception result (status={result.status.value}): {err_msg}"
            )

        if not result.evidence:
            raise IndexError("No evidence in perception result to normalize")

        if evidence_index < 0 or evidence_index >= len(result.evidence):
            raise IndexError(
                f"Evidence index {evidence_index} out of range (total evidence: {len(result.evidence)})"
            )

        evidence = result.evidence[evidence_index]

        # Extract provenance hints if available
        source_id = evidence.source_id or result.processing_metadata.provider_id
        source_type = str(evidence.provenance.get("source_type", "perception_provider"))
        device_id = evidence.provenance.get("device_id")
        correlation_id = evidence.correlation_id or result.request_id
        causation_id = evidence.causation_id or result.input_id

        # Normalize timestamp (ensure strictly positive)
        ts = evidence.timestamp if evidence.timestamp > 0.0 else result.created_at
        if ts <= 0.0:
            ts = 1.0

        # Build payload
        payload: Dict[str, Any] = {
            "semantic_type": evidence.semantic_type,
            "label": evidence.label,
            "attributes": dict(evidence.attributes),
            "spatial": evidence.spatial.to_dict() if evidence.spatial else None,
        }

        # Build normalized metadata
        privacy_val = (
            result.processing_metadata.privacy_classification.value
            if hasattr(result.processing_metadata.privacy_classification, "value")
            else str(result.processing_metadata.privacy_classification)
        )

        metadata: Dict[str, Any] = {
            "provider_id": result.processing_metadata.provider_id,
            "provider_version": result.processing_metadata.provider_version,
            "model_id": result.processing_metadata.model_id,
            "privacy_class": privacy_val,
            "request_id": result.request_id,
            "input_id": result.input_id,
            "processing_time_ms": result.processing_metadata.processing_time_ms,
            "provenance": dict(evidence.provenance),
        }

        # Extract location if available in spatial evidence
        location = evidence.spatial.location if evidence.spatial else None

        observation = MultimodalObservation(
            observation_id=f"obs_{result.request_id}_{evidence.evidence_id}",
            source_id=source_id,
            source_type=source_type,
            modality=evidence.modality,
            timestamp=ts,
            payload=payload,
            confidence=evidence.confidence,
            location=location,
            device_id=device_id,
            correlation_id=correlation_id,
            causation_id=causation_id,
            metadata=metadata,
        )

        return observation

    def normalize_all(
        self,
        result: PerceptionResult,
    ) -> Sequence[MultimodalObservation]:
        """
        Normalize all evidence items in a perception result to a sequence of
        MultimodalObservation objects. Returns empty sequence if result has no evidence.
        """
        if not isinstance(result, PerceptionResult):
            raise ValueError(f"Expected PerceptionResult, got {type(result)}")

        if not result.is_success():
            err_msg = result.errors[0].message if result.errors else "Unknown error"
            raise ValueError(
                f"Cannot normalize unsuccessful perception result (status={result.status.value}): {err_msg}"
            )

        if not result.evidence:
            return []

        return [self.normalize(result, i) for i in range(len(result.evidence))]
