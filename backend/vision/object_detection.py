"""
Visual Region & Object Detection Boundary (Phase 6.5b)

Model-neutral detector strictly distinguishing DETECTED_REGION (classical contour/color
segment) from SEMANTIC_OBJECT (explicitly verified semantic entities).
Clamps all bounding boxes within frame dimensions.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from core.models.orchestration import ModalityType
from core.models.perception import (
    BoundingBox,
    PerceptionEvidence,
    SpatialEvidence,
)
from vision.limits import VisionLimits

logger = logging.getLogger(__name__)


class VisualRegionDetector:
    """
    Model-neutral visual detector. Emits DETECTED_REGION for classical CV contours,
    or SEMANTIC_OBJECT when deterministic fixtures/classifiers are supplied.
    """

    def __init__(
        self,
        min_region_area: int = 250,
        max_regions: int = 20,
        limits: Optional[VisionLimits] = None,
        min_area: Optional[int] = None,
        fixtures: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.min_region_area = int(min_area if min_area is not None else min_region_area)
        self.max_regions = max_regions
        self.limits = limits or VisionLimits()

        # Deterministic fixtures for simulation/testing: pattern_name -> list of (label, bbox, confidence, attributes)
        self._semantic_fixtures: Dict[str, List[Dict[str, Any]]] = {}
        if fixtures:
            for k, v in fixtures.items():
                if isinstance(v, list):
                    self._semantic_fixtures[k] = v
                elif isinstance(v, dict):
                    self._semantic_fixtures[k] = [v]

    def register_semantic_fixture(self, fixture_key: str, objects: List[Dict[str, Any]]) -> None:
        """Register deterministic semantic objects for simulation/testing."""
        self._semantic_fixtures[fixture_key] = objects

    def clear_fixtures(self) -> None:
        """Clear all registered fixtures."""
        self._semantic_fixtures.clear()

    def detect(
        self,
        frame_bgr: np.ndarray,
        source_id: str = "default_cam",
        timestamp: Optional[Union[float, Any]] = None,
        provenance: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
        input_id: Optional[str] = None,
        fixture_key: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> List[PerceptionEvidence]:
        """
        Detect visual regions and semantic objects in the frame.

        Returns:
            List of PerceptionEvidence items.
        """
        import time
        ts = float(timestamp.timestamp() if hasattr(timestamp, "timestamp") else (timestamp or time.time()))
        req_id = request_id or "req_region_auto"
        inp_id = input_id or "inp_region_auto"
        prov = dict(provenance or {})
        prov.setdefault("provider_id", "visual_region_detector")
        prov.setdefault("provider_version", "1.0.0")
        prov.setdefault("source_id", source_id)
        prov.setdefault("input_id", inp_id)
        prov.setdefault("request_id", req_id)
        prov.setdefault("observation_timestamp", ts)
        prov.setdefault("correlation_id", req_id)
        prov.setdefault("causation_id", inp_id)
        provenance = prov
        timestamp = ts
        request_id = req_id
        input_id = inp_id
        params = parameters or {}
        frame_h, frame_w = frame_bgr.shape[:2]
        evidence_list: List[PerceptionEvidence] = []

        # 1. Check for registered semantic fixtures first
        effective_key = fixture_key or params.get("fixture_key")
        if effective_key and effective_key in self._semantic_fixtures:
            fixture_items = self._semantic_fixtures[effective_key]
            for idx, item in enumerate(fixture_items[: self.max_regions]):
                box_dict = item.get("box", {})
                x = max(0, min(int(box_dict.get("x", 0)), frame_w - 1))
                y = max(0, min(int(box_dict.get("y", 0)), frame_h - 1))
                w = max(1, min(int(box_dict.get("width", 50)), frame_w - x))
                h = max(1, min(int(box_dict.get("height", 50)), frame_h - y))

                conf = float(item.get("confidence", 0.9))
                conf = min(1.0, max(0.0, conf))

                label = str(item.get("label", "semantic_object"))
                bbox = BoundingBox(x=x, y=y, width=w, height=h)
                spatial = SpatialEvidence(
                    bounding_box=bbox,
                    region_label=f"obj_{idx}",
                    spatial_confidence=conf,
                )

                ev_prov = dict(provenance)
                ev_prov.setdefault("source_id", source_id)
                ev_prov.setdefault("input_id", input_id)
                ev_prov.setdefault("request_id", request_id)
                ev_prov.setdefault("observation_timestamp", timestamp)
                ev_prov.setdefault("correlation_id", request_id)
                ev_prov.setdefault("causation_id", input_id)

                ev = PerceptionEvidence(
                    evidence_id=f"ev_obj_{request_id}_{idx}",
                    semantic_type="semantic_object",
                    label=label,
                    confidence=conf,
                    source_id=source_id,
                    modality=ModalityType.IMAGE,
                    timestamp=timestamp,
                    spatial=spatial,
                    attributes={
                        "detection_method": "semantic_fixture",
                        "object_class": label,
                        "extra": item.get("attributes", {}),
                    },
                    provenance=ev_prov,
                    correlation_id=request_id,
                    causation_id=input_id,
                )
                evidence_list.append(ev)

            return evidence_list

        # 2. Classical CV Region Detection (Foreground / Salient Regions)
        # Convert to grayscale & compute adaptive edge/threshold saliency
        if len(frame_bgr.shape) == 3:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame_bgr.copy()

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY)
        edges = cv2.Canny(blurred, 50, 150)
        combined = cv2.bitwise_or(thresh, edges)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        valid_regions: List[Tuple[int, int, int, int, float]] = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_region_area:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            x = max(0, min(x, frame_w - 1))
            y = max(0, min(y, frame_h - 1))
            w = max(1, min(w, frame_w - x))
            h = max(1, min(h, frame_h - y))
            valid_regions.append((x, y, w, h, float(area)))

        valid_regions.sort(key=lambda r: r[4], reverse=True)
        valid_regions = valid_regions[: self.max_regions]

        for idx, (x, y, w, h, area) in enumerate(valid_regions):
            bbox = BoundingBox(x=x, y=y, width=w, height=h)
            spatial = SpatialEvidence(
                bounding_box=bbox,
                region_label=f"region_{idx}",
                spatial_confidence=0.85,
            )

            ev_prov = dict(provenance)
            ev_prov.setdefault("source_id", source_id)
            ev_prov.setdefault("input_id", input_id)
            ev_prov.setdefault("request_id", request_id)
            ev_prov.setdefault("observation_timestamp", timestamp)
            ev_prov.setdefault("correlation_id", request_id)
            ev_prov.setdefault("causation_id", input_id)

            # Clearly labeled as DETECTED_REGION, NOT semantic object
            ev = PerceptionEvidence(
                evidence_id=f"ev_reg_{request_id}_{idx}",
                semantic_type="detected_region",
                label="foreground_region",
                confidence=0.85,
                source_id=source_id,
                modality=ModalityType.IMAGE,
                timestamp=timestamp,
                spatial=spatial,
                attributes={
                    "detection_method": "classical_contour",
                    "region_area": area,
                    "region_index": idx,
                },
                provenance=ev_prov,
                correlation_id=request_id,
                causation_id=input_id,
            )
            evidence_list.append(ev)

        return evidence_list
