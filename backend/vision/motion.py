"""
Deterministic Motion Detector (Phase 6.5b)

Extracts objective motion evidence via frame differencing, grayscale difference,
thresholding, and contour analysis without deep learning or intent inference.
Enforces the invariant: reports 'motion_detected', NEVER 'intruder'.
"""

from __future__ import annotations

import logging
import threading
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


class MotionDetector:
    """
    Deterministic frame-differencing motion detector.
    Tracks prior frame baselines per source_id in a bounded cache.
    """

    def __init__(
        self,
        diff_threshold: int = 25,
        min_motion_area: int = 150,
        max_regions: int = 10,
        limits: Optional[VisionLimits] = None,
        threshold: Optional[int] = None,
        min_area: Optional[int] = None,
    ) -> None:
        self.diff_threshold = int(threshold if threshold is not None else diff_threshold)
        self.min_motion_area = int(min_area if min_area is not None else min_motion_area)
        self.max_regions = max_regions
        self.limits = limits or VisionLimits()

        # Bounded cache of previous frame grayscale arrays: source_id -> (gray_arr, timestamp)
        self._cache: Dict[str, Tuple[np.ndarray, float]] = {}
        self._lock = threading.RLock()

    def reset_source(self, source_id: str) -> None:
        """Clear cached prior frame for a given source."""
        with self._lock:
            self._cache.pop(source_id, None)

    def clear(self) -> None:
        """Clear all cached baselines."""
        with self._lock:
            self._cache.clear()

    def detect_motion(
        self,
        current_bgr: np.ndarray,
        source_id: str = "default_cam",
        timestamp: Optional[Union[float, Any]] = None,
        provenance: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
        input_id: Optional[str] = None,
        previous_bgr: Optional[np.ndarray] = None,
    ) -> List[PerceptionEvidence]:
        """Convenience alias for detect with default metadata for testability."""
        import time
        ts = float(timestamp.timestamp() if hasattr(timestamp, "timestamp") else (timestamp or time.time()))
        req_id = request_id or "req_motion_auto"
        inp_id = input_id or "inp_motion_auto"
        prov = dict(provenance or {})
        prov.setdefault("provider_id", "motion_detector")
        prov.setdefault("provider_version", "1.0.0")
        prov.setdefault("source_id", source_id)
        prov.setdefault("input_id", inp_id)
        prov.setdefault("request_id", req_id)
        prov.setdefault("observation_timestamp", ts)
        prov.setdefault("correlation_id", req_id)
        prov.setdefault("causation_id", inp_id)
        return self.detect(
            current_bgr=current_bgr,
            source_id=source_id,
            timestamp=ts,
            provenance=prov,
            request_id=req_id,
            input_id=inp_id,
            previous_bgr=previous_bgr,
        )

    def detect(
        self,
        current_bgr: np.ndarray,
        source_id: str,
        timestamp: float,
        provenance: Dict[str, Any],
        request_id: str,
        input_id: str,
        previous_bgr: Optional[np.ndarray] = None,
    ) -> List[PerceptionEvidence]:
        """
        Detect motion between current frame and prior frame.

        Returns:
            List of PerceptionEvidence items with semantic_type="motion", label="motion_detected".
        """
        # Convert current frame to grayscale
        if len(current_bgr.shape) == 3:
            curr_gray = cv2.cvtColor(current_bgr, cv2.COLOR_BGR2GRAY)
        else:
            curr_gray = current_bgr.copy()

        # Blur to suppress high-frequency noise
        curr_gray = cv2.GaussianBlur(curr_gray, (5, 5), 0)

        prev_gray: Optional[np.ndarray] = None

        with self._lock:
            if previous_bgr is not None:
                if len(previous_bgr.shape) == 3:
                    prev_gray = cv2.cvtColor(previous_bgr, cv2.COLOR_BGR2GRAY)
                else:
                    prev_gray = previous_bgr.copy()
                prev_gray = cv2.GaussianBlur(prev_gray, (5, 5), 0)
            elif source_id in self._cache:
                prev_gray, _ = self._cache[source_id]

            # Update cache with current frame (bounded cache eviction)
            if len(self._cache) >= self.limits.max_active_tracks:
                # Evict oldest entry
                oldest_key = min(self._cache, key=lambda k: self._cache[k][1])
                del self._cache[oldest_key]
            self._cache[source_id] = (curr_gray, timestamp)

        # First frame baseline establishes reference: zero motion
        if prev_gray is None:
            return []

        # Ensure dimensions match before differencing
        if prev_gray.shape != curr_gray.shape:
            # Dimension mismatch (e.g. camera mode changed): reset baseline
            return []

        # 1. Compute absolute difference
        frame_diff = cv2.absdiff(prev_gray, curr_gray)

        # 2. Thresholding
        _, thresh = cv2.threshold(frame_diff, self.diff_threshold, 255, cv2.THRESH_BINARY)

        # 3. Dilate thresholded image to fill holes
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations=2)

        # 4. Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        frame_h, frame_w = curr_gray.shape[:2]
        frame_area = frame_h * frame_w
        evidence_list: List[PerceptionEvidence] = []

        valid_boxes: List[Tuple[int, int, int, int, float]] = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_motion_area:
                continue

            x, y, w, h = cv2.boundingRect(cnt)

            # Clamp coordinates to frame bounds
            x = max(0, min(x, frame_w - 1))
            y = max(0, min(y, frame_h - 1))
            w = max(1, min(w, frame_w - x))
            h = max(1, min(h, frame_h - y))

            valid_boxes.append((x, y, w, h, float(area)))

        # Sort by area descending and cap to max_regions
        valid_boxes.sort(key=lambda b: b[4], reverse=True)
        valid_boxes = valid_boxes[: self.max_regions]

        for idx, (x, y, w, h, area) in enumerate(valid_boxes):
            # Deterministic bounded confidence: normalized scale between 0.65 and 0.98
            area_ratio = min(1.0, area / max(1.0, frame_area * 0.1))
            confidence = round(min(0.98, max(0.65, 0.65 + 0.33 * area_ratio)), 3)

            bbox = BoundingBox(x=x, y=y, width=w, height=h)
            spatial = SpatialEvidence(
                bounding_box=bbox,
                region_label=f"motion_region_{idx}",
                spatial_confidence=confidence,
            )

            # Ensure complete provenance keys
            ev_provenance = dict(provenance)
            ev_provenance.setdefault("source_id", source_id)
            ev_provenance.setdefault("input_id", input_id)
            ev_provenance.setdefault("request_id", request_id)
            ev_provenance.setdefault("observation_timestamp", timestamp)
            ev_provenance.setdefault("correlation_id", request_id)
            ev_provenance.setdefault("causation_id", input_id)

            ev = PerceptionEvidence(
                evidence_id=f"ev_mot_{request_id}_{idx}",
                semantic_type="motion",
                label="motion_detected",
                confidence=confidence,
                source_id=source_id,
                modality=ModalityType.VIDEO_FRAME,
                timestamp=timestamp,
                spatial=spatial,
                attributes={
                    "motion_area": area,
                    "frame_coverage_ratio": round(area / max(1.0, frame_area), 4),
                    "region_index": idx,
                },
                provenance=ev_provenance,
                correlation_id=request_id,
                causation_id=input_id,
            )
            evidence_list.append(ev)

        return evidence_list
