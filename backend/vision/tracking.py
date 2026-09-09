"""Bounded Visual Target Tracker (Phase 6.5b)

Maintains temporal target identity across successive detections within a bounded session.
Enforces strict lifecycle expiration and bounded history without claiming global identity.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from core.models.orchestration import ModalityType
from core.models.perception import (
    BoundingBox,
    PerceptionEvidence,
    SpatialEvidence,
)
from vision.limits import VisionLimits

logger = logging.getLogger(__name__)


@dataclass
class TrackState:
    """Internal temporal state for a single visual track."""
    track_id: str
    label: str
    current_box: BoundingBox
    history: List[Tuple[float, float, float]]  # [(cx, cy, timestamp), ...]
    first_seen: float
    last_seen: float
    consecutive_misses: int = 0
    total_hits: int = 1
    confidence: float = 0.9


class VisualTracker:
    """
    Deterministic centroid & IoU matching multi-object tracker.
    Generates structured track evidence and enforces bounded memory.
    """

    def __init__(
        self,
        iou_threshold: float = 0.25,
        max_distance: float = 120.0,
        limits: Optional[VisionLimits] = None,
        max_history: Optional[int] = None,
        max_missing_frames: Optional[int] = None,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.max_distance = max_distance
        self.limits = limits or VisionLimits()

        if max_history is not None:
            self.max_history_points = max_history
        else:
            self.max_history_points = self.limits.max_track_history_points

        if max_missing_frames is not None:
            self.max_missing_frames = max_missing_frames
        else:
            self.max_missing_frames = self.limits.track_expiration_frames

        self._tracks: Dict[str, TrackState] = {}
        self._next_id = 1
        self._lock = threading.RLock()

    def reset(self) -> None:
        """Reset all active tracks."""
        with self._lock:
            self._tracks.clear()
            self._next_id = 1

    def get_active_tracks(self) -> List[TrackState]:
        """Return list of currently active track states."""
        with self._lock:
            return list(self._tracks.values())

    def update(
        self,
        detections: Sequence[Union[PerceptionEvidence, SpatialEvidence, Any]],
        timestamp: Optional[Union[float, Any]] = None,
        source_id: str = "default_cam",
        provenance: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
        input_id: Optional[str] = None,
    ) -> List[PerceptionEvidence]:
        """
        Associate detections with existing tracks, spawn new tracks, and expire stale tracks.

        Returns:
            List of PerceptionEvidence items with semantic_type="visual_track".
        """
        ts = float(timestamp.timestamp() if hasattr(timestamp, "timestamp") else (timestamp or time.time()))
        req_id = request_id or "req_track_auto"
        inp_id = input_id or "inp_track_auto"
        prov = dict(provenance or {})
        prov.setdefault("provider_id", "visual_tracker")
        prov.setdefault("provider_version", "1.0.0")
        prov.setdefault("source_id", source_id)
        prov.setdefault("input_id", inp_id)
        prov.setdefault("request_id", req_id)
        prov.setdefault("observation_timestamp", ts)
        prov.setdefault("correlation_id", req_id)
        prov.setdefault("causation_id", inp_id)

        with self._lock:
            # 1. Filter detections with bounding boxes
            valid_dets: List[Tuple[str, float, BoundingBox, Tuple[float, float]]] = []
            for d in detections:
                box = None
                conf = 0.9
                label = "object"
                if isinstance(d, PerceptionEvidence):
                    if d.spatial and d.spatial.bounding_box:
                        box = d.spatial.bounding_box
                        conf = d.confidence
                        label = d.label
                elif isinstance(d, SpatialEvidence):
                    if d.bounding_box:
                        box = d.bounding_box
                        conf = d.spatial_confidence
                        label = d.region_label or "target"
                elif hasattr(d, "bounding_box") and d.bounding_box:
                    box = d.bounding_box
                    conf = getattr(d, "confidence", 0.9)

                if box is not None:
                    cx = box.x + box.width / 2.0
                    cy = box.y + box.height / 2.0
                    valid_dets.append((label, conf, box, (cx, cy)))

            matched_track_ids: set = set()
            matched_det_indices: set = set()

            # 2. Greedy IoU / Distance matching
            for t_id, track in list(self._tracks.items()):
                best_det_idx = -1
                best_score = -1.0

                tcx, tcy, _ = track.history[-1]

                for d_idx, (_, _, d_box, (dcx, dcy)) in enumerate(valid_dets):
                    if d_idx in matched_det_indices:
                        continue

                    # Calculate IoU
                    iou = track.current_box.iou(d_box)
                    dist = math.hypot(tcx - dcx, tcy - dcy)

                    score = 0.0
                    if iou >= self.iou_threshold:
                        score = 1.0 + iou
                    elif dist <= self.max_distance:
                        score = 1.0 - (dist / self.max_distance)

                    if score > best_score and score > 0.0:
                        best_score = score
                        best_det_idx = d_idx

                if best_det_idx >= 0:
                    matched_det_indices.add(best_det_idx)
                    matched_track_ids.add(t_id)
                    det_label, det_conf, new_box, (dcx, dcy) = valid_dets[best_det_idx]

                    # Update track state
                    track.current_box = new_box
                    track.last_seen = ts
                    track.consecutive_misses = 0
                    track.total_hits += 1
                    track.confidence = round(min(1.0, max(0.5, det_conf)), 3)
                    track.history.append((dcx, dcy, ts))
                    if len(track.history) > self.max_history_points:
                        track.history.pop(0)
                else:
                    track.consecutive_misses += 1

            # 3. Spawn new tracks for unmatched detections
            for d_idx, (det_label, det_conf, box, (dcx, dcy)) in enumerate(valid_dets):
                if d_idx in matched_det_indices:
                    continue

                if len(self._tracks) >= self.limits.max_active_tracks:
                    # Evict oldest track
                    oldest = min(self._tracks, key=lambda k: self._tracks[k].last_seen)
                    del self._tracks[oldest]

                t_id = f"track_{self._next_id:03d}"
                self._next_id += 1

                new_track = TrackState(
                    track_id=t_id,
                    label=det_label,
                    current_box=box,
                    history=[(dcx, dcy, ts)],
                    first_seen=ts,
                    last_seen=ts,
                    consecutive_misses=0,
                    total_hits=1,
                    confidence=round(det_conf, 3),
                )
                self._tracks[t_id] = new_track
                matched_track_ids.add(t_id)

            # 4. Expire inactive tracks
            expired_ids = []
            for t_id, track in self._tracks.items():
                if track.consecutive_misses > self.max_missing_frames:
                    expired_ids.append(t_id)
                elif ts - track.last_seen > self.limits.track_expiration_seconds:
                    expired_ids.append(t_id)

            for t_id in expired_ids:
                del self._tracks[t_id]

            # 5. Emit PerceptionEvidence for active tracks
            track_evidence: List[PerceptionEvidence] = []
            for t_id, track in self._tracks.items():
                is_active = track.consecutive_misses == 0
                status_str = "active" if is_active else "occluded"

                spatial = SpatialEvidence(
                    bounding_box=track.current_box,
                    region_label=t_id,
                    spatial_confidence=track.confidence,
                )

                ev = PerceptionEvidence(
                    evidence_id=f"ev_trk_{req_id}_{t_id}",
                    semantic_type="visual_track",
                    label=f"{track.label}_{t_id}",
                    confidence=track.confidence,
                    source_id=source_id,
                    modality=ModalityType.VIDEO_FRAME,
                    timestamp=ts,
                    spatial=spatial,
                    attributes={
                        "track_id": t_id,
                        "status": status_str,
                        "track_status": status_str,
                        "track_length": len(track.history),
                        "total_hits": track.total_hits,
                        "consecutive_misses": track.consecutive_misses,
                        "trajectory_length": len(track.history),
                    },
                    provenance=prov,
                    correlation_id=req_id,
                    causation_id=inp_id,
                )
                track_evidence.append(ev)

            return track_evidence
