"""Lightweight Scene Analyzer (Phase 6.5b)

Deterministic, bounded scene property extraction based on perceptual luminance,
spatial color balance, and edge density. Does not claim VLM capabilities.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np

from core.models.orchestration import ModalityType
from core.models.perception import PerceptionEvidence
from vision.limits import VisionLimits

logger = logging.getLogger(__name__)


class SceneEvidenceList(list):
    """List of scene evidences providing transparent single-item property access."""

    @property
    def evidence_type(self) -> Any:
        return getattr(self[0], "semantic_type", None) if self else None

    @property
    def semantic_type(self) -> str:
        return self[0].semantic_type if self else ""

    @property
    def label(self) -> str:
        return self[0].label if self else ""

    @property
    def semantic_label(self) -> str:
        return self[0].label if self else ""

    @property
    def attributes(self) -> Dict[str, Any]:
        return self[0].attributes if self else {}

    @property
    def metadata(self) -> Dict[str, Any]:
        return self[0].attributes if self else {}

    @property
    def confidence(self) -> float:
        return self[0].confidence if self else 0.0


class SceneAnalyzer:
    """
    Lightweight scene analysis computing illumination, basic indoor/outdoor heuristic,
    and visual complexity without deep neural networks.
    """

    def __init__(self, limits: Optional[VisionLimits] = None) -> None:
        self.limits = limits or VisionLimits()

    def analyze(
        self,
        frame_bgr: np.ndarray,
        source_id: str = "cam_scene",
        timestamp: Optional[Union[float, Any]] = None,
        provenance: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
        input_id: Optional[str] = None,
    ) -> SceneEvidenceList:
        """
        Analyze scene illumination, environment estimation, and visual density.

        Returns:
            SceneEvidenceList with PerceptionEvidence items.
        """
        ts = float(timestamp.timestamp() if hasattr(timestamp, "timestamp") else (timestamp or time.time()))
        req_id = request_id or "req_scene_auto"
        inp_id = input_id or "inp_scene_auto"
        prov = dict(provenance or {})
        prov.setdefault("provider_id", "scene_analyzer")
        prov.setdefault("provider_version", "1.0.0")
        prov.setdefault("source_id", source_id)
        prov.setdefault("input_id", inp_id)
        prov.setdefault("request_id", req_id)
        prov.setdefault("observation_timestamp", ts)
        prov.setdefault("correlation_id", req_id)
        prov.setdefault("causation_id", inp_id)

        frame_h, frame_w = frame_bgr.shape[:2]

        # 1. Luminance computation (OpenCV BGR -> Grayscale)
        if len(frame_bgr.shape) == 3:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame_bgr.copy()

        mean_lum = float(np.mean(gray))
        std_lum = float(np.std(gray))

        if mean_lum < 60.0:
            brightness_class = "dark"
        elif mean_lum > 185.0:
            brightness_class = "bright"
        else:
            brightness_class = "moderate"

        # 2. Indoor vs outdoor heuristic (Sky / top gradient vs bottom floor)
        top_half = gray[: max(1, frame_h // 3), :]
        bottom_half = gray[max(1, frame_h // 3) :, :]
        top_mean = float(np.mean(top_half))
        bot_mean = float(np.mean(bottom_half))

        is_outdoor = False
        if len(frame_bgr.shape) == 3:
            top_bgr = frame_bgr[: max(1, frame_h // 3), :]
            mean_b = float(np.mean(top_bgr[:, :, 0]))
            mean_r = float(np.mean(top_bgr[:, :, 2]))
            if (top_mean > bot_mean + 20) and (mean_b > mean_r + 10):
                is_outdoor = True
            elif top_mean > bot_mean + 40:
                is_outdoor = True
        elif top_mean > bot_mean + 40:
            is_outdoor = True

        env_estimate = "outdoor_like" if is_outdoor else "indoor_like"

        # 3. Visual complexity / clutter (Laplacian variance)
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if lap_var < 50.0:
            density_class = "empty"
        elif lap_var > 300.0:
            density_class = "cluttered"
        else:
            density_class = "moderate_activity"

        label = f"{brightness_class}_{env_estimate}"

        evidence = PerceptionEvidence(
            evidence_id=f"ev_scn_{req_id}_0",
            semantic_type="scene_analysis",
            label=label,
            confidence=0.88,
            source_id=source_id,
            modality=ModalityType.IMAGE,
            timestamp=ts,
            attributes={
                "brightness": brightness_class,
                "luminance_class": brightness_class,
                "environment_estimate": env_estimate,
                "density": density_class,
                "activity_level": density_class,
                "mean_luminance": round(mean_lum, 2),
                "contrast_std": round(std_lum, 2),
                "laplacian_variance": round(lap_var, 2),
            },
            provenance=prov,
            correlation_id=req_id,
            causation_id=inp_id,
        )

        return SceneEvidenceList([evidence])
