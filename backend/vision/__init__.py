"""ATLAS Phase 6.5b — Visual Perception Module.

This module provides model-neutral, hardware-neutral, deterministic, and bounded
visual perception capabilities for ATLAS.

It exposes concrete visual perception providers, frame loaders, and sub-detectors
for motion, region/object detection, scene analysis, tracking, and OCR.
"""

from backend.vision.limits import VisionLimits
from backend.vision.frame_loader import (
    load_and_validate_frame,
    validate_input_modality,
    create_synthetic_image,
    FrameLoadingError,
)
from backend.vision.motion import MotionDetector
from backend.vision.object_detection import VisualRegionDetector
from backend.vision.scene_analysis import SceneAnalyzer
from backend.vision.tracking import VisualTracker
from backend.vision.ocr import VisualOCRDetector
from backend.vision.provider import VisualPerceptionProvider
from backend.vision.reference_provider import (
    MockVisualProvider,
    ReferenceVisualProvider,
)

__all__ = [
    "VisionLimits",
    "load_and_validate_frame",
    "validate_input_modality",
    "create_synthetic_image",
    "FrameLoadingError",
    "MotionDetector",
    "VisualRegionDetector",
    "SceneAnalyzer",
    "VisualTracker",
    "VisualOCRDetector",
    "VisualPerceptionProvider",
    "MockVisualProvider",
    "ReferenceVisualProvider",
]
