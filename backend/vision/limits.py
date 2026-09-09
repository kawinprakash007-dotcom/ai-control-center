"""
Vision Subsystem Limits & Constraints (Phase 6.5b)

Defines upper bounds for image dimensions, payload sizes, tracking histories,
detection regions, and temporal horizons to enforce deterministic resource utilization.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class VisionLimits:
    """Explicit bounded limits for visual perception processing."""

    # Dimension and payload limits
    max_image_width: int = 4096
    max_image_height: int = 4096
    max_pixel_count: int = 16_777_216  # 16 MP
    max_frame_bytes: int = 10 * 1024 * 1024  # 10 MB

    # Detection and tracking limits
    max_detected_regions: int = 50
    max_evidence_items: int = 100
    max_active_tracks: int = 50
    max_track_history_points: int = 10
    track_expiration_frames: int = 5
    track_expiration_seconds: float = 30.0

    # OCR and scene analysis limits
    max_ocr_text_length: int = 1024
    max_scene_attributes: int = 32

    # Temporal limits
    max_staleness_seconds: float = 300.0
