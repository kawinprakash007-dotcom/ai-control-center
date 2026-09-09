"""ATLAS Phase 6.5d — Modality Compatibility, Source Diversity & Contradiction Subsystem.

Provides:
- Model-neutral modality compatibility evaluation
- Source diversity calculation (preventing single-source domination)
- Contradiction identification between opposing observations
- Corroboration and support scoring across independent perception streams
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from core.models.multimodal_fusion import (
    EvidenceRelationType,
    ModalityCompatibilityLevel,
)
from core.models.orchestration import ModalityType


# General modality categorizations
_SPATIAL_MODALITIES: Set[ModalityType] = {ModalityType.GPS}
_VISUAL_MODALITIES: Set[ModalityType] = {ModalityType.IMAGE, ModalityType.VIDEO_FRAME}
_TELEMETRY_MODALITIES: Set[ModalityType] = {ModalityType.TELEMETRY, ModalityType.DEVICE_STATE}
_ACOUSTIC_MODALITIES: Set[ModalityType] = {ModalityType.AUDIO_EVENT, ModalityType.VOICE_TRANSCRIPT}
_LINGUISTIC_MODALITIES: Set[ModalityType] = {ModalityType.TEXT, ModalityType.VOICE_TRANSCRIPT, ModalityType.USER_ACTION}


def evaluate_modality_compatibility(
    mod1: ModalityType,
    mod2: ModalityType,
) -> ModalityCompatibilityLevel:
    """
    Evaluate the semantic compatibility level between two modalities.
    Complementary modalities provide mutual corroboration (e.g. vision + gps, vision + audio).
    """
    if mod1 == mod2:
        return ModalityCompatibilityLevel.COMPATIBLE

    # Visual + Spatial / Telemetry / Acoustic are complementary
    if (mod1 in _VISUAL_MODALITIES and mod2 in _SPATIAL_MODALITIES) or (
        mod2 in _VISUAL_MODALITIES and mod1 in _SPATIAL_MODALITIES
    ):
        return ModalityCompatibilityLevel.COMPLEMENTARY

    if (mod1 in _VISUAL_MODALITIES and mod2 in _TELEMETRY_MODALITIES) or (
        mod2 in _VISUAL_MODALITIES and mod1 in _TELEMETRY_MODALITIES
    ):
        return ModalityCompatibilityLevel.COMPLEMENTARY

    if (mod1 in _VISUAL_MODALITIES and mod2 in _ACOUSTIC_MODALITIES) or (
        mod2 in _VISUAL_MODALITIES and mod1 in _ACOUSTIC_MODALITIES
    ):
        return ModalityCompatibilityLevel.COMPLEMENTARY

    # GPS + Telemetry are complementary
    if (mod1 in _SPATIAL_MODALITIES and mod2 in _TELEMETRY_MODALITIES) or (
        mod2 in _SPATIAL_MODALITIES and mod1 in _TELEMETRY_MODALITIES
    ):
        return ModalityCompatibilityLevel.COMPLEMENTARY

    # Acoustic + Linguistic are complementary
    if (mod1 in _ACOUSTIC_MODALITIES and mod2 in _LINGUISTIC_MODALITIES) or (
        mod2 in _ACOUSTIC_MODALITIES and mod1 in _LINGUISTIC_MODALITIES
    ):
        return ModalityCompatibilityLevel.COMPLEMENTARY

    return ModalityCompatibilityLevel.COMPATIBLE


def calculate_source_diversity(source_ids: Sequence[str]) -> float:
    """
    Calculate source diversity score in [0.0, 1.0].
    Guarantees that multiple observations from the same source (e.g. 4 camera frames)
    do not receive the same weight as cross-product corroboration (e.g. camera + drone).
    """
    if not source_ids:
        return 0.0
    unique_count = len(set(source_ids))
    total_count = len(source_ids)
    if total_count <= 1:
        return 1.0
    if unique_count <= 1:
        return 0.0
    return round((unique_count - 1) / (total_count - 1), 4)


def detect_contradiction(
    label_1: str,
    label_2: str,
    attributes_1: Dict[str, Any],
    attributes_2: Dict[str, Any],
) -> Tuple[bool, str]:
    """
    Check if two observations in the same spatiotemporal context contradict each other.
    Example:
    - One reports 'person_detected' while another reports 'no_person_detected' or 'clear'.
    - One reports 'door_open' while another reports 'door_closed'.
    """
    l1 = str(label_1 or "").strip().lower()
    l2 = str(label_2 or "").strip().lower()

    # Direct negations across labels or attributes
    contradictory_pairs = [
        ("detected", "not_detected"),
        ("present", "absent"),
        ("occupied", "clear"),
        ("open", "closed"),
        ("moving", "stationary"),
        ("online", "offline"),
        ("healthy", "fault"),
        ("clear", "blocked"),
        ("clear", "obstacle"),
        ("normal", "error"),
        ("normal", "critical"),
        ("nominal", "fault"),
        ("nominal", "critical"),
        ("ok", "fail"),
    ]

    for pos, neg in contradictory_pairs:
        if (pos in l1 and neg in l2) or (neg in l1 and pos in l2):
            return True, f"Opposing condition labels: '{l1}' vs '{l2}'"

    # Status attribute contradiction
    s1 = str(attributes_1.get("status", "")).strip().lower()
    s2 = str(attributes_2.get("status", "")).strip().lower()
    if s1 and s2 and s1 != s2:
        for pos, neg in contradictory_pairs:
            if (pos in s1 and neg in s2) or (neg in s1 and pos in s2):
                return True, f"Opposing status attributes: '{s1}' vs '{s2}'"

    # Attribute-level presence contradiction
    p1 = attributes_1.get("present", attributes_1.get("detected", attributes_1.get("is_present")))
    p2 = attributes_2.get("present", attributes_2.get("detected", attributes_2.get("is_present")))
    if p1 is not None and p2 is not None and isinstance(p1, bool) and isinstance(p2, bool):
        if p1 != p2:
            return True, f"Direct boolean contradiction: {p1} vs {p2}"

    # Speed attribute contradiction (e.g. 10.0 m/s moving vs 0.0 stopped)
    sp1 = attributes_1.get("speed")
    sp2 = attributes_2.get("speed")
    if sp1 is not None and sp2 is not None:
        try:
            v1, v2 = float(sp1), float(sp2)
            if (v1 > 5.0 and v2 == 0.0) or (v2 > 5.0 and v1 == 0.0):
                return True, f"Contradictory speed attributes: {v1} vs {v2}"
        except (ValueError, TypeError):
            pass

    return False, ""
