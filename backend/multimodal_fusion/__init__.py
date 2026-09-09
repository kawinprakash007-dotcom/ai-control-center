"""ATLAS Phase 6.5d — Temporal & Cross-Modal Fusion Module.

Provides pre-situation evidence organization across time, modality, space, and source:
- Temporal alignment & bounded temporal windows
- Geodesy-based spatial evidence linking (reusing Phase 6.5c)
- Modality compatibility & source diversity scoring
- Contradiction exposure without WorldState mutation
- Deterministic fusion clustering
- Emission of typed FusionResult objects
"""

from core.models.multimodal_fusion import (
    CrossModalCorrelation,
    EvidenceRelationship,
    EvidenceRelationType,
    FusionCluster,
    FusionFreshnessStatus,
    FusionLimits,
    FusionResult,
    ModalityCompatibilityLevel,
    SpatialEvidenceLink,
    TemporalRelation,
    TemporalWindow,
)
from multimodal_fusion.compatibility import (
    calculate_source_diversity,
    detect_contradiction,
    evaluate_modality_compatibility,
)
from multimodal_fusion.engine import TemporalCrossModalFusionEngine
from multimodal_fusion.spatial import correlate_spatial_evidence
from multimodal_fusion.temporal import (
    build_temporal_window,
    evaluate_temporal_relation,
    filter_by_temporal_window,
)

__all__ = [
    # Enums
    "TemporalRelation",
    "EvidenceRelationType",
    "ModalityCompatibilityLevel",
    "FusionFreshnessStatus",
    # Domain Models
    "FusionLimits",
    "TemporalWindow",
    "SpatialEvidenceLink",
    "EvidenceRelationship",
    "CrossModalCorrelation",
    "FusionCluster",
    "FusionResult",
    # Subsystem Functions
    "evaluate_temporal_relation",
    "build_temporal_window",
    "filter_by_temporal_window",
    "correlate_spatial_evidence",
    "evaluate_modality_compatibility",
    "calculate_source_diversity",
    "detect_contradiction",
    # Engine
    "TemporalCrossModalFusionEngine",
]
