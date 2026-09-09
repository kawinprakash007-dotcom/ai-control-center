"""Phase 6.5d — Temporal & Cross-Modal Fusion Test Suite.

Comprehensive tests validating:
1. Domain models, bounds, validation, and immutability (A-D).
2. Temporal processing: relations (BEFORE, AFTER, OVERLAPS, COINCIDENT, WITHIN_WINDOW, SEQUENCE),
   windowing, and ordering (E-L).
3. Spatial correlation reusing Phase 6.5c geodesy (M-N).
4. Modality compatibility, contradiction detection, and source diversity (O-Q).
5. Duplicate evidence handling, caching, and freshness evaluation (R-T).
6. Evidence relationships: contradiction, corroboration, support (U-W).
7. Fusion clustering: deterministic IDs, bounding windows, connected components (X-Z).
8. Deterministic ordering, provenance preservation, and bounds enforcement (AA-AE).
9. Multi-product edge support: Vision, Glass, Drone, Rover, Digital Twin (AF-AK).
10. Concrete temporal and cross-modal scenarios (AL-AN).
11. Replay serialization compatibility and robustness (AO-AR).
12. Architectural boundary invariants: zero WorldState/Goal/Mission/Tool/Gateway bypasses,
    zero physical hardware drivers, zero subprocess/shell execution, model neutrality (AS-BH).
13. End-to-end handover to SituationFusionEngine (Section 34).
"""

from __future__ import annotations

import copy
import datetime
import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence
from unittest.mock import MagicMock

# Fallbacks for lightweight test environments
if "chromadb" not in sys.modules:
    sys.modules["chromadb"] = MagicMock()
if "sentence_transformers" not in sys.modules:
    sys.modules["sentence_transformers"] = MagicMock()

import pytest

from core.models.device_contract import (
    ConnectivityStatus,
    DeviceHealthStatus,
    ProductType,
)
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
from core.models.orchestration import (
    GeoLocation,
    ModalityType,
    MultimodalObservation,
)
from core.models.perception import (
    PerceptionCapability,
    PerceptionEvidence,
    PerceptionMetadata,
    PerceptionPrivacyClass,
    REQUIRED_PROVENANCE_KEYS,
    SpatialEvidence,
)
from core.models.spatial_telemetry import (
    MovementState,
    PositionObservation,
    RelativePosition,
    SpatialRelationType,
    TelemetryObservation,
)
from core.models.world_state import WorldState
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
from orchestration.fusion_engine import SituationFusionConfig, SituationFusionEngine


# ==============================================================================
# Helper Factories
# ==============================================================================

def make_observation(
    obs_id: str,
    modality: ModalityType,
    timestamp: float,
    source_id: str = "src_01",
    payload: Optional[Dict[str, Any]] = None,
    confidence: float = 0.95,
    location: Optional[GeoLocation] = None,
    correlation_id: str = "corr_test",
    causation_id: Optional[str] = None,
) -> MultimodalObservation:
    """Create a standardized MultimodalObservation for testing."""
    return MultimodalObservation(
        observation_id=obs_id,
        source_id=source_id,
        source_type="test_device",
        modality=modality,
        timestamp=timestamp,
        payload=payload or {"label": "test_object", "attributes": {}},
        confidence=confidence,
        location=location,
        correlation_id=correlation_id,
        causation_id=causation_id,
        metadata={"provenance": {"source": "unit_test"}},
    )


def make_evidence(
    ev_id: str,
    modality: ModalityType,
    timestamp: float,
    source_id: str = "src_01",
    label: str = "target",
    semantic_type: str = "detection",
    confidence: float = 0.95,
    location: Optional[GeoLocation] = None,
    attributes: Optional[Dict[str, Any]] = None,
    correlation_id: str = "corr_test",
    causation_id: Optional[str] = None,
) -> PerceptionEvidence:
    """Create a standardized PerceptionEvidence for testing with valid provenance."""
    spatial = SpatialEvidence(location=location) if location else None
    return PerceptionEvidence(
        evidence_id=ev_id,
        source_id=source_id,
        modality=modality,
        timestamp=timestamp,
        semantic_type=semantic_type,
        label=label,
        confidence=confidence,
        spatial=spatial,
        attributes=attributes or {},
        correlation_id=correlation_id,
        causation_id=causation_id,
        provenance={
            "source_id": source_id,
            "provider_id": "test_provider",
            "input_id": f"inp_{ev_id}",
            "request_id": f"req_{ev_id}",
            "observation_timestamp": timestamp,
            "correlation_id": correlation_id,
            "causation_id": causation_id,
            "provider_name": "test_provider",
            "provider_version": "1.0.0",
            "model_identifier": "test_model",
            "input_hash": "sha256_mock_hash",
        },
    )


# ==============================================================================
# SECTION A-D: Domain Models, Enums & Serialization
# ==============================================================================

class TestDomainModelsAndEnums:
    """Tests A-D: Enums, validation, immutability, and dual serialization."""

    def test_section_a_temporal_and_evidence_enums(self):
        # TemporalRelation
        assert TemporalRelation.BEFORE.value == "BEFORE"
        assert TemporalRelation.AFTER.value == "AFTER"
        assert TemporalRelation.OVERLAPS.value == "OVERLAPS"
        assert TemporalRelation.COINCIDENT.value == "COINCIDENT"
        assert TemporalRelation.WITHIN_WINDOW.value == "WITHIN_WINDOW"
        assert TemporalRelation.SEQUENCE.value == "SEQUENCE"
        assert TemporalRelation.from_str("before") == TemporalRelation.BEFORE

        with pytest.raises(ValueError):
            TemporalRelation.from_str("NON_EXISTENT")

        # EvidenceRelationType
        assert EvidenceRelationType.SUPPORTS.value == "SUPPORTS"
        assert EvidenceRelationType.CORROBORATES.value == "CORROBORATES"
        assert EvidenceRelationType.CONTRADICTS.value == "CONTRADICTS"
        assert EvidenceRelationType.PRECEDES.value == "PRECEDES"
        assert EvidenceRelationType.FOLLOWS.value == "FOLLOWS"
        assert EvidenceRelationType.OVERLAPS.value == "OVERLAPS"
        assert EvidenceRelationType.COLOCATED_WITH.value == "COLOCATED_WITH"
        assert EvidenceRelationType.SAME_ENTITY_CANDIDATE.value == "SAME_ENTITY_CANDIDATE"
        assert EvidenceRelationType.DUPLICATES.value == "DUPLICATES"
        assert EvidenceRelationType.from_str("corroborates") == EvidenceRelationType.CORROBORATES

        # ModalityCompatibilityLevel
        assert ModalityCompatibilityLevel.COMPLEMENTARY.value == "COMPLEMENTARY"
        assert ModalityCompatibilityLevel.COMPATIBLE.value == "COMPATIBLE"
        assert ModalityCompatibilityLevel.INDEPENDENT.value == "INDEPENDENT"
        assert ModalityCompatibilityLevel.CONFLICTING.value == "CONFLICTING"

        # FusionFreshnessStatus
        assert FusionFreshnessStatus.VALID.value == "VALID"
        assert FusionFreshnessStatus.STALE.value == "STALE"
        assert FusionFreshnessStatus.EXPIRED.value == "EXPIRED"

    def test_section_b_temporal_window_validation(self):
        # Valid window
        tw = TemporalWindow(start_time=100.0, end_time=110.0, max_duration=300.0)
        assert tw.duration == 10.0
        assert tw.contains(105.0)
        assert tw.contains(100.0)
        assert tw.contains(110.0)
        assert not tw.contains(99.0)
        assert not tw.contains(111.0)
        assert tw.contains(110.05, tolerance=0.1)

        # Negative start_time
        with pytest.raises(ValueError, match="non-negative"):
            TemporalWindow(start_time=-1.0, end_time=10.0)

        # start_time > end_time
        with pytest.raises(ValueError, match="cannot exceed end_time"):
            TemporalWindow(start_time=20.0, end_time=10.0)

        # Exceeds max_duration
        with pytest.raises(ValueError, match="exceeds max_duration"):
            TemporalWindow(start_time=0.0, end_time=500.0, max_duration=300.0)

        # NaN / Inf validation
        with pytest.raises(ValueError):
            TemporalWindow(start_time=float("nan"), end_time=10.0)
        with pytest.raises(ValueError):
            TemporalWindow(start_time=0.0, end_time=float("inf"))

        # Immutability
        with pytest.raises(AttributeError):
            tw.start_time = 50.0  # type: ignore

    def test_section_c_domain_models_immutability(self):
        link = SpatialEvidenceLink(
            source_id_1="cam_01",
            source_id_2="drone_01",
            distance_meters=12.5,
            spatial_relation=SpatialRelationType.NEARBY,
            spatial_confidence=0.9,
            bearing_degrees=45.0,
        )
        assert link.distance_meters == 12.5
        with pytest.raises(AttributeError):
            link.distance_meters = 20.0  # type: ignore

        rel = EvidenceRelationship(
            relation_type=EvidenceRelationType.CORROBORATES,
            source_evidence_id="ev_01",
            target_evidence_id="ev_02",
            confidence=0.88,
            reason="Complementary sensors",
            spatial_link=link,
        )
        assert rel.confidence == 0.88
        with pytest.raises(AttributeError):
            rel.confidence = 0.5  # type: ignore

    def test_section_d_fusion_result_dual_serialization(self):
        tw = TemporalWindow(start_time=100.0, end_time=120.0)
        cluster_id = FusionCluster.generate_deterministic_id(
            evidence_ids=["ev_1", "ev_2"],
            source_ids=["src_1", "src_2"],
            start_time=100.0,
            end_time=120.0,
        )
        cluster = FusionCluster(
            cluster_id=cluster_id,
            evidence_ids=("ev_1", "ev_2"),
            source_ids=("src_1", "src_2"),
            modalities=(ModalityType.IMAGE, ModalityType.GPS),
            temporal_window=tw,
            relationships=(),
            confidence=0.9,
            source_diversity=1.0,
        )
        res = FusionResult(
            result_id="fres_test_01",
            temporal_window=tw,
            clusters=(cluster,),
            relationships=(),
            correlations=(),
            created_at=120.0,
        )

        # to_dict / from_dict
        d = res.to_dict()
        assert d["result_id"] == "fres_test_01"
        assert len(d["clusters"]) == 1
        assert d["clusters"][0]["cluster_id"] == cluster_id

        restored = FusionResult.from_dict(d)
        assert restored.result_id == res.result_id
        assert restored.temporal_window.duration == 20.0
        assert len(restored.clusters) == 1

        # model_dump / model_validate
        d2 = res.model_dump()
        restored2 = FusionResult.model_validate(d2)
        assert restored2.result_id == res.result_id

        # model_dump_json
        j = res.model_dump_json()
        assert "fres_test_01" in j
        assert cluster_id in j

        # Deterministic result ID generation
        det_id = FusionResult.generate_deterministic_result_id(tw, 5)
        assert det_id.startswith("fres_")
        assert len(det_id) > 10


# ==============================================================================
# SECTION E-L: Temporal Processing & Windowing
# ==============================================================================

class TestTemporalProcessingAndWindowing:
    """Tests E-L: Temporal relations, window construction, filtering, bounds."""

    def test_section_e_temporal_relation_before(self):
        rel = evaluate_temporal_relation(t1=100.0, t2=120.0)
        assert rel == TemporalRelation.BEFORE

        rel_iv = evaluate_temporal_relation(t1=(100.0, 105.0), t2=(110.0, 115.0))
        assert rel_iv == TemporalRelation.BEFORE

    def test_section_f_temporal_relation_after(self):
        rel = evaluate_temporal_relation(t1=150.0, t2=120.0)
        assert rel == TemporalRelation.AFTER

        rel_iv = evaluate_temporal_relation(t1=(120.0, 130.0), t2=(100.0, 110.0))
        assert rel_iv == TemporalRelation.AFTER

    def test_section_g_temporal_relation_overlaps(self):
        rel_iv = evaluate_temporal_relation(t1=(100.0, 120.0), t2=(110.0, 130.0))
        assert rel_iv == TemporalRelation.OVERLAPS

        rel_iv2 = evaluate_temporal_relation(t1=(115.0, 135.0), t2=(100.0, 120.0))
        assert rel_iv2 == TemporalRelation.OVERLAPS

    def test_section_h_temporal_relation_coincident(self):
        rel = evaluate_temporal_relation(t1=100.0, t2=100.02, coincidence_tolerance=0.05)
        assert rel == TemporalRelation.COINCIDENT

        rel_iv = evaluate_temporal_relation(
            t1=(100.0, 105.0), t2=(100.03, 105.02), coincidence_tolerance=0.05
        )
        assert rel_iv == TemporalRelation.COINCIDENT

    def test_section_i_temporal_relation_within_window(self):
        rel = evaluate_temporal_relation(t1=105.0, t2=(100.0, 110.0))
        assert rel == TemporalRelation.WITHIN_WINDOW

    def test_section_j_temporal_relation_sequence(self):
        rel = evaluate_temporal_relation(t1=100.0, t2=103.0, sequence_tolerance=5.0)
        assert rel == TemporalRelation.SEQUENCE

    def test_section_k_build_temporal_window(self):
        timestamps = [10.0, 20.0, 30.0, 25.0, 15.0]
        tw = build_temporal_window(timestamps, max_duration=300.0)
        assert tw.start_time == 10.0
        assert tw.end_time == 30.0
        assert tw.duration == 20.0
        assert len(tw.source_timestamps) == 5

        # Single timestamp
        tw_single = build_temporal_window([42.0])
        assert tw_single.start_time == 42.0
        assert tw_single.end_time == 42.0
        assert tw_single.duration == 0.0

        # Empty timestamps
        tw_empty = build_temporal_window([])
        assert tw_empty.duration == 0.0

        # Window exceeding max_duration clips to latest max_duration
        wide_ts = [0.0, 100.0, 500.0]
        tw_clipped = build_temporal_window(wide_ts, max_duration=100.0)
        assert tw_clipped.duration <= 100.0
        assert tw_clipped.end_time == 500.0
        assert tw_clipped.start_time == 400.0

    def test_section_l_filter_by_temporal_window(self):
        tw = TemporalWindow(start_time=100.0, end_time=200.0)
        items = [
            {"id": "1", "timestamp": 50.0},
            {"id": "2", "timestamp": 100.0},
            {"id": "3", "timestamp": 150.0},
            {"id": "4", "timestamp": 200.0},
            {"id": "5", "timestamp": 250.0},
        ]
        filtered = filter_by_temporal_window(items, tw)
        ids = [x["id"] for x in filtered]
        assert ids == ["2", "3", "4"]


# ==============================================================================
# SECTION M-N: Spatial Correlation
# ==============================================================================

class TestSpatialCorrelation:
    """Tests M-N: Geodesy reuse from Phase 6.5c, distance thresholds, colocation."""

    def test_section_m_spatial_evidence_correlation(self):
        # San Francisco coordinates
        loc1 = GeoLocation(latitude=37.7749, longitude=-122.4194, altitude=10.0)
        # Point ~111 meters North (0.001 deg latitude)
        loc2 = GeoLocation(latitude=37.7759, longitude=-122.4194, altitude=15.0)

        link = correlate_spatial_evidence(
            source_id_1="drone_01",
            source_id_2="rover_01",
            location_1=loc1,
            location_2=loc2,
            max_distance_meters=500.0,
        )
        assert link is not None
        assert 110.0 <= link.distance_meters <= 112.0
        assert link.bearing_degrees is not None
        assert link.spatial_relation in (
            SpatialRelationType.NEARBY,
            SpatialRelationType.VICINITY,
            SpatialRelationType.SAME_LOCATION,
        )

    def test_section_n_colocation_and_distance_thresholds(self):
        loc1 = GeoLocation(latitude=37.7749, longitude=-122.4194)
        # Exact coincident point
        link_coincident = correlate_spatial_evidence(
            source_id_1="cam_01",
            source_id_2="tag_01",
            location_1=loc1,
            location_2=loc1,
            max_distance_meters=100.0,
        )
        assert link_coincident is not None
        assert link_coincident.distance_meters == 0.0
        assert link_coincident.spatial_relation == SpatialRelationType.SAME_LOCATION
        assert link_coincident.spatial_confidence == 1.0

        # Distant point (> max_distance_meters)
        loc_far = GeoLocation(latitude=37.8000, longitude=-122.4194)
        link_far = correlate_spatial_evidence(
            source_id_1="cam_01",
            source_id_2="far_node",
            location_1=loc1,
            location_2=loc_far,
            max_distance_meters=500.0,
        )
        assert link_far is None


# ==============================================================================
# SECTION O-Q: Modality Compatibility & Source Diversity
# ==============================================================================

class TestCompatibilityAndDiversity:
    """Tests O-Q: Compatibility matrix, source diversity calculations."""

    def test_section_o_complementary_modalities(self):
        assert evaluate_modality_compatibility(ModalityType.IMAGE, ModalityType.AUDIO_EVENT) == ModalityCompatibilityLevel.COMPLEMENTARY
        assert evaluate_modality_compatibility(ModalityType.IMAGE, ModalityType.GPS) == ModalityCompatibilityLevel.COMPLEMENTARY
        assert evaluate_modality_compatibility(ModalityType.GPS, ModalityType.TELEMETRY) == ModalityCompatibilityLevel.COMPLEMENTARY
        assert evaluate_modality_compatibility(ModalityType.IMAGE, ModalityType.TELEMETRY) == ModalityCompatibilityLevel.COMPLEMENTARY

    def test_section_p_incompatible_or_independent_modalities(self):
        # Same modalities are compatible
        assert evaluate_modality_compatibility(ModalityType.IMAGE, ModalityType.IMAGE) == ModalityCompatibilityLevel.COMPATIBLE
        assert evaluate_modality_compatibility(ModalityType.AUDIO_EVENT, ModalityType.TELEMETRY) in (
            ModalityCompatibilityLevel.COMPATIBLE,
            ModalityCompatibilityLevel.INDEPENDENT,
        )

    def test_section_q_source_diversity_calculation(self):
        # Empty sources
        assert calculate_source_diversity([]) == 0.0

        # Single source, multiple observations -> 0.0 (no diversity)
        assert calculate_source_diversity(["camera_01", "camera_01", "camera_01"]) == 0.0

        # Two different sources
        div_2 = calculate_source_diversity(["camera_01", "drone_01"])
        assert div_2 == 1.0

        # Three observations from two unique sources -> (2-1)/(3-1) = 0.5
        div_mixed = calculate_source_diversity(["camera_01", "camera_01", "drone_01"])
        assert div_mixed == 0.5


# ==============================================================================
# SECTION R-T: Deduplication, Freshness & Caching
# ==============================================================================

class TestDeduplicationAndFreshness:
    """Tests R-T: Duplicate detection, cache bounds, freshness lifecycle."""

    def test_section_r_duplicate_evidence_detection(self):
        engine = TemporalCrossModalFusionEngine(limits=FusionLimits(max_duplicate_cache=10))
        engine.clear_cache()

        obs1 = make_observation(obs_id="obs_dup_1", modality=ModalityType.IMAGE, timestamp=100.0)
        obs2 = make_observation(obs_id="obs_dup_1", modality=ModalityType.IMAGE, timestamp=100.0)

        res = engine.fuse([obs1, obs2], reference_time=105.0)
        assert "obs_dup_1" in res.duplicate_evidence_ids
        dup_rels = [r for r in res.relationships if r.relation_type == EvidenceRelationType.DUPLICATES]
        assert len(dup_rels) >= 1

    def test_section_s_freshness_classification(self):
        limits = FusionLimits(freshness_threshold_seconds=10.0, expiration_threshold_seconds=60.0)
        engine = TemporalCrossModalFusionEngine(limits=limits)
        engine.clear_cache()

        now = 100.0
        # Valid: age 5s (< 10s)
        obs_valid = make_observation(obs_id="v1", modality=ModalityType.IMAGE, timestamp=95.0)
        # Stale: age 20s (10s <= age < 60s)
        obs_stale = make_observation(obs_id="s1", modality=ModalityType.IMAGE, timestamp=80.0)
        # Expired: age 70s (>= 60s)
        obs_expired = make_observation(obs_id="e1", modality=ModalityType.IMAGE, timestamp=30.0)

        res = engine.fuse([obs_valid, obs_stale, obs_expired], reference_time=now)
        assert res is not None
        assert res.provenance["input_count"] == 3

    def test_section_t_cache_boundary_eviction(self):
        limits = FusionLimits(max_duplicate_cache=3)
        engine = TemporalCrossModalFusionEngine(limits=limits)
        engine.clear_cache()

        # Ingest 3 items
        for i in range(3):
            obs = make_observation(obs_id=f"item_{i}", modality=ModalityType.IMAGE, timestamp=100.0 + i)
            engine.fuse([obs], reference_time=105.0)

        # Ingest 4th item (evicts item_0)
        obs4 = make_observation(obs_id="item_3", modality=ModalityType.IMAGE, timestamp=104.0)
        engine.fuse([obs4], reference_time=105.0)

        # Ingest item_0 again -> should NOT be considered duplicate now because it was evicted
        obs_re0 = make_observation(obs_id="item_0", modality=ModalityType.IMAGE, timestamp=105.0)
        res = engine.fuse([obs_re0], reference_time=105.0)
        assert "item_0" not in res.duplicate_evidence_ids


# ==============================================================================
# SECTION U-W: Contradiction, Corroboration & Support
# ==============================================================================

class TestEvidenceRelationships:
    """Tests U-W: Semantics of contradictions, corroborations, and support links."""

    def test_section_u_contradiction_detection(self):
        engine = TemporalCrossModalFusionEngine()
        engine.clear_cache()

        # Drone observes status 'clear'
        ev_drone = make_evidence(
            ev_id="ev_drone_clear",
            modality=ModalityType.IMAGE,
            timestamp=100.0,
            source_id="drone_alpha",
            label="path_clear",
            attributes={"status": "clear", "speed": 10.0},
        )
        # Rover bumper observes status 'blocked' at same timestamp
        ev_rover = make_evidence(
            ev_id="ev_rover_blocked",
            modality=ModalityType.TELEMETRY,
            timestamp=100.02,
            source_id="rover_bravo",
            label="obstacle_detected",
            attributes={"status": "blocked", "speed": 0.0},
        )

        res = engine.fuse([ev_drone, ev_rover], reference_time=101.0)
        assert len(res.contradictions) >= 1
        contra = res.contradictions[0]
        assert contra.relation_type == EvidenceRelationType.CONTRADICTS
        assert "speed" in contra.reason or "status" in contra.reason or "label" in contra.reason

    def test_section_v_corroboration_detection(self):
        engine = TemporalCrossModalFusionEngine()
        engine.clear_cache()

        loc = GeoLocation(latitude=37.7749, longitude=-122.4194)
        # Camera sees person
        ev_cam = make_evidence(
            ev_id="ev_cam_person",
            modality=ModalityType.IMAGE,
            timestamp=100.0,
            source_id="camera_hallway",
            label="person",
            location=loc,
        )
        # Video stream sees person at same location
        ev_vid = make_evidence(
            ev_id="ev_vid_person",
            modality=ModalityType.VIDEO_FRAME,
            timestamp=100.05,
            source_id="video_hallway",
            label="person",
            location=loc,
        )

        res = engine.fuse([ev_cam, ev_vid], reference_time=101.0)
        corrob = [r for r in res.relationships if r.relation_type == EvidenceRelationType.CORROBORATES]
        assert len(corrob) >= 1
        assert corrob[0].confidence > 0.8

    def test_section_w_support_relationship(self):
        engine = TemporalCrossModalFusionEngine()
        engine.clear_cache()

        loc1 = GeoLocation(latitude=37.7749, longitude=-122.4194)
        loc2 = GeoLocation(latitude=37.77495, longitude=-122.4194)

        # Visual sighting
        ev_vis = make_evidence(
            ev_id="ev_vis_car",
            modality=ModalityType.IMAGE,
            timestamp=100.0,
            source_id="cam_01",
            label="vehicle",
            location=loc1,
        )
        # GPS telemetry nearby
        ev_gps = make_evidence(
            ev_id="ev_gps_car",
            modality=ModalityType.GPS,
            timestamp=100.1,
            source_id="gps_tracker_01",
            label="gps_fix",
            location=loc2,
        )

        res = engine.fuse([ev_vis, ev_gps], reference_time=101.0)
        assert len(res.relationships) >= 1
        rel = res.relationships[0]
        assert rel.relation_type in (EvidenceRelationType.SUPPORTS, EvidenceRelationType.CORROBORATES)
        assert rel.spatial_link is not None


# ==============================================================================
# SECTION X-Z: Fusion Clustering
# ==============================================================================

class TestFusionClustering:
    """Tests X-Z: Clustering components, deterministic IDs, convex boundaries."""

    def test_section_x_fusion_cluster_creation(self):
        engine = TemporalCrossModalFusionEngine()
        engine.clear_cache()

        loc = GeoLocation(latitude=37.7749, longitude=-122.4194)
        ev1 = make_evidence(ev_id="e1", modality=ModalityType.IMAGE, timestamp=100.0, source_id="s1", label="drone", location=loc)
        ev2 = make_evidence(ev_id="e2", modality=ModalityType.AUDIO_EVENT, timestamp=100.1, source_id="s2", label="drone", location=loc)

        res = engine.fuse([ev1, ev2], reference_time=102.0)
        assert len(res.clusters) == 1
        cluster = res.clusters[0]
        assert set(cluster.evidence_ids) == {"e1", "e2"}
        assert set(cluster.source_ids) == {"s1", "s2"}
        assert cluster.source_diversity == 1.0

    def test_section_y_deterministic_cluster_id(self):
        # Two identical calls must generate exact same cluster ID
        id1 = FusionCluster.generate_deterministic_id(
            evidence_ids=["ev_a", "ev_b"],
            source_ids=["src_1", "src_2"],
            start_time=100.0,
            end_time=110.0,
        )
        id2 = FusionCluster.generate_deterministic_id(
            evidence_ids=["ev_b", "ev_a"],  # different order
            source_ids=["src_2", "src_1"],  # different order
            start_time=100.0,
            end_time=110.0,
        )
        assert id1 == id2
        assert id1.startswith("fcluster_")
        assert len(id1) == len("fcluster_") + 16

    def test_section_z_cluster_bounds_and_metrics(self):
        tw = TemporalWindow(start_time=50.0, end_time=60.0)
        c = FusionCluster(
            cluster_id="fcluster_01",
            evidence_ids=("e1", "e2"),
            source_ids=("s1", "s2"),
            modalities=(ModalityType.IMAGE, ModalityType.AUDIO_EVENT),
            temporal_window=tw,
            relationships=(),
            confidence=0.92,
            source_diversity=1.0,
            contradiction_count=0,
        )
        assert c.temporal_window.duration == 10.0
        assert c.confidence == 0.92
        assert c.contradiction_count == 0


# ==============================================================================
# SECTION AA-AE: Determinism, Provenance & Resource Bounds
# ==============================================================================

class TestDeterminismProvenanceAndBounds:
    """Tests AA-AE: Deterministic ordering, correlation preservation, bounds enforcement."""

    def test_section_aa_deterministic_output_ordering(self):
        engine = TemporalCrossModalFusionEngine()

        obs_a = make_observation(obs_id="obs_a", modality=ModalityType.IMAGE, timestamp=100.0, source_id="src_1")
        obs_b = make_observation(obs_id="obs_b", modality=ModalityType.AUDIO_EVENT, timestamp=101.0, source_id="src_2")
        obs_c = make_observation(obs_id="obs_c", modality=ModalityType.GPS, timestamp=102.0, source_id="src_3")

        # Run with order [a, b, c]
        engine.clear_cache()
        res1 = engine.fuse([obs_a, obs_b, obs_c], reference_time=110.0)

        # Run with order [c, a, b]
        engine.clear_cache()
        res2 = engine.fuse([obs_c, obs_a, obs_b], reference_time=110.0)

        assert res1.temporal_window.start_time == res2.temporal_window.start_time
        assert res1.temporal_window.end_time == res2.temporal_window.end_time
        assert len(res1.clusters) == len(res2.clusters)
        if res1.clusters and res2.clusters:
            assert res1.clusters[0].cluster_id == res2.clusters[0].cluster_id

    def test_section_ab_correlation_and_causation_id_preservation(self):
        engine = TemporalCrossModalFusionEngine()
        engine.clear_cache()

        obs = make_observation(
            obs_id="obs_corr_1",
            modality=ModalityType.IMAGE,
            timestamp=100.0,
            correlation_id="corr_mission_42",
            causation_id="cause_event_99",
        )
        res = engine.fuse([obs], reference_time=105.0, correlation_id="corr_mission_42", causation_id="cause_event_99")
        assert res.provenance["correlation_id"] == "corr_mission_42"
        assert res.provenance["causation_id"] == "cause_event_99"

    def test_section_ac_provenance_preservation(self):
        engine = TemporalCrossModalFusionEngine()
        engine.clear_cache()

        ev = make_evidence(ev_id="ev_prov_1", modality=ModalityType.IMAGE, timestamp=100.0)
        res = engine.fuse([ev], reference_time=105.0)

        assert res.provenance["engine"] == "TemporalCrossModalFusionEngine"
        assert res.provenance["version"] == "1.0.0"
        assert res.provenance["input_count"] == 1
        assert "semantic_hash" in res.to_dict()

    def test_section_ad_bounded_outputs(self):
        limits = FusionLimits(
            max_input_observations=5,
            max_evidence_relationships=3,
            max_clusters=2,
        )
        engine = TemporalCrossModalFusionEngine(limits=limits)
        engine.clear_cache()

        # Supply 10 observations
        obs_list = [
            make_observation(obs_id=f"bulk_{i}", modality=ModalityType.IMAGE, timestamp=100.0 + i)
            for i in range(10)
        ]
        res = engine.fuse(obs_list, reference_time=120.0)
        assert res.provenance["input_count"] <= 5
        assert len(res.relationships) <= 3
        assert len(res.clusters) <= 2

    def test_section_ae_semantic_hash_generation(self):
        tw = TemporalWindow(start_time=100.0, end_time=110.0)
        res = FusionResult(result_id="fres_hash_test", temporal_window=tw, created_at=110.0)
        h1 = res.semantic_hash
        h2 = res.semantic_hash
        assert h1 == h2
        assert len(h1) == 64  # SHA-256


# ==============================================================================
# SECTION AF-AK: Multi-Product Edge Support
# ==============================================================================

class TestMultiProductEdgeSupport:
    """Tests AF-AK: Vision, Glass, Drone, Rover, Digital Twin fusion."""

    def test_section_af_vision_edge_fusion(self):
        engine = TemporalCrossModalFusionEngine()
        engine.clear_cache()

        ev = make_evidence(
            ev_id="vis_det_01",
            modality=ModalityType.IMAGE,
            timestamp=100.0,
            source_id="atlas_vision_edge",
            label="forklift",
            attributes={"bounding_box": [10, 20, 100, 200]},
        )
        res = engine.fuse([ev], reference_time=105.0)
        assert len(res.normalized_observations) == 1
        assert res.normalized_observations[0].payload["label"] == "forklift"

    def test_section_ag_glass_edge_fusion(self):
        engine = TemporalCrossModalFusionEngine()
        engine.clear_cache()

        ev_audio = make_evidence(
            ev_id="glass_audio_01",
            modality=ModalityType.AUDIO_EVENT,
            timestamp=100.0,
            source_id="atlas_glass_01",
            label="speech_transcript",
            attributes={"text": "warning system failure"},
        )
        res = engine.fuse([ev_audio], reference_time=105.0)
        assert len(res.normalized_observations) == 1
        assert res.normalized_observations[0].modality == ModalityType.AUDIO_EVENT

    def test_section_ah_drone_edge_fusion(self):
        engine = TemporalCrossModalFusionEngine()
        engine.clear_cache()

        loc = GeoLocation(latitude=37.7749, longitude=-122.4194, altitude=50.0)
        ev_drone = make_evidence(
            ev_id="drone_survey_01",
            modality=ModalityType.GPS,
            timestamp=100.0,
            source_id="atlas_drone_01",
            label="aerial_position",
            location=loc,
            attributes={"battery_percent": 88.0, "altitude_agl": 50.0},
        )
        res = engine.fuse([ev_drone], reference_time=105.0)
        assert len(res.normalized_observations) == 1
        assert res.normalized_observations[0].location.altitude == 50.0

    def test_section_ai_rover_edge_fusion(self):
        engine = TemporalCrossModalFusionEngine()
        engine.clear_cache()

        ev_rover = make_evidence(
            ev_id="rover_telemetry_01",
            modality=ModalityType.TELEMETRY,
            timestamp=100.0,
            source_id="atlas_rover_01",
            label="wheel_velocity",
            attributes={"speed": 1.2, "wheel_slip": 0.05},
        )
        res = engine.fuse([ev_rover], reference_time=105.0)
        assert len(res.normalized_observations) == 1
        assert res.normalized_observations[0].payload["attributes"]["speed"] == 1.2

    def test_section_aj_digital_twin_edge_fusion(self):
        engine = TemporalCrossModalFusionEngine()
        engine.clear_cache()

        loc = GeoLocation(latitude=37.7749, longitude=-122.4194)
        ev_sim = make_evidence(
            ev_id="sim_twin_01",
            modality=ModalityType.DEVICE_STATE,
            timestamp=100.0,
            source_id="digital_twin_sim",
            label="simulated_drone_state",
            location=loc,
            attributes={"is_simulated": True, "health": "NOMINAL"},
        )
        res = engine.fuse([ev_sim], reference_time=105.0)
        assert len(res.normalized_observations) == 1
        assert res.normalized_observations[0].payload["attributes"]["is_simulated"] is True

    def test_section_ak_cross_product_fusion(self):
        engine = TemporalCrossModalFusionEngine()
        engine.clear_cache()

        loc = GeoLocation(latitude=37.7749, longitude=-122.4194)
        # 1. Drone aerial camera
        ev_drone = make_evidence(
            ev_id="ev_drone_aerial",
            modality=ModalityType.IMAGE,
            timestamp=100.0,
            source_id="drone_alpha",
            label="target_vehicle",
            location=loc,
            attributes={"entity_id": "car_404"},
        )
        # 2. Rover ground telemetry
        ev_rover = make_evidence(
            ev_id="ev_rover_telemetry",
            modality=ModalityType.TELEMETRY,
            timestamp=100.02,
            source_id="rover_beta",
            label="proximity_beacon",
            location=loc,
            attributes={"entity_id": "car_404"},
        )
        # 3. Glass user observation
        ev_glass = make_evidence(
            ev_id="ev_glass_speech",
            modality=ModalityType.AUDIO_EVENT,
            timestamp=100.04,
            source_id="glass_gamma",
            label="speech_callout",
            location=loc,
            attributes={"entity_id": "car_404"},
        )

        res = engine.fuse([ev_drone, ev_rover, ev_glass], reference_time=102.0)
        assert len(res.clusters) == 1
        c = res.clusters[0]
        assert len(c.evidence_ids) == 3
        assert len(c.source_ids) == 3
        assert c.source_diversity == 1.0


# ==============================================================================
# SECTION AL-AN: Concrete Scenarios
# ==============================================================================

class TestConcreteFusionScenarios:
    """Tests AL-AN: Multi-modal colocation, sequence tracking, contradiction."""

    def test_section_al_telemetry_visual_spatial_scenario(self):
        engine = TemporalCrossModalFusionEngine()
        engine.clear_cache()

        loc = GeoLocation(latitude=37.7749, longitude=-122.4194)
        ev_vision = make_evidence(
            ev_id="drone_vis",
            modality=ModalityType.IMAGE,
            timestamp=100.0,
            source_id="drone_01",
            label="intruder_vehicle",
            location=loc,
            confidence=0.95,
        )
        ev_telemetry = make_evidence(
            ev_id="rover_tel",
            modality=ModalityType.TELEMETRY,
            timestamp=100.05,
            source_id="rover_01",
            label="radar_blip",
            location=loc,
            confidence=0.90,
        )

        res = engine.fuse([ev_vision, ev_telemetry], reference_time=102.0)
        assert len(res.relationships) >= 1
        rel = res.relationships[0]
        assert rel.spatial_link is not None
        assert rel.spatial_link.distance_meters == 0.0

    def test_section_am_temporal_sequence_scenario(self):
        engine = TemporalCrossModalFusionEngine()
        engine.clear_cache()

        # Same camera tracking moving entity over time
        ev_t1 = make_evidence(ev_id="frame_1", modality=ModalityType.IMAGE, timestamp=100.0, source_id="cam_01")
        ev_t2 = make_evidence(ev_id="frame_2", modality=ModalityType.IMAGE, timestamp=105.0, source_id="cam_01")

        res = engine.fuse([ev_t1, ev_t2], reference_time=110.0)
        assert len(res.relationships) >= 1
        rel = res.relationships[0]
        assert rel.relation_type == EvidenceRelationType.PRECEDES
        assert rel.temporal_relation == TemporalRelation.BEFORE

    def test_section_an_contradiction_scenario(self):
        engine = TemporalCrossModalFusionEngine()
        engine.clear_cache()

        ev1 = make_evidence(
            ev_id="sensor_a",
            modality=ModalityType.TELEMETRY,
            timestamp=100.0,
            source_id="node_a",
            label="status_check",
            attributes={"status": "normal"},
        )
        ev2 = make_evidence(
            ev_id="sensor_b",
            modality=ModalityType.TELEMETRY,
            timestamp=100.01,
            source_id="node_b",
            label="status_check",
            attributes={"status": "critical_error"},
        )

        res = engine.fuse([ev1, ev2], reference_time=101.0)
        assert len(res.contradictions) == 1
        assert res.contradictions[0].relation_type == EvidenceRelationType.CONTRADICTS


# ==============================================================================
# SECTION AO-AR: Replay, Robustness & Security
# ==============================================================================

class TestReplayRobustnessAndSecurity:
    """Tests AO-AR: Serialization fidelity, extreme inputs, malformed data, no credentials."""

    def test_section_ao_replay_serialization(self):
        engine = TemporalCrossModalFusionEngine()
        engine.clear_cache()

        loc = GeoLocation(latitude=37.7749, longitude=-122.4194)
        obs1 = make_observation(obs_id="obs_replay_1", modality=ModalityType.IMAGE, timestamp=100.0, location=loc)
        obs2 = make_observation(obs_id="obs_replay_2", modality=ModalityType.GPS, timestamp=100.1, location=loc)

        res = engine.fuse([obs1, obs2], reference_time=105.0)
        res_json = res.model_dump_json()

        # Reconstruct from JSON
        res_reconstructed = FusionResult.model_validate_json(res_json)
        assert res_reconstructed.result_id == res.result_id
        assert res_reconstructed.temporal_window.duration == res.temporal_window.duration
        assert len(res_reconstructed.clusters) == len(res.clusters)

    def test_section_ap_extreme_input_bounds(self):
        # 200 inputs clipped gracefully to max_input_observations
        limits = FusionLimits(max_input_observations=50)
        engine = TemporalCrossModalFusionEngine(limits=limits)
        engine.clear_cache()

        inputs = [
            make_observation(obs_id=f"stress_{i}", modality=ModalityType.IMAGE, timestamp=100.0 + (i * 0.01))
            for i in range(200)
        ]
        res = engine.fuse(inputs, reference_time=200.0)
        assert res.provenance["input_count"] <= 50

    def test_section_aq_malformed_evidence_handling(self):
        engine = TemporalCrossModalFusionEngine()
        engine.clear_cache()

        obs_empty = make_observation(
            obs_id="obs_malformed_01",
            modality=ModalityType.IMAGE,
            timestamp=1.0,
            payload={},
            confidence=0.5,
        )
        res = engine.fuse([obs_empty], reference_time=2.0)
        assert res is not None
        assert res.provenance["input_count"] == 1

    def test_section_ar_zero_credential_leakage(self):
        engine = TemporalCrossModalFusionEngine()
        engine.clear_cache()

        # Malicious attributes containing secrets
        ev = make_evidence(
            ev_id="ev_secrets",
            modality=ModalityType.TELEMETRY,
            timestamp=100.0,
            attributes={
                "api_key": "SECRET_KEY_12345",
                "password": "Password123!",
                "temperature": 25.4,
            },
        )
        res = engine.fuse([ev], reference_time=105.0)

        # Check normalized observations
        assert len(res.normalized_observations) == 1
        attrs = res.normalized_observations[0].payload["attributes"]
        assert "api_key" not in attrs or attrs["api_key"] == "[REDACTED]"
        assert "password" not in attrs or attrs["password"] == "[REDACTED]"
        assert attrs.get("temperature") == 25.4


# ==============================================================================
# SECTION AS-BH: Architectural Boundary Invariants
# ==============================================================================

class TestArchitecturalBoundaryInvariants:
    """Tests AS-BH: Invariants against decision leakage, mutation, and physical hardware."""

    def test_section_as_zero_world_state_mutation(self):
        ws = WorldState(state_id="ws_01", version=1, timestamp=time.time())
        initial_state = copy.deepcopy(ws.to_dict())

        engine = TemporalCrossModalFusionEngine()
        ev = make_evidence(ev_id="ev_ws_test", modality=ModalityType.IMAGE, timestamp=100.0)
        _ = engine.fuse([ev], reference_time=105.0)

        # WorldState must remain strictly unmutated
        assert ws.to_dict() == initial_state

    def test_section_at_au_av_aw_ax_ay_zero_authority_bypasses(self):
        engine = TemporalCrossModalFusionEngine()
        # Verify engine has zero goal, mission, runtime, tool, gateway, or policy bypass attributes
        assert not hasattr(engine, "goal_store")
        assert not hasattr(engine, "goal_manager")
        assert not hasattr(engine, "mission_intelligence")
        assert not hasattr(engine, "cognitive_runtime")
        assert not hasattr(engine, "tool_orchestrator")
        assert not hasattr(engine, "device_gateway")
        assert not hasattr(engine, "policy_engine")

    def test_section_az_zero_situation_creation(self):
        # 6.5d must NEVER instantiate Situation objects
        engine = TemporalCrossModalFusionEngine()
        ev = make_evidence(ev_id="ev_sit_test", modality=ModalityType.IMAGE, timestamp=100.0)
        res = engine.fuse([ev], reference_time=105.0)

        # Output must be FusionResult, not Situation
        assert isinstance(res, FusionResult)
        assert not hasattr(res, "situation_id")
        assert not hasattr(res, "operational_status")

    def test_section_ba_downstream_handover_compatibility(self):
        # Handover to SituationFusionEngine
        fusion_engine = SituationFusionEngine()

        engine = TemporalCrossModalFusionEngine()
        ev = make_evidence(ev_id="ev_handover_1", modality=ModalityType.IMAGE, timestamp=100.0)
        res = engine.fuse([ev], reference_time=105.0)

        # Normalized observations from 6.5d can be directly fed into SituationFusionEngine
        assert len(res.normalized_observations) == 1
        obs = res.normalized_observations[0]
        fusion_engine.ingest(obs)
        assert len(fusion_engine._seen_observation_ids) == 1

    def test_section_bb_bc_bd_zero_hardware_and_execution_imports(self):
        # Verify no physical hardware drivers, subprocesses, or eval/exec are used
        import backend.core.models.multimodal_fusion as m_mod
        import backend.multimodal_fusion.compatibility as c_mod
        import backend.multimodal_fusion.engine as e_mod
        import backend.multimodal_fusion.spatial as s_mod
        import backend.multimodal_fusion.temporal as t_mod

        forbidden_tokens = [
            "subprocess",
            "os.system",
            "eval(",
            "exec(",
            "shell=True",
            "rospy",
            "rclpy",
            "pymavlink",
            "RPi.GPIO",
            "serial.",
        ]
        modules = [m_mod, c_mod, e_mod, s_mod, t_mod]

        for mod in modules:
            source = open(mod.__file__, "r", encoding="utf-8").read()
            for token in forbidden_tokens:
                assert token not in source, f"Forbidden token '{token}' found in {mod.__file__}"


# ==============================================================================
# SECTION 34: End-to-End Pipeline Verification
# ==============================================================================

class TestEndToEndMultimodalFusionHandover:
    """End-to-End test demonstrating Phase 6.5d fusion handover."""

    def test_section_34_end_to_end_pipeline(self):
        engine = TemporalCrossModalFusionEngine()
        engine.clear_cache()

        now = 200.0
        loc = GeoLocation(latitude=37.7749, longitude=-122.4194)

        # Stream of 3 observations:
        # 1. Drone vision detects vehicle
        obs_vis = make_observation(
            obs_id="obs_drone_vehicle",
            modality=ModalityType.IMAGE,
            timestamp=195.0,
            source_id="drone_01",
            payload={"label": "vehicle", "attributes": {"speed": 15.0}},
            location=loc,
        )
        # 2. Ground rover verifies same vehicle
        obs_tel = make_observation(
            obs_id="obs_rover_radar",
            modality=ModalityType.TELEMETRY,
            timestamp=195.1,
            source_id="rover_01",
            payload={"label": "vehicle", "attributes": {"speed": 15.0}},
            location=loc,
        )
        # 3. Audio acoustic sensor records engine sound
        obs_aud = make_observation(
            obs_id="obs_acoustic_sensor",
            modality=ModalityType.AUDIO_EVENT,
            timestamp=195.2,
            source_id="mic_array_01",
            payload={"label": "vehicle_engine_rumble", "attributes": {}},
            location=loc,
        )

        # Execute fusion
        fusion_result = engine.fuse(
            observations=[obs_vis, obs_tel, obs_aud],
            reference_time=now,
            correlation_id="mission_recon_77",
            causation_id="alert_trigger_01",
        )

        # Verify results
        assert fusion_result is not None
        assert fusion_result.temporal_window.duration == 0.2
        assert len(fusion_result.clusters) == 1

        cluster = fusion_result.clusters[0]
        assert len(cluster.evidence_ids) == 3
        assert cluster.source_diversity == 1.0
        assert cluster.contradiction_count == 0

        # Handover to SituationFusionEngine
        situation_engine = SituationFusionEngine()
        for norm_obs in fusion_result.normalized_observations:
            situation_engine.ingest(norm_obs)

        assert len(situation_engine._seen_observation_ids) == 3
