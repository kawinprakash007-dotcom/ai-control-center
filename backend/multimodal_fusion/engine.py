"""ATLAS Phase 6.5d — Temporal & Cross-Modal Fusion Engine.

Authoritative implementation of pre-situation evidence organization:
1. Normalizes input ordering deterministically.
2. Detects and tracks duplicate evidence.
3. Evaluates freshness (VALID, STALE, EXPIRED).
4. Evaluates temporal and spatial relationships between observation pairs.
5. Evaluates modality compatibility and source diversity.
6. Flags contradictions without resolving or mutating WorldState.
7. Groups related evidence into bounded FusionCluster instances with deterministic IDs.
8. Produces immutable FusionResult records.

CRITICAL ARCHITECTURAL BOUNDARIES:
- NEVER creates Situations, Missions, Goals, or Tasks.
- NEVER mutates WorldState, GoalStore, or application state.
- NEVER executes tools, actuates devices, or invokes model inference.
"""

from __future__ import annotations

import collections
import hashlib
import logging
import math
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

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
from core.models.device_contract import sanitize_contract_metadata
from core.models.orchestration import GeoLocation, ModalityType, MultimodalObservation
from core.models.perception import PerceptionEvidence
from multimodal_fusion.compatibility import (
    calculate_source_diversity,
    detect_contradiction,
    evaluate_modality_compatibility,
)
from multimodal_fusion.spatial import correlate_spatial_evidence
from multimodal_fusion.temporal import (
    build_temporal_window,
    evaluate_temporal_relation,
)

logger = logging.getLogger("atlas.multimodal_fusion")


class TemporalCrossModalFusionEngine:
    """
    Deterministic, model-neutral Temporal & Cross-Modal Fusion Engine.
    Organizes multi-source perception streams into structured evidence clusters.
    """

    def __init__(self, limits: Optional[FusionLimits] = None) -> None:
        self.limits = limits or FusionLimits()
        # Bounded duplicate cache: keeps track of recent seen identifiers
        self._duplicate_cache: collections.deque = collections.deque(
            maxlen=self.limits.max_duplicate_cache
        )
        self._seen_ids: Set[str] = set()

    def clear_cache(self) -> None:
        """Clear duplicate tracking cache (useful for test isolation)."""
        self._duplicate_cache.clear()
        self._seen_ids.clear()

    def fuse(
        self,
        observations: Sequence[Union[MultimodalObservation, PerceptionEvidence]],
        reference_time: Optional[float] = None,
        correlation_id: str = "",
        causation_id: Optional[str] = None,
    ) -> FusionResult:
        """
        Execute deterministic temporal and cross-modal fusion on a batch of observations/evidence.
        """
        if not observations:
            # Empty fusion result
            now = reference_time if reference_time is not None else 0.0
            tw = TemporalWindow(start_time=now, end_time=now, max_duration=self.limits.max_temporal_window_seconds)
            return FusionResult(
                result_id="fres_empty",
                temporal_window=tw,
                created_at=now,
            )

        # Enforce input capacity limit
        capped_inputs = observations[: self.limits.max_input_observations]

        # 1. Normalize Item Descriptors & Deterministic Sorting
        items = [self._extract_item_descriptor(obs) for obs in capped_inputs]
        # Sort deterministically by (timestamp, source_id, id)
        items.sort(key=lambda x: (x["timestamp"], x["source_id"], x["id"]))

        now = reference_time if reference_time is not None else max(x["timestamp"] for x in items)

        # 2. Build Temporal Window
        timestamps = [x["timestamp"] for x in items]
        tw = build_temporal_window(
            timestamps=timestamps,
            max_duration=self.limits.max_temporal_window_seconds,
            reference_timestamp=now,
        )

        # 3. Duplicate Detection and Freshness Evaluation
        duplicate_ids: List[str] = []
        relationships: List[EvidenceRelationship] = []
        contradictions: List[EvidenceRelationship] = []
        valid_items: List[Dict[str, Any]] = []

        for item in items:
            item_id = item["id"]
            # Deduplication
            if item_id in self._seen_ids:
                duplicate_ids.append(item_id)
                relationships.append(
                    EvidenceRelationship(
                        relation_type=EvidenceRelationType.DUPLICATES,
                        source_evidence_id=item_id,
                        target_evidence_id=item_id,
                        confidence=1.0,
                        reason=f"Observation '{item_id}' previously ingested",
                        created_at=now,
                    )
                )
                continue

            # Record in bounded cache
            if len(self._duplicate_cache) >= self.limits.max_duplicate_cache:
                oldest = self._duplicate_cache.popleft()
                self._seen_ids.discard(oldest)
            self._duplicate_cache.append(item_id)
            self._seen_ids.add(item_id)

            # Freshness
            age = max(0.0, now - item["timestamp"])
            if age > self.limits.expiration_threshold_seconds:
                item["freshness"] = FusionFreshnessStatus.EXPIRED
            elif age > self.limits.freshness_threshold_seconds:
                item["freshness"] = FusionFreshnessStatus.STALE
            else:
                item["freshness"] = FusionFreshnessStatus.VALID

            valid_items.append(item)

        # 4. Pairwise Relationship Analysis
        num_items = len(valid_items)
        for i in range(num_items):
            for j in range(i + 1, num_items):
                if len(relationships) >= self.limits.max_evidence_relationships:
                    break

                it1 = valid_items[i]
                it2 = valid_items[j]

                # A. Temporal Relation
                t_rel = evaluate_temporal_relation(it1["timestamp"], it2["timestamp"])

                # B. Spatial Relation
                s_link = correlate_spatial_evidence(
                    source_id_1=it1["source_id"],
                    source_id_2=it2["source_id"],
                    location_1=it1["location"],
                    location_2=it2["location"],
                    max_distance_meters=self.limits.max_spatial_distance_meters,
                )

                # C. Modality Compatibility
                m_comp = evaluate_modality_compatibility(it1["modality"], it2["modality"])

                # D. Contradiction Check
                is_contra, contra_reason = detect_contradiction(
                    label_1=it1["label"],
                    label_2=it2["label"],
                    attributes_1=it1["attributes"],
                    attributes_2=it2["attributes"],
                )

                if is_contra:
                    contra_rel = EvidenceRelationship(
                        relation_type=EvidenceRelationType.CONTRADICTS,
                        source_evidence_id=it1["id"],
                        target_evidence_id=it2["id"],
                        confidence=round(min(it1["confidence"], it2["confidence"]), 4),
                        reason=contra_reason,
                        temporal_relation=t_rel,
                        spatial_link=s_link,
                        created_at=now,
                    )
                    contradictions.append(contra_rel)
                    relationships.append(contra_rel)
                    continue

                # E. Corroboration & Support Evaluation
                is_corroboration = False
                rel_type = EvidenceRelationType.SUPPORTS

                # Complementary modalities or colocated independent sources corroborate
                if it1["source_id"] != it2["source_id"]:
                    if s_link is not None or (t_rel in (TemporalRelation.COINCIDENT, TemporalRelation.OVERLAPS)):
                        if m_comp == ModalityCompatibilityLevel.COMPLEMENTARY:
                            is_corroboration = True
                            rel_type = EvidenceRelationType.CORROBORATES
                        elif it1["label"] and it2["label"] and it1["label"] == it2["label"]:
                            is_corroboration = True
                            rel_type = EvidenceRelationType.CORROBORATES

                # Time sequence from same source
                if it1["source_id"] == it2["source_id"] and t_rel == TemporalRelation.BEFORE:
                    rel_type = EvidenceRelationType.PRECEDES

                # Entity candidate matching
                e1 = it1["attributes"].get("entity_id", it1["attributes"].get("target_id"))
                e2 = it2["attributes"].get("entity_id", it2["attributes"].get("target_id"))
                if e1 and e2 and e1 == e2:
                    rel_type = EvidenceRelationType.SAME_ENTITY_CANDIDATE

                rel_conf = round(min(it1["confidence"], it2["confidence"]) * (1.0 if not s_link else s_link.spatial_confidence), 4)

                rel = EvidenceRelationship(
                    relation_type=rel_type,
                    source_evidence_id=it1["id"],
                    target_evidence_id=it2["id"],
                    confidence=rel_conf,
                    reason=f"{it1['modality'].value} <-> {it2['modality'].value} ({m_comp.value})",
                    temporal_relation=t_rel,
                    spatial_link=s_link,
                    created_at=now,
                )
                relationships.append(rel)

        # 5. Cluster Construction
        clusters = self._construct_clusters(valid_items, relationships, tw, now)

        # 6. Cross-Modal Correlations
        correlations = self._build_correlations(clusters, relationships, now)

        # 7. Normalized Multimodal Observations
        normalized_obs = tuple(
            obs if isinstance(obs, MultimodalObservation) else self._evidence_to_observation(obs)
            for obs in capped_inputs
        )

        res_id = FusionResult.generate_deterministic_result_id(tw, len(capped_inputs))

        return FusionResult(
            result_id=res_id,
            temporal_window=tw,
            clusters=tuple(clusters[: self.limits.max_clusters]),
            relationships=tuple(relationships[: self.limits.max_evidence_relationships]),
            correlations=tuple(correlations[: self.limits.max_clusters]),
            contradictions=tuple(contradictions),
            duplicate_evidence_ids=tuple(duplicate_ids),
            normalized_observations=tuple(normalized_obs),
            provenance={
                "engine": "TemporalCrossModalFusionEngine",
                "version": "1.0.0",
                "correlation_id": correlation_id or res_id,
                "causation_id": causation_id,
                "input_count": len(capped_inputs),
                "cluster_count": len(clusters),
            },
            created_at=now,
        )

    def _construct_clusters(
        self,
        items: List[Dict[str, Any]],
        relationships: List[EvidenceRelationship],
        window: TemporalWindow,
        now: float,
    ) -> List[FusionCluster]:
        """Group connected and corroborated items into pre-situation FusionClusters."""
        if not items:
            return []

        # Build adjacency graph based on non-contradictory relationships
        adj: Dict[str, Set[str]] = {it["id"]: set() for it in items}
        for rel in relationships:
            if rel.relation_type != EvidenceRelationType.CONTRADICTS:
                adj[rel.source_evidence_id].add(rel.target_evidence_id)
                adj[rel.target_evidence_id].add(rel.source_evidence_id)

        # Connected components traversal
        visited: Set[str] = set()
        components: List[List[str]] = []

        for it in items:
            iid = it["id"]
            if iid not in visited:
                comp: List[str] = []
                queue = [iid]
                visited.add(iid)
                while queue:
                    curr = queue.pop(0)
                    comp.append(curr)
                    for neighbor in sorted(adj.get(curr, set())):
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)
                components.append(comp)

        item_map = {it["id"]: it for it in items}
        clusters: List[FusionCluster] = []

        for comp in components:
            if len(clusters) >= self.limits.max_clusters:
                break

            comp_items = [item_map[cid] for cid in comp if cid in item_map]
            comp_ev_ids = [it["id"] for it in comp_items][: self.limits.max_evidence_per_cluster]
            comp_src_ids = list(set(it["source_id"] for it in comp_items))[: self.limits.max_sources_per_cluster]
            comp_mods = list(set(it["modality"] for it in comp_items))[: self.limits.max_modalities_per_cluster]

            cid = FusionCluster.generate_deterministic_id(
                evidence_ids=comp_ev_ids,
                source_ids=comp_src_ids,
                start_time=window.start_time,
                end_time=window.end_time,
            )

            # Cluster relationships
            comp_set = set(comp_ev_ids)
            comp_rels = tuple(
                r for r in relationships
                if r.source_evidence_id in comp_set and r.target_evidence_id in comp_set
            )

            contra_count = sum(1 for r in comp_rels if r.relation_type == EvidenceRelationType.CONTRADICTS)
            diversity = calculate_source_diversity([it["source_id"] for it in comp_items])

            avg_conf = sum(it["confidence"] for it in comp_items) / len(comp_items) if comp_items else 1.0

            cluster = FusionCluster(
                cluster_id=cid,
                evidence_ids=tuple(comp_ev_ids),
                source_ids=tuple(comp_src_ids),
                modalities=tuple(comp_mods),
                temporal_window=window,
                relationships=comp_rels,
                confidence=round(avg_conf, 4),
                source_diversity=diversity,
                contradiction_count=contra_count,
            )
            clusters.append(cluster)

        # Sort clusters deterministically by cluster_id
        clusters.sort(key=lambda c: c.cluster_id)
        return clusters

    def _build_correlations(
        self,
        clusters: List[FusionCluster],
        relationships: List[EvidenceRelationship],
        now: float,
    ) -> List[CrossModalCorrelation]:
        """Translate multi-item clusters into CrossModalCorrelation summaries."""
        correlations: List[CrossModalCorrelation] = []
        for cluster in clusters:
            corr_id = f"corr_{cluster.cluster_id}"
            has_corroboration = any(
                r.relation_type == EvidenceRelationType.CORROBORATES for r in cluster.relationships
            )
            rel_type = EvidenceRelationType.CORROBORATES if has_corroboration else EvidenceRelationType.SUPPORTS

            corr = CrossModalCorrelation(
                correlation_id=corr_id,
                evidence_ids=cluster.evidence_ids,
                participating_modalities=cluster.modalities,
                participating_sources=cluster.source_ids,
                temporal_relationship=TemporalRelation.WITHIN_WINDOW,
                relationship_type=rel_type,
                confidence=cluster.confidence,
                source_diversity=cluster.source_diversity,
                created_at=now,
            )
            correlations.append(corr)
        return correlations

    def _extract_item_descriptor(
        self, item: Union[MultimodalObservation, PerceptionEvidence]
    ) -> Dict[str, Any]:
        """Extract uniform semantic fields from either MultimodalObservation or PerceptionEvidence."""
        if isinstance(item, MultimodalObservation):
            label = ""
            attrs: Dict[str, Any] = {}
            if isinstance(item.payload, dict):
                label = item.payload.get("label", item.payload.get("semantic_type", ""))
                attrs = sanitize_contract_metadata(dict(item.payload.get("attributes", item.payload)))
            elif isinstance(item.payload, str):
                label = item.payload

            return {
                "id": item.observation_id,
                "source_id": item.source_id,
                "modality": item.modality,
                "timestamp": item.timestamp,
                "confidence": item.confidence,
                "location": item.location,
                "label": label,
                "attributes": attrs,
                "correlation_id": item.correlation_id,
                "causation_id": item.causation_id,
            }
        elif isinstance(item, PerceptionEvidence):
            loc = item.spatial.location if item.spatial else None
            return {
                "id": item.evidence_id,
                "source_id": item.source_id,
                "modality": item.modality,
                "timestamp": item.timestamp,
                "confidence": item.confidence,
                "location": loc,
                "label": item.label or item.semantic_type,
                "attributes": sanitize_contract_metadata(dict(item.attributes)),
                "correlation_id": item.correlation_id,
                "causation_id": item.causation_id,
            }
        else:
            raise TypeError(f"Expected MultimodalObservation or PerceptionEvidence, got {type(item)}")

    def _evidence_to_observation(self, ev: PerceptionEvidence) -> MultimodalObservation:
        """Convert a PerceptionEvidence item into a canonical MultimodalObservation."""
        loc = ev.spatial.location if ev.spatial else None
        sanitized_attrs = sanitize_contract_metadata(dict(ev.attributes))
        return MultimodalObservation(
            observation_id=f"obs_fused_{ev.evidence_id}",
            source_id=ev.source_id,
            source_type="perception_evidence",
            modality=ev.modality,
            timestamp=ev.timestamp if ev.timestamp > 0.0 else 1.0,
            payload={"semantic_type": ev.semantic_type, "label": ev.label, "attributes": sanitized_attrs},
            confidence=ev.confidence,
            location=loc,
            correlation_id=ev.correlation_id,
            causation_id=ev.causation_id,
            metadata={"provenance": ev.provenance},
        )
