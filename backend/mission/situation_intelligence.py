"""
ATLAS Phase 6.4 — Multi-Product Situation Intelligence Engine.

Synthesizes unified MultiProductSituation instances by correlating evidence across
multiple heterogeneous edge products (Vision, Glass, Drone, Rover).
Implements source-diverse corroboration, explicit contradiction tracking,
and deterministic entity correlation.

CRITICAL ARCHITECTURAL RULES:
1. NO SECOND BRAIN: Consumes Situation objects from SituationFusionEngine.
2. NO DIRECT EXECUTION: Does not invoke tools, device gateways, or LLMs.
3. PRESERVES EVIDENCE: Contradictions are preserved explicitly; no 'last observation wins'.
4. SOURCE DIVERSITY: Multi-product corroboration strictly requires independent sources.
"""

from collections import deque
import hashlib
import math
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from core.interfaces.mission_interface import MultiProductSituationIntelligenceInterface
from core.models.device_contract import ProductRole, ProductType
from core.models.mission import (
    EntityCorrelation,
    EntityCorrelationStatus,
    MissionLimits,
    MultiProductSituation,
    ProductEvidence,
    SituationContradiction,
)
from core.models.orchestration import (
    GeoLocation,
    ModalityType,
    Situation,
    SituationCategory,
    SituationSeverity,
    SituationStatus,
)


class MultiProductSituationIntelligenceEngine(MultiProductSituationIntelligenceInterface):
    """
    Deterministic intelligence engine for cross-product situation synthesis,
    entity correlation, and contradiction tracking.
    """

    def __init__(self, limits: Optional[MissionLimits] = None):
        self._lock = threading.RLock()
        self.limits = limits or MissionLimits()
        self._active_multi_situations: Dict[str, MultiProductSituation] = {}
        self._entity_correlations: Dict[str, EntityCorrelation] = {}
        self._contradictions: Dict[str, SituationContradiction] = {}
        self._history: deque = deque(maxlen=self.limits.max_situations_correlated)

    # ========================================================================
    # Primary Ingress & Evaluation
    # ========================================================================

    def evaluate_situations(
        self,
        situations: Sequence[Situation],
        world_state: Optional[Any] = None,
        events: Optional[Sequence[Any]] = None,
        now: Optional[float] = None,
    ) -> Sequence[MultiProductSituation]:
        """
        Evaluate and correlate individual Situation instances across products.
        Synthesizes unified MultiProductSituation records with source-diversity corroboration.
        """
        current_time = float(now if now is not None else time.time())
        with self._lock:
            if not situations:
                return tuple(self._active_multi_situations.values())

            # 1. Cluster situations by spatial proximity, category, or correlation_id
            clusters = self._cluster_situations(situations)

            results: List[MultiProductSituation] = []
            for cluster in clusters:
                mps = self._synthesize_cluster(cluster, current_time)
                self._active_multi_situations[mps.situation_id] = mps
                self._history.append(mps)
                results.append(mps)

            # Enforce capacity
            if len(self._active_multi_situations) > self.limits.max_situations_correlated:
                to_remove = len(self._active_multi_situations) - self.limits.max_situations_correlated
                keys = list(self._active_multi_situations.keys())[:to_remove]
                for k in keys:
                    del self._active_multi_situations[k]

            return tuple(results)

    # ========================================================================
    # Clustering & Synthesis
    # ========================================================================

    def _cluster_situations(self, situations: Sequence[Situation]) -> List[List[Situation]]:
        """
        Deterministic clustering based on shared correlation_id, spatial proximity,
        or compatible category and overlapping temporal windows.
        """
        clusters: List[List[Situation]] = []
        assigned: Set[str] = set()

        for sit in situations:
            if sit.situation_id in assigned:
                continue

            current_cluster = [sit]
            assigned.add(sit.situation_id)

            for other in situations:
                if other.situation_id in assigned:
                    continue

                if self._should_merge_situations(sit, other):
                    current_cluster.append(other)
                    assigned.add(other.situation_id)

            clusters.append(current_cluster)

        return clusters

    def _should_merge_situations(self, sit_a: Situation, sit_b: Situation) -> bool:
        """Evaluate if two situations represent aspects of the same multi-product incident."""
        # Rule 1: Explicit shared non-empty correlation ID
        if sit_a.correlation_id and sit_b.correlation_id and sit_a.correlation_id == sit_b.correlation_id:
            return True

        # Temporal boundary: situations separated by more than window cannot merge
        time_delta = abs(sit_a.updated_at - sit_b.updated_at)
        if time_delta > self.limits.temporal_correlation_window_s:
            return False

        # Rule 2: Shared involved entities
        shared_entities = set(sit_a.involved_entities).intersection(set(sit_b.involved_entities))
        if shared_entities:
            return True

        # Rule 3: Same category and spatial proximity (within 50 meters)
        if sit_a.category == sit_b.category and sit_a.location and sit_b.location:
            dist = self._calculate_distance(sit_a.location, sit_b.location)
            if dist <= 50.0:
                return True

        return False

    def _synthesize_cluster(self, cluster: Sequence[Situation], current_time: float) -> MultiProductSituation:
        """Synthesize a unified MultiProductSituation from a cluster of situations."""
        # Deterministic situation_id from sorted constituent situation IDs
        sorted_ids = sorted(s.situation_id for s in cluster)
        seed_str = ":".join(sorted_ids)
        sit_hash = hashlib.sha256(seed_str.encode()).hexdigest()[:12]
        mps_id = f"mps_{sit_hash}"

        # Aggregate evidence references
        evidence_list: List[ProductEvidence] = []
        involved_products: Set[ProductType] = set()
        involved_product_ids: Set[str] = set()
        involved_entities: Set[str] = set()
        supporting_situation_ids: List[str] = list(sorted_ids)
        categories: List[SituationCategory] = []
        severities: List[SituationSeverity] = []
        locations: List[GeoLocation] = []

        correlation_id = ""
        causation_id: Optional[str] = None

        for sit in cluster:
            categories.append(sit.category)
            severities.append(sit.severity)
            if sit.location:
                locations.append(sit.location)
            for ent in sit.involved_entities:
                involved_entities.add(ent)
            if sit.correlation_id and not correlation_id:
                correlation_id = sit.correlation_id
            if sit.causation_id and not causation_id:
                causation_id = sit.causation_id

            # Extract or convert evidence
            for ev in sit.supporting_evidence:
                prod_type, prod_role = self._infer_product_type_and_role(ev.source_id)
                involved_products.add(prod_type)
                involved_product_ids.add(ev.source_id)

                pe = ProductEvidence(
                    evidence_id=f"pe_{ev.evidence_id}",
                    source_id=ev.source_id,
                    product_type=prod_type,
                    product_role=prod_role,
                    observation_id=ev.observation_id,
                    situation_id=sit.situation_id,
                    timestamp=ev.timestamp,
                    confidence=ev.evidence_weight,
                    modality=ev.modality,
                    summary=ev.concise_summary,
                    data={"provenance": ev.provenance} if ev.provenance else {},
                    correlation_id=sit.correlation_id,
                    causation_id=sit.causation_id,
                )
                evidence_list.append(pe)

        # Detect contradictions within the cluster
        contradictions = self._detect_contradictions_for_cluster(cluster, evidence_list, current_time)

        # Correlate entities within the cluster
        entity_correlations = self._correlate_entities_for_cluster(cluster, evidence_list, current_time)

        # Synthesize confidence via source diversity & contradiction penalty
        confidence = self._synthesize_multi_confidence(evidence_list, bool(contradictions))

        # Synthesize dominant category and max severity
        dominant_cat = self._select_dominant_category(categories)
        max_sev = self._select_max_severity(severities)

        # Centroid location if available
        centroid_loc = self._compute_centroid(locations)

        # Recommended missions
        recommended_missions = self._recommend_missions(dominant_cat, max_sev, tuple(involved_products))

        title = f"Multi-Product {dominant_cat.value}: {len(involved_products)} products, {len(cluster)} situations"
        desc = (
            f"Synthesized from situations {sorted_ids}. "
            f"Involved products: {[p.value for p in sorted(involved_products, key=lambda p: p.value)]}. "
            f"Evidence count: {len(evidence_list)}. Contradictions: {len(contradictions)}."
        )

        return MultiProductSituation(
            situation_id=mps_id,
            category=dominant_cat,
            title=title,
            description=desc,
            severity=max_sev,
            confidence=confidence,
            status=SituationStatus.ACTIVE,
            involved_products=tuple(sorted(involved_products, key=lambda p: p.value)),
            involved_product_ids=tuple(sorted(involved_product_ids)),
            involved_entities=tuple(sorted(involved_entities)),
            supporting_situation_ids=tuple(supporting_situation_ids),
            evidence_references=tuple(evidence_list),
            contradictions=tuple(contradictions),
            entity_correlations=tuple(entity_correlations),
            location=centroid_loc,
            created_at=min((s.created_at for s in cluster), default=current_time),
            updated_at=current_time,
            correlation_id=correlation_id or f"corr_{sit_hash}",
            causation_id=causation_id,
            recommended_missions=tuple(recommended_missions),
        )

    # ========================================================================
    # Confidence Synthesis & Source Diversity
    # ========================================================================

    def _synthesize_multi_confidence(
        self,
        evidence: Sequence[ProductEvidence],
        has_contradiction: bool,
    ) -> float:
        """
        Deterministic, bounded confidence calculation.

        Rules:
        1. Base confidence is the weighted average of individual evidence confidences.
        2. Source diversity boost: Distinct product types (e.g. Vision + Drone + Glass)
           provide significant corroboration boost (+0.08 per extra distinct product, up to +0.20 max).
        3. Same-product repetition (e.g. 10 Vision frames) provides ZERO corroboration boost.
        4. Contradiction penalty: If unresolved contradiction exists, penalize -0.15.
        5. Strictly bounded in [0.0, 1.0].
        """
        if not evidence:
            return 0.0

        # Weighted average base confidence
        total_weight = sum(e.confidence for e in evidence)
        if total_weight <= 0.0:
            return 0.0

        base_conf = sum(e.confidence * e.confidence for e in evidence) / total_weight

        # Count distinct product types
        distinct_products = {e.product_type for e in evidence if e.product_type != ProductType.UNKNOWN}

        corroboration_boost = 0.0
        if len(distinct_products) >= 2:
            # Independent multi-product corroboration
            corroboration_boost = min(0.20, (len(distinct_products) - 1) * 0.08)

        # Contradiction penalty
        penalty = 0.15 if has_contradiction else 0.0

        final_conf = base_conf + corroboration_boost - penalty
        return round(max(0.0, min(1.0, final_conf)), 3)

    # ========================================================================
    # Contradiction Detection
    # ========================================================================

    def detect_contradictions(
        self,
        situations: Sequence[Situation],
        now: Optional[float] = None,
    ) -> Sequence[SituationContradiction]:
        """Detect and return all contradictions across situations."""
        current_time = float(now if now is not None else time.time())
        with self._lock:
            contradictions: List[SituationContradiction] = []
            for i in range(len(situations)):
                for j in range(i + 1, len(situations)):
                    c = self._evaluate_pairwise_contradiction(situations[i], situations[j], current_time)
                    if c:
                        contradictions.append(c)
                        self._contradictions[c.contradiction_id] = c
            return tuple(contradictions)

    def _detect_contradictions_for_cluster(
        self,
        cluster: Sequence[Situation],
        evidence_list: Sequence[ProductEvidence],
        current_time: float,
    ) -> List[SituationContradiction]:
        """Detect contradictions among situations and evidence inside a cluster."""
        contradictions: List[SituationContradiction] = []

        # 1. Situation-level pairwise check
        for i in range(len(cluster)):
            for j in range(i + 1, len(cluster)):
                c = self._evaluate_pairwise_contradiction(cluster[i], cluster[j], current_time)
                if c:
                    contradictions.append(c)
                    self._contradictions[c.contradiction_id] = c

        # 2. Evidence-level contradictory payload check (e.g. detected: True vs detected: False)
        for i in range(len(evidence_list)):
            for j in range(i + 1, len(evidence_list)):
                ev_a = evidence_list[i]
                ev_b = evidence_list[j]
                if ev_a.source_id == ev_b.source_id:
                    continue  # Only cross-source checks

                claim_a = str(ev_a.data.get("claim", ev_a.summary))
                claim_b = str(ev_b.data.get("claim", ev_b.summary))

                if self._detect_contradiction(claim_a, claim_b):
                    cid = f"contra_{ev_a.evidence_id}_{ev_b.evidence_id}"
                    contra = SituationContradiction(
                        contradiction_id=cid,
                        situation_id=ev_a.situation_id or "cluster",
                        source_a=ev_a.source_id,
                        source_b=ev_b.source_id,
                        claim_a=ev_a.summary,
                        claim_b=ev_b.summary,
                        confidence_a=ev_a.confidence,
                        confidence_b=ev_b.confidence,
                        timestamp_a=ev_a.timestamp,
                        timestamp_b=ev_b.timestamp,
                        resolution_status="UNRESOLVED",
                        detected_at=current_time,
                    )
                    contradictions.append(contra)
                    self._contradictions[cid] = contra

        return contradictions

    def _detect_contradiction(self, claim_a: str, claim_b: str) -> bool:
        """Helper to test semantic contradiction between two claims."""
        a = claim_a.lower()
        b = claim_b.lower()
        pairs = [
            ("detected", "not detected"),
            ("detected", "no person"),
            ("detected", "no target"),
            ("detected", "no intruder"),
            ("present", "not present"),
            ("present", "absent"),
            ("present", "no person"),
            ("breach", "clear"),
            ("breach", "secure"),
            ("breached", "clear"),
            ("breached", "secure"),
            ("movement", "false alarm"),
            ("observed", "false alarm"),
            ("blocked", "clear"),
            ("occupied", "empty"),
        ]
        for term1, term2 in pairs:
            if (term1 in a and term2 in b) or (term1 in b and term2 in a):
                return True
        return False

    def _evaluate_pairwise_contradiction(
        self,
        sit_a: Situation,
        sit_b: Situation,
        current_time: float,
    ) -> Optional[SituationContradiction]:
        """Check if two situations assert mutually contradictory facts."""
        text_a = f"{sit_a.title} {sit_a.description}"
        text_b = f"{sit_b.title} {sit_b.description}"

        if self._detect_contradiction(text_a, text_b):
            cid = f"contra_{sit_a.situation_id}_{sit_b.situation_id}"
            return SituationContradiction(
                contradiction_id=cid,
                situation_id=sit_a.situation_id,
                source_a=sit_a.situation_id,
                source_b=sit_b.situation_id,
                claim_a=sit_a.title,
                claim_b=sit_b.title,
                confidence_a=sit_a.confidence,
                confidence_b=sit_b.confidence,
                timestamp_a=sit_a.updated_at,
                timestamp_b=sit_b.updated_at,
                resolution_status="UNRESOLVED",
                detected_at=current_time,
            )
        return None

    # ========================================================================
    # Entity Correlation
    # ========================================================================

    def correlate_entities(
        self,
        situations: Sequence[Situation],
        now: Optional[float] = None,
    ) -> Sequence[EntityCorrelation]:
        """Correlate entities observed across different products."""
        current_time = float(now if now is not None else time.time())
        with self._lock:
            evidence_all: List[ProductEvidence] = []
            for s in situations:
                for ev in s.supporting_evidence:
                    ptype, prole = self._infer_product_type_and_role(ev.source_id)
                    evidence_all.append(
                        ProductEvidence(
                            evidence_id=f"pe_{ev.evidence_id}",
                            source_id=ev.source_id,
                            product_type=ptype,
                            product_role=prole,
                            observation_id=ev.observation_id,
                            situation_id=s.situation_id,
                            timestamp=ev.timestamp,
                            confidence=ev.evidence_weight,
                            modality=ev.modality,
                            summary=ev.concise_summary,
                        )
                    )
            return tuple(self._correlate_entities_for_cluster(situations, evidence_all, current_time))

    def _correlate_entities_for_cluster(
        self,
        cluster: Sequence[Situation],
        evidence_list: Sequence[ProductEvidence],
        current_time: float,
    ) -> List[EntityCorrelation]:
        """Identify candidate or correlated entities across products."""
        correlations: List[EntityCorrelation] = []

        all_entities: List[Tuple[str, str, float]] = []  # (entity_id, source_id, timestamp)
        for sit in cluster:
            for ent in sit.involved_entities:
                all_entities.append((ent, sit.situation_id, sit.updated_at))

        for i in range(len(all_entities)):
            for j in range(i + 1, len(all_entities)):
                e1, src1, t1 = all_entities[i]
                e2, src2, t2 = all_entities[j]
                if e1 == e2:
                    continue  # Identical entity name

                # Semantic entity type compatibility (e.g. 'person_17' and 'person_A')
                type1 = self._infer_entity_type(e1)
                type2 = self._infer_entity_type(e2)

                if type1 == type2 and type1 != "UNKNOWN":
                    time_delta = abs(t1 - t2)
                    cid = f"ecorr_{e1}_{e2}"

                    # If within 60s, candidate or correlated
                    if time_delta <= 60.0:
                        status = EntityCorrelationStatus.CORRELATED if time_delta <= 15.0 else EntityCorrelationStatus.CANDIDATE
                        conf = round(max(0.2, 0.95 - (time_delta * 0.01)), 2)
                        ecorr = EntityCorrelation(
                            correlation_id=cid,
                            primary_entity_id=e1,
                            correlated_entity_id=e2,
                            entity_type=type1,
                            status=status,
                            confidence=conf,
                            supporting_evidence_ids=(src1, src2),
                            time_delta_seconds=time_delta,
                            updated_at=current_time,
                        )
                        correlations.append(ecorr)
                        self._entity_correlations[cid] = ecorr

        return correlations

    def _infer_entity_type(self, entity_str: str) -> str:
        """Derive standard entity type from string token."""
        norm = entity_str.upper()
        if "PERSON" in norm or "HUMAN" in norm or "INTRUDER" in norm:
            return "PERSON"
        if "VEHICLE" in norm or "CAR" in norm or "TRUCK" in norm:
            return "VEHICLE"
        if "OBSTACLE" in norm or "GATE" in norm or "DOOR" in norm:
            return "OBSTACLE"
        if "HAZARD" in norm or "FIRE" in norm or "SMOKE" in norm:
            return "HAZARD"
        if "DRONE" in norm or "ROVER" in norm or "GLASS" in norm or "VISION" in norm:
            return "DEVICE"
        return "UNKNOWN"

    # ========================================================================
    # State Accessors
    # ========================================================================

    def get_active_multi_situations(self) -> Sequence[MultiProductSituation]:
        with self._lock:
            return tuple(self._active_multi_situations.values())

    def get_multi_situation(self, situation_id: str) -> Optional[MultiProductSituation]:
        with self._lock:
            return self._active_multi_situations.get(situation_id)

    def get_contradictions(self) -> Sequence[SituationContradiction]:
        with self._lock:
            return tuple(self._contradictions.values())

    def get_entity_correlations(self) -> Sequence[EntityCorrelation]:
        with self._lock:
            return tuple(self._entity_correlations.values())

    def record_entity_correlation(
        self,
        primary_entity_id: str,
        correlated_entity_id: str,
        entity_type: str,
        status: EntityCorrelationStatus,
        confidence: float,
        supporting_evidence_ids: Sequence[str] = (),
        time_delta_seconds: float = 0.0,
        now: Optional[float] = None,
    ) -> EntityCorrelation:
        """Explicitly record or update an entity correlation."""
        current_time = float(now if now is not None else time.time())
        cid = f"ecorr_{primary_entity_id}_{correlated_entity_id}"
        ecorr = EntityCorrelation(
            correlation_id=cid,
            primary_entity_id=primary_entity_id,
            correlated_entity_id=correlated_entity_id,
            entity_type=entity_type,
            status=status,
            confidence=confidence,
            supporting_evidence_ids=tuple(supporting_evidence_ids),
            time_delta_seconds=time_delta_seconds,
            updated_at=current_time,
        )
        with self._lock:
            self._entity_correlations[cid] = ecorr
            return ecorr

    def get_entity_correlation(self, primary_id: str, correlated_id: str) -> Optional[EntityCorrelation]:
        """Look up an entity correlation in either direction."""
        with self._lock:
            cid1 = f"ecorr_{primary_id}_{correlated_id}"
            cid2 = f"ecorr_{correlated_id}_{primary_id}"
            return self._entity_correlations.get(cid1) or self._entity_correlations.get(cid2)

    def clear(self) -> None:
        with self._lock:
            self._active_multi_situations.clear()
            self._entity_correlations.clear()
            self._contradictions.clear()
            self._history.clear()

    # ========================================================================
    # Helpers
    # ========================================================================

    def _infer_product_type_and_role(self, source_id: str) -> Tuple[ProductType, ProductRole]:
        """Deterministically infer ProductType and ProductRole from device ID string."""
        sid = source_id.upper()
        if "VISION" in sid or "CAMERA" in sid:
            return ProductType.VISION, ProductRole.OBSERVATION_SOURCE
        if "GLASS" in sid or "WEARABLE" in sid:
            return ProductType.GLASS, ProductRole.HYBRID
        if "DRONE" in sid or "UAV" in sid:
            return ProductType.DRONE, ProductRole.HYBRID
        if "ROVER" in sid or "UGV" in sid:
            return ProductType.ROVER, ProductRole.HYBRID
        return ProductType.UNKNOWN, ProductRole.OBSERVATION_SOURCE

    def _calculate_distance(self, loc1: GeoLocation, loc2: GeoLocation) -> float:
        """Euclidean ground distance approximation in meters."""
        dlat = (loc1.latitude - loc2.latitude) * 111000.0
        dlon = (loc1.longitude - loc2.longitude) * 111000.0 * math.cos(math.radians(loc1.latitude))
        return math.sqrt(dlat * dlat + dlon * dlon)

    def _compute_centroid(self, locations: Sequence[GeoLocation]) -> Optional[GeoLocation]:
        if not locations:
            return None
        avg_lat = sum(loc.latitude for loc in locations) / len(locations)
        avg_lon = sum(loc.longitude for loc in locations) / len(locations)
        avg_alt = sum(loc.altitude or 0.0 for loc in locations) / len(locations)
        return GeoLocation(latitude=avg_lat, longitude=avg_lon, altitude=avg_alt)

    def _select_dominant_category(self, categories: Sequence[SituationCategory]) -> SituationCategory:
        """Select highest priority category (SECURITY > ANOMALY > NAVIGATIONAL > ENVIRONMENTAL > OPERATIONAL)."""
        priority_order = [
            SituationCategory.SECURITY,
            SituationCategory.ANOMALY,
            SituationCategory.NAVIGATIONAL,
            SituationCategory.ENVIRONMENTAL,
            SituationCategory.OPERATIONAL,
            SituationCategory.SYSTEM_HEALTH,
            SituationCategory.USER_INTERACTION,
            SituationCategory.UNKNOWN,
        ]
        for p in priority_order:
            if p in categories:
                return p
        return SituationCategory.UNKNOWN

    def _select_max_severity(self, severities: Sequence[SituationSeverity]) -> SituationSeverity:
        """Select highest severity."""
        sev_rank = {
            SituationSeverity.CRITICAL: 5,
            SituationSeverity.HIGH: 4,
            SituationSeverity.MEDIUM: 3,
            SituationSeverity.LOW: 2,
            SituationSeverity.INFO: 1,
        }
        return max(severities, key=lambda s: sev_rank.get(s, 0), default=SituationSeverity.INFO)

    def _recommend_missions(
        self,
        category: SituationCategory,
        severity: SituationSeverity,
        involved_products: Tuple[ProductType, ...],
    ) -> List[str]:
        """Recommend candidate tactical mission templates based on situation profile."""
        recs: List[str] = []
        if category == SituationCategory.SECURITY:
            recs.append("PERIMETER_SECURITY_RESPONSE")
            recs.append("LOCATE_AND_VERIFY")
        elif category == SituationCategory.ANOMALY:
            recs.append("ANOMALY_INVESTIGATION")
        elif category == SituationCategory.ENVIRONMENTAL:
            recs.append("HAZARD_CONTAINMENT")
        else:
            recs.append("SURVEILLANCE_PATROL")
        return recs
