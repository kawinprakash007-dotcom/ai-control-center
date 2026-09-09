import collections
from dataclasses import dataclass, field
import hashlib
import json
import logging
import math
import threading
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from core.interfaces.orchestration_interface import SituationFusionInterface
from core.interfaces.runtime_interface import CognitiveEventSinkInterface
from core.interfaces.world_interface import ConflictResolverInterface, WorldStateStoreInterface
from core.models.orchestration import (
    GeoLocation,
    ModalityType,
    MultimodalObservation,
    Situation,
    SituationCategory,
    SituationEvidence,
    SituationSeverity,
    SituationStatus,
)
from core.models.runtime import CognitiveEvent, CognitiveEventType, CognitiveStage
from core.models.world_state import ConflictPolicy, StateConflict, WorldCondition
from world.resolver import DeterministicConflictResolver

logger = logging.getLogger("atlas.orchestration.fusion")


@dataclass(frozen=True)
class SituationFusionConfig:
    """
    Deterministic configuration parameters for SituationFusionEngine.
    """
    max_temporal_distance_seconds: float = 30.0
    max_spatial_distance_meters: float = 50.0
    max_batch_size: int = 500
    max_active_situations: int = 100
    max_evidence_per_situation: int = 50
    max_history_entries: int = 500
    stale_observation_threshold_seconds: float = 300.0
    future_slack_seconds: float = 5.0
    default_validity_window_seconds: float = 120.0
    conflict_confidence_penalty: float = 0.20
    independent_source_boost: float = 0.05
    independent_modality_boost: float = 0.05
    max_corroboration_boost: float = 0.20
    min_corroboration_count_for_active: int = 2


class SituationFusionEngine(SituationFusionInterface):
    """
    Deterministic, model-neutral SituationFusionEngine for ATLAS Central Orchestration Layer.

    Responsibilities:
    - Ingests and correlates MultimodalObservations across modalities, time, and space.
    - Deterministically merges related evidence into unified Situation objects.
    - Synthesizes confidence from source diversity, modality diversity, and contradictions.
    - Detects conflicting observations reusing DeterministicConflictResolver.
    - Maintains immutable Situation lifecycles (DETECTED -> ACTIVE -> UPDATING -> RESOLVED/EXPIRED).

    CRITICAL ARCHITECTURAL BOUNDARIES:
    - Zero execution of ToolCalls or external tool execution engines.
    - Zero direct hardware, device gateway, or adapter calls.
    - Zero process spawning, shell execution, or socket/network calls.
    - Zero model/LLM/VLM inference or model router reasoning.
    - Zero world state mutations (read-only queries only).
    - Zero goal creation or goal store mutations.
    """

    def __init__(
        self,
        config: Optional[SituationFusionConfig] = None,
        conflict_resolver: Optional[ConflictResolverInterface] = None,
        world_state_store: Optional[WorldStateStoreInterface] = None,
        event_sink: Optional[CognitiveEventSinkInterface] = None,
        clock: Optional[Callable[[], float]] = None,
        conflict_policy: Optional[ConflictPolicy] = None,
    ):
        self.config = config or SituationFusionConfig()
        self.conflict_resolver = conflict_resolver or DeterministicConflictResolver()
        self.world_state_store = world_state_store
        self.event_sink = event_sink
        self.clock = clock or time.time
        self.conflict_policy = conflict_policy or ConflictPolicy()

        self._lock = threading.Lock()
        self._active_situations: Dict[str, Situation] = {}
        self._situation_signatures: Dict[str, str] = {}
        self._seen_observation_ids: collections.deque = collections.deque(maxlen=5000)
        self._seen_observation_set: Set[str] = set()
        self._recent_source_hashes: Dict[str, Tuple[int, float]] = {}
        self._situation_history: collections.deque = collections.deque(
            maxlen=self.config.max_history_entries
        )

    # ========================================================================
    # Public Ingestion & Interface API
    # ========================================================================

    def ingest(
        self,
        observation: MultimodalObservation,
        now: Optional[float] = None,
    ) -> Optional[Situation]:
        """
        Ingest and correlate a single MultimodalObservation.
        Returns the created or updated Situation, or None if suppressed.
        """
        results = self.ingest_batch([observation], now=now)
        return results[0] if results else None

    def ingest_batch(
        self,
        observations: Sequence[MultimodalObservation],
        now: Optional[float] = None,
    ) -> Sequence[Situation]:
        """
        Ingest, validate, sort, and correlate a batch of MultimodalObservations.
        Guarantees deterministic ordering regardless of arrival sequence.
        Returns the latest state of all affected Situations.
        """
        if not observations:
            return ()

        if len(observations) > self.config.max_batch_size:
            raise ValueError(
                f"Batch size ({len(observations)}) exceeds maximum limit of {self.config.max_batch_size}"
            )

        current_time = now if now is not None else self.clock()

        # Deterministic sorting by observation timestamp, then ID for stable tie-breaking
        sorted_obs = sorted(observations, key=lambda o: (o.timestamp, o.observation_id))

        affected_situations: Dict[str, Situation] = {}

        with self._lock:
            for obs in sorted_obs:
                sit = self._process_single_observation(obs, current_time)
                if sit is not None:
                    affected_situations[sit.situation_id] = sit

            self._enforce_capacity(current_time)

        # Deterministic output ordering by situation_id
        return tuple(affected_situations[k] for k in sorted(affected_situations.keys()))

    def evaluate_observations(
        self,
        observations: Sequence[MultimodalObservation],
    ) -> Sequence[Situation]:
        """
        Implements SituationFusionInterface.evaluate_observations.
        """
        return self.ingest_batch(observations)

    def get_active_situations(self, now: Optional[float] = None) -> Sequence[Situation]:
        """
        Implements SituationFusionInterface.get_active_situations.
        Returns all unexpired, active, or detected Situations.
        """
        current_time = now if now is not None else self.clock()
        with self._lock:
            active = []
            for s in self._active_situations.values():
                if s.is_valid(current_time):
                    active.append(s)
            # Deterministic sorting by updated_at descending, then situation_id
            return tuple(sorted(active, key=lambda x: (-x.updated_at, x.situation_id)))

    def get_situation(self, situation_id: str) -> Optional[Situation]:
        """Retrieve a specific Situation by ID if active or in recent history."""
        with self._lock:
            if situation_id in self._active_situations:
                return self._active_situations[situation_id]
            for s in reversed(self._situation_history):
                if s.situation_id == situation_id:
                    return s
            return None

    def resolve_situation(
        self,
        situation_id: str,
        reason: str = "",
        now: Optional[float] = None,
    ) -> Optional[Situation]:
        """
        Explicitly transition an active situation to RESOLVED.
        """
        current_time = now if now is not None else self.clock()
        with self._lock:
            sit = self._active_situations.get(situation_id)
            if not sit:
                return None

            resolved = Situation(
                situation_id=sit.situation_id,
                category=sit.category,
                title=sit.title,
                description=sit.description if not reason else f"{sit.description} [Resolved: {reason}]",
                severity=SituationSeverity.INFO,
                confidence=sit.confidence,
                status=SituationStatus.RESOLVED,
                involved_entities=sit.involved_entities,
                supporting_evidence=sit.supporting_evidence,
                location=sit.location,
                created_at=sit.created_at,
                updated_at=current_time,
                valid_until=sit.valid_until,
                related_event_ids=sit.related_event_ids,
                related_goal_ids=sit.related_goal_ids,
                correlation_id=sit.correlation_id,
                causation_id=sit.causation_id,
                metadata={**sit.metadata, "resolution_reason": reason, "resolved_at": current_time},
            )

            self._active_situations[situation_id] = resolved
            self._emit_event(CognitiveEventType.SITUATION_UPDATED, resolved, "Situation resolved")
            return resolved

    def dismiss_situation(
        self,
        situation_id: str,
        reason: str = "",
        now: Optional[float] = None,
    ) -> Optional[Situation]:
        """Explicitly dismiss a false-positive or invalidated situation."""
        current_time = now if now is not None else self.clock()
        with self._lock:
            sit = self._active_situations.get(situation_id)
            if not sit:
                return None

            dismissed = Situation(
                situation_id=sit.situation_id,
                category=sit.category,
                title=sit.title,
                description=sit.description if not reason else f"{sit.description} [Dismissed: {reason}]",
                severity=SituationSeverity.INFO,
                confidence=sit.confidence,
                status=SituationStatus.DISMISSED,
                involved_entities=sit.involved_entities,
                supporting_evidence=sit.supporting_evidence,
                location=sit.location,
                created_at=sit.created_at,
                updated_at=current_time,
                valid_until=sit.valid_until,
                related_event_ids=sit.related_event_ids,
                related_goal_ids=sit.related_goal_ids,
                correlation_id=sit.correlation_id,
                causation_id=sit.causation_id,
                metadata={**sit.metadata, "dismissed_reason": reason, "dismissed_at": current_time},
            )
            self._active_situations[situation_id] = dismissed
            return dismissed

    def expire_stale_situations(self, now: Optional[float] = None) -> Sequence[Situation]:
        """Check all active situations and transition expired ones to EXPIRED."""
        current_time = now if now is not None else self.clock()
        expired_list = []
        with self._lock:
            for sit_id, sit in list(self._active_situations.items()):
                if sit.status not in (SituationStatus.RESOLVED, SituationStatus.DISMISSED, SituationStatus.EXPIRED):
                    if sit.valid_until is not None and current_time > sit.valid_until:
                        expired = Situation(
                            situation_id=sit.situation_id,
                            category=sit.category,
                            title=sit.title,
                            description=sit.description,
                            severity=SituationSeverity.INFO,
                            confidence=sit.confidence,
                            status=SituationStatus.EXPIRED,
                            involved_entities=sit.involved_entities,
                            supporting_evidence=sit.supporting_evidence,
                            location=sit.location,
                            created_at=sit.created_at,
                            updated_at=current_time,
                            valid_until=sit.valid_until,
                            related_event_ids=sit.related_event_ids,
                            related_goal_ids=sit.related_goal_ids,
                            correlation_id=sit.correlation_id,
                            causation_id=sit.causation_id,
                            metadata={**sit.metadata, "expired_at": current_time},
                        )
                        self._active_situations[sit_id] = expired
                        expired_list.append(expired)
                        self._emit_event(CognitiveEventType.SITUATION_EXPIRED, expired, "Situation expired")
        return tuple(expired_list)

    def clear(self) -> None:
        """Reset internal engine state."""
        with self._lock:
            self._active_situations.clear()
            self._situation_signatures.clear()
            self._seen_observation_ids.clear()
            self._seen_observation_set.clear()
            self._recent_source_hashes.clear()
            self._situation_history.clear()

    # ========================================================================
    # Core Internal Processing & Correlation
    # ========================================================================

    def _process_single_observation(
        self,
        obs: MultimodalObservation,
        current_time: float,
    ) -> Optional[Situation]:
        """
        Process a single observation through duplicate detection, correlation,
        and situation merge or creation. Must be called under lock.
        """
        # 1. Primary Duplicate Observation Suppression
        if obs.observation_id in self._seen_observation_set:
            self._emit_simple_event(
                CognitiveEventType.SITUATION_SUPPRESSED,
                obs.correlation_id,
                f"Suppressed duplicate observation {obs.observation_id}",
            )
            # Return existing situation matching this observation if available
            for s in self._active_situations.values():
                if any(e.observation_id == obs.observation_id for e in s.supporting_evidence):
                    return s
            return None

        # Record observation ID
        if len(self._seen_observation_ids) == self._seen_observation_ids.maxlen:
            removed_id = self._seen_observation_ids.popleft()
            self._seen_observation_set.discard(removed_id)
        self._seen_observation_ids.append(obs.observation_id)
        self._seen_observation_set.add(obs.observation_id)

        # 2. Same-Source Redundant Frame Suppression
        is_same_source_duplicate = self._check_same_source_duplicate(obs)

        # 3. Freshness Evaluation
        is_stale = False
        if obs.is_expired(current_time):
            is_stale = True
        elif (current_time - obs.timestamp) > self.config.stale_observation_threshold_seconds:
            is_stale = True

        # Future timestamp handling
        if obs.timestamp > (current_time + self.config.future_slack_seconds):
            logger.debug(
                "Observation %s has future timestamp (%.2f > %.2f + %.2f)",
                obs.observation_id,
                obs.timestamp,
                current_time,
                self.config.future_slack_seconds,
            )

        # 4. Search for Matching Active Situation
        matching_sit = self._find_matching_situation(obs)

        if matching_sit is not None:
            # Merge into existing situation
            return self._merge_into_situation(matching_sit, obs, is_same_source_duplicate, current_time)

        # 5. Stale data must NOT silently create a NEW situation
        if is_stale:
            self._emit_simple_event(
                CognitiveEventType.SITUATION_SUPPRESSED,
                obs.correlation_id,
                f"Suppressed stale observation {obs.observation_id} from creating a situation",
            )
            return None

        # 6. Create New Situation
        return self._create_new_situation(obs, current_time)

    def _check_same_source_duplicate(self, obs: MultimodalObservation) -> bool:
        """
        Detect repeated identical or nearly identical frames from the same source within 1.0s.
        Prevents camera frame spam from inflating confidence.
        """
        payload_repr = str(obs.payload)
        h = hash((obs.source_id, obs.modality.value, payload_repr))
        source_key = f"{obs.source_id}:{obs.modality.value}"
        now_ts = obs.timestamp

        if source_key in self._recent_source_hashes:
            last_hash, last_ts = self._recent_source_hashes[source_key]
            if last_hash == h and abs(now_ts - last_ts) <= 1.0:
                self._recent_source_hashes[source_key] = (h, now_ts)
                return True

        self._recent_source_hashes[source_key] = (h, now_ts)
        return False

    def _find_matching_situation(self, obs: MultimodalObservation) -> Optional[Situation]:
        """
        Find the best matching active Situation for an incoming observation.
        Evaluates temporal, spatial, entity, and correlation/causation criteria.
        """
        obs_entities = set(self._extract_entities(obs))
        obs_cat = self._classify_category(obs)

        candidates: List[Tuple[float, Situation]] = []

        for sit in self._active_situations.values():
            if sit.status in (SituationStatus.RESOLVED, SituationStatus.DISMISSED, SituationStatus.EXPIRED):
                continue

            # Check 1: Explicit Correlation ID or Causation Link
            has_explicit_link = False
            if obs.correlation_id and obs.correlation_id == sit.correlation_id:
                has_explicit_link = True
            elif obs.causation_id and (
                obs.causation_id == sit.situation_id
                or any(e.observation_id == obs.causation_id for e in sit.supporting_evidence)
            ):
                has_explicit_link = True
            elif sit.causation_id and sit.causation_id == obs.observation_id:
                has_explicit_link = True

            # Check 2: Temporal Proximity
            dt = abs(obs.timestamp - sit.updated_at)
            max_dt = (
                self.config.max_temporal_distance_seconds * 2.0
                if has_explicit_link
                else self.config.max_temporal_distance_seconds
            )
            if dt > max_dt:
                continue

            # Check 3: Spatial Proximity (if coordinates exist for both)
            spatial_match = True
            if obs.location is not None and sit.location is not None:
                dist = obs.location.distance_to(sit.location)
                if dist > self.config.max_spatial_distance_meters:
                    spatial_match = False
                    continue

            # Check 4: Entity / Context Compatibility
            entity_overlap = bool(obs_entities & set(sit.involved_entities))
            category_match = (
                obs_cat == sit.category
                or sit.category == SituationCategory.UNKNOWN
                or obs_cat == SituationCategory.UNKNOWN
                or self._are_categories_compatible(obs_cat, sit.category)
            )

            # Match Decision Rule
            if has_explicit_link:
                candidates.append((0.0 + dt / 100.0, sit))
            elif entity_overlap and spatial_match:
                # Strong entity match within temporal/spatial bounds
                candidates.append((1.0 + dt / 100.0, sit))
            elif spatial_match and (obs.device_id and any(obs.device_id in e.source_id for e in sit.supporting_evidence)):
                candidates.append((2.0 + dt / 100.0, sit))
            elif category_match and spatial_match and obs.location is not None and sit.location is not None:
                candidates.append((3.0 + dt / 100.0, sit))

        if not candidates:
            return None

        # Sort by best match score (lowest score = highest match priority)
        candidates.sort(key=lambda c: c[0])
        return candidates[0][1]

    def _are_categories_compatible(self, cat1: SituationCategory, cat2: SituationCategory) -> bool:
        """Check if two categories represent corroborating aspects of the same incident."""
        if cat1 == cat2:
            return True
        pairs = {
            (SituationCategory.ANOMALY, SituationCategory.SECURITY),
            (SituationCategory.ANOMALY, SituationCategory.NAVIGATIONAL),
            (SituationCategory.SECURITY, SituationCategory.USER_INTERACTION),
            (SituationCategory.OPERATIONAL, SituationCategory.SYSTEM_HEALTH),
            (SituationCategory.OPERATIONAL, SituationCategory.NAVIGATIONAL),
        }
        return (cat1, cat2) in pairs or (cat2, cat1) in pairs

    # ========================================================================
    # Situation Construction & Merging
    # ========================================================================

    def _create_new_situation(
        self,
        obs: MultimodalObservation,
        current_time: float,
    ) -> Situation:
        """Construct a new Situation from an initial observation."""
        category = self._classify_category(obs)
        entities = tuple(sorted(self._extract_entities(obs)))
        evidence = self._create_evidence_record(obs, weight=obs.confidence)

        # Generate deterministic situation signature and ID
        sig = self._generate_signature(category, entities, obs.location, obs.timestamp)
        sit_id = f"sit_{uuid.uuid5(uuid.NAMESPACE_DNS, sig).hex[:12]}"

        # Title and description
        title = self._generate_title(category, obs)
        description = self._generate_description(category, obs)

        severity = self._synthesize_severity(category, (evidence,), obs)
        confidence = obs.confidence
        valid_until = obs.timestamp + self.config.default_validity_window_seconds

        correlation_id = obs.correlation_id if obs.correlation_id else sit_id

        sit = Situation(
            situation_id=sit_id,
            category=category,
            title=title,
            description=description,
            severity=severity,
            confidence=confidence,
            status=SituationStatus.DETECTED,
            involved_entities=entities,
            supporting_evidence=(evidence,),
            location=obs.location,
            created_at=obs.timestamp,
            updated_at=obs.timestamp,
            valid_until=valid_until,
            related_event_ids=(),
            related_goal_ids=(),
            correlation_id=correlation_id,
            causation_id=obs.causation_id,
            metadata={
                "signature": sig,
                "created_by_observation": obs.observation_id,
                "primary_device": obs.device_id or obs.source_id,
            },
        )

        self._active_situations[sit_id] = sit
        self._situation_signatures[sig] = sit_id
        self._emit_event(CognitiveEventType.SITUATION_CREATED, sit, f"Created situation {sit.title}")
        return sit

    def _merge_into_situation(
        self,
        sit: Situation,
        obs: MultimodalObservation,
        is_same_source_duplicate: bool,
        current_time: float,
    ) -> Situation:
        """
        Merge an incoming observation into an existing active Situation.
        Returns a new immutable Situation instance.
        """
        # Archive current version into bounded history
        self._situation_history.append(sit)

        # Create new evidence record
        # If identical frame from same source within 1s, assign low weight so it doesn't inflate confidence
        evidence_weight = 0.05 if is_same_source_duplicate else obs.confidence
        new_evidence_item = self._create_evidence_record(obs, weight=evidence_weight)

        # Append evidence without duplicate observation IDs
        existing_evidence = list(sit.supporting_evidence)
        if not any(e.observation_id == obs.observation_id for e in existing_evidence):
            existing_evidence.append(new_evidence_item)

        # Bound evidence list if needed
        if len(existing_evidence) > self.config.max_evidence_per_situation:
            first = existing_evidence[0]
            last = existing_evidence[-1]
            middle = sorted(existing_evidence[1:-1], key=lambda e: (-e.evidence_weight, -e.timestamp))
            existing_evidence = [first] + middle[: self.config.max_evidence_per_situation - 2] + [last]

        evidence_tuple = tuple(existing_evidence)

        # Combine involved entities
        obs_entities = self._extract_entities(obs)
        combined_entities = tuple(sorted(set(sit.involved_entities) | set(obs_entities)))

        # Update Location: use most recent location with best accuracy
        best_loc = sit.location
        if obs.location is not None:
            if best_loc is None:
                best_loc = obs.location
            elif (obs.location.accuracy or 999.0) < (best_loc.accuracy or 999.0) or obs.timestamp >= sit.updated_at:
                best_loc = obs.location

        # Temporal bounds
        new_created_at = min(sit.created_at, obs.timestamp)
        new_updated_at = max(sit.updated_at, obs.timestamp)
        new_valid_until = new_updated_at + self.config.default_validity_window_seconds

        # Conflict detection
        has_conflict, conflict_info = self._detect_and_resolve_conflicts(sit, obs)

        # Re-synthesize confidence
        new_confidence = self._synthesize_confidence(evidence_tuple, has_conflict)

        # Re-synthesize category (refine if previously UNKNOWN or higher specificity)
        new_category = sit.category
        obs_cat = self._classify_category(obs)
        if new_category == SituationCategory.UNKNOWN:
            new_category = obs_cat
        elif obs_cat in (SituationCategory.ANOMALY, SituationCategory.SECURITY) and new_category not in (SituationCategory.ANOMALY, SituationCategory.SECURITY):
            new_category = obs_cat

        # Re-synthesize severity
        new_severity = self._synthesize_severity(new_category, evidence_tuple, obs)

        # Lifecycle state transition
        distinct_sources = len({e.source_id for e in evidence_tuple})
        distinct_modalities = len({e.modality for e in evidence_tuple})

        if sit.status == SituationStatus.DETECTED:
            if (
                distinct_sources >= self.config.min_corroboration_count_for_active
                or distinct_modalities >= self.config.min_corroboration_count_for_active
                or new_confidence >= 0.85
            ):
                new_status = SituationStatus.ACTIVE
            else:
                new_status = SituationStatus.DETECTED
        elif has_conflict:
            new_status = SituationStatus.UPDATING
        else:
            new_status = SituationStatus.ACTIVE

        # Create updated immutable Situation
        merged_metadata = dict(sit.metadata)
        if conflict_info:
            merged_metadata["latest_conflict"] = conflict_info

        updated_sit = Situation(
            situation_id=sit.situation_id,
            category=new_category,
            title=sit.title,
            description=self._update_description(sit.description, obs),
            severity=new_severity,
            confidence=new_confidence,
            status=new_status,
            involved_entities=combined_entities,
            supporting_evidence=evidence_tuple,
            location=best_loc,
            created_at=new_created_at,
            updated_at=new_updated_at,
            valid_until=new_valid_until,
            related_event_ids=sit.related_event_ids,
            related_goal_ids=sit.related_goal_ids,
            correlation_id=sit.correlation_id,
            causation_id=sit.causation_id or obs.causation_id,
            metadata=merged_metadata,
        )

        self._active_situations[sit.situation_id] = updated_sit
        self._emit_event(
            CognitiveEventType.SITUATION_MERGED,
            updated_sit,
            f"Merged observation {obs.observation_id} into {sit.situation_id}",
        )
        return updated_sit

    # ========================================================================
    # Conflict Detection & DeterministicConflictResolver Reuse
    # ========================================================================

    def _detect_and_resolve_conflicts(
        self,
        sit: Situation,
        obs: MultimodalObservation,
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Reuse Phase 4.4 DeterministicConflictResolver to evaluate opposing observations.
        Does NOT invent a new resolver. Preserves both evidence sources.
        """
        if not isinstance(obs.payload, dict):
            return False, None

        entity_id = (
            obs.payload.get("entity_id")
            or obs.device_id
            or (sit.involved_entities[0] if sit.involved_entities else obs.source_id)
        )
        property_name = obs.payload.get("property_name") or obs.payload.get("metric")

        if not entity_id or not property_name or "value" not in obs.payload:
            return False, None

        obs_val = obs.payload["value"]

        # Check existing evidence in situation for conflicting property values
        for ev in sit.supporting_evidence:
            if ev.provenance and isinstance(ev.provenance, dict):
                ev_entity = ev.provenance.get("entity_id") or (sit.involved_entities[0] if sit.involved_entities else None)
                ev_prop = ev.provenance.get("property_name")
                ev_val = ev.provenance.get("value")

                if ev_entity == entity_id and ev_prop == property_name and ev_val is not None:
                    if ev_val != obs_val:
                        # Construct Phase 4.4 WorldCondition and StateConflict
                        from core.models.world_state import StateProvenance
                        cond = WorldCondition(
                            entity_id=entity_id,
                            property_name=property_name,
                            value=ev_val,
                            confidence=ev.evidence_weight,
                            observed_at=ev.timestamp,
                            provenance=StateProvenance(
                                source_id=ev.source_id,
                                source_type="sensor",
                                observation_id=ev.observation_id,
                                recorded_at=ev.timestamp,
                            ),
                        )
                        real_conflict = StateConflict(
                            conflict_id=f"conf_{uuid.uuid4().hex[:8]}",
                            entity_id=entity_id,
                            property_name=property_name,
                            existing_condition=cond,
                            competing_observation=obs.to_world_state_observation(entity_id, property_name, obs_val),
                            detected_at=obs.timestamp,
                        )

                        resolution = self.conflict_resolver.resolve(real_conflict, self.conflict_policy)
                        conflict_data = {
                            "conflict_id": real_conflict.conflict_id,
                            "entity_id": entity_id,
                            "property": property_name,
                            "val_a": ev_val,
                            "val_b": obs_val,
                            "resolved": resolution is not None,
                            "resolution_strategy": resolution.strategy.value if resolution else None,
                            "winner": resolution.winning_source if resolution else None,
                        }

                        self._emit_simple_event(
                            CognitiveEventType.SITUATION_CONFLICT_DETECTED,
                            sit.correlation_id,
                            f"Conflict detected on {entity_id}.{property_name}: '{ev_val}' vs '{obs_val}'",
                        )

                        # Flag conflict (both sources remain preserved in supporting_evidence)
                        return True, conflict_data

        return False, None

    # ========================================================================
    # Deterministic Confidence & Severity Models
    # ========================================================================

    def _synthesize_confidence(
        self,
        evidence: Sequence[SituationEvidence],
        has_unresolved_conflict: bool,
    ) -> float:
        """
        Deterministic, documented confidence calculation.

        Rules:
        1. Base confidence is the evidence-weight-weighted average of corroborating items.
        2. Source diversity: Independent sources (|distinct source_ids| >= 2) provide corroboration boost.
        3. Modality diversity: Independent modalities (|distinct modalities| >= 2) provide corroboration boost.
        4. Same-source repetition (e.g. 5 camera frames from 1 camera) provides ZERO corroboration boost.
        5. Contradiction penalty applied if unresolved conflict exists.
        6. Strictly bounded in [0.0, 1.0].
        """
        if not evidence:
            return 0.0

        total_weight = sum(e.evidence_weight for e in evidence)
        if total_weight <= 0.0:
            return 0.0

        # Weighted average base confidence
        base_confidence = sum(e.evidence_weight * e.evidence_weight for e in evidence) / total_weight

        # Count distinct sources and modalities
        distinct_sources = {e.source_id for e in evidence}
        distinct_modalities = {e.modality for e in evidence}

        corroboration_boost = 0.0
        # Multi-source corroboration rule: Must have at least 2 independent sources OR distinct modalities
        if len(distinct_sources) >= 2:
            src_boost = (len(distinct_sources) - 1) * self.config.independent_source_boost
            mod_boost = (len(distinct_modalities) - 1) * self.config.independent_modality_boost
            corroboration_boost = min(self.config.max_corroboration_boost, src_boost + mod_boost)
        elif len(distinct_modalities) >= 2:
            # Different modalities from same device (e.g. Atlas Glass camera + microphone + imu)
            corroboration_boost = min(
                self.config.max_corroboration_boost,
                (len(distinct_modalities) - 1) * self.config.independent_modality_boost,
            )

        # Conflict penalty
        penalty = self.config.conflict_confidence_penalty if has_unresolved_conflict else 0.0

        final_conf = base_confidence + corroboration_boost - penalty
        return round(max(0.0, min(1.0, final_conf)), 3)

    def _synthesize_severity(
        self,
        category: SituationCategory,
        evidence: Sequence[SituationEvidence],
        latest_obs: MultimodalObservation,
    ) -> SituationSeverity:
        """
        Derive severity deterministically from category, corroboration, and payload signals.
        Never collapses severity into confidence.
        """
        distinct_sources = len({e.source_id for e in evidence})
        distinct_modalities = len({e.modality for e in evidence})

        payload_text = str(latest_obs.payload).lower()
        all_evidence_text = " ".join(e.concise_summary.lower() for e in evidence) + " " + payload_text
        has_critical_indicators = any(
            k in all_evidence_text
            for k in ("distress", "fall", "prone", "unauthorized", "collision", "breach", "fire", "emergency", "sudden stop")
        )

        if has_critical_indicators and (distinct_sources >= 2 or distinct_modalities >= 2):
            return SituationSeverity.CRITICAL

        if category == SituationCategory.SECURITY:
            return SituationSeverity.HIGH if distinct_sources >= 2 else SituationSeverity.MEDIUM
        elif category == SituationCategory.ANOMALY:
            return SituationSeverity.HIGH if has_critical_indicators else SituationSeverity.MEDIUM
        elif category == SituationCategory.SYSTEM_HEALTH:
            if "low" in payload_text or "critical" in payload_text or "fail" in payload_text:
                return SituationSeverity.HIGH
            return SituationSeverity.MEDIUM
        elif category == SituationCategory.NAVIGATIONAL:
            return SituationSeverity.MEDIUM
        elif category == SituationCategory.OPERATIONAL:
            return SituationSeverity.LOW
        elif category == SituationCategory.USER_INTERACTION:
            return SituationSeverity.INFO
        elif category == SituationCategory.ENVIRONMENTAL:
            return SituationSeverity.MEDIUM
        return SituationSeverity.INFO

    # ========================================================================
    # Classification & Signature Generation
    # ========================================================================

    def _classify_category(self, obs: MultimodalObservation) -> SituationCategory:
        """Deterministic lightweight rule-based mapping using observable evidence only."""
        payload_str = str(obs.payload).lower()

        # Check explicit category in metadata
        meta_cat = obs.metadata.get("category")
        if meta_cat:
            return SituationCategory.from_str(meta_cat)

        if obs.modality == ModalityType.GPS or any(k in payload_str for k in ("drift", "obstacle", "waypoint", "nav")):
            return SituationCategory.NAVIGATIONAL

        if obs.modality == ModalityType.USER_ACTION or any(k in payload_str for k in ("command", "button", "voice_command")):
            return SituationCategory.USER_INTERACTION

        if any(k in payload_str for k in ("prone", "distress", "fall", "collision", "anomaly", "abnormal")):
            return SituationCategory.ANOMALY

        if any(k in payload_str for k in ("intrusion", "unauthorized", "face_unknown", "perimeter")):
            return SituationCategory.SECURITY

        if obs.modality == ModalityType.TELEMETRY or any(k in payload_str for k in ("battery", "temp", "voltage", "cpu", "memory", "packet_loss")):
            return SituationCategory.SYSTEM_HEALTH

        if any(k in payload_str for k in ("smoke", "heat", "weather", "ambient", "flood")):
            return SituationCategory.ENVIRONMENTAL

        if obs.modality in (ModalityType.DEVICE_STATE, ModalityType.WORLD_STATE):
            return SituationCategory.OPERATIONAL

        return SituationCategory.UNKNOWN

    def _generate_signature(
        self,
        category: SituationCategory,
        entities: Tuple[str, ...],
        location: Optional[GeoLocation],
        timestamp: float,
    ) -> str:
        """Deterministic, reproducible situation signature for deduplication and clustering."""
        ent_key = ",".join(sorted(entities)) if entities else "global"
        loc_key = (
            f"{round(location.latitude, 3)}_{round(location.longitude, 3)}"
            if location is not None
            else "no_loc"
        )
        time_bucket = int(timestamp // self.config.max_temporal_distance_seconds)
        return f"{category.value}:{ent_key}:{loc_key}:{time_bucket}"

    def _extract_entities(self, obs: MultimodalObservation) -> List[str]:
        """Extract involved entity IDs from observation without creating a duplicate registry."""
        entities = []
        if obs.device_id:
            entities.append(obs.device_id)

        if isinstance(obs.payload, dict):
            ent = obs.payload.get("entity_id") or obs.payload.get("device_id")
            if ent and ent not in entities:
                entities.append(str(ent))
            if "entities" in obs.payload and isinstance(obs.payload["entities"], (list, tuple)):
                for e in obs.payload["entities"]:
                    if str(e) not in entities:
                        entities.append(str(e))

        # Optional read-only lookup in world_state_store if present
        if not entities and self.world_state_store is not None:
            ent_obj = self.world_state_store.get_entity(obs.source_id)
            if ent_obj:
                entities.append(ent_obj.entity_id)

        if not entities:
            entities.append(obs.source_id)

        return sorted(entities)

    def _create_evidence_record(
        self,
        obs: MultimodalObservation,
        weight: float,
    ) -> SituationEvidence:
        """Build lean SituationEvidence reference. DOES NOT duplicate large binary payloads."""
        summary = self._summarize_observation(obs)
        prov: Dict[str, Any] = {
            "device_id": obs.device_id,
            "correlation_id": obs.correlation_id,
        }
        if obs.artifact_reference:
            prov["artifact_reference"] = obs.artifact_reference
        if isinstance(obs.payload, dict):
            if "entity_id" in obs.payload:
                prov["entity_id"] = obs.payload["entity_id"]
            if "property_name" in obs.payload:
                prov["property_name"] = obs.payload["property_name"]
            if "value" in obs.payload:
                prov["value"] = obs.payload["value"]

        return SituationEvidence(
            evidence_id=f"ev_{obs.observation_id}",
            observation_id=obs.observation_id,
            source_id=obs.source_id,
            modality=obs.modality,
            evidence_weight=weight,
            timestamp=obs.timestamp,
            concise_summary=summary,
            provenance=prov,
        )

    def _summarize_observation(self, obs: MultimodalObservation) -> str:
        """Create a bounded, concise summary string for an observation."""
        if isinstance(obs.payload, str):
            return obs.payload[:120]
        elif isinstance(obs.payload, dict):
            desc = obs.payload.get("description") or obs.payload.get("event") or obs.payload.get("status")
            if desc:
                return str(desc)[:120]
            items = [f"{k}={v}" for k, v in list(obs.payload.items())[:3]]
            return f"{obs.modality.value}: {', '.join(items)}"[:120]
        return f"{obs.modality.value} from {obs.source_id}"

    def _generate_title(self, category: SituationCategory, obs: MultimodalObservation) -> str:
        """Generate a concise, model-neutral situation title."""
        summary = self._summarize_observation(obs)
        return f"{category.value.replace('_', ' ').title()}: {summary[:60]}"

    def _generate_description(self, category: SituationCategory, obs: MultimodalObservation) -> str:
        """Generate initial description."""
        summary = self._summarize_observation(obs)
        return f"Detected {category.value} incident via {obs.modality.value} from {obs.source_id}. Detail: {summary}"

    def _update_description(self, current_desc: str, obs: MultimodalObservation) -> str:
        """Incrementally update description with new corroborating evidence."""
        new_item = f"+[{obs.modality.value}:{obs.source_id}]"
        if new_item not in current_desc:
            return f"{current_desc} {new_item}"
        return current_desc

    def _enforce_capacity(self, current_time: float) -> None:
        """Enforce maximum active situation limit by evicting oldest/resolved entries."""
        if len(self._active_situations) <= self.config.max_active_situations:
            return

        inactive = [
            (s.situation_id, s)
            for s in self._active_situations.values()
            if s.status in (SituationStatus.RESOLVED, SituationStatus.DISMISSED, SituationStatus.EXPIRED)
        ]
        if inactive:
            inactive.sort(key=lambda item: item[1].updated_at)
            for sit_id, s in inactive:
                if len(self._active_situations) <= self.config.max_active_situations:
                    break
                self._situation_history.append(s)
                del self._active_situations[sit_id]

        while len(self._active_situations) > self.config.max_active_situations:
            oldest_id = min(self._active_situations.keys(), key=lambda k: self._active_situations[k].updated_at)
            evicted = self._active_situations.pop(oldest_id)
            self._situation_history.append(evicted)

    # ========================================================================
    # Observability & Event Emission
    # ========================================================================

    def _emit_event(self, event_type: CognitiveEventType, sit: Situation, message: str) -> None:
        """Emit structured cognitive telemetry event if an event_sink is provided."""
        if not self.event_sink:
            return
        event = CognitiveEvent(
            event_id=f"cog_sit_{uuid.uuid4().hex[:8]}",
            turn_id=sit.correlation_id or sit.situation_id,
            session_id="situation_fusion",
            stage=CognitiveStage.OBSERVATION,
            event_type=event_type,
            timestamp=sit.updated_at,
            component="situation_fusion",
            summary=message,
            metadata={
                "situation_id": sit.situation_id,
                "category": sit.category.value,
                "title": sit.title,
                "severity": sit.severity.value,
                "confidence": sit.confidence,
                "status": sit.status.value,
                "evidence_count": len(sit.supporting_evidence),
            },
        )
        self.event_sink.publish(event)

    def _emit_simple_event(self, event_type: CognitiveEventType, turn_id: str, message: str) -> None:
        """Emit simple telemetry event."""
        if not self.event_sink:
            return
        event = CognitiveEvent(
            event_id=f"cog_sit_{uuid.uuid4().hex[:8]}",
            turn_id=turn_id or "fusion_root",
            session_id="situation_fusion",
            stage=CognitiveStage.OBSERVATION,
            event_type=event_type,
            timestamp=self.clock(),
            component="situation_fusion",
            summary=message,
        )
        self.event_sink.publish(event)
