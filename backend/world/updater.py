import logging
import threading
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from core.interfaces.runtime_interface import CognitiveEventSinkInterface
from core.interfaces.world_interface import (
    ConflictResolverInterface,
    WorldStateStoreInterface,
    WorldStateUpdaterInterface,
)
from core.models.runtime import CognitiveEvent, CognitiveEventType, CognitiveStage
from core.models.world_state import (
    ConflictPolicy,
    ConflictResolution,
    ConflictStatus,
    FreshnessConfig,
    FreshnessStatus,
    Observation,
    ResolutionStrategy,
    StateConflict,
    StateProvenance,
    TransitionType,
    WorldCondition,
    WorldEntity,
    WorldState,
    WorldStateTransition,
    WorldStateUpdateResult,
)
from world.resolver import DeterministicConflictResolver

logger = logging.getLogger(__name__)


class DeterministicWorldStateUpdater(WorldStateUpdaterInterface):
    """
    Deterministic, model-neutral state update boundary for Phase 4.4.
    Every accepted world-state mutation must pass through this boundary.
    Guarantees validation, identity resolution, freshness checks, conflict detection,
    deterministic conflict resolution, version increments, and audit event emission.
    """

    def __init__(
        self,
        store: WorldStateStoreInterface,
        resolver: Optional[ConflictResolverInterface] = None,
        freshness_config: Optional[FreshnessConfig] = None,
        conflict_policy: Optional[ConflictPolicy] = None,
        event_sink: Optional[CognitiveEventSinkInterface] = None,
        clock: Optional[Callable[[], float]] = None,
    ):
        self.store = store
        self.resolver = resolver or DeterministicConflictResolver()
        self.freshness_config = freshness_config or FreshnessConfig()
        self.conflict_policy = conflict_policy or ConflictPolicy()
        self.event_sink = event_sink
        self.clock = clock or time.time
        self._update_lock = threading.Lock()

    def _publish_event(
        self,
        event_type: CognitiveEventType,
        summary: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Publish structured event to event sink with world state correlation."""
        if not self.event_sink:
            return

        now = self.clock()
        data = {
            "summary": summary,
            **(metadata or {}),
        }

        event = CognitiveEvent(
            event_id=f"evt_ws_{uuid.uuid4().hex[:12]}",
            turn_id=f"ws_ver_{metadata.get('version', 'unknown') if metadata else 'unknown'}",
            session_id="world_state",
            stage=CognitiveStage.RECEIVED,
            event_type=event_type,
            timestamp=now,
            summary=summary,
            metadata=data,
        )

        try:
            if hasattr(self.event_sink, "publish"):
                self.event_sink.publish(event)
            elif hasattr(self.event_sink, "receive_event"):
                self.event_sink.receive_event(event)
        except Exception as ex:
            logger.debug("Failed publishing world state event: %s", ex)

    def evaluate_freshness(
        self,
        condition: WorldCondition,
        now: Optional[float] = None,
    ) -> FreshnessStatus:
        current = now if now is not None else self.clock()
        return self.freshness_config.evaluate(
            observed_at=condition.observed_at,
            expires_at=condition.expires_at,
            now=current,
        )

    def check_expirations(self, now: Optional[float] = None) -> List[WorldCondition]:
        current = now if now is not None else self.clock()
        cur_state = self.store.get_current_state()
        expired: List[WorldCondition] = []

        for cond in cur_state.conditions:
            status = self.evaluate_freshness(cond, now=current)
            if status == FreshnessStatus.EXPIRED:
                expired.append(cond)
                self._publish_event(
                    event_type=CognitiveEventType.WORLD_CONDITION_EXPIRED,
                    summary=f"World condition '{cond.entity_id}.{cond.property_name}' expired.",
                    metadata={
                        "entity_id": cond.entity_id,
                        "property_name": cond.property_name,
                        "observed_at": cond.observed_at,
                        "expires_at": cond.expires_at,
                        "version": cur_state.version,
                    },
                )
        return expired

    def apply_observation(self, observation: Observation) -> WorldStateUpdateResult:
        with self._update_lock:
            now = self.clock()

            # 1. Validation
            if not isinstance(observation, Observation):
                return WorldStateUpdateResult(
                    success=False,
                    transition_type=TransitionType.REJECTED,
                    previous_version=0,
                    current_version=0,
                    error="Input must be a valid Observation object.",
                )

            if not observation.observation_id or not observation.entity_id or not observation.property_name:
                return WorldStateUpdateResult(
                    success=False,
                    transition_type=TransitionType.REJECTED,
                    previous_version=0,
                    current_version=0,
                    error="Observation missing required identity fields.",
                )

            if not (0.0 <= observation.confidence <= 1.0):
                return WorldStateUpdateResult(
                    success=False,
                    transition_type=TransitionType.REJECTED,
                    previous_version=0,
                    current_version=0,
                    error=f"Observation confidence must be in [0.0, 1.0], got {observation.confidence}",
                )

            if observation.expires_at is not None and now >= observation.expires_at:
                return WorldStateUpdateResult(
                    success=False,
                    transition_type=TransitionType.REJECTED,
                    previous_version=0,
                    current_version=0,
                    error="Observation is already expired upon arrival.",
                )

            current_state = self.store.get_current_state()
            prev_ver = current_state.version
            next_ver = prev_ver + 1

            existing_cond = current_state.get_condition(observation.entity_id, observation.property_name)

            # -------------------------------------------------------------
            # CASE A: No Existing Condition -> ADD
            # -------------------------------------------------------------
            if existing_cond is None:
                new_cond = WorldCondition(
                    entity_id=observation.entity_id,
                    property_name=observation.property_name,
                    value=observation.value,
                    confidence=observation.confidence,
                    observed_at=observation.timestamp,
                    expires_at=observation.expires_at,
                    provenance=observation.to_provenance(),
                    metadata=observation.metadata,
                )

                # Ensure entity exists in world state
                entity_list = list(current_state.entities)
                created_entity = False
                if current_state.get_entity(observation.entity_id) is None:
                    new_ent = WorldEntity(
                        entity_id=observation.entity_id,
                        entity_type=observation.metadata.get("entity_type", "generic"),
                        attributes={"created_by_observation": observation.observation_id},
                        confidence=observation.confidence,
                        provenance=observation.to_provenance(),
                    )
                    entity_list.append(new_ent)
                    created_entity = True

                new_conditions = current_state.conditions + (new_cond,)
                next_state = WorldState(
                    state_id=f"ws_v{next_ver}_{uuid.uuid4().hex[:8]}",
                    version=next_ver,
                    timestamp=now,
                    entities=tuple(entity_list),
                    conditions=new_conditions,
                    relationships=current_state.relationships,
                    metadata={"source_observation_id": observation.observation_id},
                )

                trans = WorldStateTransition(
                    transition_id=f"tr_{uuid.uuid4().hex[:12]}",
                    from_version=prev_ver,
                    to_version=next_ver,
                    transition_type=TransitionType.ADD,
                    observation_id=observation.observation_id,
                    entity_id=observation.entity_id,
                    property_name=observation.property_name,
                    old_value=None,
                    new_value=observation.value,
                    timestamp=now,
                    provenance=observation.to_provenance(),
                )

                self.store.apply_transition(trans, next_state)

                if created_entity:
                    self._publish_event(
                        event_type=CognitiveEventType.WORLD_ENTITY_ADDED,
                        summary=f"World entity '{observation.entity_id}' added.",
                        metadata={"entity_id": observation.entity_id, "version": next_ver},
                    )

                self._publish_event(
                    event_type=CognitiveEventType.WORLD_CONDITION_CHANGED,
                    summary=f"World condition '{observation.entity_id}.{observation.property_name}' added: {observation.value}",
                    metadata={
                        "entity_id": observation.entity_id,
                        "property_name": observation.property_name,
                        "new_value": observation.value,
                        "version": next_ver,
                    },
                )

                self._publish_event(
                    event_type=CognitiveEventType.WORLD_STATE_UPDATED,
                    summary=f"World state transitioned to version {next_ver}.",
                    metadata={"version": next_ver, "transition_type": TransitionType.ADD.value},
                )

                return WorldStateUpdateResult(
                    success=True,
                    transition_type=TransitionType.ADD,
                    previous_version=prev_ver,
                    current_version=next_ver,
                    condition=new_cond,
                    transition=trans,
                )

            # -------------------------------------------------------------
            # CASE B: Existing Condition Exists
            # -------------------------------------------------------------

            # Subcase B1: Values Match -> UNCHANGED (Refreshes timestamp / confidence)
            if existing_cond.value == observation.value:
                # If observation has newer timestamp or higher confidence, refresh
                if observation.timestamp > existing_cond.observed_at or observation.confidence > existing_cond.confidence:
                    refreshed_cond = WorldCondition(
                        entity_id=existing_cond.entity_id,
                        property_name=existing_cond.property_name,
                        value=existing_cond.value,
                        confidence=max(existing_cond.confidence, observation.confidence),
                        observed_at=max(existing_cond.observed_at, observation.timestamp),
                        expires_at=observation.expires_at or existing_cond.expires_at,
                        provenance=observation.to_provenance(),
                        metadata=observation.metadata or existing_cond.metadata,
                    )

                    new_conditions = tuple(
                        refreshed_cond if (c.entity_id == refreshed_cond.entity_id and c.property_name == refreshed_cond.property_name) else c
                        for c in current_state.conditions
                    )

                    next_state = WorldState(
                        state_id=f"ws_v{next_ver}_{uuid.uuid4().hex[:8]}",
                        version=next_ver,
                        timestamp=now,
                        entities=current_state.entities,
                        conditions=new_conditions,
                        relationships=current_state.relationships,
                        metadata={"refreshed_by": observation.observation_id},
                    )

                    trans = WorldStateTransition(
                        transition_id=f"tr_{uuid.uuid4().hex[:12]}",
                        from_version=prev_ver,
                        to_version=next_ver,
                        transition_type=TransitionType.UNCHANGED,
                        observation_id=observation.observation_id,
                        entity_id=observation.entity_id,
                        property_name=observation.property_name,
                        old_value=existing_cond.value,
                        new_value=observation.value,
                        timestamp=now,
                        provenance=observation.to_provenance(),
                    )

                    self.store.apply_transition(trans, next_state)

                    self._publish_event(
                        event_type=CognitiveEventType.WORLD_STATE_UPDATED,
                        summary=f"World condition '{observation.entity_id}.{observation.property_name}' refreshed (UNCHANGED).",
                        metadata={"version": next_ver, "transition_type": TransitionType.UNCHANGED.value},
                    )

                    return WorldStateUpdateResult(
                        success=True,
                        transition_type=TransitionType.UNCHANGED,
                        previous_version=prev_ver,
                        current_version=next_ver,
                        condition=refreshed_cond,
                        transition=trans,
                    )

                # Pure duplicate without newer timestamp/confidence: no-op
                return WorldStateUpdateResult(
                    success=True,
                    transition_type=TransitionType.UNCHANGED,
                    previous_version=prev_ver,
                    current_version=prev_ver,
                    condition=existing_cond,
                )

            # Subcase B2: Values Differ -> Evaluate Expiration or Conflict
            existing_freshness = self.evaluate_freshness(existing_cond, now=now)

            if existing_freshness == FreshnessStatus.EXPIRED:
                # Existing fact is expired; new observation cleanly supersedes without conflict
                updated_cond = WorldCondition(
                    entity_id=observation.entity_id,
                    property_name=observation.property_name,
                    value=observation.value,
                    confidence=observation.confidence,
                    observed_at=observation.timestamp,
                    expires_at=observation.expires_at,
                    provenance=observation.to_provenance(),
                    metadata=observation.metadata,
                )

                new_conditions = tuple(
                    updated_cond if (c.entity_id == updated_cond.entity_id and c.property_name == updated_cond.property_name) else c
                    for c in current_state.conditions
                )

                next_state = WorldState(
                    state_id=f"ws_v{next_ver}_{uuid.uuid4().hex[:8]}",
                    version=next_ver,
                    timestamp=now,
                    entities=current_state.entities,
                    conditions=new_conditions,
                    relationships=current_state.relationships,
                    metadata={"superseded_expired": existing_cond.property_name},
                )

                trans = WorldStateTransition(
                    transition_id=f"tr_{uuid.uuid4().hex[:12]}",
                    from_version=prev_ver,
                    to_version=next_ver,
                    transition_type=TransitionType.UPDATE,
                    observation_id=observation.observation_id,
                    entity_id=observation.entity_id,
                    property_name=observation.property_name,
                    old_value=existing_cond.value,
                    new_value=observation.value,
                    timestamp=now,
                    provenance=observation.to_provenance(),
                )

                self.store.apply_transition(trans, next_state)

                self._publish_event(
                    event_type=CognitiveEventType.WORLD_CONDITION_EXPIRED,
                    summary=f"World condition '{existing_cond.entity_id}.{existing_cond.property_name}' was EXPIRED and superseded.",
                    metadata={"version": next_ver, "old_value": existing_cond.value},
                )

                self._publish_event(
                    event_type=CognitiveEventType.WORLD_CONDITION_CHANGED,
                    summary=f"World condition '{observation.entity_id}.{observation.property_name}' updated from expired state.",
                    metadata={"version": next_ver, "new_value": observation.value},
                )

                self._publish_event(
                    event_type=CognitiveEventType.WORLD_STATE_UPDATED,
                    summary=f"World state transitioned to version {next_ver}.",
                    metadata={"version": next_ver, "transition_type": TransitionType.UPDATE.value},
                )

                return WorldStateUpdateResult(
                    success=True,
                    transition_type=TransitionType.UPDATE,
                    previous_version=prev_ver,
                    current_version=next_ver,
                    condition=updated_cond,
                    transition=trans,
                )

            # Subcase B3: Contradiction / Conflict Detected
            conflict = StateConflict(
                conflict_id=f"conf_{uuid.uuid4().hex[:12]}",
                entity_id=observation.entity_id,
                property_name=observation.property_name,
                existing_condition=existing_cond,
                competing_observation=observation,
                detected_at=now,
                status=ConflictStatus.DETECTED,
            )
            self.store.save_conflict(conflict)

            self._publish_event(
                event_type=CognitiveEventType.WORLD_STATE_CONFLICT,
                summary=f"State conflict on '{observation.entity_id}.{observation.property_name}': {existing_cond.value} vs {observation.value}",
                metadata={
                    "conflict_id": conflict.conflict_id,
                    "entity_id": observation.entity_id,
                    "property_name": observation.property_name,
                    "existing_value": existing_cond.value,
                    "competing_value": observation.value,
                },
            )

            # Resolve conflict deterministically
            resolution = self.resolver.resolve(conflict, self.conflict_policy)

            if resolution is not None:
                self.store.resolve_conflict(resolution)
                resolved_conflict = StateConflict(
                    conflict_id=conflict.conflict_id,
                    entity_id=conflict.entity_id,
                    property_name=conflict.property_name,
                    existing_condition=conflict.existing_condition,
                    competing_observation=conflict.competing_observation,
                    detected_at=conflict.detected_at,
                    status=ConflictStatus.RESOLVED,
                    resolution=resolution,
                )

                self._publish_event(
                    event_type=CognitiveEventType.WORLD_STATE_CONFLICT_RESOLVED,
                    summary=f"Conflict '{conflict.conflict_id}' resolved: {resolution.resolved_value} (strategy: {resolution.strategy.value})",
                    metadata={
                        "conflict_id": conflict.conflict_id,
                        "resolved_value": resolution.resolved_value,
                        "winning_source": resolution.winning_source,
                        "strategy": resolution.strategy.value,
                    },
                )

                # If competing observation won, apply UPDATE transition
                if resolution.resolved_value == observation.value:
                    updated_cond = WorldCondition(
                        entity_id=observation.entity_id,
                        property_name=observation.property_name,
                        value=observation.value,
                        confidence=observation.confidence,
                        observed_at=observation.timestamp,
                        expires_at=observation.expires_at,
                        provenance=observation.to_provenance(),
                        metadata=observation.metadata,
                    )

                    new_conditions = tuple(
                        updated_cond if (c.entity_id == updated_cond.entity_id and c.property_name == updated_cond.property_name) else c
                        for c in current_state.conditions
                    )

                    next_state = WorldState(
                        state_id=f"ws_v{next_ver}_{uuid.uuid4().hex[:8]}",
                        version=next_ver,
                        timestamp=now,
                        entities=current_state.entities,
                        conditions=new_conditions,
                        relationships=current_state.relationships,
                        metadata={"conflict_resolved": conflict.conflict_id},
                    )

                    trans = WorldStateTransition(
                        transition_id=f"tr_{uuid.uuid4().hex[:12]}",
                        from_version=prev_ver,
                        to_version=next_ver,
                        transition_type=TransitionType.UPDATE,
                        observation_id=observation.observation_id,
                        entity_id=observation.entity_id,
                        property_name=observation.property_name,
                        old_value=existing_cond.value,
                        new_value=observation.value,
                        timestamp=now,
                        provenance=observation.to_provenance(),
                        conflict_id=conflict.conflict_id,
                    )

                    self.store.apply_transition(trans, next_state)

                    self._publish_event(
                        event_type=CognitiveEventType.WORLD_CONDITION_CHANGED,
                        summary=f"World condition '{observation.entity_id}.{observation.property_name}' updated after conflict resolution.",
                        metadata={"version": next_ver, "new_value": observation.value},
                    )

                    self._publish_event(
                        event_type=CognitiveEventType.WORLD_STATE_UPDATED,
                        summary=f"World state transitioned to version {next_ver}.",
                        metadata={"version": next_ver, "transition_type": TransitionType.UPDATE.value},
                    )

                    return WorldStateUpdateResult(
                        success=True,
                        transition_type=TransitionType.UPDATE,
                        previous_version=prev_ver,
                        current_version=next_ver,
                        condition=updated_cond,
                        transition=trans,
                        conflict=resolved_conflict,
                    )
                else:
                    # Existing condition won: state value remains unchanged
                    return WorldStateUpdateResult(
                        success=True,
                        transition_type=TransitionType.CONFLICT,
                        previous_version=prev_ver,
                        current_version=prev_ver,
                        condition=existing_cond,
                        conflict=resolved_conflict,
                    )

            # Unresolved Conflict: Do NOT overwrite. Keep observable.
            return WorldStateUpdateResult(
                success=False,
                transition_type=TransitionType.CONFLICT,
                previous_version=prev_ver,
                current_version=prev_ver,
                condition=existing_cond,
                conflict=conflict,
                error=f"Unresolved state conflict on '{observation.entity_id}.{observation.property_name}'.",
            )
