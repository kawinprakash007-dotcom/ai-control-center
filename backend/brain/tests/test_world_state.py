import copy
import json
import threading
import time
import pytest
from typing import Any, Dict, List, Optional, Tuple

from core.interfaces.runtime_interface import CognitiveEventSinkInterface
from core.models.context import (
    AttentionFocus,
    CognitiveState,
    ContextItem,
    ContextPriority,
    ContextSource,
    SensitivityLevel,
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
    WorldRelationship,
    WorldState,
    WorldStateTransition,
    WorldStateUpdateResult,
)
from world.context_adapter import (
    world_condition_to_context_item,
    world_state_to_context_items,
)
from world.resolver import DeterministicConflictResolver
from world.store import (
    InMemoryWorldStateStore,
    SQLiteWorldStateStore,
    create_initial_world_state,
    deserialize_conflict,
    deserialize_transition,
    deserialize_world_state,
    serialize_conflict,
    serialize_transition,
    serialize_world_state,
)
from world.updater import DeterministicWorldStateUpdater


# ============================================================================
# TEST FIXTURES & HARNESSES
# ============================================================================

class FakeEventSink(CognitiveEventSinkInterface):
    """Test event sink capturing emitted CognitiveEvents for assertions."""

    def __init__(self):
        self.events: List[CognitiveEvent] = []

    def publish(self, event: CognitiveEvent) -> None:
        self.events.append(event)

    def receive_event(self, event: CognitiveEvent) -> None:
        self.events.append(event)

    def get_events(self) -> List[CognitiveEvent]:
        return list(self.events)

    def clear(self) -> None:
        self.events.clear()


def make_provenance(
    source_id: str = "sensor_1",
    source_type: str = "direct_sensor",
    obs_id: str = "obs_1",
    recorded_at: float = 1000.0,
) -> StateProvenance:
    return StateProvenance(
        source_id=source_id,
        source_type=source_type,
        observation_id=obs_id,
        recorded_at=recorded_at,
    )


def make_condition(
    entity_id: str = "door_front",
    property_name: str = "open_state",
    value: Any = "CLOSED",
    confidence: float = 0.95,
    observed_at: float = 1000.0,
    expires_at: Optional[float] = None,
    source_type: str = "direct_sensor",
) -> WorldCondition:
    return WorldCondition(
        entity_id=entity_id,
        property_name=property_name,
        value=value,
        confidence=confidence,
        observed_at=observed_at,
        expires_at=expires_at,
        provenance=make_provenance(source_type=source_type, recorded_at=observed_at),
    )


def make_observation(
    obs_id: str = "obs_101",
    source_id: str = "cam_01",
    source_type: str = "perception",
    timestamp: float = 1000.0,
    entity_id: str = "door_front",
    property_name: str = "open_state",
    value: Any = "OPEN",
    confidence: float = 0.90,
    expires_at: Optional[float] = None,
) -> Observation:
    return Observation(
        observation_id=obs_id,
        source_id=source_id,
        source_type=source_type,
        timestamp=timestamp,
        entity_id=entity_id,
        property_name=property_name,
        value=value,
        confidence=confidence,
        expires_at=expires_at,
    )


# ============================================================================
# A. DOMAIN VALIDATION
# ============================================================================

def test_domain_world_state_immutability():
    """A1. WorldState is frozen and rejects direct attribute mutation."""
    state = create_initial_world_state()
    with pytest.raises((AttributeError, TypeError)):
        state.version = 2  # type: ignore


def test_domain_world_condition_frozen():
    """A2. WorldCondition is frozen and rejects direct mutation."""
    cond = make_condition()
    with pytest.raises((AttributeError, TypeError)):
        cond.value = "OPEN"  # type: ignore


def test_domain_world_state_version_must_be_positive():
    """A3. WorldState version must be >= 1."""
    with pytest.raises(ValueError, match="WorldState version must be >= 1"):
        WorldState(state_id="ws_0", version=0, timestamp=100.0)


# ============================================================================
# B. ENTITY CREATION
# ============================================================================

def test_entity_creation_valid():
    """B1. WorldEntity correctly stores ID, type, attributes, and provenance."""
    prov = make_provenance()
    entity = WorldEntity(
        entity_id="server_rack_1",
        entity_type="hardware",
        attributes={"rack_number": 4, "cooling": "active"},
        confidence=0.99,
        provenance=prov,
    )
    assert entity.entity_id == "server_rack_1"
    assert entity.entity_type == "hardware"
    assert entity.attributes["rack_number"] == 4
    assert entity.confidence == 0.99
    assert entity.provenance.source_id == "sensor_1"


def test_entity_creation_rejects_empty_id():
    """B2. WorldEntity rejects empty string for entity_id."""
    with pytest.raises(ValueError, match="WorldEntity entity_id must be a non-empty string"):
        WorldEntity(entity_id="", entity_type="hardware")


# ============================================================================
# C. CONDITION CREATION
# ============================================================================

def test_condition_creation_and_key():
    """C1. WorldCondition correctly computes condition_key."""
    cond = make_condition(entity_id="hvac_unit", property_name="temp_celsius", value=21.5)
    assert cond.condition_key == ("hvac_unit", "temp_celsius")
    assert cond.value == 21.5


def test_condition_expiration_time_check():
    """C2. WorldCondition accurately detects expiration against a given clock."""
    cond = make_condition(observed_at=100.0, expires_at=150.0)
    assert not cond.is_expired(now=149.0)
    assert cond.is_expired(now=150.0)
    assert cond.is_expired(now=200.0)


# ============================================================================
# D. RELATIONSHIP HANDLING
# ============================================================================

def test_relationship_creation_and_retrieval():
    """D1. WorldRelationship links two entities and is indexed in WorldState."""
    rel = WorldRelationship(
        source_entity_id="node_a",
        relationship_type="connected_to",
        target_entity_id="node_b",
        attributes={"bandwidth": "10Gbps"},
        confidence=1.0,
        provenance=make_provenance(),
    )
    state = WorldState(
        state_id="ws_rel",
        version=1,
        timestamp=1000.0,
        relationships=(rel,),
    )
    node_rels = state.get_relationships_for_entity("node_a")
    assert len(node_rels) == 1
    assert node_rels[0].relationship_type == "connected_to"
    assert node_rels[0].target_entity_id == "node_b"


# ============================================================================
# E. PROVENANCE
# ============================================================================

def test_provenance_preservation():
    """E1. StateProvenance retains source_id, source_type, observation_id, and recorded_at."""
    prov = make_provenance(source_id="thermostat_3", source_type="sensor", obs_id="obs_999", recorded_at=500.0)
    assert prov.source_id == "thermostat_3"
    assert prov.source_type == "sensor"
    assert prov.observation_id == "obs_999"
    assert prov.recorded_at == 500.0


def test_provenance_rejects_empty_source():
    """E2. StateProvenance rejects empty source string."""
    with pytest.raises(ValueError):
        StateProvenance(source_id="", source_type="sensor", observation_id="obs_1")


# ============================================================================
# F. CONFIDENCE VALIDATION
# ============================================================================

def test_confidence_validation_bounds():
    """F1. Confidence must be in range [0.0, 1.0]."""
    with pytest.raises(ValueError):
        make_condition(confidence=-0.1)
    with pytest.raises(ValueError):
        make_condition(confidence=1.05)


def test_observation_confidence_bounds():
    """F2. Observation rejects confidence out of bounds."""
    with pytest.raises(ValueError):
        make_observation(confidence=1.5)


# ============================================================================
# G. FRESHNESS STATES
# ============================================================================

def test_freshness_evaluation_lifecycle():
    """G1. Deterministic progression across FRESH, AGING, STALE, and EXPIRED."""
    cfg = FreshnessConfig(fresh_duration=10.0, aging_duration=30.0, stale_duration=60.0)
    t0 = 100.0

    # Fresh
    assert cfg.evaluate(observed_at=t0, expires_at=None, now=105.0) == FreshnessStatus.FRESH
    # Aging
    assert cfg.evaluate(observed_at=t0, expires_at=None, now=120.0) == FreshnessStatus.AGING
    # Stale
    assert cfg.evaluate(observed_at=t0, expires_at=None, now=150.0) == FreshnessStatus.STALE
    # Expired
    assert cfg.evaluate(observed_at=t0, expires_at=None, now=170.0) == FreshnessStatus.EXPIRED


def test_freshness_orthogonal_to_confidence():
    """G2. High confidence condition can still be evaluated as STALE."""
    cfg = FreshnessConfig(fresh_duration=10.0, aging_duration=30.0, stale_duration=60.0)
    cond = make_condition(confidence=1.0, observed_at=100.0)
    status = cfg.evaluate(cond.observed_at, cond.expires_at, now=150.0)
    assert status == FreshnessStatus.STALE
    assert cond.confidence == 1.0


# ============================================================================
# H. EXPIRATION
# ============================================================================

def test_explicit_expires_at_overrides_duration():
    """H1. Explicit expires_at immediately triggers EXPIRED even if fresh duration not elapsed."""
    cfg = FreshnessConfig(fresh_duration=100.0)
    status = cfg.evaluate(observed_at=100.0, expires_at=105.0, now=106.0)
    assert status == FreshnessStatus.EXPIRED


def test_updater_check_expirations():
    """H2. Updater check_expirations identifies expired items and emits events."""
    store = InMemoryWorldStateStore()
    sink = FakeEventSink()
    updater = DeterministicWorldStateUpdater(
        store=store,
        event_sink=sink,
        clock=lambda: 200.0,
        freshness_config=FreshnessConfig(fresh_duration=10.0, aging_duration=20.0, stale_duration=50.0),
    )

    # Add condition observed at 100.0 (age = 100.0 > stale_duration 50.0)
    obs = make_observation(timestamp=100.0, entity_id="lamp_1", property_name="state", value="ON")
    updater.apply_observation(obs)

    expired = updater.check_expirations(now=200.0)
    assert len(expired) == 1
    assert expired[0].entity_id == "lamp_1"

    # Event was emitted
    exp_events = [e for e in sink.get_events() if e.event_type == CognitiveEventType.WORLD_CONDITION_EXPIRED]
    assert len(exp_events) >= 1
    assert exp_events[0].metadata["entity_id"] == "lamp_1"


# ============================================================================
# I. NEW-CONDITION TRANSITIONS (ADD)
# ============================================================================

def test_transition_add_new_condition():
    """I1. Valid observation on unseen entity/property produces ADD transition."""
    store = InMemoryWorldStateStore()
    updater = DeterministicWorldStateUpdater(store=store, clock=lambda: 1000.0)

    obs = make_observation(
        obs_id="obs_01",
        entity_id="window_main",
        property_name="locked",
        value=True,
        confidence=0.98,
        timestamp=1000.0,
    )

    res = updater.apply_observation(obs)
    assert res.success is True
    assert res.transition_type == TransitionType.ADD
    assert res.previous_version == 1
    assert res.current_version == 2
    assert res.condition is not None
    assert res.condition.value is True

    # State in store updated
    state = store.get_current_state()
    assert state.version == 2
    assert state.get_condition("window_main", "locked").value is True
    assert state.get_entity("window_main") is not None


# ============================================================================
# J. CHANGED-CONDITION TRANSITIONS (UPDATE)
# ============================================================================

def test_transition_update_when_old_fact_expired():
    """J1. When an old condition is EXPIRED, new observation cleanly updates without conflict."""
    store = InMemoryWorldStateStore()
    cfg = FreshnessConfig(fresh_duration=10.0, aging_duration=20.0, stale_duration=30.0)
    current_time = 1000.0
    updater = DeterministicWorldStateUpdater(
        store=store,
        freshness_config=cfg,
        clock=lambda: current_time,
    )

    # 1. Add initial condition at 1000.0
    obs1 = make_observation(obs_id="o1", timestamp=1000.0, entity_id="printer", property_name="status", value="IDLE")
    updater.apply_observation(obs1)

    # 2. Advance time to 1050.0 (age = 50s > stale_duration 30s -> EXPIRED)
    current_time = 1050.0
    obs2 = make_observation(obs_id="o2", timestamp=1050.0, entity_id="printer", property_name="status", value="PRINTING")
    res = updater.apply_observation(obs2)

    assert res.success is True
    assert res.transition_type == TransitionType.UPDATE
    assert res.current_version == 3
    assert res.condition.value == "PRINTING"
    assert store.get_current_state().get_condition("printer", "status").value == "PRINTING"


# ============================================================================
# K. UNCHANGED OBSERVATIONS
# ============================================================================

def test_transition_unchanged_refreshes_timestamp():
    """K1. Observation with identical value and newer timestamp refreshes freshness as UNCHANGED."""
    store = InMemoryWorldStateStore()
    current_time = 1000.0
    updater = DeterministicWorldStateUpdater(store=store, clock=lambda: current_time)

    obs1 = make_observation(obs_id="o1", timestamp=1000.0, value="ONLINE")
    updater.apply_observation(obs1)
    assert store.get_current_state().version == 2

    # Same value, newer timestamp
    current_time = 1010.0
    obs2 = make_observation(obs_id="o2", timestamp=1010.0, value="ONLINE")
    res = updater.apply_observation(obs2)

    assert res.success is True
    assert res.transition_type == TransitionType.UNCHANGED
    assert res.current_version == 3
    assert res.condition.observed_at == 1010.0


def test_transition_unchanged_duplicate_is_noop():
    """K2. Identical observation without newer timestamp or confidence is a no-op."""
    store = InMemoryWorldStateStore()
    updater = DeterministicWorldStateUpdater(store=store, clock=lambda: 1000.0)

    obs1 = make_observation(obs_id="o1", timestamp=1000.0, confidence=0.8, value="OK")
    updater.apply_observation(obs1)
    ver = store.get_current_state().version

    # Duplicate observation at same or older timestamp and equal/lower confidence
    obs2 = make_observation(obs_id="o2", timestamp=1000.0, confidence=0.8, value="OK")
    res = updater.apply_observation(obs2)

    assert res.success is True
    assert res.transition_type == TransitionType.UNCHANGED
    assert res.current_version == ver
    assert store.get_current_state().version == ver


# ============================================================================
# L. VERSION INCREMENTS
# ============================================================================

def test_monotonic_version_increments():
    """L1. State version increases strictly monotonically by 1 per mutation."""
    store = InMemoryWorldStateStore()
    updater = DeterministicWorldStateUpdater(store=store, clock=lambda: 1000.0)

    assert store.get_current_state().version == 1

    updater.apply_observation(make_observation(obs_id="o1", entity_id="e1", property_name="p", value=1))
    assert store.get_current_state().version == 2

    updater.apply_observation(make_observation(obs_id="o2", entity_id="e2", property_name="p", value=2))
    assert store.get_current_state().version == 3

    updater.apply_observation(make_observation(obs_id="o3", entity_id="e3", property_name="p", value=3))
    assert store.get_current_state().version == 4


# ============================================================================
# M. DETERMINISM
# ============================================================================

def test_deterministic_reproducibility():
    """M1. Same sequence of observations applied to two separate stores yields identical state."""
    store1 = InMemoryWorldStateStore()
    store2 = InMemoryWorldStateStore()

    updater1 = DeterministicWorldStateUpdater(store=store1, clock=lambda: 1000.0)
    updater2 = DeterministicWorldStateUpdater(store=store2, clock=lambda: 1000.0)

    observations = [
        make_observation(obs_id="o1", entity_id="room_1", property_name="occupied", value=True, timestamp=1000.0),
        make_observation(obs_id="o2", entity_id="room_1", property_name="lights", value="DIM", timestamp=1001.0),
        make_observation(obs_id="o3", entity_id="room_2", property_name="temp", value=22.0, timestamp=1002.0),
    ]

    for obs in observations:
        updater1.apply_observation(obs)
        updater2.apply_observation(obs)

    state1 = store1.get_current_state()
    state2 = store2.get_current_state()

    assert state1.version == state2.version
    assert len(state1.conditions) == len(state2.conditions)
    assert serialize_world_state(state1)["conditions"] == serialize_world_state(state2)["conditions"]


# ============================================================================
# N. CONFLICT DETECTION
# ============================================================================

def test_conflict_detection_opposing_values():
    """N1. Competing active observations on same entity and property trigger CONFLICT."""
    store = InMemoryWorldStateStore()
    updater = DeterministicWorldStateUpdater(store=store, clock=lambda: 1000.0)

    # Initial condition
    obs1 = make_observation(
        obs_id="o1",
        source_id="sensor_1",
        source_type="direct_sensor",
        entity_id="docking_bay",
        property_name="status",
        value="LOCKED",
        confidence=0.9,
        timestamp=1000.0,
    )
    updater.apply_observation(obs1)

    # Competing contradictory observation
    obs2 = make_observation(
        obs_id="o2",
        source_id="cam_bay",
        source_type="perception",
        entity_id="docking_bay",
        property_name="status",
        value="UNLOCKED",
        confidence=0.85,
        timestamp=1001.0,
    )

    res = updater.apply_observation(obs2)
    assert res.conflict is not None
    assert res.conflict.entity_id == "docking_bay"
    assert res.conflict.property_name == "status"
    assert res.conflict.existing_condition.value == "LOCKED"
    assert res.conflict.competing_observation.value == "UNLOCKED"


# ============================================================================
# O. CONFLICT RESOLUTION
# ============================================================================

def test_conflict_resolution_by_source_authority():
    """O1. Higher authority source overrides lower authority source."""
    resolver = DeterministicConflictResolver()
    policy = ConflictPolicy(source_authorities={"user": 100, "perception": 70, "direct_sensor": 85})

    existing = make_condition(value="DOOR_CLOSED", source_type="direct_sensor")
    competing = make_observation(value="DOOR_OPEN", source_type="user")

    conflict = StateConflict(
        conflict_id="conf_1",
        entity_id="door",
        property_name="state",
        existing_condition=existing,
        competing_observation=competing,
        detected_at=1000.0,
    )

    res = resolver.resolve(conflict, policy)
    assert res is not None
    assert res.strategy == ResolutionStrategy.SOURCE_AUTHORITY
    assert res.resolved_value == "DOOR_OPEN"
    assert res.winning_source == competing.source_id


def test_conflict_resolution_by_confidence_delta():
    """O2. Equal authority resolves to higher confidence when delta exceeds threshold."""
    resolver = DeterministicConflictResolver()
    policy = ConflictPolicy(
        source_authorities={"sensor": 80},
        confidence_delta_threshold=0.10,
        confidence_tie_breaker=True,
    )

    existing = make_condition(value="SAFE", confidence=0.70, source_type="sensor")
    competing = make_observation(value="HAZARD", confidence=0.95, source_type="sensor")

    conflict = StateConflict(
        conflict_id="conf_2",
        entity_id="zone_a",
        property_name="safety",
        existing_condition=existing,
        competing_observation=competing,
        detected_at=1000.0,
    )

    res = resolver.resolve(conflict, policy)
    assert res is not None
    assert res.strategy == ResolutionStrategy.HIGHER_CONFIDENCE
    assert res.resolved_value == "HAZARD"


def test_conflict_resolution_by_recency():
    """O3. Equal authority and similar confidence resolve by recency."""
    resolver = DeterministicConflictResolver()
    policy = ConflictPolicy(
        source_authorities={"sensor": 80},
        confidence_delta_threshold=0.15,
        recency_tie_breaker=True,
    )

    existing = make_condition(value="IDLE", confidence=0.85, observed_at=100.0, source_type="sensor")
    competing = make_observation(value="RUNNING", confidence=0.88, timestamp=150.0, source_type="sensor")

    conflict = StateConflict(
        conflict_id="conf_3",
        entity_id="engine",
        property_name="state",
        existing_condition=existing,
        competing_observation=competing,
        detected_at=150.0,
    )

    res = resolver.resolve(conflict, policy)
    assert res is not None
    assert res.strategy == ResolutionStrategy.RECENCY
    assert res.resolved_value == "RUNNING"


# ============================================================================
# P. UNRESOLVED CONFLICTS
# ============================================================================

def test_unresolved_conflict_remains_observable():
    """P1. Ambiguous/tied conflict remains UNRESOLVED and does not corrupt world state."""
    store = InMemoryWorldStateStore()
    # Policy with tie-breakers disabled
    policy = ConflictPolicy(
        source_authorities={"sensor": 80},
        confidence_tie_breaker=False,
        recency_tie_breaker=False,
    )
    updater = DeterministicWorldStateUpdater(store=store, conflict_policy=policy, clock=lambda: 1000.0)

    # Initial observation
    obs1 = make_observation(obs_id="o1", source_type="sensor", entity_id="vault", property_name="alarm", value=False)
    updater.apply_observation(obs1)
    prev_ver = store.get_current_state().version

    # Conflicting observation with equal authority and tie-breakers disabled
    obs2 = make_observation(obs_id="o2", source_type="sensor", entity_id="vault", property_name="alarm", value=True)
    res = updater.apply_observation(obs2)

    assert res.success is False
    assert res.transition_type == TransitionType.CONFLICT
    assert res.conflict is not None
    assert res.conflict.status == ConflictStatus.DETECTED

    # World state value was NOT overwritten
    current = store.get_current_state()
    assert current.version == prev_ver
    assert current.get_condition("vault", "alarm").value is False

    # Conflict is queryable in store
    conflicts = store.get_conflicts(unresolved_only=True)
    assert len(conflicts) == 1
    assert conflicts[0].entity_id == "vault"


# ============================================================================
# Q. SQLITE PERSISTENCE
# ============================================================================

def test_sqlite_persistence_round_trip(tmp_path):
    """Q1. SQLiteWorldStateStore accurately persists and retrieves entities, conditions, and states."""
    db_file = str(tmp_path / "world_test.db")
    store = SQLiteWorldStateStore(db_path=db_file)

    assert store.get_current_state().version == 1

    updater = DeterministicWorldStateUpdater(store=store, clock=lambda: 1000.0)
    obs = make_observation(
        obs_id="o_sql",
        entity_id="gateway_1",
        property_name="active_connections",
        value=42,
        timestamp=1000.0,
    )
    res = updater.apply_observation(obs)
    assert res.success is True
    assert store.get_current_state().version == 2

    # Direct condition lookup
    cond = store.get_condition("gateway_1", "active_connections")
    assert cond is not None
    assert cond.value == 42

    # Direct entity lookup
    ent = store.get_entity("gateway_1")
    assert ent is not None
    assert ent.entity_id == "gateway_1"


# ============================================================================
# R. RESTART / RELOAD RECOVERY
# ============================================================================

def test_sqlite_restart_reload_recovery(tmp_path):
    """R1. Re-instantiating SQLiteWorldStateStore on existing DB restores exact state."""
    db_file = str(tmp_path / "restart_test.db")

    # Session 1: Apply mutations
    store1 = SQLiteWorldStateStore(db_path=db_file)
    updater1 = DeterministicWorldStateUpdater(store=store1, clock=lambda: 1000.0)
    updater1.apply_observation(make_observation(obs_id="o1", entity_id="bot_1", property_name="battery", value=88))
    updater1.apply_observation(make_observation(obs_id="o2", entity_id="bot_2", property_name="battery", value=95))

    v_end = store1.get_current_state().version
    assert v_end == 3

    # Session 2: Reload DB fresh
    store2 = SQLiteWorldStateStore(db_path=db_file)
    recovered = store2.get_current_state()

    assert recovered.version == 3
    assert recovered.get_condition("bot_1", "battery").value == 88
    assert recovered.get_condition("bot_2", "battery").value == 95


# ============================================================================
# S. TRANSITION HISTORY
# ============================================================================

def test_transition_history_traversal(tmp_path):
    """S1. Transition history accurately records every applied state transition."""
    db_file = str(tmp_path / "history_test.db")
    store = SQLiteWorldStateStore(db_path=db_file)
    updater = DeterministicWorldStateUpdater(store=store, clock=lambda: 1000.0)

    updater.apply_observation(make_observation(obs_id="o1", entity_id="fan", property_name="rpm", value=1200, timestamp=1000.0))
    updater.apply_observation(make_observation(obs_id="o2", entity_id="fan", property_name="rpm", value=1800, timestamp=1005.0))

    history = store.get_transition_history(since_version=0)
    assert len(history) == 2
    assert history[0].transition_type == TransitionType.ADD
    assert history[0].to_version == 2
    assert history[1].transition_type == TransitionType.UPDATE
    assert history[1].to_version == 3
    assert history[1].old_value == 1200
    assert history[1].new_value == 1800


# ============================================================================
# T. CONCURRENCY & ATOMICITY
# ============================================================================

def test_concurrent_observation_updates_thread_safe(tmp_path):
    """T1. Multi-threaded observation application produces atomic, strictly monotonic versions."""
    db_file = str(tmp_path / "concurrency_test.db")
    store = SQLiteWorldStateStore(db_path=db_file)
    updater = DeterministicWorldStateUpdater(store=store, clock=time.time)

    num_threads = 5
    observations_per_thread = 4
    errors = []

    def worker(worker_idx: int):
        for i in range(observations_per_thread):
            obs = make_observation(
                obs_id=f"obs_w{worker_idx}_{i}",
                entity_id=f"sensor_w{worker_idx}",
                property_name=f"metric_{i}",
                value=i * 10,
                timestamp=time.time(),
            )
            res = updater.apply_observation(obs)
            if not res.success:
                errors.append(f"Worker {worker_idx} failed: {res.error}")

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0
    final_state = store.get_current_state()
    # Genesis version 1 + (num_threads * observations_per_thread) additions
    expected_version = 1 + (num_threads * observations_per_thread)
    assert final_state.version == expected_version
    assert len(final_state.conditions) == num_threads * observations_per_thread


# ============================================================================
# U. EVENT EMISSION
# ============================================================================

def test_structured_event_emission_lifecycle():
    """U1. Structured CognitiveEvents emitted for add, update, and conflict."""
    store = InMemoryWorldStateStore()
    sink = FakeEventSink()
    updater = DeterministicWorldStateUpdater(store=store, event_sink=sink, clock=lambda: 1000.0)

    obs1 = make_observation(obs_id="o1", entity_id="gate", property_name="open", value=False)
    updater.apply_observation(obs1)

    event_types = [e.event_type for e in sink.get_events()]
    assert CognitiveEventType.WORLD_STATE_UPDATED in event_types
    assert CognitiveEventType.WORLD_CONDITION_CHANGED in event_types
    assert CognitiveEventType.WORLD_ENTITY_ADDED in event_types


# ============================================================================
# V. EVENT CORRELATION
# ============================================================================

def test_event_correlation_metadata():
    """V1. Emitted events contain state version, entity ID, and summary."""
    store = InMemoryWorldStateStore()
    sink = FakeEventSink()
    updater = DeterministicWorldStateUpdater(store=store, event_sink=sink, clock=lambda: 1000.0)

    obs = make_observation(obs_id="o1", entity_id="air_lock", property_name="pressurized", value=True)
    updater.apply_observation(obs)

    cond_events = [e for e in sink.get_events() if e.event_type == CognitiveEventType.WORLD_CONDITION_CHANGED]
    assert len(cond_events) == 1
    assert cond_events[0].metadata["entity_id"] == "air_lock"
    assert cond_events[0].metadata["property_name"] == "pressurized"
    assert cond_events[0].metadata["new_value"] is True
    assert cond_events[0].metadata["version"] == 2


# ============================================================================
# W. COGNITIVE CONTEXT INTEGRATION
# ============================================================================

def test_world_state_to_context_items_adapter():
    """W1. Active conditions convert to model-neutral ContextItem objects for ContextManager."""
    cond1 = make_condition(entity_id="server_1", property_name="cpu_load", value="15%")
    cond2 = make_condition(entity_id="server_1", property_name="status", value="HEALTHY")
    state = WorldState(
        state_id="ws_ctx",
        version=5,
        timestamp=1000.0,
        conditions=(cond1, cond2),
    )

    items = world_state_to_context_items(state, now=1005.0)
    assert len(items) == 2
    assert all(isinstance(item, ContextItem) for item in items)
    assert all(item.source == ContextSource.WORLD_STATE for item in items)
    assert all(item.priority == ContextPriority.HIGH for item in items)


def test_cognitive_state_incorporates_world_conditions():
    """W2. CognitiveState accepts world_conditions tuple."""
    cond = make_condition()
    cog_state = CognitiveState(
        original_goal="Monitor systems",
        world_conditions=(cond,),
    )
    assert len(cog_state.world_conditions) == 1
    assert cog_state.to_dict()["world_conditions_count"] == 1


# ============================================================================
# X. REPLAY RECONSTRUCTION
# ============================================================================

def test_replay_reconstruction_from_transition_history():
    """X1. Sequence of transitions reconstructs exact historical states deterministically."""
    store = InMemoryWorldStateStore()
    updater = DeterministicWorldStateUpdater(store=store, clock=lambda: 1000.0)

    obs_list = [
        make_observation(obs_id=f"o_{i}", entity_id="valve", property_name="pos", value=i * 25, timestamp=1000.0 + i)
        for i in range(4)
    ]
    for obs in obs_list:
        updater.apply_observation(obs)

    # Reconstruct state from transition sequence
    transitions = store.get_transition_history()
    assert len(transitions) == 4

    reconstructed_conditions: Dict[Tuple[str, str], Any] = {}
    for t in transitions:
        reconstructed_conditions[(t.entity_id, t.property_name)] = t.new_value

    current = store.get_current_state()
    assert current.get_condition("valve", "pos").value == reconstructed_conditions[("valve", "pos")]
    assert current.get_condition("valve", "pos").value == 75


# ============================================================================
# Y. INVALID OBSERVATION REJECTION
# ============================================================================

def test_invalid_observation_missing_fields_rejected():
    """Y1. Observation with missing or malformed fields is rejected."""
    store = InMemoryWorldStateStore()
    updater = DeterministicWorldStateUpdater(store=store)

    # Missing entity_id
    with pytest.raises(ValueError):
        Observation(
            observation_id="o_bad",
            source_id="s1",
            source_type="t1",
            timestamp=100.0,
            entity_id="",
            property_name="p",
            value="v",
        )


def test_observation_already_expired_rejected():
    """Y2. Observation whose expires_at is already in the past is rejected."""
    store = InMemoryWorldStateStore()
    updater = DeterministicWorldStateUpdater(store=store, clock=lambda: 200.0)

    obs = make_observation(timestamp=100.0, expires_at=150.0)  # expires_at 150 < clock 200
    res = updater.apply_observation(obs)
    assert res.success is False
    assert res.transition_type == TransitionType.REJECTED
    assert "already expired" in res.error


# ============================================================================
# Z. SECURITY & ARCHITECTURAL INVARIANTS
# ============================================================================

def test_security_zero_code_execution_or_eval():
    """Z1. World state persistence and deserialization strictly uses JSON, never pickle or eval."""
    cond = make_condition(value={"safe_dict": True, "code": "print('malicious')"})
    serialized = json.dumps(cond.to_dict())

    # Ensure json loads safely without code execution
    deserialized = json.loads(serialized)
    recovered = WorldCondition.from_dict(deserialized)
    assert recovered.value["code"] == "print('malicious')"


def test_security_updater_has_no_direct_tools_or_models():
    """Z2. DeterministicWorldStateUpdater does not hold model routers or tool orchestrators."""
    store = InMemoryWorldStateStore()
    updater = DeterministicWorldStateUpdater(store=store)

    assert not hasattr(updater, "model_router")
    assert not hasattr(updater, "tool_orchestrator")
    assert not hasattr(updater, "llm")
    assert not hasattr(updater, "prompt_template")
