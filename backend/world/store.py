import copy
import json
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from core.interfaces.world_interface import WorldStateStoreInterface
from core.models.world_state import (
    ConflictResolution,
    ConflictStatus,
    ResolutionStrategy,
    StateConflict,
    StateProvenance,
    TransitionType,
    WorldCondition,
    WorldEntity,
    WorldRelationship,
    WorldState,
    WorldStateTransition,
)


def serialize_world_state(state: WorldState) -> Dict[str, Any]:
    """Serialize a WorldState snapshot to a JSON-compatible dictionary."""
    return state.to_dict()


def deserialize_world_state(data: Dict[str, Any]) -> WorldState:
    """Reconstruct an immutable WorldState snapshot from a serialized dictionary."""
    return WorldState.from_dict(data)


def serialize_transition(transition: WorldStateTransition) -> Dict[str, Any]:
    """Serialize a WorldStateTransition to a JSON-compatible dictionary."""
    return transition.to_dict()


def deserialize_transition(data: Dict[str, Any]) -> WorldStateTransition:
    """Reconstruct a WorldStateTransition from a serialized dictionary."""
    return WorldStateTransition.from_dict(data)


def serialize_conflict(conflict: StateConflict) -> Dict[str, Any]:
    """Serialize a StateConflict to a JSON-compatible dictionary."""
    return conflict.to_dict()


def deserialize_conflict(data: Dict[str, Any]) -> StateConflict:
    """Reconstruct a StateConflict from a serialized dictionary."""
    return StateConflict.from_dict(data)


def create_initial_world_state(state_id: Optional[str] = None, timestamp: Optional[float] = None) -> WorldState:
    """Generate the baseline genesis WorldState (version 1)."""
    return WorldState(
        state_id=state_id or f"ws_init_{int(time.time())}",
        version=1,
        timestamp=timestamp or time.time(),
        entities=(),
        conditions=(),
        relationships=(),
        metadata={"genesis": True},
    )


class InMemoryWorldStateStore(WorldStateStoreInterface):
    """
    Thread-safe in-memory store for WorldState, transitions, and conflicts.
    Guarantees monotonic version ordering and immutable snapshots.
    """

    def __init__(self, initial_state: Optional[WorldState] = None):
        self._lock = threading.Lock()
        genesis = initial_state or create_initial_world_state()
        self._current_state: WorldState = genesis
        self._versions: Dict[int, WorldState] = {genesis.version: genesis}
        self._transitions: List[WorldStateTransition] = []
        self._conflicts: Dict[str, StateConflict] = {}

    def get_current_state(self) -> WorldState:
        with self._lock:
            return self._current_state

    def get_condition(self, entity_id: str, property_name: str) -> Optional[WorldCondition]:
        with self._lock:
            return self._current_state.get_condition(entity_id, property_name)

    def get_entity(self, entity_id: str) -> Optional[WorldEntity]:
        with self._lock:
            return self._current_state.get_entity(entity_id)

    def apply_transition(self, transition: WorldStateTransition, next_state: WorldState) -> WorldState:
        with self._lock:
            expected_version = self._current_state.version + 1
            if next_state.version != expected_version:
                raise ValueError(
                    f"Version monotonicity violation: expected version {expected_version}, got {next_state.version}"
                )
            if transition.to_version != next_state.version:
                raise ValueError(
                    f"Transition to_version ({transition.to_version}) does not match next_state.version ({next_state.version})"
                )

            self._transitions.append(transition)
            self._versions[next_state.version] = next_state
            self._current_state = next_state
            return self._current_state

    def get_state_version(self, version: int) -> Optional[WorldState]:
        with self._lock:
            return self._versions.get(version)

    def get_transition_history(self, since_version: int = 0, limit: int = 100) -> List[WorldStateTransition]:
        with self._lock:
            filtered = [t for t in self._transitions if t.to_version > since_version]
            return filtered[:limit]

    def get_conflicts(self, unresolved_only: bool = True) -> List[StateConflict]:
        with self._lock:
            if unresolved_only:
                return [c for c in self._conflicts.values() if c.status != ConflictStatus.RESOLVED]
            return list(self._conflicts.values())

    def save_conflict(self, conflict: StateConflict) -> StateConflict:
        with self._lock:
            self._conflicts[conflict.conflict_id] = conflict
            return conflict

    def resolve_conflict(self, resolution: ConflictResolution) -> None:
        with self._lock:
            if resolution.conflict_id not in self._conflicts:
                raise KeyError(f"Conflict '{resolution.conflict_id}' not found.")
            old = self._conflicts[resolution.conflict_id]
            updated = StateConflict(
                conflict_id=old.conflict_id,
                entity_id=old.entity_id,
                property_name=old.property_name,
                existing_condition=old.existing_condition,
                competing_observation=old.competing_observation,
                detected_at=old.detected_at,
                status=ConflictStatus.RESOLVED,
                resolution=resolution,
            )
            self._conflicts[resolution.conflict_id] = updated


class SQLiteWorldStateStore(WorldStateStoreInterface):
    """
    Thread-safe, atomic SQLite-backed persistent world state store.
    Uses locked internal helpers to guarantee non-recursive locking and atomicity.
    """

    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=15.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout = 5000")
        conn.execute("PRAGMA foreign_keys = ON")
        if self.db_path != ":memory:":
            conn.execute("PRAGMA journal_mode = WAL")
        return conn

    def _init_db(self) -> None:
        with self._lock:
            with self._get_connection() as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS world_states (
                        version INTEGER PRIMARY KEY,
                        state_id TEXT NOT NULL,
                        timestamp REAL NOT NULL,
                        payload_json TEXT NOT NULL
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS world_conditions (
                        entity_id TEXT NOT NULL,
                        property_name TEXT NOT NULL,
                        value_json TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        observed_at REAL NOT NULL,
                        expires_at REAL,
                        provenance_json TEXT NOT NULL,
                        version INTEGER NOT NULL,
                        PRIMARY KEY (entity_id, property_name)
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS world_entities (
                        entity_id TEXT PRIMARY KEY,
                        entity_type TEXT NOT NULL,
                        attributes_json TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        provenance_json TEXT NOT NULL,
                        version INTEGER NOT NULL
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS world_relationships (
                        source_entity_id TEXT NOT NULL,
                        relationship_type TEXT NOT NULL,
                        target_entity_id TEXT NOT NULL,
                        attributes_json TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        provenance_json TEXT NOT NULL,
                        version INTEGER NOT NULL,
                        PRIMARY KEY (source_entity_id, relationship_type, target_entity_id)
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS world_transitions (
                        transition_id TEXT PRIMARY KEY,
                        from_version INTEGER NOT NULL,
                        to_version INTEGER NOT NULL,
                        transition_type TEXT NOT NULL,
                        observation_id TEXT NOT NULL,
                        entity_id TEXT NOT NULL,
                        property_name TEXT NOT NULL,
                        payload_json TEXT NOT NULL,
                        timestamp REAL NOT NULL
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS world_conflicts (
                        conflict_id TEXT PRIMARY KEY,
                        entity_id TEXT NOT NULL,
                        property_name TEXT NOT NULL,
                        status TEXT NOT NULL,
                        payload_json TEXT NOT NULL,
                        detected_at REAL NOT NULL,
                        resolved_at REAL
                    )
                    """
                )
                conn.execute("CREATE INDEX IF NOT EXISTS idx_transitions_to_ver ON world_transitions(to_version)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_conflicts_status ON world_conflicts(status)")
                conn.commit()

                # Seed genesis state if empty
                cursor = conn.execute("SELECT COUNT(*) as cnt FROM world_states")
                if cursor.fetchone()["cnt"] == 0:
                    genesis = create_initial_world_state()
                    payload = json.dumps(serialize_world_state(genesis))
                    conn.execute(
                        "INSERT INTO world_states (version, state_id, timestamp, payload_json) VALUES (?, ?, ?, ?)",
                        (genesis.version, genesis.state_id, genesis.timestamp, payload),
                    )
                    conn.commit()

    # ------------------------------------------------------------------------
    # Locked internal helpers (assuming self._lock is already held)
    # ------------------------------------------------------------------------

    def _get_current_state_locked(self, conn: sqlite3.Connection) -> WorldState:
        cursor = conn.execute(
            "SELECT payload_json FROM world_states ORDER BY version DESC LIMIT 1"
        )
        row = cursor.fetchone()
        if row is None:
            return create_initial_world_state()
        data = json.loads(row["payload_json"])
        return deserialize_world_state(data)

    def _apply_transition_locked(
        self,
        conn: sqlite3.Connection,
        transition: WorldStateTransition,
        next_state: WorldState,
    ) -> WorldState:
        cur_state = self._get_current_state_locked(conn)
        expected_version = cur_state.version + 1
        if next_state.version != expected_version:
            raise ValueError(
                f"Version monotonicity violation: expected version {expected_version}, got {next_state.version}"
            )
        if transition.to_version != next_state.version:
            raise ValueError(
                f"Transition to_version ({transition.to_version}) does not match next_state.version ({next_state.version})"
            )

        # 1. Insert next state
        state_payload = json.dumps(serialize_world_state(next_state))
        conn.execute(
            "INSERT INTO world_states (version, state_id, timestamp, payload_json) VALUES (?, ?, ?, ?)",
            (next_state.version, next_state.state_id, next_state.timestamp, state_payload),
        )

        # 2. Insert transition
        trans_payload = json.dumps(serialize_transition(transition))
        conn.execute(
            """
            INSERT INTO world_transitions (
                transition_id, from_version, to_version, transition_type,
                observation_id, entity_id, property_name, payload_json, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                transition.transition_id,
                transition.from_version,
                transition.to_version,
                transition.transition_type.value,
                transition.observation_id,
                transition.entity_id,
                transition.property_name,
                trans_payload,
                transition.timestamp,
            ),
        )

        # 3. Synchronize conditions table
        for cond in next_state.conditions:
            conn.execute(
                """
                INSERT INTO world_conditions (
                    entity_id, property_name, value_json, confidence,
                    observed_at, expires_at, provenance_json, version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(entity_id, property_name) DO UPDATE SET
                    value_json = excluded.value_json,
                    confidence = excluded.confidence,
                    observed_at = excluded.observed_at,
                    expires_at = excluded.expires_at,
                    provenance_json = excluded.provenance_json,
                    version = excluded.version
                """,
                (
                    cond.entity_id,
                    cond.property_name,
                    json.dumps(cond.value),
                    cond.confidence,
                    cond.observed_at,
                    cond.expires_at,
                    json.dumps(cond.provenance.to_dict()),
                    next_state.version,
                ),
            )

        # 4. Synchronize entities table
        for ent in next_state.entities:
            conn.execute(
                """
                INSERT INTO world_entities (
                    entity_id, entity_type, attributes_json, confidence, provenance_json, version
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(entity_id) DO UPDATE SET
                    entity_type = excluded.entity_type,
                    attributes_json = excluded.attributes_json,
                    confidence = excluded.confidence,
                    provenance_json = excluded.provenance_json,
                    version = excluded.version
                """,
                (
                    ent.entity_id,
                    ent.entity_type,
                    json.dumps(ent.attributes),
                    ent.confidence,
                    json.dumps(ent.provenance.to_dict()),
                    next_state.version,
                ),
            )

        # 5. Synchronize relationships table
        for rel in next_state.relationships:
            conn.execute(
                """
                INSERT INTO world_relationships (
                    source_entity_id, relationship_type, target_entity_id,
                    attributes_json, confidence, provenance_json, version
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_entity_id, relationship_type, target_entity_id) DO UPDATE SET
                    attributes_json = excluded.attributes_json,
                    confidence = excluded.confidence,
                    provenance_json = excluded.provenance_json,
                    version = excluded.version
                """,
                (
                    rel.source_entity_id,
                    rel.relationship_type,
                    rel.target_entity_id,
                    json.dumps(rel.attributes),
                    rel.confidence,
                    json.dumps(rel.provenance.to_dict()),
                    next_state.version,
                ),
            )

        conn.commit()
        return next_state

    # ------------------------------------------------------------------------
    # Public interface implementations
    # ------------------------------------------------------------------------

    def get_current_state(self) -> WorldState:
        with self._lock:
            with self._get_connection() as conn:
                return self._get_current_state_locked(conn)

    def get_condition(self, entity_id: str, property_name: str) -> Optional[WorldCondition]:
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    """
                    SELECT value_json, confidence, observed_at, expires_at, provenance_json
                    FROM world_conditions
                    WHERE entity_id = ? AND property_name = ?
                    """,
                    (entity_id, property_name),
                )
                row = cursor.fetchone()
                if row is None:
                    return None
                val = json.loads(row["value_json"])
                prov = StateProvenance.from_dict(json.loads(row["provenance_json"]))
                return WorldCondition(
                    entity_id=entity_id,
                    property_name=property_name,
                    value=val,
                    confidence=row["confidence"],
                    observed_at=row["observed_at"],
                    expires_at=row["expires_at"],
                    provenance=prov,
                )

    def get_entity(self, entity_id: str) -> Optional[WorldEntity]:
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    """
                    SELECT entity_type, attributes_json, confidence, provenance_json
                    FROM world_entities
                    WHERE entity_id = ?
                    """,
                    (entity_id,),
                )
                row = cursor.fetchone()
                if row is None:
                    return None
                attrs = json.loads(row["attributes_json"])
                prov = StateProvenance.from_dict(json.loads(row["provenance_json"]))
                return WorldEntity(
                    entity_id=entity_id,
                    entity_type=row["entity_type"],
                    attributes=attrs,
                    confidence=row["confidence"],
                    provenance=prov,
                )

    def apply_transition(self, transition: WorldStateTransition, next_state: WorldState) -> WorldState:
        with self._lock:
            with self._get_connection() as conn:
                return self._apply_transition_locked(conn, transition, next_state)

    def get_state_version(self, version: int) -> Optional[WorldState]:
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    "SELECT payload_json FROM world_states WHERE version = ?",
                    (version,),
                )
                row = cursor.fetchone()
                if row is None:
                    return None
                data = json.loads(row["payload_json"])
                return deserialize_world_state(data)

    def get_transition_history(self, since_version: int = 0, limit: int = 100) -> List[WorldStateTransition]:
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    """
                    SELECT payload_json FROM world_transitions
                    WHERE to_version > ?
                    ORDER BY to_version ASC
                    LIMIT ?
                    """,
                    (since_version, limit),
                )
                rows = cursor.fetchall()
                results = []
                for r in rows:
                    data = json.loads(r["payload_json"])
                    results.append(deserialize_transition(data))
                return results

    def get_conflicts(self, unresolved_only: bool = True) -> List[StateConflict]:
        with self._lock:
            with self._get_connection() as conn:
                if unresolved_only:
                    cursor = conn.execute(
                        "SELECT payload_json FROM world_conflicts WHERE status != ? ORDER BY detected_at DESC",
                        (ConflictStatus.RESOLVED.value,),
                    )
                else:
                    cursor = conn.execute(
                        "SELECT payload_json FROM world_conflicts ORDER BY detected_at DESC"
                    )
                rows = cursor.fetchall()
                return [deserialize_conflict(json.loads(r["payload_json"])) for r in rows]

    def save_conflict(self, conflict: StateConflict) -> StateConflict:
        with self._lock:
            payload = json.dumps(serialize_conflict(conflict))
            with self._get_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO world_conflicts (
                        conflict_id, entity_id, property_name, status, payload_json, detected_at, resolved_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(conflict_id) DO UPDATE SET
                        status = excluded.status,
                        payload_json = excluded.payload_json,
                        resolved_at = excluded.resolved_at
                    """,
                    (
                        conflict.conflict_id,
                        conflict.entity_id,
                        conflict.property_name,
                        conflict.status.value,
                        payload,
                        conflict.detected_at,
                        conflict.resolution.resolved_at if conflict.resolution else None,
                    ),
                )
                conn.commit()
            return conflict

    def resolve_conflict(self, resolution: ConflictResolution) -> None:
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    "SELECT payload_json FROM world_conflicts WHERE conflict_id = ?",
                    (resolution.conflict_id,),
                )
                row = cursor.fetchone()
                if row is None:
                    raise KeyError(f"Conflict '{resolution.conflict_id}' not found.")
                old = deserialize_conflict(json.loads(row["payload_json"]))
                updated = StateConflict(
                    conflict_id=old.conflict_id,
                    entity_id=old.entity_id,
                    property_name=old.property_name,
                    existing_condition=old.existing_condition,
                    competing_observation=old.competing_observation,
                    detected_at=old.detected_at,
                    status=ConflictStatus.RESOLVED,
                    resolution=resolution,
                )
                payload = json.dumps(serialize_conflict(updated))
                conn.execute(
                    """
                    UPDATE world_conflicts
                    SET status = ?, payload_json = ?, resolved_at = ?
                    WHERE conflict_id = ?
                    """,
                    (
                        ConflictStatus.RESOLVED.value,
                        payload,
                        resolution.resolved_at,
                        resolution.conflict_id,
                    ),
                )
                conn.commit()
