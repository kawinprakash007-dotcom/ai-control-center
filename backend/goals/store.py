import copy
import json
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from core.interfaces.goal_interface import GoalStoreInterface
from core.models.goal import (
    Goal,
    GoalCompletionCriteria,
    GoalConstraints,
    GoalProgress,
    GoalStatus,
    Objective,
    ObjectiveStatus,
)
from core.models.goal_management import GoalPriority


def _serialize_objective(obj: Objective) -> Dict[str, Any]:
    return {
        "objective_id": obj.objective_id,
        "description": obj.description,
        "order": obj.order,
        "dependencies": list(obj.dependencies),
        "status": obj.status.value,
        "completion_criteria": obj.completion_criteria,
        "attempts": obj.attempts,
        "max_attempts": obj.max_attempts,
        "progress": obj.progress,
        "result_summary": obj.result_summary,
        "blocker_reason": obj.blocker_reason,
        "metadata": obj.metadata,
    }


def _deserialize_objective(data: Dict[str, Any]) -> Objective:
    return Objective(
        objective_id=data["objective_id"],
        description=data["description"],
        order=data.get("order", 1),
        dependencies=tuple(data.get("dependencies", ())),
        status=ObjectiveStatus(data.get("status", "pending")),
        completion_criteria=data.get("completion_criteria"),
        attempts=data.get("attempts", 0),
        max_attempts=data.get("max_attempts", 3),
        progress=float(data.get("progress", 0.0)),
        result_summary=data.get("result_summary"),
        blocker_reason=data.get("blocker_reason"),
        metadata=data.get("metadata", {}),
    )


def serialize_goal(goal: Goal) -> Dict[str, Any]:
    """Convert a Goal instance into a JSON-serializable dictionary."""
    return {
        "goal_id": goal.goal_id,
        "original_goal": goal.original_goal,
        "status": goal.status.value,
        "active_objective_id": goal.active_objective_id,
        "created_at": goal.created_at,
        "updated_at": goal.updated_at,
        "constraints": {
            "deadline": goal.constraints.deadline,
            "allowed_capabilities": list(goal.constraints.allowed_capabilities)
            if goal.constraints.allowed_capabilities
            else None,
            "max_turns_total": goal.constraints.max_turns_total,
            "max_turns_per_objective": goal.constraints.max_turns_per_objective,
            "max_failed_objectives": goal.constraints.max_failed_objectives,
            "max_consecutive_no_progress": goal.constraints.max_consecutive_no_progress,
            "privacy_requirement": goal.constraints.privacy_requirement,
            "autonomy_level": goal.constraints.autonomy_level,
        },
        "completion_criteria": {
            "require_all_objectives": goal.completion_criteria.require_all_objectives,
            "required_objective_ids": list(goal.completion_criteria.required_objective_ids)
            if goal.completion_criteria.required_objective_ids
            else None,
            "min_completion_percentage": goal.completion_criteria.min_completion_percentage,
            "user_confirmation_required": goal.completion_criteria.user_confirmation_required,
        },
        "progress": {
            "completed_objectives": goal.progress.completed_objectives,
            "total_objectives": goal.progress.total_objectives,
            "percentage": goal.progress.percentage,
            "active_objective_id": goal.progress.active_objective_id,
            "blockers": list(goal.progress.blockers),
            "last_update": goal.progress.last_update,
            "progress_reason": goal.progress.progress_reason,
        },
        "objectives": [_serialize_objective(obj) for obj in goal.objectives],
        "metadata": goal.metadata,
        "priority": goal.priority.value if hasattr(goal.priority, "value") else str(goal.priority),
    }


def deserialize_goal(data: Dict[str, Any]) -> Goal:
    """Parse a serialized dictionary into a Goal instance."""
    raw_constraints = data.get("constraints", {})
    allowed_caps = raw_constraints.get("allowed_capabilities")
    constraints = GoalConstraints(
        deadline=raw_constraints.get("deadline"),
        allowed_capabilities=tuple(allowed_caps) if allowed_caps else None,
        max_turns_total=raw_constraints.get("max_turns_total", 20),
        max_turns_per_objective=raw_constraints.get("max_turns_per_objective", 5),
        max_failed_objectives=raw_constraints.get("max_failed_objectives", 2),
        max_consecutive_no_progress=raw_constraints.get("max_consecutive_no_progress", 3),
        privacy_requirement=raw_constraints.get("privacy_requirement"),
        autonomy_level=raw_constraints.get("autonomy_level", "supervised"),
    )

    raw_criteria = data.get("completion_criteria", {})
    req_ids = raw_criteria.get("required_objective_ids")
    completion_criteria = GoalCompletionCriteria(
        require_all_objectives=raw_criteria.get("require_all_objectives", True),
        required_objective_ids=tuple(req_ids) if req_ids else None,
        min_completion_percentage=float(raw_criteria.get("min_completion_percentage", 100.0)),
        user_confirmation_required=raw_criteria.get("user_confirmation_required", False),
    )

    raw_prog = data.get("progress", {})
    progress = GoalProgress(
        completed_objectives=raw_prog.get("completed_objectives", 0),
        total_objectives=raw_prog.get("total_objectives", 0),
        percentage=float(raw_prog.get("percentage", 0.0)),
        active_objective_id=raw_prog.get("active_objective_id"),
        blockers=tuple(raw_prog.get("blockers", ())),
        last_update=raw_prog.get("last_update", time.time()),
        progress_reason=raw_prog.get("progress_reason", ""),
    )

    raw_objs = data.get("objectives", [])
    objectives = tuple(_deserialize_objective(item) for item in raw_objs)
    raw_priority = data.get("priority", "normal")
    priority = GoalPriority.from_str(raw_priority)

    return Goal(
        original_goal=data["original_goal"],
        goal_id=data["goal_id"],
        objectives=objectives,
        status=GoalStatus(data.get("status", "created")),
        constraints=constraints,
        completion_criteria=completion_criteria,
        progress=progress,
        active_objective_id=data.get("active_objective_id"),
        created_at=data.get("created_at", time.time()),
        updated_at=data.get("updated_at", time.time()),
        metadata=data.get("metadata", {}),
        priority=priority,
    )


class InMemoryGoalStore(GoalStoreInterface):
    """
    In-memory, thread-safe implementation of GoalStoreInterface.
    Ideal for testing and ephemeral interactive sessions.
    """

    def __init__(self):
        self._goals: Dict[str, Goal] = {}
        self._claim: Optional[Tuple[str, str, float]] = None  # (goal_id, owner_id, expires_at)
        self._lock = threading.Lock()

    def claim_goal(self, goal_id: str, owner_id: str, lease_duration: float = 60.0) -> bool:
        now = time.time()
        with self._lock:
            if goal_id not in self._goals:
                return False
            if self._claim is not None:
                cur_goal, cur_owner, cur_exp = self._claim
                if cur_exp > now and (cur_goal != goal_id or cur_owner != owner_id):
                    return False
            self._claim = (goal_id, owner_id, now + lease_duration)
            return True

    def release_goal(self, goal_id: str, owner_id: str) -> bool:
        with self._lock:
            if self._claim is not None:
                cur_goal, cur_owner, _ = self._claim
                if cur_goal == goal_id and cur_owner == owner_id:
                    self._claim = None
                    return True
            return False

    def get_claimed_goal(self) -> Optional[Tuple[str, str, float]]:
        now = time.time()
        with self._lock:
            if self._claim is not None:
                cur_goal, cur_owner, cur_exp = self._claim
                if cur_exp > now:
                    return (cur_goal, cur_owner, cur_exp)
                else:
                    self._claim = None
            return None

    def update_priority(self, goal_id: str, priority: Any) -> Goal:
        with self._lock:
            if goal_id not in self._goals:
                raise KeyError(f"Cannot update non-existent goal '{goal_id}'.")
            g = self._goals[goal_id]
            g.priority = GoalPriority.from_str(priority) if isinstance(priority, str) else priority
            g.updated_at = time.time()
            self._goals[goal_id] = copy.deepcopy(g)
            return copy.deepcopy(g)

    def update_deadline(self, goal_id: str, deadline: Optional[float]) -> Goal:
        with self._lock:
            if goal_id not in self._goals:
                raise KeyError(f"Cannot update non-existent goal '{goal_id}'.")
            g = self._goals[goal_id]
            c = g.constraints
            g.constraints = GoalConstraints(
                deadline=deadline,
                allowed_capabilities=c.allowed_capabilities,
                max_turns_total=c.max_turns_total,
                max_turns_per_objective=c.max_turns_per_objective,
                max_failed_objectives=c.max_failed_objectives,
                max_consecutive_no_progress=c.max_consecutive_no_progress,
                privacy_requirement=c.privacy_requirement,
                autonomy_level=c.autonomy_level,
            )
            g.updated_at = time.time()
            self._goals[goal_id] = copy.deepcopy(g)
            return copy.deepcopy(g)

    def create_goal(self, goal: Goal) -> Goal:
        with self._lock:
            if goal.goal_id in self._goals:
                raise ValueError(f"Goal with ID '{goal.goal_id}' already exists.")
            self._goals[goal.goal_id] = copy.deepcopy(goal)
            return copy.deepcopy(goal)

    def get_goal(self, goal_id: str) -> Optional[Goal]:
        with self._lock:
            goal = self._goals.get(goal_id)
            return copy.deepcopy(goal) if goal is not None else None

    def update_goal(self, goal: Goal) -> Goal:
        with self._lock:
            if goal.goal_id not in self._goals:
                raise KeyError(f"Cannot update non-existent goal '{goal.goal_id}'.")
            goal.updated_at = time.time()
            self._goals[goal.goal_id] = copy.deepcopy(goal)
            return copy.deepcopy(goal)

    def list_goals(
        self,
        status: Optional[GoalStatus] = None,
        limit: int = 50,
    ) -> List[Goal]:
        with self._lock:
            results = []
            for g in self._goals.values():
                if status is None or g.status == status:
                    results.append(copy.deepcopy(g))
                if len(results) >= limit:
                    break
            return results

    def get_all_goals(self) -> List[Goal]:
        """Return all stored goals."""
        return self.list_goals(limit=10000)

    def delete_goal(self, goal_id: str) -> bool:
        with self._lock:
            if goal_id in self._goals:
                del self._goals[goal_id]
                return True
            return False


class SQLiteGoalStore(GoalStoreInterface):
    """
    SQLite-backed persistent implementation of GoalStoreInterface.
    Ensures goal state survives across cognitive turns and process restarts.
    """

    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._lock:
            with self._get_connection() as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS goals (
                        goal_id TEXT PRIMARY KEY,
                        original_goal TEXT NOT NULL,
                        status TEXT NOT NULL,
                        percentage REAL NOT NULL,
                        created_at REAL NOT NULL,
                        updated_at REAL NOT NULL,
                        payload_json TEXT NOT NULL
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS goal_claims (
                        claim_id INTEGER PRIMARY KEY CHECK (claim_id = 1),
                        goal_id TEXT NOT NULL,
                        owner_id TEXT NOT NULL,
                        expires_at REAL NOT NULL
                    )
                    """
                )
                conn.execute("CREATE INDEX IF NOT EXISTS idx_goals_status ON goals(status)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_goals_updated ON goals(updated_at)")
                conn.commit()

    def _get_goal_locked(self, goal_id: str) -> Optional[Goal]:
        """Internal helper for reading a goal when caller already holds self._lock."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT payload_json FROM goals WHERE goal_id = ?",
                (goal_id,),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            data = json.loads(row["payload_json"])
            return deserialize_goal(data)

    def _update_goal_locked(self, goal: Goal) -> Goal:
        """Internal helper for updating a goal when caller already holds self._lock."""
        goal.updated_at = time.time()
        serialized = serialize_goal(goal)
        payload = json.dumps(serialized)
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                UPDATE goals
                SET status = ?, percentage = ?, updated_at = ?, payload_json = ?
                WHERE goal_id = ?
                """,
                (
                    goal.status.value,
                    goal.progress.percentage,
                    goal.updated_at,
                    payload,
                    goal.goal_id,
                ),
            )
            if cursor.rowcount == 0:
                raise KeyError(f"Cannot update non-existent goal '{goal.goal_id}'.")
            conn.commit()
        return copy.deepcopy(goal)

    def claim_goal(self, goal_id: str, owner_id: str, lease_duration: float = 60.0) -> bool:
        now = time.time()
        with self._lock:
            if self._get_goal_locked(goal_id) is None:
                return False
            with self._get_connection() as conn:
                cursor = conn.execute("SELECT goal_id, owner_id, expires_at FROM goal_claims WHERE claim_id = 1")
                row = cursor.fetchone()
                if row is not None:
                    cur_goal = row["goal_id"]
                    cur_owner = row["owner_id"]
                    cur_exp = row["expires_at"]
                    if cur_exp > now and (cur_goal != goal_id or cur_owner != owner_id):
                        return False
                conn.execute(
                    """
                    INSERT INTO goal_claims (claim_id, goal_id, owner_id, expires_at)
                    VALUES (1, ?, ?, ?)
                    ON CONFLICT(claim_id) DO UPDATE SET
                        goal_id = excluded.goal_id,
                        owner_id = excluded.owner_id,
                        expires_at = excluded.expires_at
                    """,
                    (goal_id, owner_id, now + lease_duration),
                )
                conn.commit()
                return True

    def release_goal(self, goal_id: str, owner_id: str) -> bool:
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    "DELETE FROM goal_claims WHERE claim_id = 1 AND goal_id = ? AND owner_id = ?",
                    (goal_id, owner_id),
                )
                conn.commit()
                return cursor.rowcount > 0

    def get_claimed_goal(self) -> Optional[Tuple[str, str, float]]:
        now = time.time()
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.execute("SELECT goal_id, owner_id, expires_at FROM goal_claims WHERE claim_id = 1")
                row = cursor.fetchone()
                if row is None:
                    return None
                if row["expires_at"] > now:
                    return (row["goal_id"], row["owner_id"], row["expires_at"])
                conn.execute("DELETE FROM goal_claims WHERE claim_id = 1")
                conn.commit()
                return None

    def update_priority(self, goal_id: str, priority: Any) -> Goal:
        with self._lock:
            goal = self._get_goal_locked(goal_id)
            if goal is None:
                raise KeyError(f"Cannot update non-existent goal '{goal_id}'.")
            goal.priority = GoalPriority.from_str(priority) if isinstance(priority, str) else priority
            return self._update_goal_locked(goal)

    def update_deadline(self, goal_id: str, deadline: Optional[float]) -> Goal:
        with self._lock:
            goal = self._get_goal_locked(goal_id)
            if goal is None:
                raise KeyError(f"Cannot update non-existent goal '{goal_id}'.")
            c = goal.constraints
            goal.constraints = GoalConstraints(
                deadline=deadline,
                allowed_capabilities=c.allowed_capabilities,
                max_turns_total=c.max_turns_total,
                max_turns_per_objective=c.max_turns_per_objective,
                max_failed_objectives=c.max_failed_objectives,
                max_consecutive_no_progress=c.max_consecutive_no_progress,
                privacy_requirement=c.privacy_requirement,
                autonomy_level=c.autonomy_level,
            )
            return self._update_goal_locked(goal)

    def get_goal(self, goal_id: str) -> Optional[Goal]:
        with self._lock:
            return self._get_goal_locked(goal_id)

    def update_goal(self, goal: Goal) -> Goal:
        with self._lock:
            return self._update_goal_locked(goal)

    def create_goal(self, goal: Goal) -> Goal:
        with self._lock:
            serialized = serialize_goal(goal)
            payload = json.dumps(serialized)
            with self._get_connection() as conn:
                try:
                    conn.execute(
                        """
                        INSERT INTO goals (goal_id, original_goal, status, percentage, created_at, updated_at, payload_json)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            goal.goal_id,
                            goal.original_goal,
                            goal.status.value,
                            goal.progress.percentage,
                            goal.created_at,
                            goal.updated_at,
                            payload,
                        ),
                    )
                    conn.commit()
                except sqlite3.IntegrityError:
                    raise ValueError(f"Goal with ID '{goal.goal_id}' already exists.")
            return copy.deepcopy(goal)


    def list_goals(
        self,
        status: Optional[GoalStatus] = None,
        limit: int = 50,
    ) -> List[Goal]:
        with self._lock:
            with self._get_connection() as conn:
                if status is not None:
                    cursor = conn.execute(
                        "SELECT payload_json FROM goals WHERE status = ? ORDER BY updated_at DESC LIMIT ?",
                        (status.value, limit),
                    )
                else:
                    cursor = conn.execute(
                        "SELECT payload_json FROM goals ORDER BY updated_at DESC LIMIT ?",
                        (limit,),
                    )
                rows = cursor.fetchall()
                return [deserialize_goal(json.loads(r["payload_json"])) for r in rows]

    def delete_goal(self, goal_id: str) -> bool:
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.execute("DELETE FROM goals WHERE goal_id = ?", (goal_id,))
                conn.commit()
                return cursor.rowcount > 0

    def get_all_goals(self) -> List[Goal]:
        """Return all stored goals."""
        return self.list_goals(limit=10000)
