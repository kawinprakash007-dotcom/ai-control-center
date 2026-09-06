import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

from core.interfaces.trace_store_interface import TraceStoreInterface
from core.models.runtime import CognitiveTrace
from runtime.serialization import TraceSerializer

logger = logging.getLogger("atlas.trace_store")


class FileTraceStore(TraceStoreInterface):
    """
    Lightweight, file-based implementation of TraceStoreInterface.
    Persists serialized CognitiveTrace documents as deterministic JSON files.
    """

    def __init__(self, directory: str = "traces"):
        self.directory = os.path.abspath(directory)
        os.makedirs(self.directory, exist_ok=True)

    def _sanitize_turn_id(self, turn_id: str) -> str:
        """Prevent path traversal and sanitize filename characters."""
        clean = re.sub(r"[^a-zA-Z0-9_\-]", "_", str(turn_id))
        return clean or "unknown_turn"

    def _get_trace_path(self, turn_id: str) -> str:
        clean_id = self._sanitize_turn_id(turn_id)
        return os.path.join(self.directory, f"trace_{clean_id}.json")

    def save_trace(self, trace: CognitiveTrace, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Serialize and save a CognitiveTrace to disk."""
        path = self._get_trace_path(trace.turn_id)
        doc = TraceSerializer.serialize_trace(trace, metadata=metadata)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(doc, f, indent=2, sort_keys=True)
            return path
        except Exception as ex:
            logger.error("Failed to save trace %s to %s: %s", trace.turn_id, path, ex)
            raise

    def load_trace(self, turn_id: str) -> Optional[CognitiveTrace]:
        """Load and deserialize a CognitiveTrace by turn_id."""
        path = self._get_trace_path(turn_id)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return TraceSerializer.deserialize_trace(data)
        except Exception as ex:
            logger.error("Failed to load trace %s from %s: %s", turn_id, path, ex)
            return None

    def list_traces(self, limit: int = 50) -> List[Dict[str, Any]]:
        """List summaries of stored traces up to limit."""
        results: List[Dict[str, Any]] = []
        if not os.path.exists(self.directory):
            return results

        files = sorted(os.listdir(self.directory), reverse=True)
        for fname in files:
            if not fname.startswith("trace_") or not fname.endswith(".json"):
                continue
            if len(results) >= limit:
                break
            full_path = os.path.join(self.directory, fname)
            try:
                with open(full_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                results.append({
                    "turn_id": data.get("turn_id"),
                    "session_id": data.get("session_id"),
                    "final_status": data.get("final_status"),
                    "event_count": data.get("event_count", len(data.get("events", []))),
                    "schema_version": data.get("schema_version"),
                    "file_path": full_path,
                })
            except Exception as ex:
                logger.debug("Skipping unreadable trace file %s: %s", full_path, ex)
        return results

    def delete_trace(self, turn_id: str) -> bool:
        """Remove a stored trace file."""
        path = self._get_trace_path(turn_id)
        if os.path.exists(path):
            try:
                os.remove(path)
                return True
            except OSError as ex:
                logger.error("Failed to delete trace file %s: %s", path, ex)
                return False
        return False
