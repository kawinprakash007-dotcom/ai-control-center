import hashlib
import json
from typing import Any, Dict, List, Optional, Tuple, Union

from core.models.runtime import (
    CognitiveEvent,
    CognitiveEventType,
    CognitiveStage,
    CognitiveTrace,
    TurnStatus,
    sanitize_event_metadata,
)

SCHEMA_VERSION = 1
MAX_SERIALIZED_EVENTS = 1000


class TraceSerializationError(ValueError):
    """Raised when trace data is malformed, invalid, or fails schema validation."""
    pass


class UnsupportedSchemaVersionError(TraceSerializationError):
    """Raised when trace schema version is unrecognized or unsupported."""
    pass


def compute_trace_hash(trace_or_data: Union[CognitiveTrace, Dict[str, Any]]) -> str:
    """
    Compute a deterministic SHA-256 hash over the canonical representation of trace events.
    Focuses on semantic integrity: turn_id, event_id, stage, event_type, status, summary, and metadata.
    """
    if isinstance(trace_or_data, CognitiveTrace):
        data = TraceSerializer.serialize_trace(trace_or_data)
    elif isinstance(trace_or_data, dict):
        data = trace_or_data
    else:
        raise TraceSerializationError(f"Expected CognitiveTrace or dict, got {type(trace_or_data)}")

    # Extract semantic event properties in order
    canonical_items = []
    events_data = data.get("events", [])
    for ev in events_data:
        canonical_items.append({
            "event_id": str(ev.get("event_id", "")),
            "stage": str(ev.get("stage", "")),
            "event_type": str(ev.get("event_type", "")),
            "status": str(ev.get("status", "")),
            "summary": str(ev.get("summary", "")),
            "metadata": ev.get("metadata", {}),
            "parent_event_id": ev.get("parent_event_id"),
        })

    payload = {
        "turn_id": str(data.get("turn_id", "")),
        "session_id": str(data.get("session_id", "")),
        "events": canonical_items,
    }

    canonical_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


class TraceSerializer:
    """
    Deterministic serializer and deserializer for CognitiveEvent and CognitiveTrace.
    Ensures safe JSON-compatible document export and import without arbitrary code execution.
    """

    @classmethod
    def serialize_event(cls, event: CognitiveEvent) -> Dict[str, Any]:
        """Convert a CognitiveEvent to a JSON-safe dictionary."""
        if not isinstance(event, CognitiveEvent):
            raise TraceSerializationError(f"Expected CognitiveEvent, got {type(event)}")

        # Ensure metadata is fully sanitized before export
        sanitized_meta = sanitize_event_metadata(event.metadata)

        return {
            "event_id": event.event_id,
            "turn_id": event.turn_id,
            "session_id": event.session_id,
            "stage": event.stage.value,
            "event_type": event.event_type.value,
            "timestamp": event.timestamp,
            "duration": event.duration,
            "status": event.status,
            "component": event.component,
            "summary": event.summary,
            "metadata": sanitized_meta,
            "parent_event_id": event.parent_event_id,
        }

    @classmethod
    def deserialize_event(cls, data: Dict[str, Any]) -> CognitiveEvent:
        """Parse and validate a dictionary into a CognitiveEvent."""
        if not isinstance(data, dict):
            raise TraceSerializationError(f"Event data must be a dictionary, got {type(data)}")

        required_fields = ("event_id", "turn_id", "session_id", "stage", "event_type", "timestamp")
        missing = [f for f in required_fields if f not in data]
        if missing:
            raise TraceSerializationError(f"Malformed event, missing required fields: {missing}")

        try:
            stage = CognitiveStage(data["stage"])
        except ValueError as ex:
            raise TraceSerializationError(f"Unknown CognitiveStage '{data['stage']}': {ex}")

        try:
            event_type = CognitiveEventType(data["event_type"])
        except ValueError as ex:
            raise TraceSerializationError(f"Unknown CognitiveEventType '{data['event_type']}': {ex}")

        try:
            timestamp = float(data["timestamp"])
        except (ValueError, TypeError) as ex:
            raise TraceSerializationError(f"Invalid event timestamp: {ex}")

        duration = float(data.get("duration", 0.0))
        status = str(data.get("status", "OK"))
        component = str(data.get("component", "runtime"))
        summary = str(data.get("summary", ""))
        metadata = data.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        parent_event_id = data.get("parent_event_id")

        return CognitiveEvent(
            event_id=str(data["event_id"]),
            turn_id=str(data["turn_id"]),
            session_id=str(data["session_id"]),
            stage=stage,
            event_type=event_type,
            timestamp=timestamp,
            duration=duration,
            status=status,
            component=component,
            summary=summary,
            metadata=metadata,
            parent_event_id=str(parent_event_id) if parent_event_id is not None else None,
        )

    @classmethod
    def serialize_trace(
        cls,
        trace: CognitiveTrace,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Convert a CognitiveTrace to a versioned, portable, JSON-safe document."""
        if not isinstance(trace, CognitiveTrace):
            raise TraceSerializationError(f"Expected CognitiveTrace, got {type(trace)}")

        events_data = [cls.serialize_event(ev) for ev in trace.events]

        doc = {
            "schema_version": SCHEMA_VERSION,
            "turn_id": trace.turn_id,
            "session_id": trace.session_id,
            "start_time": trace.start_time,
            "end_time": trace.end_time,
            "final_status": trace.final_status.value,
            "max_events": trace.max_events,
            "event_count": len(events_data),
            "events": events_data,
            "metadata": metadata or {},
        }
        # Include computed integrity hash
        doc["integrity_hash"] = compute_trace_hash(doc)
        return doc

    @classmethod
    def deserialize_trace(
        cls,
        data: Dict[str, Any],
        max_events_bound: int = MAX_SERIALIZED_EVENTS,
    ) -> CognitiveTrace:
        """Parse and validate a serialized trace document into a CognitiveTrace."""
        if not isinstance(data, dict):
            raise TraceSerializationError(f"Trace data must be a dictionary, got {type(data)}")

        schema_version = data.get("schema_version")
        if schema_version is None:
            raise TraceSerializationError("Missing required 'schema_version' in trace document")
        if schema_version != SCHEMA_VERSION:
            raise UnsupportedSchemaVersionError(
                f"Unsupported trace schema version {schema_version}. Supported: {SCHEMA_VERSION}"
            )

        required_fields = ("turn_id", "session_id", "events")
        missing = [f for f in required_fields if f not in data]
        if missing:
            raise TraceSerializationError(f"Missing required fields in trace document: {missing}")

        raw_events = data.get("events")
        if not isinstance(raw_events, list):
            raise TraceSerializationError(f"'events' must be a list, got {type(raw_events)}")

        if len(raw_events) > max_events_bound:
            raise TraceSerializationError(
                f"Trace exceeds maximum event bound ({len(raw_events)} > {max_events_bound})"
            )

        events: List[CognitiveEvent] = []
        for idx, ev_data in enumerate(raw_events):
            try:
                events.append(cls.deserialize_event(ev_data))
            except Exception as ex:
                raise TraceSerializationError(f"Error parsing event at index {idx}: {ex}")

        try:
            final_status = TurnStatus(data.get("final_status", TurnStatus.RUNNING.value))
        except ValueError:
            final_status = TurnStatus.RUNNING

        start_time = float(data.get("start_time", 0.0))
        end_time = float(data["end_time"]) if data.get("end_time") is not None else None
        max_events = int(data.get("max_events", 100))

        return CognitiveTrace(
            turn_id=str(data["turn_id"]),
            session_id=str(data["session_id"]),
            events=tuple(events),
            start_time=start_time,
            end_time=end_time,
            final_status=final_status,
            max_events=max_events,
        )

    @classmethod
    def to_json(cls, trace: CognitiveTrace, indent: Optional[int] = None) -> str:
        """Serialize trace to JSON string."""
        doc = cls.serialize_trace(trace)
        return json.dumps(doc, indent=indent, sort_keys=True)

    @classmethod
    def from_json(cls, json_str: str) -> CognitiveTrace:
        """Parse a JSON string into a CognitiveTrace."""
        try:
            data = json.loads(json_str)
        except Exception as ex:
            raise TraceSerializationError(f"Invalid JSON string: {ex}")
        return cls.deserialize_trace(data)

    # Convenient aliases
    serialize = serialize_trace
    deserialize = deserialize_trace
