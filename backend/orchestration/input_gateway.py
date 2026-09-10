import collections
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import logging
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

from core.interfaces.orchestration_interface import CentralInputGatewayInterface
from core.interfaces.runtime_interface import CognitiveEventSinkInterface
from core.models.orchestration import (
    DeviceIdentity,
    GeoLocation,
    ModalityType,
    MultimodalObservation,
)
from core.models.runtime import CognitiveEvent, CognitiveEventType, CognitiveStage, sanitize_event_metadata

logger = logging.getLogger("atlas.orchestration.gateway")


# ============================================================================
# Enums and Status Codes
# ============================================================================

class IngressStatus(str, Enum):
    """Lifecycle disposition of an ingress message."""
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    DUPLICATE_SUPPRESSED = "DUPLICATE_SUPPRESSED"
    ACCEPTED_DUPLICATE = "ACCEPTED_DUPLICATE"


class IngressRejectionReason(str, Enum):
    """
    Explicit, deterministic rejection categories for incoming messages.
    """
    INVALID_SCHEMA = "INVALID_SCHEMA"
    UNKNOWN_SOURCE = "UNKNOWN_SOURCE"
    INVALID_TIMESTAMP = "INVALID_TIMESTAMP"
    STALE = "STALE"
    FUTURE_TIMESTAMP = "FUTURE_TIMESTAMP"
    INVALID_CONFIDENCE = "INVALID_CONFIDENCE"
    INVALID_LOCATION = "INVALID_LOCATION"
    DUPLICATE = "DUPLICATE"
    RATE_LIMITED = "RATE_LIMITED"
    PAYLOAD_TOO_LARGE = "PAYLOAD_TOO_LARGE"
    UNSUPPORTED_MODALITY = "UNSUPPORTED_MODALITY"
    AUTHENTICATION_FAILED = "AUTHENTICATION_FAILED"
    CAPACITY_EXCEEDED = "CAPACITY_EXCEEDED"


class DuplicatePolicy(str, Enum):
    """Configured duplicate handling strategy."""
    SUPPRESS = "SUPPRESS"
    ACCEPT_WITH_METADATA = "ACCEPT_WITH_METADATA"


class BackpressurePolicy(str, Enum):
    """Configured backpressure strategy when buffer capacity is reached."""
    REJECT = "REJECT"
    DROP_LOWEST_PRIORITY = "DROP_LOWEST_PRIORITY"


# Modality mapping table for edge participants and transports
_MODALITY_MAP: Dict[str, ModalityType] = {
    "text": ModalityType.TEXT,
    "user_text": ModalityType.TEXT,
    "voice_transcript": ModalityType.VOICE_TRANSCRIPT,
    "transcript": ModalityType.VOICE_TRANSCRIPT,
    "glass_voice": ModalityType.VOICE_TRANSCRIPT,
    "audio_event": ModalityType.AUDIO_EVENT,
    "audio": ModalityType.AUDIO_EVENT,
    "image": ModalityType.IMAGE,
    "glass_image": ModalityType.IMAGE,
    "camera_frame": ModalityType.IMAGE,
    "video_frame": ModalityType.VIDEO_FRAME,
    "video": ModalityType.VIDEO_FRAME,
    "gps": ModalityType.GPS,
    "drone_gps": ModalityType.GPS,
    "location": ModalityType.GPS,
    "telemetry": ModalityType.TELEMETRY,
    "rover_telemetry": ModalityType.TELEMETRY,
    "sensor": ModalityType.TELEMETRY,
    "esp32_sensor": ModalityType.TELEMETRY,
    "device_state": ModalityType.DEVICE_STATE,
    "world_state": ModalityType.WORLD_STATE,
    "event": ModalityType.EVENT,
    "user_action": ModalityType.USER_ACTION,
    "visual": ModalityType.IMAGE,
    "spatial": ModalityType.GPS,
    "temporal": ModalityType.TELEMETRY,
}


def normalize_modality(raw_modality: Any) -> ModalityType:
    """
    Deterministically normalize any string, enum, or label into canonical ModalityType.
    Returns ModalityType.UNKNOWN if unrecognized.
    """
    if isinstance(raw_modality, ModalityType):
        return raw_modality
    if not isinstance(raw_modality, str):
        return ModalityType.UNKNOWN

    cleaned = raw_modality.strip().lower()
    if cleaned in _MODALITY_MAP:
        return _MODALITY_MAP[cleaned]

    return ModalityType.from_str(raw_modality)


# ============================================================================
# Transport-Neutral Ingress Envelope
# ============================================================================

@dataclass(frozen=True)
class IngressEnvelope:
    """
    Transport-independent envelope for external messages entering ATLAS Central.
    Separates transport metadata and authentication credentials from observation semantics.
    """
    message_id: str
    source_id: str
    source_type: str
    modality: Union[ModalityType, str]
    timestamp: float
    payload: Any = None
    confidence: float = 1.0
    location: Optional[Union[GeoLocation, Dict[str, Any]]] = None
    device_id: Optional[str] = None
    correlation_id: str = ""
    causation_id: Optional[str] = None
    artifact_reference: Optional[str] = None
    expires_at: Optional[float] = None
    auth_metadata: Optional[Dict[str, Any]] = None
    auth_token: Optional[str] = None
    schema_version: str = "1.0"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_auth: bool = False) -> Dict[str, Any]:
        """
        Deterministic JSON-safe representation.
        By default, authentication credentials are NEVER serialized to prevent leaks.
        """
        data: Dict[str, Any] = {
            "message_id": self.message_id,
            "source_id": self.source_id,
            "source_type": self.source_type,
            "modality": self.modality.value if isinstance(self.modality, ModalityType) else str(self.modality),
            "timestamp": self.timestamp,
            "payload": self.payload,
            "confidence": self.confidence,
            "correlation_id": self.correlation_id,
            "schema_version": self.schema_version,
        }
        if self.location is not None:
            if isinstance(self.location, GeoLocation):
                data["location"] = self.location.to_dict()
            elif isinstance(self.location, dict):
                data["location"] = dict(self.location)
        if self.device_id is not None:
            data["device_id"] = self.device_id
        if self.causation_id is not None:
            data["causation_id"] = self.causation_id
        if self.artifact_reference is not None:
            data["artifact_reference"] = self.artifact_reference
        if self.expires_at is not None:
            data["expires_at"] = self.expires_at
        if self.metadata:
            data["metadata"] = dict(self.metadata)
        if include_auth:
            if self.auth_metadata:
                data["auth_metadata"] = dict(self.auth_metadata)
            if self.auth_token:
                data["auth_token"] = self.auth_token
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IngressEnvelope":
        """Reconstruct IngressEnvelope from a dictionary."""
        loc = None
        if data.get("location") is not None:
            loc_val = data["location"]
            if isinstance(loc_val, GeoLocation):
                loc = loc_val
            elif isinstance(loc_val, dict):
                loc = GeoLocation.from_dict(loc_val)

        return cls(
            message_id=str(data.get("message_id", "")),
            source_id=str(data.get("source_id", "")),
            source_type=str(data.get("source_type", "")),
            modality=data.get("modality", ModalityType.UNKNOWN),
            timestamp=float(data.get("timestamp", 0.0)),
            payload=data.get("payload"),
            confidence=float(data.get("confidence", 1.0)),
            location=loc,
            device_id=data.get("device_id"),
            correlation_id=str(data.get("correlation_id", "")),
            causation_id=data.get("causation_id"),
            artifact_reference=data.get("artifact_reference"),
            expires_at=float(data["expires_at"]) if data.get("expires_at") is not None else None,
            auth_metadata=data.get("auth_metadata"),
            auth_token=data.get("auth_token"),
            schema_version=str(data.get("schema_version", "1.0")),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class IngressResult:
    """
    Deterministic result of an ingress ingestion operation.
    """
    status: IngressStatus
    observation: Optional[MultimodalObservation] = None
    rejection_reason: Optional[IngressRejectionReason] = None
    error_message: str = ""
    message_id: str = ""
    timestamp: float = 0.0

    @property
    def is_accepted(self) -> bool:
        return self.status in (IngressStatus.ACCEPTED, IngressStatus.ACCEPTED_DUPLICATE)


# ============================================================================
# Gateway Configuration and Metrics
# ============================================================================

@dataclass(frozen=True)
class GatewayConfig:
    """
    Deterministic operational constraints for CentralInputGateway.
    """
    supported_schema_versions: Tuple[str, ...] = ("1.0",)
    max_payload_size_bytes: int = 65536  # 64 KB inline payload limit
    clock_skew_tolerance_seconds: float = 5.0
    max_stale_seconds: float = 300.0  # 5 minutes stale threshold
    max_future_seconds: float = 5.0
    max_messages_per_source_per_second: int = 50
    max_batch_size: int = 500
    max_buffer_size: int = 5000
    strict_source_registration: bool = False
    allow_simulation_sources: bool = True
    duplicate_policy: DuplicatePolicy = DuplicatePolicy.SUPPRESS
    duplicate_cache_size: int = 5000
    backpressure_policy: BackpressurePolicy = BackpressurePolicy.REJECT
    rate_limit_window_seconds: float = 1.0


@dataclass
class GatewayMetrics:
    """
    Internal bounded counters and statistics for gateway health and observability.
    """
    received_count: int = 0
    accepted_count: int = 0
    rejected_count: int = 0
    duplicate_count: int = 0
    stale_count: int = 0
    rate_limited_count: int = 0
    bytes_received: int = 0
    active_sources: int = 0
    error_counts: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "received_count": self.received_count,
            "accepted_count": self.accepted_count,
            "rejected_count": self.rejected_count,
            "duplicate_count": self.duplicate_count,
            "stale_count": self.stale_count,
            "rate_limited_count": self.rate_limited_count,
            "bytes_received": self.bytes_received,
            "active_sources": self.active_sources,
            "error_counts": dict(self.error_counts),
        }


# ============================================================================
# Authentication Boundary Interface
# ============================================================================

class AuthenticationValidatorInterface:
    """
    Contract for ingress authentication/authorization validation.
    CRITICAL: Keeps authentication credentials decoupled from observation semantics.
    """
    def validate_auth(self, envelope: IngressEnvelope) -> bool:
        """Evaluate whether envelope credentials/tokens are valid."""
        raise NotImplementedError


class AllowAllAuthValidator(AuthenticationValidatorInterface):
    """Default permissive validator for open / local testing."""
    def validate_auth(self, envelope: IngressEnvelope) -> bool:
        return True


class TokenAuthValidator(AuthenticationValidatorInterface):
    """Deterministic token matching validator."""
    def __init__(self, valid_tokens: Sequence[str]):
        self._valid_tokens: Set[str] = set(valid_tokens)

    def validate_auth(self, envelope: IngressEnvelope) -> bool:
        token = envelope.auth_token
        if not token and envelope.auth_metadata:
            token = envelope.auth_metadata.get("token") or envelope.auth_metadata.get("auth_token")
        if not token:
            return False
        return token in self._valid_tokens


# ============================================================================
# Central Input Gateway Implementation
# ============================================================================

class CentralInputGateway(CentralInputGatewayInterface):
    """
    Deterministic Ingress Gateway for the ATLAS Central Orchestration Layer.

    Responsibilities:
    - Ingress boundary for external messages (Glass, Drone, Rover, sensors, simulation).
    - Authentication and source identity verification.
    - Schema validation and explicit rejection.
    - Message normalization into canonical MultimodalObservation objects.
    - Freshness, clock skew, and timestamp validation.
    - Bounded rate limiting per source.
    - Duplicate detection and suppression.
    - Bounded backpressure management.
    - Preserves correlation and causation identifiers.
    - Routes validated observations strictly to SituationFusionEngine (if configured).

    CRITICAL ARCHITECTURAL BOUNDARIES:
    - Zero tool execution or ToolOrchestrator calls.
    - Zero DeviceGateway or DeviceAdapter invocations.
    - Zero LLM, VLM, ReasoningEngine, or ModelRouter invocations.
    - Zero Goal creation or GoalStore mutations.
    - Zero WorldState mutations.
    - Zero hardware, socket, or network library imports.
    - Zero background threads or uncontrolled daemon processes.
    """

    def __init__(
        self,
        config: Optional[GatewayConfig] = None,
        auth_validator: Optional[AuthenticationValidatorInterface] = None,
        fusion_engine: Optional[Any] = None,
        event_sink: Optional[CognitiveEventSinkInterface] = None,
        clock: Optional[Callable[[], float]] = None,
    ):
        self.config = config or GatewayConfig()
        self.auth_validator = auth_validator or AllowAllAuthValidator()
        self.fusion_engine = fusion_engine
        self.event_sink = event_sink
        self.clock = clock or time.time

        self._lock = threading.Lock()
        self._registered_devices: Dict[str, DeviceIdentity] = {}
        self._seen_message_ids: collections.deque = collections.deque(
            maxlen=self.config.duplicate_cache_size
        )
        self._seen_message_id_set: Set[str] = set()
        self._seen_signatures: collections.deque = collections.deque(
            maxlen=self.config.duplicate_cache_size
        )
        self._seen_signature_set: Set[str] = set()

        # Sliding window timestamps per source_id: deque of float timestamps
        self._source_rate_windows: Dict[str, collections.deque] = {}

        # Bounded recent observation store
        self._recent_observations: collections.deque = collections.deque(
            maxlen=self.config.max_buffer_size
        )

        self._metrics = GatewayMetrics()

    # ========================================================================
    # Registration API
    # ========================================================================

    def register_device(self, device: DeviceIdentity) -> None:
        """Register a known edge participant identity."""
        with self._lock:
            self._registered_devices[device.device_id] = device

    def unregister_device(self, device_id: str) -> bool:
        """Unregister an edge participant."""
        with self._lock:
            return self._registered_devices.pop(device_id, None) is not None

    def is_source_registered(self, source_id: str) -> bool:
        """Check whether source_id is a registered device."""
        with self._lock:
            return source_id in self._registered_devices

    def _is_simulation_source(self, envelope: IngressEnvelope) -> bool:
        """Check whether the envelope originates from a recognized simulation source."""
        s_type = (envelope.source_type or "").upper()
        s_id = (envelope.source_id or "").upper()
        if "SIM" in s_type or "SIM" in s_id or "VIRTUAL" in s_type or "VIRTUAL" in s_id:
            return True
        with self._lock:
            dev = self._registered_devices.get(envelope.source_id)
            if dev and dev.is_simulation:
                return True
        return False

    # ========================================================================
    # Ingress Processing Pipeline
    # ========================================================================

    def receive_message(
        self,
        data: Dict[str, Any],
        now: Optional[float] = None,
    ) -> IngressResult:
        """
        Ingest an external message serialized as a dictionary.
        """
        try:
            envelope = IngressEnvelope.from_dict(data)
        except Exception as e:
            current_time = now if now is not None else self.clock()
            with self._lock:
                self._record_rejection(IngressRejectionReason.INVALID_SCHEMA, 0)
            return IngressResult(
                status=IngressStatus.REJECTED,
                rejection_reason=IngressRejectionReason.INVALID_SCHEMA,
                error_message=f"Failed to parse IngressEnvelope: {e}",
                timestamp=current_time,
            )
        return self.receive_envelope(envelope, now=now)

    def receive_envelope(
        self,
        envelope: IngressEnvelope,
        now: Optional[float] = None,
    ) -> IngressResult:
        """
        Primary ingress pipeline.
        Performs validation, normalization, rate limiting, duplicate check, and routing.
        """
        current_time = now if now is not None else self.clock()
        payload_bytes = self._estimate_payload_size(envelope.payload)

        with self._lock:
            self._metrics.received_count += 1
            self._metrics.bytes_received += payload_bytes

        # 1. Schema version validation
        if envelope.schema_version not in self.config.supported_schema_versions:
            return self._reject(
                envelope=envelope,
                reason=IngressRejectionReason.INVALID_SCHEMA,
                message=f"Unsupported schema version: '{envelope.schema_version}'",
                now=current_time,
            )

        # 2. Basic identifier validation
        if not envelope.message_id or not isinstance(envelope.message_id, str):
            return self._reject(
                envelope=envelope,
                reason=IngressRejectionReason.INVALID_SCHEMA,
                message="Message ID must be a non-empty string",
                now=current_time,
            )
        if not envelope.source_id or not isinstance(envelope.source_id, str):
            return self._reject(
                envelope=envelope,
                reason=IngressRejectionReason.INVALID_SCHEMA,
                message="Source ID must be a non-empty string",
                now=current_time,
            )
        if not envelope.source_type or not isinstance(envelope.source_type, str):
            return self._reject(
                envelope=envelope,
                reason=IngressRejectionReason.INVALID_SCHEMA,
                message="Source type must be a non-empty string",
                now=current_time,
            )

        # 3. Authentication validation
        try:
            if not self.auth_validator.validate_auth(envelope):
                return self._reject(
                    envelope=envelope,
                    reason=IngressRejectionReason.AUTHENTICATION_FAILED,
                    message="Ingress authentication failed",
                    now=current_time,
                )
        except Exception as e:
            return self._reject(
                envelope=envelope,
                reason=IngressRejectionReason.AUTHENTICATION_FAILED,
                message=f"Ingress authentication error: {e}",
                now=current_time,
            )

        # 4. Source registration validation (strict mode)
        if self.config.strict_source_registration:
            is_sim = self.config.allow_simulation_sources and self._is_simulation_source(envelope)
            if not is_sim and not self.is_source_registered(envelope.source_id):
                return self._reject(
                    envelope=envelope,
                    reason=IngressRejectionReason.UNKNOWN_SOURCE,
                    message=f"Unregistered source '{envelope.source_id}' rejected under strict registration policy",
                    now=current_time,
                )

        # 5. Modality normalization and validation
        modality = normalize_modality(envelope.modality)
        if modality == ModalityType.UNKNOWN:
            return self._reject(
                envelope=envelope,
                reason=IngressRejectionReason.UNSUPPORTED_MODALITY,
                message=f"Unsupported or unrecognized modality: '{envelope.modality}'",
                now=current_time,
            )

        # 6. Confidence validation
        try:
            conf = float(envelope.confidence)
            if not (0.0 <= conf <= 1.0) or (conf != conf):  # NaN check
                raise ValueError("Out of range")
        except Exception:
            return self._reject(
                envelope=envelope,
                reason=IngressRejectionReason.INVALID_CONFIDENCE,
                message=f"Confidence must be a float in [0.0, 1.0], got '{envelope.confidence}'",
                now=current_time,
            )

        # 7. GeoLocation validation
        norm_location: Optional[GeoLocation] = None
        if envelope.location is not None:
            try:
                if isinstance(envelope.location, GeoLocation):
                    norm_location = envelope.location
                elif isinstance(envelope.location, dict):
                    norm_location = GeoLocation.from_dict(envelope.location)
                else:
                    raise ValueError("Location must be GeoLocation or dict")
            except Exception as e:
                return self._reject(
                    envelope=envelope,
                    reason=IngressRejectionReason.INVALID_LOCATION,
                    message=f"Invalid location coordinate: {e}",
                    now=current_time,
                )

        # 8. Timestamp freshness & clock skew validation
        ts = envelope.timestamp
        try:
            ts = float(ts)
            if ts <= 0.0 or (ts != ts):
                raise ValueError("Timestamp non-positive or NaN")
        except Exception:
            return self._reject(
                envelope=envelope,
                reason=IngressRejectionReason.INVALID_TIMESTAMP,
                message=f"Invalid timestamp: '{envelope.timestamp}'",
                now=current_time,
            )

        # Future timestamp check (allowing clock skew)
        if ts > current_time + self.config.clock_skew_tolerance_seconds:
            return self._reject(
                envelope=envelope,
                reason=IngressRejectionReason.FUTURE_TIMESTAMP,
                message=f"Timestamp {ts} is in the future beyond tolerance (now={current_time}, skew={self.config.clock_skew_tolerance_seconds})",
                now=current_time,
            )

        # Stale timestamp check
        age = current_time - ts
        if age > self.config.max_stale_seconds:
            with self._lock:
                self._metrics.stale_count += 1
            return self._reject(
                envelope=envelope,
                reason=IngressRejectionReason.STALE,
                message=f"Message is stale by {age:.1f}s (max_stale={self.config.max_stale_seconds}s)",
                now=current_time,
            )

        # 9. Payload size check
        if payload_bytes > self.config.max_payload_size_bytes:
            return self._reject(
                envelope=envelope,
                reason=IngressRejectionReason.PAYLOAD_TOO_LARGE,
                message=f"Payload size {payload_bytes} bytes exceeds maximum allowed {self.config.max_payload_size_bytes} bytes. Use artifact_reference.",
                now=current_time,
            )

        # 10. Rate limiting (per source_id)
        with self._lock:
            if not self._check_and_update_rate_limit(envelope.source_id, current_time):
                self._metrics.rate_limited_count += 1
                return self._reject_unlocked(
                    envelope=envelope,
                    reason=IngressRejectionReason.RATE_LIMITED,
                    message=f"Source '{envelope.source_id}' exceeded rate limit of {self.config.max_messages_per_source_per_second} msg/sec",
                    now=current_time,
                )

        # 11. Duplicate detection
        canonical_sig = self._calculate_signature(envelope, modality)
        is_duplicate = False
        with self._lock:
            if envelope.message_id in self._seen_message_id_set or canonical_sig in self._seen_signature_set:
                is_duplicate = True
                self._metrics.duplicate_count += 1

        if is_duplicate:
            if self.config.duplicate_policy == DuplicatePolicy.SUPPRESS:
                self._emit_event(
                    event_type=CognitiveEventType.INGRESS_DUPLICATE_SUPPRESSED,
                    envelope=envelope,
                    metadata={"reason": "DUPLICATE_SUPPRESSED"},
                    now=current_time,
                )
                return IngressResult(
                    status=IngressStatus.DUPLICATE_SUPPRESSED,
                    rejection_reason=IngressRejectionReason.DUPLICATE,
                    error_message=f"Duplicate message '{envelope.message_id}' suppressed",
                    message_id=envelope.message_id,
                    timestamp=current_time,
                )

        # 12. Backpressure and capacity management
        with self._lock:
            if len(self._recent_observations) >= self.config.max_buffer_size:
                if self.config.backpressure_policy == BackpressurePolicy.REJECT:
                    return self._reject_unlocked(
                        envelope=envelope,
                        reason=IngressRejectionReason.CAPACITY_EXCEEDED,
                        message=f"Gateway buffer capacity ({self.config.max_buffer_size}) exceeded",
                        now=current_time,
                    )
                elif self.config.backpressure_policy == BackpressurePolicy.DROP_LOWEST_PRIORITY:
                    self._drop_lowest_priority_unlocked()

        # 13. Normalization into canonical MultimodalObservation
        # Clean metadata: NEVER include auth_token or auth_metadata
        obs_metadata = dict(envelope.metadata)
        if is_duplicate:
            obs_metadata["is_duplicate"] = True

        correlation_id = envelope.correlation_id if envelope.correlation_id else envelope.message_id
        device_id = envelope.device_id if envelope.device_id else envelope.source_id

        observation = MultimodalObservation(
            observation_id=envelope.message_id,
            source_id=envelope.source_id,
            source_type=envelope.source_type,
            modality=modality,
            timestamp=ts,  # Preserve original source timestamp!
            payload=envelope.payload,
            confidence=conf,
            location=norm_location,
            device_id=device_id,
            correlation_id=correlation_id,
            causation_id=envelope.causation_id,
            artifact_reference=envelope.artifact_reference,
            expires_at=envelope.expires_at,
            metadata=obs_metadata,
        )

        # 14. Commit to state and metrics
        with self._lock:
            self._remember_seen_unlocked(envelope.message_id, canonical_sig)
            self._recent_observations.append(observation)
            self._metrics.accepted_count += 1
            self._metrics.active_sources = len(self._source_rate_windows)

        # 15. Forward strictly to SituationFusionEngine if present
        if self.fusion_engine is not None:
            try:
                self.fusion_engine.ingest(observation, now=current_time)
            except Exception as e:
                logger.error(f"Failed to forward observation to SituationFusionEngine: {e}")

        # 16. Observability
        self._emit_event(
            event_type=CognitiveEventType.INGRESS_OBSERVATION_ACCEPTED,
            envelope=envelope,
            metadata={
                "observation_id": observation.observation_id,
                "modality": modality.value,
                "is_duplicate": is_duplicate,
            },
            now=current_time,
        )

        final_status = IngressStatus.ACCEPTED_DUPLICATE if is_duplicate else IngressStatus.ACCEPTED
        return IngressResult(
            status=final_status,
            observation=observation,
            message_id=envelope.message_id,
            timestamp=current_time,
        )

    # ========================================================================
    # Batch Ingestion
    # ========================================================================

    def ingest_batch(
        self,
        envelopes: Sequence[Union[IngressEnvelope, Dict[str, Any], MultimodalObservation]],
        now: Optional[float] = None,
    ) -> Sequence[IngressResult]:
        """
        Bounded batch ingestion.
        Processes each message deterministically. Partial failure does not crash the batch.
        """
        current_time = now if now is not None else self.clock()
        if not envelopes:
            return ()

        # Bound batch size
        bounded_batch = envelopes[: self.config.max_batch_size]
        results: List[IngressResult] = []

        for item in bounded_batch:
            if isinstance(item, IngressEnvelope):
                res = self.receive_envelope(item, now=current_time)
            elif isinstance(item, dict):
                res = self.receive_message(item, now=current_time)
            elif isinstance(item, MultimodalObservation):
                # Convert canonical observation into an envelope to pass full gateway ingress checks
                env = IngressEnvelope(
                    message_id=item.observation_id,
                    source_id=item.source_id,
                    source_type=item.source_type,
                    modality=item.modality,
                    timestamp=item.timestamp,
                    payload=item.payload,
                    confidence=item.confidence,
                    location=item.location,
                    device_id=item.device_id,
                    correlation_id=item.correlation_id,
                    causation_id=item.causation_id,
                    artifact_reference=item.artifact_reference,
                    expires_at=item.expires_at,
                    metadata=item.metadata,
                )
                res = self.receive_envelope(env, now=current_time)
            else:
                res = IngressResult(
                    status=IngressStatus.REJECTED,
                    rejection_reason=IngressRejectionReason.INVALID_SCHEMA,
                    error_message=f"Unsupported batch item type: {type(item)}",
                    timestamp=current_time,
                )
                with self._lock:
                    self._record_rejection(IngressRejectionReason.INVALID_SCHEMA, 0)
            results.append(res)

        return tuple(results)

    # ========================================================================
    # CentralInputGatewayInterface Implementation
    # ========================================================================

    def ingest_observation(self, observation: MultimodalObservation, now: Optional[float] = None) -> None:
        """
        Direct ingestion of a MultimodalObservation complying with CentralInputGatewayInterface.
        """
        env = IngressEnvelope(
            message_id=observation.observation_id,
            source_id=observation.source_id,
            source_type=observation.source_type,
            modality=observation.modality,
            timestamp=observation.timestamp,
            payload=observation.payload,
            confidence=observation.confidence,
            location=observation.location,
            device_id=observation.device_id,
            correlation_id=observation.correlation_id,
            causation_id=observation.causation_id,
            artifact_reference=observation.artifact_reference,
            expires_at=observation.expires_at,
            metadata=observation.metadata,
        )
        res = self.receive_envelope(env, now=now)
        if not res.is_accepted:
            raise ValueError(f"Observation rejected by gateway: {res.rejection_reason} - {res.error_message}")

    def get_recent_observations(
        self,
        modality: Optional[ModalityType] = None,
        limit: int = 100,
    ) -> Sequence[MultimodalObservation]:
        """Retrieve recent observations filtered by modality up to bounded limit."""
        with self._lock:
            items = list(self._recent_observations)

        if modality is not None:
            norm_mod = normalize_modality(modality)
            items = [obs for obs in items if obs.modality == norm_mod]

        limit = max(0, min(limit, self.config.max_buffer_size))
        return tuple(items[-limit:])

    # ========================================================================
    # Metrics and Operational Status
    # ========================================================================

    def get_metrics(self) -> GatewayMetrics:
        """Expose a snapshot copy of internal counters."""
        with self._lock:
            return GatewayMetrics(
                received_count=self._metrics.received_count,
                accepted_count=self._metrics.accepted_count,
                rejected_count=self._metrics.rejected_count,
                duplicate_count=self._metrics.duplicate_count,
                stale_count=self._metrics.stale_count,
                rate_limited_count=self._metrics.rate_limited_count,
                bytes_received=self._metrics.bytes_received,
                active_sources=len(self._source_rate_windows),
                error_counts=dict(self._metrics.error_counts),
            )

    def reset_metrics(self) -> None:
        """Reset internal metrics counters."""
        with self._lock:
            self._metrics = GatewayMetrics()

    # ========================================================================
    # Internal Helpers (Rejection, Rate Limiting, Observability)
    # ========================================================================

    def _estimate_payload_size(self, payload: Any) -> int:
        """Estimate memory/byte size of inline payload."""
        if payload is None:
            return 0
        if isinstance(payload, bytes):
            return len(payload)
        if isinstance(payload, str):
            return len(payload.encode("utf-8", errors="replace"))
        if isinstance(payload, (int, float, bool)):
            return 8
        try:
            return len(json.dumps(payload, default=str).encode("utf-8"))
        except Exception:
            return 64

    def _calculate_signature(self, envelope: IngressEnvelope, modality: ModalityType) -> str:
        """Deterministic canonical signature for secondary duplicate detection."""
        hasher = hashlib.sha256()
        hasher.update(envelope.source_id.encode("utf-8"))
        hasher.update(modality.value.encode("utf-8"))
        hasher.update(f"{envelope.timestamp:.3f}".encode("utf-8"))
        if envelope.artifact_reference:
            hasher.update(envelope.artifact_reference.encode("utf-8"))
        elif envelope.payload is not None:
            if isinstance(envelope.payload, (bytes, str)):
                p_bytes = (
                    envelope.payload
                    if isinstance(envelope.payload, bytes)
                    else envelope.payload.encode("utf-8", errors="replace")
                )
                hasher.update(hashlib.md5(p_bytes).digest())
            else:
                hasher.update(str(envelope.payload).encode("utf-8"))
        return hasher.hexdigest()

    def _check_and_update_rate_limit(self, source_id: str, current_time: float) -> bool:
        """
        Deterministic sliding-window rate check per source_id.
        Caller MUST hold self._lock.
        """
        window = self._source_rate_windows.setdefault(source_id, collections.deque())
        cutoff = current_time - self.config.rate_limit_window_seconds

        # Evict timestamps older than sliding window
        while window and window[0] <= cutoff:
            window.popleft()

        if len(window) >= self.config.max_messages_per_source_per_second:
            return False

        window.append(current_time)
        return True

    def _remember_seen_unlocked(self, message_id: str, signature: str) -> None:
        """
        Record message_id and signature into bounded LRU caches.
        Caller MUST hold self._lock.
        """
        if len(self._seen_message_ids) >= self.config.duplicate_cache_size:
            oldest_id = self._seen_message_ids.popleft()
            self._seen_message_id_set.discard(oldest_id)
        self._seen_message_ids.append(message_id)
        self._seen_message_id_set.add(message_id)

        if len(self._seen_signatures) >= self.config.duplicate_cache_size:
            oldest_sig = self._seen_signatures.popleft()
            self._seen_signature_set.discard(oldest_sig)
        self._seen_signatures.append(signature)
        self._seen_signature_set.add(signature)

    def _drop_lowest_priority_unlocked(self) -> None:
        """
        Backpressure: drop telemetry/sensor or oldest item to keep capacity bounded.
        Caller MUST hold self._lock.
        """
        if not self._recent_observations:
            return

        # Prefer dropping TELEMETRY, DEVICE_STATE, or oldest
        for i, obs in enumerate(self._recent_observations):
            if obs.modality in (ModalityType.TELEMETRY, ModalityType.DEVICE_STATE):
                del self._recent_observations[i]
                return

        # Otherwise pop oldest
        self._recent_observations.popleft()

    def _reject(
        self,
        envelope: IngressEnvelope,
        reason: IngressRejectionReason,
        message: str,
        now: float,
    ) -> IngressResult:
        """Reject message, update counters, and emit telemetry event."""
        with self._lock:
            return self._reject_unlocked(envelope, reason, message, now)

    def _reject_unlocked(
        self,
        envelope: IngressEnvelope,
        reason: IngressRejectionReason,
        message: str,
        now: float,
    ) -> IngressResult:
        """Reject message under lock."""
        self._record_rejection(reason, self._estimate_payload_size(envelope.payload))
        self._emit_event(
            event_type=CognitiveEventType.INGRESS_OBSERVATION_REJECTED,
            envelope=envelope,
            metadata={
                "rejection_reason": reason.value,
                "error_message": message,
            },
            now=now,
        )
        return IngressResult(
            status=IngressStatus.REJECTED,
            rejection_reason=reason,
            error_message=message,
            message_id=envelope.message_id,
            timestamp=now,
        )

    def _record_rejection(self, reason: IngressRejectionReason, payload_bytes: int) -> None:
        """Update metrics counters for rejection."""
        self._metrics.rejected_count += 1
        r_name = reason.value
        self._metrics.error_counts[r_name] = self._metrics.error_counts.get(r_name, 0) + 1

    def _emit_event(
        self,
        event_type: CognitiveEventType,
        envelope: IngressEnvelope,
        metadata: Dict[str, Any],
        now: float,
    ) -> None:
        """
        Emit a safe CognitiveEvent without leaking auth credentials or private tokens.
        """
        if self.event_sink is None:
            return

        # Sanitize event metadata: NEVER include raw credentials
        safe_meta = {
            "message_id": envelope.message_id,
            "source_id": envelope.source_id,
            "source_type": envelope.source_type,
            "correlation_id": envelope.correlation_id or envelope.message_id,
            **metadata,
        }
        sanitized = sanitize_event_metadata(safe_meta)

        try:
            event_id = f"evt_{envelope.message_id}_{int(now * 1000)}"
            turn_id = envelope.correlation_id or envelope.message_id
            session_id = str(envelope.metadata.get("session_id", "ingress_gateway"))
            event = CognitiveEvent(
                event_id=event_id,
                turn_id=turn_id,
                session_id=session_id,
                stage=CognitiveStage.OBSERVATION,
                event_type=event_type,
                timestamp=now,
                metadata=sanitized,
            )
            if hasattr(self.event_sink, "publish"):
                self.event_sink.publish(event)
            elif hasattr(self.event_sink, "record_event"):
                self.event_sink.record_event(event)
        except Exception as e:
            logger.debug(f"Failed to record cognitive event: {e}")
