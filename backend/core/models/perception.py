"""
ATLAS Phase 6.5a & Phase 4.4 — Multimodal Perception Domain Models.

Establishes transport-neutral, immutable perception contracts across:
- Edge products: ATLAS Vision, ATLAS Glass, ATLAS Drone, ATLAS Rover
- Perception providers: Mock, OCR, CV, Audio, Speech, Spatial, Telemetry
- Canonical multimodal observation normalization boundary

CRITICAL ARCHITECTURAL RULES:
1. Model-neutral & hardware-neutral: ZERO PyTorch, YOLO, Whisper, OpenCV, ROS, or GPIO specifics.
2. Perception does NOT reason, plan missions, create goals, mutate WorldState, or execute tools.
3. Models are frozen/immutable dataclasses with deterministic serialization and validation.
4. Confidence is strictly bounded [0.0, 1.0] and rejects NaN/Infinity.
5. All accepted evidence MUST carry complete provenance.
6. Credentials and secrets are NEVER serialized or leaked in error strings.
"""

from dataclasses import dataclass, field
from enum import Enum
import math
import time
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Set, Tuple, Union

from core.models.computer import ScreenDimensions
from core.models.device_contract import ProductType, sanitize_contract_metadata
from core.models.orchestration import GeoLocation, ModalityType, MultimodalObservation


# ============================================================================
# 1. Enums
# ============================================================================

class PerceptionStatus(str, Enum):
    """Execution status of a perception request."""
    SUCCESS = "SUCCESS"
    PARTIAL = "PARTIAL"
    NO_DETECTION = "NO_DETECTION"
    INVALID_INPUT = "INVALID_INPUT"
    UNSUPPORTED = "UNSUPPORTED"
    UNSUPPORTED_MODALITY = "UNSUPPORTED_MODALITY"
    TIMEOUT = "TIMEOUT"
    FAILED = "FAILED"
    UNKNOWN = "UNKNOWN"

    @classmethod
    def from_str(cls, val: Any) -> "PerceptionStatus":
        if isinstance(val, cls):
            return val
        norm = str(val or "").strip().upper()
        for member in cls:
            if member.value == norm or member.name == norm:
                return member
        return cls.UNKNOWN


class PerceptionSourceType(str, Enum):
    """Source classification of the perception input or provider."""
    DEVICE = "DEVICE"
    SIMULATION = "SIMULATION"
    LOCAL_MODEL = "LOCAL_MODEL"
    REMOTE_MODEL = "REMOTE_MODEL"
    RULE_BASED = "RULE_BASED"
    HUMAN = "HUMAN"
    SYSTEM = "SYSTEM"
    SENSOR = "SENSOR"
    UNKNOWN = "UNKNOWN"

    @classmethod
    def from_str(cls, val: Any) -> "PerceptionSourceType":
        if isinstance(val, cls):
            return val
        norm = str(val or "").strip().upper()
        for member in cls:
            if member.value == norm or member.name == norm:
                return member
        return cls.UNKNOWN


class PerceptionPrivacyClass(str, Enum):
    """Privacy classification governing perception evidence handling."""
    PUBLIC = "PUBLIC"
    INTERNAL = "INTERNAL"
    CONFIDENTIAL = "CONFIDENTIAL"
    SENSITIVE = "SENSITIVE"
    RESTRICTED = "RESTRICTED"

    @classmethod
    def from_str(cls, val: Any) -> "PerceptionPrivacyClass":
        if isinstance(val, cls):
            return val
        norm = str(val or "").strip().upper()
        for member in cls:
            if member.value == norm or member.name == norm:
                return member
        return cls.CONFIDENTIAL


class PerceptionCapability(str, Enum):
    """Standardized perception capability categories."""
    VISION_OBJECT_DETECTION = "VISION_OBJECT_DETECTION"
    VISION_MOTION_DETECTION = "VISION_MOTION_DETECTION"
    VISION_SCENE_CLASSIFICATION = "VISION_SCENE_CLASSIFICATION"
    VISION_OCR = "VISION_OCR"
    AUDIO_EVENT_CLASSIFICATION = "AUDIO_EVENT_CLASSIFICATION"
    SPEECH_TRANSCRIPTION = "SPEECH_TRANSCRIPTION"
    SPATIAL_LOCALIZATION = "SPATIAL_LOCALIZATION"
    TELEMETRY_NORMALIZATION = "TELEMETRY_NORMALIZATION"
    MULTIMODAL_CORRELATION = "MULTIMODAL_CORRELATION"

    @classmethod
    def from_str(cls, val: Any) -> "PerceptionCapability":
        if isinstance(val, cls):
            return val
        norm = str(val or "").strip().upper()
        for member in cls:
            if member.value == norm or member.name == norm:
                return member
        raise ValueError(f"Unknown PerceptionCapability: '{val}'")


# ============================================================================
# 2. Phase 4.4 Visual Desktop / Grounding Models (Preserved for Backward Compatibility)
# ============================================================================

class ElementType(str, Enum):
    """Bounded vocabulary of visual and semantic UI element categories."""
    BUTTON = "button"
    TEXT = "text"
    INPUT = "input"
    LINK = "link"
    CHECKBOX = "checkbox"
    RADIO = "radio"
    IMAGE = "image"
    ICON = "icon"
    MENU = "menu"
    WINDOW = "window"
    PANEL = "panel"
    UNKNOWN = "unknown"


class SpatialRelation(str, Enum):
    """Bounded set of deterministic spatial relationships for scene querying."""
    ABOVE = "above"
    BELOW = "below"
    LEFT_OF = "left_of"
    RIGHT_OF = "right_of"
    NEAR = "near"
    INSIDE = "inside"
    CONTAINS = "contains"


@dataclass(frozen=True)
class BoundingBox:
    """
    Immutable representation of an axis-aligned bounding box.
    Enforces non-negative coordinates and non-negative dimensions.
    """
    x: int
    y: int
    width: int
    height: int

    def __post_init__(self):
        if self.x < 0 or self.y < 0:
            raise ValueError(f"Bounding box origin coordinates cannot be negative: ({self.x}, {self.y})")
        if self.width < 0 or self.height < 0:
            raise ValueError(f"Bounding box dimensions cannot be negative: {self.width}x{self.height}")

    @property
    def center(self) -> Tuple[int, int]:
        """Return the (x, y) center coordinate of the box."""
        return (self.x + self.width // 2, self.y + self.height // 2)

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height

    def contains_point(self, px: int, py: int) -> bool:
        """Check if point (px, py) falls inside or on the boundary of the box."""
        return self.x <= px <= self.right and self.y <= py <= self.bottom

    def contains_box(self, other: "BoundingBox") -> bool:
        """Check if this box completely encloses other box."""
        return (
            self.x <= other.x
            and self.y <= other.y
            and self.right >= other.right
            and self.bottom >= other.bottom
        )

    def intersects(self, other: "BoundingBox") -> bool:
        """Check if this box overlaps with other box."""
        return not (
            self.right <= other.x
            or other.right <= self.x
            or self.bottom <= other.y
            or other.bottom <= self.y
        )

    def intersection_area(self, other: "BoundingBox") -> int:
        inter_x = max(self.x, other.x)
        inter_y = max(self.y, other.y)
        inter_right = min(self.right, other.right)
        inter_bottom = min(self.bottom, other.bottom)
        if inter_right > inter_x and inter_bottom > inter_y:
            return (inter_right - inter_x) * (inter_bottom - inter_y)
        return 0

    def iou(self, other: "BoundingBox") -> float:
        """Intersection over Union between two bounding boxes."""
        inter = self.intersection_area(other)
        union = self.area + other.area - inter
        if union <= 0:
            return 0.0
        return inter / float(union)

    def distance_to(self, other: "BoundingBox") -> float:
        """Euclidean distance between bounding box centers."""
        cx1, cy1 = self.center
        cx2, cy2 = other.center
        return math.hypot(cx1 - cx2, cy1 - cy2)

    def relation_to(self, other: "BoundingBox", max_distance: float = 150.0) -> Optional[SpatialRelation]:
        """
        Determine spatial relationship of self relative to other.
        e.g., if self is below other, returns SpatialRelation.BELOW.
        """
        if self.contains_box(other):
            return SpatialRelation.CONTAINS
        if other.contains_box(self):
            return SpatialRelation.INSIDE

        dist = self.distance_to(other)
        if dist > max_distance:
            return None

        cx1, cy1 = self.center
        cx2, cy2 = other.center
        dx = cx1 - cx2
        dy = cy1 - cy2

        if abs(dy) > abs(dx):
            return SpatialRelation.BELOW if dy > 0 else SpatialRelation.ABOVE
        else:
            return SpatialRelation.RIGHT_OF if dx > 0 else SpatialRelation.LEFT_OF

    def to_dict(self) -> Dict[str, Any]:
        return {"x": self.x, "y": self.y, "width": self.width, "height": self.height}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BoundingBox":
        return cls(
            x=int(data["x"]),
            y=int(data["y"]),
            width=int(data["width"]),
            height=int(data["height"]),
        )


class _LegacyPerceptionSourceMember(str):
    """Helper enabling backward compatibility for Phase 4.4 PerceptionSource enum members."""
    @property
    def value(self) -> str:
        return str(self)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, _LegacyPerceptionSourceMember):
            return str(self) == str(other)
        if isinstance(other, str):
            return str(self) == other
        if hasattr(other, "value"):
            return str(self) == str(other.value)
        return False

    def __hash__(self) -> int:
        return hash(str(self))


@dataclass(frozen=True)
class PerceptionSource:
    """
    Provenance tracking and metadata for multi-product edge and simulation perception sources.
    Maintains full backwards compatibility with Phase 4.4 legacy visual perception constants.
    """
    source_id: str = "UNKNOWN"
    source_type: PerceptionSourceType = PerceptionSourceType.SYSTEM
    product_id: Optional[str] = None
    device_id: Optional[str] = None
    provider_id: str = ""
    provider_version: str = "1.0.0"

    # Class-level legacy compatibility constants for Phase 4.4 visual perception
    OCR: ClassVar[_LegacyPerceptionSourceMember] = _LegacyPerceptionSourceMember("ocr")
    CV: ClassVar[_LegacyPerceptionSourceMember] = _LegacyPerceptionSourceMember("cv")
    ACCESSIBILITY: ClassVar[_LegacyPerceptionSourceMember] = _LegacyPerceptionSourceMember("accessibility")
    MANUAL: ClassVar[_LegacyPerceptionSourceMember] = _LegacyPerceptionSourceMember("manual")
    FUTURE_VISION_MODEL: ClassVar[_LegacyPerceptionSourceMember] = _LegacyPerceptionSourceMember("future_vision_model")

    def __post_init__(self):
        if not self.source_id or not isinstance(self.source_id, str):
            raise ValueError("PerceptionSource source_id must be a non-empty string.")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "source_type": self.source_type.value,
            "product_id": self.product_id,
            "device_id": self.device_id,
            "provider_id": self.provider_id,
            "provider_version": self.provider_version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PerceptionSource":
        return cls(
            source_id=str(data.get("source_id", "UNKNOWN")),
            source_type=PerceptionSourceType.from_str(data.get("source_type", "SYSTEM")),
            product_id=data.get("product_id"),
            device_id=data.get("device_id"),
            provider_id=str(data.get("provider_id", "")),
            provider_version=str(data.get("provider_version", "1.0.0")),
        )


VisualPerceptionSource = PerceptionSource


@dataclass(frozen=True)
class VisualElement:
    """Immutable representation of an individual detected UI element."""
    element_id: str
    element_type: ElementType
    bounding_box: BoundingBox
    text: Optional[str] = None
    confidence: float = 1.0
    source: Union[PerceptionSource, _LegacyPerceptionSourceMember, str] = field(
        default_factory=lambda: PerceptionSource.OCR
    )
    properties: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"Element confidence must be in range [0.0, 1.0], got {self.confidence}")

    @property
    def center(self) -> Tuple[int, int]:
        return self.bounding_box.center

    def matches_text(self, query: str, exact: bool = False) -> bool:
        if not self.text:
            return False
        if exact:
            return self.text.strip().lower() == query.strip().lower()
        return query.strip().lower() in self.text.strip().lower()

    def is_valid_for_screen(self, screen: ScreenDimensions) -> bool:
        return (
            screen.contains(self.bounding_box.x, self.bounding_box.y)
            and self.bounding_box.right <= screen.width
            and self.bounding_box.bottom <= screen.height
        )


@dataclass(frozen=True)
class VisualScene:
    """Structured, immutable representation of a visual desktop state."""
    source_observation_id: str
    screen_dimensions: ScreenDimensions
    elements: Tuple[VisualElement, ...] = field(default_factory=tuple)
    active_window_title: Optional[str] = None
    process_name: Optional[str] = None
    timestamp: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def element_count(self) -> int:
        return len(self.elements)

    def find_text(self, query: str, exact: bool = False) -> List[VisualElement]:
        return [el for el in self.elements if el.matches_text(query, exact=exact)]

    def find_by_type(self, element_type: ElementType) -> List[VisualElement]:
        return [el for el in self.elements if el.element_type == element_type]

    def find_in_region(self, region: BoundingBox) -> List[VisualElement]:
        return [el for el in self.elements if region.intersects(el.bounding_box)]

    def find_near(self, target: VisualElement, max_distance: float = 100.0) -> List[VisualElement]:
        results = []
        for el in self.elements:
            if el.element_id == target.element_id:
                continue
            if el.bounding_box.distance_to(target.bounding_box) <= max_distance:
                results.append(el)
        return sorted(results, key=lambda x: x.bounding_box.distance_to(target.bounding_box))

    def find_by_relation(
        self,
        reference: VisualElement,
        relation: SpatialRelation,
        max_distance: float = 200.0,
    ) -> List[VisualElement]:
        results = []
        for el in self.elements:
            if el.element_id == reference.element_id:
                continue
            rel = el.bounding_box.relation_to(reference.bounding_box, max_distance=max_distance)
            if rel == relation:
                results.append(el)
        return sorted(results, key=lambda x: x.bounding_box.distance_to(reference.bounding_box))

    def get_element_by_id(self, element_id: str) -> Optional[VisualElement]:
        for el in self.elements:
            if el.element_id == element_id:
                return el
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_observation_id": self.source_observation_id,
            "dimensions": f"{self.screen_dimensions.width}x{self.screen_dimensions.height}",
            "element_count": self.element_count,
            "active_window": self.active_window_title,
            "process_name": self.process_name,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class GroundingRequest:
    """Model-neutral query specification for resolving visual targets."""
    text: Optional[str] = None
    element_type: Optional[ElementType] = None
    semantic_role: Optional[str] = None
    relation: Optional[SpatialRelation] = None
    relation_target_text: Optional[str] = None
    approximate_region: Optional[BoundingBox] = None
    application_context: Optional[str] = None
    min_confidence: float = 0.3
    max_candidates: int = 10


@dataclass(frozen=True)
class GroundedTarget:
    """Deterministic result of visual target grounding."""
    target_id: str
    element: VisualElement
    click_coordinate: Tuple[int, int]
    confidence: float
    source_observation_id: str
    observation_timestamp: float
    provenance: Dict[str, Any] = field(default_factory=dict)

    def is_stale(self, current_observation_id: str, max_age_seconds: float = 30.0, current_time: Optional[float] = None) -> bool:
        if self.source_observation_id != current_observation_id:
            return True
        if current_time is not None and (current_time - self.observation_timestamp > max_age_seconds):
            return True
        return False

    def to_tool_call_params(self, action: str = "click") -> Dict[str, Any]:
        return {
            "action": action,
            "x": self.click_coordinate[0],
            "y": self.click_coordinate[1],
        }


@dataclass(frozen=True)
class GroundingAmbiguity:
    """Diagnostic data when multiple candidates match a grounding query with similar confidence."""
    is_ambiguous: bool
    candidate_count: int
    candidates: Tuple[VisualElement, ...]
    reason: Optional[str] = None


@dataclass(frozen=True)
class GroundingResult:
    """Structured outcome of target grounding."""
    success: bool
    target: Optional[GroundedTarget] = None
    ambiguity: Optional[GroundingAmbiguity] = None
    error_code: Optional[str] = None
    reason: Optional[str] = None
    candidates: Tuple[VisualElement, ...] = field(default_factory=tuple)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# 3. Phase 6.5a Perception Limits & Safety Bounds
# ============================================================================

@dataclass(frozen=True)
class PerceptionLimits:
    """Strict resource and payload bounds for multimodal perception contracts."""
    max_evidence_per_result: int = 100
    max_attributes: int = 50
    max_attribute_key_length: int = 64
    max_attribute_value_length: int = 1024
    max_provider_count: int = 50
    max_capabilities_per_provider: int = 50
    max_modalities_per_provider: int = 20
    max_errors_per_result: int = 20
    max_metadata_size_bytes: int = 16384
    max_correlation_string_length: int = 128
    max_payload_ref_length: int = 256

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_evidence_per_result": self.max_evidence_per_result,
            "max_attributes": self.max_attributes,
            "max_attribute_key_length": self.max_attribute_key_length,
            "max_attribute_value_length": self.max_attribute_value_length,
            "max_provider_count": self.max_provider_count,
            "max_capabilities_per_provider": self.max_capabilities_per_provider,
            "max_modalities_per_provider": self.max_modalities_per_provider,
            "max_errors_per_result": self.max_errors_per_result,
            "max_metadata_size_bytes": self.max_metadata_size_bytes,
            "max_correlation_string_length": self.max_correlation_string_length,
            "max_payload_ref_length": self.max_payload_ref_length,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PerceptionLimits":
        return cls(
            max_evidence_per_result=int(data.get("max_evidence_per_result", 100)),
            max_attributes=int(data.get("max_attributes", 50)),
            max_attribute_key_length=int(data.get("max_attribute_key_length", 64)),
            max_attribute_value_length=int(data.get("max_attribute_value_length", 1024)),
            max_provider_count=int(data.get("max_provider_count", 50)),
            max_capabilities_per_provider=int(data.get("max_capabilities_per_provider", 50)),
            max_modalities_per_provider=int(data.get("max_modalities_per_provider", 20)),
            max_errors_per_result=int(data.get("max_errors_per_result", 20)),
            max_metadata_size_bytes=int(data.get("max_metadata_size_bytes", 16384)),
            max_correlation_string_length=int(data.get("max_correlation_string_length", 128)),
            max_payload_ref_length=int(data.get("max_payload_ref_length", 256)),
        )


# ============================================================================
# 4. Phase 6.5a Error & Spatial Models
# ============================================================================

@dataclass(frozen=True)
class PerceptionError:
    """Structured, bounded diagnostic record for perception failures."""
    code: str
    message: str
    provider_id: str
    request_id: str
    input_id: str
    recoverable: bool = False
    details: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.code or not isinstance(self.code, str):
            raise ValueError("PerceptionError code must be a non-empty string.")
        if not self.message or not isinstance(self.message, str):
            raise ValueError("PerceptionError message must be a non-empty string.")
        object.__setattr__(self, "details", sanitize_contract_metadata(self.details))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "provider_id": self.provider_id,
            "request_id": self.request_id,
            "input_id": self.input_id,
            "recoverable": self.recoverable,
            "details": sanitize_contract_metadata(self.details),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PerceptionError":
        return cls(
            code=str(data["code"]),
            message=str(data["message"]),
            provider_id=str(data.get("provider_id", "")),
            request_id=str(data.get("request_id", "")),
            input_id=str(data.get("input_id", "")),
            recoverable=bool(data.get("recoverable", False)),
            details=dict(data.get("details", {})),
        )


@dataclass(frozen=True)
class SpatialEvidence:
    """
    Spatial localization representation for perception evidence.
    Reuses existing BoundingBox and GeoLocation models without duplication.
    """
    point: Optional[Tuple[float, float]] = None
    bounding_box: Optional[BoundingBox] = None
    location: Optional[GeoLocation] = None
    region_label: Optional[str] = None
    relative_position: Optional[str] = None
    spatial_confidence: float = 1.0

    def __post_init__(self):
        conf = float(self.spatial_confidence)
        if math.isnan(conf) or math.isinf(conf) or not (0.0 <= conf <= 1.0):
            raise ValueError(f"SpatialEvidence spatial_confidence must be in range [0.0, 1.0], got {conf}")
        object.__setattr__(self, "spatial_confidence", conf)

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "spatial_confidence": round(self.spatial_confidence, 3),
        }
        if self.point is not None:
            data["point"] = [round(self.point[0], 3), round(self.point[1], 3)]
        if self.bounding_box is not None:
            data["bounding_box"] = self.bounding_box.to_dict()
        if self.location is not None:
            data["location"] = self.location.to_dict()
        if self.region_label is not None:
            data["region_label"] = self.region_label
        if self.relative_position is not None:
            data["relative_position"] = self.relative_position
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SpatialEvidence":
        point = None
        if "point" in data and data["point"] is not None:
            p = data["point"]
            point = (float(p[0]), float(p[1]))
        bbox = None
        if "bounding_box" in data and data["bounding_box"] is not None:
            bbox = BoundingBox.from_dict(data["bounding_box"])
        loc = None
        if "location" in data and data["location"] is not None:
            loc = GeoLocation.from_dict(data["location"])
        return cls(
            point=point,
            bounding_box=bbox,
            location=loc,
            region_label=data.get("region_label"),
            relative_position=data.get("relative_position"),
            spatial_confidence=float(data.get("spatial_confidence", 1.0)),
        )


# ============================================================================
# 5. Phase 6.5a Metadata & Input Models
# ============================================================================

@dataclass(frozen=True)
class PerceptionMetadata:
    """Provenance and processing metadata for perception operations."""
    provider_id: str
    provider_version: str = "1.0.0"
    model_id: Optional[str] = None
    model_version: Optional[str] = None
    processing_time_ms: Optional[float] = None
    confidence_calibration: Optional[Dict[str, float]] = None
    privacy_classification: PerceptionPrivacyClass = PerceptionPrivacyClass.INTERNAL
    deterministic: bool = True
    schema_version: str = "1.0.0"
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.provider_id or not isinstance(self.provider_id, str):
            raise ValueError("PerceptionMetadata provider_id must be a non-empty string.")
        pclass = self.privacy_classification
        if isinstance(pclass, str):
            pclass = PerceptionPrivacyClass.from_str(pclass)
        object.__setattr__(self, "privacy_classification", pclass)
        object.__setattr__(self, "extra", sanitize_contract_metadata(self.extra))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider_id": self.provider_id,
            "provider_version": self.provider_version,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "processing_time_ms": round(self.processing_time_ms, 2) if self.processing_time_ms is not None else None,
            "confidence_calibration": dict(self.confidence_calibration) if self.confidence_calibration else None,
            "privacy_classification": self.privacy_classification.value,
            "deterministic": self.deterministic,
            "schema_version": self.schema_version,
            "extra": sanitize_contract_metadata(self.extra),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PerceptionMetadata":
        return cls(
            provider_id=str(data["provider_id"]),
            provider_version=str(data.get("provider_version", "1.0.0")),
            model_id=data.get("model_id"),
            model_version=data.get("model_version"),
            processing_time_ms=float(data["processing_time_ms"]) if data.get("processing_time_ms") is not None else None,
            confidence_calibration=dict(data["confidence_calibration"]) if data.get("confidence_calibration") else None,
            privacy_classification=PerceptionPrivacyClass.from_str(data.get("privacy_classification", "INTERNAL")),
            deterministic=bool(data.get("deterministic", True)),
            schema_version=str(data.get("schema_version", "1.0.0")),
            extra=dict(data.get("extra", {})),
        )


@dataclass(frozen=True)
class PerceptionInput:
    """
    Lightweight, bounded input descriptor for a perception request.
    References modality artifacts without embedding large binary buffers.
    """
    input_id: str
    modality: ModalityType
    payload_ref: str
    captured_at: float
    source_id: str
    correlation_id: str = ""
    causation_id: Optional[str] = None
    schema_version: str = "1.0.0"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.input_id or not isinstance(self.input_id, str):
            raise ValueError("PerceptionInput input_id must be a non-empty string.")
        if not self.source_id or not isinstance(self.source_id, str):
            raise ValueError("PerceptionInput source_id must be a non-empty string.")
        if not self.payload_ref or not isinstance(self.payload_ref, str):
            raise ValueError("PerceptionInput payload_ref must be a non-empty string.")

        limits = PerceptionLimits()
        if len(self.payload_ref) > limits.max_payload_ref_length:
            raise ValueError(f"PerceptionInput payload_ref exceeds maximum length ({limits.max_payload_ref_length})")
        if self.correlation_id and len(self.correlation_id) > limits.max_correlation_string_length:
            raise ValueError(f"PerceptionInput correlation_id exceeds maximum length ({limits.max_correlation_string_length})")

        mod = self.modality
        if isinstance(mod, str):
            mod = ModalityType.from_str(mod)
        elif not isinstance(mod, ModalityType):
            raise ValueError(f"PerceptionInput modality must be a valid ModalityType, got {type(self.modality)}")
        object.__setattr__(self, "modality", mod)

        cap_ts = float(self.captured_at)
        if cap_ts <= 0.0:
            raise ValueError(f"PerceptionInput captured_at must be positive, got {cap_ts}")
        object.__setattr__(self, "captured_at", cap_ts)
        object.__setattr__(self, "metadata", sanitize_contract_metadata(self.metadata))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_id": self.input_id,
            "modality": self.modality.value,
            "payload_ref": self.payload_ref,
            "captured_at": self.captured_at,
            "source_id": self.source_id,
            "correlation_id": self.correlation_id,
            "causation_id": self.causation_id,
            "schema_version": self.schema_version,
            "metadata": sanitize_contract_metadata(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PerceptionInput":
        return cls(
            input_id=str(data["input_id"]),
            modality=ModalityType.from_str(data["modality"]),
            payload_ref=str(data["payload_ref"]),
            captured_at=float(data["captured_at"]),
            source_id=str(data["source_id"]),
            correlation_id=str(data.get("correlation_id", "")),
            causation_id=str(data["causation_id"]) if data.get("causation_id") else None,
            schema_version=str(data.get("schema_version", "1.0.0")),
            metadata=dict(data.get("metadata", {})),
        )


# ============================================================================
# 6. Phase 6.5a Perception Evidence Model
# ============================================================================

REQUIRED_PROVENANCE_KEYS: Tuple[str, ...] = (
    "source_id",
    "provider_id",
    "provider_version",
    "input_id",
    "request_id",
    "observation_timestamp",
    "correlation_id",
    "causation_id",
)


@dataclass(frozen=True)
class PerceptionEvidence:
    """
    Immutable semantic perception evidence produced by a perception provider.
    Carries explicit, bounded confidence, spatial localization, temporal semantics,
    and mandatory provenance.
    """
    evidence_id: str
    semantic_type: str
    label: str
    confidence: float
    source_id: str
    modality: ModalityType
    timestamp: float
    spatial: Optional[SpatialEvidence] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)
    correlation_id: str = ""
    causation_id: Optional[str] = None
    duration: Optional[float] = None

    def __post_init__(self):
        if not self.evidence_id or not isinstance(self.evidence_id, str):
            raise ValueError("PerceptionEvidence evidence_id must be a non-empty string.")
        if not self.semantic_type or not isinstance(self.semantic_type, str):
            raise ValueError("PerceptionEvidence semantic_type must be a non-empty string.")
        if not self.label or not isinstance(self.label, str):
            raise ValueError("PerceptionEvidence label must be a non-empty string.")
        if not self.source_id or not isinstance(self.source_id, str):
            raise ValueError("PerceptionEvidence source_id must be a non-empty string.")

        # Confidence: strictly bounded [0.0, 1.0], rejects NaN and Inf
        conf = float(self.confidence)
        if math.isnan(conf) or math.isinf(conf) or not (0.0 <= conf <= 1.0):
            raise ValueError(f"PerceptionEvidence confidence must be a finite number in [0.0, 1.0], got {conf}")
        object.__setattr__(self, "confidence", conf)

        ts = float(self.timestamp)
        if ts <= 0.0:
            raise ValueError(f"PerceptionEvidence timestamp must be positive, got {ts}")
        object.__setattr__(self, "timestamp", ts)

        mod = self.modality
        if isinstance(mod, str):
            mod = ModalityType.from_str(mod)
        elif not isinstance(mod, ModalityType):
            raise ValueError(f"PerceptionEvidence modality must be a valid ModalityType, got {type(self.modality)}")
        object.__setattr__(self, "modality", mod)

        # Mandatory provenance validation
        if not self.provenance or not isinstance(self.provenance, dict):
            raise ValueError("PerceptionEvidence provenance must be a non-empty dictionary.")
        missing_prov = [k for k in REQUIRED_PROVENANCE_KEYS if k not in self.provenance]
        if missing_prov:
            raise ValueError(f"PerceptionEvidence provenance missing required keys: {missing_prov}")

        limits = PerceptionLimits()
        if len(self.attributes) > limits.max_attributes:
            raise ValueError(f"PerceptionEvidence attributes exceed maximum allowed count ({limits.max_attributes})")
        for k, v in self.attributes.items():
            if len(str(k)) > limits.max_attribute_key_length:
                raise ValueError(f"Attribute key exceeds maximum length ({limits.max_attribute_key_length}): {k}")
            if len(str(v)) > limits.max_attribute_value_length:
                raise ValueError(f"Attribute value exceeds maximum length ({limits.max_attribute_value_length})")

        object.__setattr__(self, "attributes", sanitize_contract_metadata(self.attributes))
        object.__setattr__(self, "provenance", sanitize_contract_metadata(self.provenance))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "semantic_type": self.semantic_type,
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "source_id": self.source_id,
            "modality": self.modality.value,
            "timestamp": self.timestamp,
            "spatial": self.spatial.to_dict() if self.spatial else None,
            "attributes": sanitize_contract_metadata(self.attributes),
            "provenance": sanitize_contract_metadata(self.provenance),
            "correlation_id": self.correlation_id,
            "causation_id": self.causation_id,
            "duration": self.duration,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PerceptionEvidence":
        spatial = None
        if data.get("spatial") is not None:
            spatial = SpatialEvidence.from_dict(data["spatial"])
        return cls(
            evidence_id=str(data["evidence_id"]),
            semantic_type=str(data["semantic_type"]),
            label=str(data["label"]),
            confidence=float(data["confidence"]),
            source_id=str(data["source_id"]),
            modality=ModalityType.from_str(data["modality"]),
            timestamp=float(data["timestamp"]),
            spatial=spatial,
            attributes=dict(data.get("attributes", {})),
            provenance=dict(data.get("provenance", {})),
            correlation_id=str(data.get("correlation_id", "")),
            causation_id=str(data["causation_id"]) if data.get("causation_id") else None,
            duration=float(data["duration"]) if data.get("duration") is not None else None,
        )


# ============================================================================
# 7. Phase 6.5a Perception Request & Result Models
# ============================================================================

@dataclass(frozen=True)
class PerceptionRequest:
    """Formal query submitted to a perception provider."""
    request_id: str
    input_data: PerceptionInput
    requested_capabilities: Tuple[PerceptionCapability, ...] = field(default_factory=tuple)
    privacy_constraints: PerceptionPrivacyClass = PerceptionPrivacyClass.INTERNAL
    confidence_threshold: float = 0.0
    deadline: Optional[float] = None
    correlation_id: str = ""
    causation_id: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.request_id or not isinstance(self.request_id, str):
            raise ValueError("PerceptionRequest request_id must be a non-empty string.")
        if not isinstance(self.input_data, PerceptionInput):
            raise TypeError(f"PerceptionRequest input_data must be PerceptionInput, got {type(self.input_data)}")
        conf = float(self.confidence_threshold)
        if math.isnan(conf) or math.isinf(conf) or not (0.0 <= conf <= 1.0):
            raise ValueError(f"PerceptionRequest confidence_threshold must be in [0.0, 1.0], got {conf}")
        object.__setattr__(self, "confidence_threshold", conf)

        cid = self.correlation_id or self.input_data.correlation_id or self.request_id
        object.__setattr__(self, "correlation_id", cid)
        if not self.causation_id and self.input_data.causation_id:
            object.__setattr__(self, "causation_id", self.input_data.causation_id)
        object.__setattr__(self, "parameters", sanitize_contract_metadata(self.parameters))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "input_data": self.input_data.to_dict(),
            "requested_capabilities": [c.value for c in self.requested_capabilities],
            "privacy_constraints": self.privacy_constraints.value,
            "confidence_threshold": round(self.confidence_threshold, 3),
            "deadline": self.deadline,
            "correlation_id": self.correlation_id,
            "causation_id": self.causation_id,
            "parameters": sanitize_contract_metadata(self.parameters),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PerceptionRequest":
        caps = tuple(PerceptionCapability.from_str(c) for c in data.get("requested_capabilities", ()))
        return cls(
            request_id=str(data["request_id"]),
            input_data=PerceptionInput.from_dict(data["input_data"]),
            requested_capabilities=caps,
            privacy_constraints=PerceptionPrivacyClass.from_str(data.get("privacy_constraints", "INTERNAL")),
            confidence_threshold=float(data.get("confidence_threshold", 0.0)),
            deadline=float(data["deadline"]) if data.get("deadline") is not None else None,
            correlation_id=str(data.get("correlation_id", "")),
            causation_id=str(data["causation_id"]) if data.get("causation_id") else None,
            parameters=dict(data.get("parameters", {})),
        )


@dataclass(frozen=True)
class PerceptionResult:
    """
    Standardized result envelope returned by a perception provider.
    Carries verified evidence, observation candidates, processing metadata, and errors.
    """
    request_id: str
    input_id: str
    status: PerceptionStatus
    evidence: Tuple[PerceptionEvidence, ...] = field(default_factory=tuple)
    processing_metadata: PerceptionMetadata = field(default_factory=lambda: PerceptionMetadata(provider_id="UNKNOWN"))
    observation_candidates: Tuple[MultimodalObservation, ...] = field(default_factory=tuple)
    errors: Tuple[PerceptionError, ...] = field(default_factory=tuple)
    warnings: Tuple[str, ...] = field(default_factory=tuple)
    created_at: float = field(default_factory=time.time)

    def __post_init__(self):
        if not self.request_id or not isinstance(self.request_id, str):
            raise ValueError("PerceptionResult request_id must be a non-empty string.")
        if not self.input_id or not isinstance(self.input_id, str):
            raise ValueError("PerceptionResult input_id must be a non-empty string.")

        st = self.status
        if isinstance(st, str):
            st = PerceptionStatus.from_str(st)
        object.__setattr__(self, "status", st)

        limits = PerceptionLimits()
        if len(self.evidence) > limits.max_evidence_per_result:
            raise ValueError(f"PerceptionResult evidence count exceeds maximum allowed ({limits.max_evidence_per_result})")
        if len(self.errors) > limits.max_errors_per_result:
            raise ValueError(f"PerceptionResult error count exceeds maximum allowed ({limits.max_errors_per_result})")

    def is_success(self) -> bool:
        """Return True if perception processing succeeded."""
        return self.status == PerceptionStatus.SUCCESS

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "input_id": self.input_id,
            "status": self.status.value,
            "evidence": [e.to_dict() for e in self.evidence],
            "processing_metadata": self.processing_metadata.to_dict(),
            "observation_candidates": [
                {
                    "observation_id": obs.observation_id,
                    "source_id": obs.source_id,
                    "source_type": obs.source_type,
                    "modality": obs.modality.value,
                    "timestamp": obs.timestamp,
                    "confidence": obs.confidence,
                }
                for obs in self.observation_candidates
            ],
            "errors": [err.to_dict() for err in self.errors],
            "warnings": list(self.warnings),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PerceptionResult":
        ev_items = tuple(PerceptionEvidence.from_dict(e) for e in data.get("evidence", ()))
        err_items = tuple(PerceptionError.from_dict(err) for err in data.get("errors", ()))
        meta = (
            PerceptionMetadata.from_dict(data["processing_metadata"])
            if "processing_metadata" in data
            else PerceptionMetadata(provider_id="UNKNOWN")
        )
        return cls(
            request_id=str(data["request_id"]),
            input_id=str(data["input_id"]),
            status=PerceptionStatus.from_str(data["status"]),
            evidence=ev_items,
            processing_metadata=meta,
            observation_candidates=(),
            errors=err_items,
            warnings=tuple(data.get("warnings", ())),
            created_at=float(data.get("created_at", time.time())),
        )
