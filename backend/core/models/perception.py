from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple, List, Dict, Any
import math

from core.models.computer import ScreenDimensions


class ElementType(str, Enum):
    """
    Bounded vocabulary of visual and semantic UI element categories.
    """
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


class PerceptionSource(str, Enum):
    """
    Provenance tracking for multi-source perception signals.
    """
    OCR = "ocr"
    CV = "cv"
    ACCESSIBILITY = "accessibility"
    MANUAL = "manual"
    FUTURE_VISION_MODEL = "future_vision_model"


class SpatialRelation(str, Enum):
    """
    Bounded set of deterministic spatial relationships for scene querying.
    """
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


@dataclass(frozen=True)
class VisualElement:
    """
    Immutable representation of an individual detected UI element.
    """
    element_id: str
    element_type: ElementType
    bounding_box: BoundingBox
    text: Optional[str] = None
    confidence: float = 1.0
    source: PerceptionSource = PerceptionSource.OCR
    properties: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"Element confidence must be in range [0.0, 1.0], got {self.confidence}")

    @property
    def center(self) -> Tuple[int, int]:
        return self.bounding_box.center

    def matches_text(self, query: str, exact: bool = False) -> bool:
        """Check if element text matches query."""
        if not self.text:
            return False
        if exact:
            return self.text.strip().lower() == query.strip().lower()
        return query.strip().lower() in self.text.strip().lower()

    def is_valid_for_screen(self, screen: ScreenDimensions) -> bool:
        """Validate that element's bounding box is fully contained within the screen."""
        return (
            screen.contains(self.bounding_box.x, self.bounding_box.y)
            and self.bounding_box.right <= screen.width
            and self.bounding_box.bottom <= screen.height
        )


@dataclass(frozen=True)
class VisualScene:
    """
    Structured, immutable representation of a visual desktop state.
    Provides deterministic spatial and semantic query methods.
    """
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
        """Find all elements whose text matches the query."""
        return [el for el in self.elements if el.matches_text(query, exact=exact)]

    def find_by_type(self, element_type: ElementType) -> List[VisualElement]:
        """Find all elements of a given element type."""
        return [el for el in self.elements if el.element_type == element_type]

    def find_in_region(self, region: BoundingBox) -> List[VisualElement]:
        """Find all elements whose bounding boxes intersect the region."""
        return [el for el in self.elements if region.intersects(el.bounding_box)]

    def find_near(self, target: VisualElement, max_distance: float = 100.0) -> List[VisualElement]:
        """Find all elements within max_distance of the target element."""
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
        """Find elements having the specified spatial relation relative to reference."""
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
    """
    Model-neutral query specification for resolving visual targets.
    """
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
    """
    Deterministic result of visual target grounding.
    Carries provenance and click coordinates.
    CRITICAL: Does NOT trigger device clicks.
    """
    target_id: str
    element: VisualElement
    click_coordinate: Tuple[int, int]
    confidence: float
    source_observation_id: str
    observation_timestamp: float
    provenance: Dict[str, Any] = field(default_factory=dict)

    def is_stale(self, current_observation_id: str, max_age_seconds: float = 30.0, current_time: Optional[float] = None) -> bool:
        """Check if grounded target is invalid against the current observation or lifecycle."""
        if self.source_observation_id != current_observation_id:
            return True
        if current_time is not None and (current_time - self.observation_timestamp > max_age_seconds):
            return True
        return False

    def to_tool_call_params(self, action: str = "click") -> Dict[str, Any]:
        """
        Format parameters for a subsequent governed ToolCall.
        Does NOT execute anything.
        """
        return {
            "action": action,
            "x": self.click_coordinate[0],
            "y": self.click_coordinate[1],
        }


@dataclass(frozen=True)
class GroundingAmbiguity:
    """
    Diagnostic data when multiple candidates match a grounding query with similar confidence.
    """
    is_ambiguous: bool
    candidate_count: int
    candidates: Tuple[VisualElement, ...]
    reason: Optional[str] = None


@dataclass(frozen=True)
class GroundingResult:
    """
    Structured outcome of target grounding.
    """
    success: bool
    target: Optional[GroundedTarget] = None
    ambiguity: Optional[GroundingAmbiguity] = None
    error_code: Optional[str] = None  # TARGET_NOT_FOUND, AMBIGUOUS_TARGET, TARGET_STALE, CONFIDENCE_TOO_LOW
    reason: Optional[str] = None
    candidates: Tuple[VisualElement, ...] = field(default_factory=tuple)
    metadata: Dict[str, Any] = field(default_factory=dict)
