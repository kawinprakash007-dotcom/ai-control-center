from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple, Dict, Any


class ComputerAction(str, Enum):
    """
    Controlled set of permissible computer interaction operations.
    Arbitrary shell, exec, or OS scripting are strictly excluded.
    """
    SCREENSHOT = "screenshot"
    MOVE = "move"
    CLICK = "click"
    DOUBLE_CLICK = "double_click"
    TYPE = "type"
    PRESS_KEY = "press_key"
    SCROLL = "scroll"
    WAIT = "wait"


class TargetType(str, Enum):
    """
    Representation type for interaction targeting.
    """
    COORDINATE = "coordinate"
    REGION = "region"
    SEMANTIC = "semantic"


@dataclass(frozen=True)
class ScreenDimensions:
    """
    Immutable representation of screen dimensions.
    """
    width: int
    height: int

    def __post_init__(self):
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"Invalid screen dimensions: {self.width}x{self.height}")

    def contains(self, x: int, y: int) -> bool:
        """Check if coordinates fall strictly within screen bounds."""
        return 0 <= x < self.width and 0 <= y < self.height


@dataclass(frozen=True)
class ComputerTarget:
    """
    Controlled target representation for GUI interaction.
    """
    target_type: TargetType = TargetType.COORDINATE
    x: Optional[int] = None
    y: Optional[int] = None
    region: Optional[Tuple[int, int, int, int]] = None  # (x, y, width, height)
    element_id: Optional[str] = None
    label: Optional[str] = None

    def validate_bounds(self, screen: ScreenDimensions) -> Tuple[bool, Optional[str]]:
        """Validate target against screen dimensions."""
        if self.target_type == TargetType.COORDINATE:
            if self.x is None or self.y is None:
                return False, "Coordinate target requires both 'x' and 'y' integer values."
            if not isinstance(self.x, int) or not isinstance(self.y, int):
                return False, f"Coordinates must be integers, got x={type(self.x).__name__}, y={type(self.y).__name__}."
            if not screen.contains(self.x, self.y):
                return False, f"Coordinates ({self.x}, {self.y}) are outside screen bounds (0,0) to ({screen.width-1},{screen.height-1})."
            return True, None

        if self.target_type == TargetType.REGION:
            if not self.region or len(self.region) != 4:
                return False, "Region target requires a 4-tuple: (x, y, width, height)."
            rx, ry, rw, rh = self.region
            if rw <= 0 or rh <= 0:
                return False, f"Region dimensions must be positive, got {rw}x{rh}."
            if not (screen.contains(rx, ry) and screen.contains(rx + rw - 1, ry + rh - 1)):
                return False, f"Region ({rx}, {ry}, {rw}, {rh}) exceeds screen bounds ({screen.width}x{screen.height})."
            return True, None

        if self.target_type == TargetType.SEMANTIC:
            if not self.element_id and not self.label:
                return False, "Semantic target requires either 'element_id' or 'label'."
            return True, None

        return False, f"Unknown target type: {self.target_type}"


@dataclass(frozen=True)
class ComputerObservation:
    """
    Immutable representation of a visual and accessibility desktop observation.
    Maintains strict separation between Observation and Action.
    """
    timestamp: float
    screen_dimensions: ScreenDimensions
    screenshot_path: Optional[str] = None
    screenshot_base64: Optional[str] = None
    active_window_title: Optional[str] = None
    process_name: Optional[str] = None
    ui_tree_metadata: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    observation_id: Optional[str] = None

    def __post_init__(self):
        if not self.observation_id:
            object.__setattr__(self, "observation_id", f"obs_{int(self.timestamp * 1000)}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "timestamp": self.timestamp,
            "dimensions": f"{self.screen_dimensions.width}x{self.screen_dimensions.height}",
            "screenshot_path": self.screenshot_path,
            "has_base64": bool(self.screenshot_base64),
            "active_window": self.active_window_title,
            "process_name": self.process_name,
            "metadata": self.metadata,
        }


# Whitelist of standard navigation and control keys
ALLOWED_SPECIAL_KEYS = {
    "enter", "return", "tab", "space", "backspace", "delete", "escape", "esc",
    "up", "down", "left", "right", "home", "end", "pageup", "pagedown",
    "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11", "f12",
}
