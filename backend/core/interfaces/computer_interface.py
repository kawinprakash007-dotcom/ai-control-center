from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Any

from core.models.computer import (
    ScreenDimensions,
    ComputerObservation,
    ComputerTarget,
)


class ComputerBackendInterface(ABC):
    """
    Model-neutral interface abstracting low-level OS GUI interactions.
    Enforces that execution is separated from reasoning, and allows clean mock injection for tests.
    """

    @abstractmethod
    def get_screen_dimensions(self) -> ScreenDimensions:
        """Return the width and height of the active display screen."""
        pass

    @abstractmethod
    def capture_screenshot(self) -> ComputerObservation:
        """Capture a visual observation of the desktop display."""
        pass

    @abstractmethod
    def move_cursor(self, x: int, y: int) -> None:
        """Move cursor to target coordinates (x, y)."""
        pass

    @abstractmethod
    def click(self, x: int, y: int, button: str = "left") -> None:
        """Click mouse at target coordinates (x, y)."""
        pass

    @abstractmethod
    def double_click(self, x: int, y: int, button: str = "left") -> None:
        """Double click mouse at target coordinates (x, y)."""
        pass

    @abstractmethod
    def type_text(self, text: str) -> None:
        """Send keystrokes for explicit text."""
        pass

    @abstractmethod
    def press_key(self, key: str) -> None:
        """Press a specific key (e.g. 'enter', 'tab', 'escape')."""
        pass

    @abstractmethod
    def scroll(self, amount: int, direction: str = "down") -> None:
        """Scroll vertical or horizontal wheels."""
        pass

    @abstractmethod
    def wait(self, seconds: float) -> None:
        """Pause execution for a bounded duration."""
        pass

    @abstractmethod
    def get_active_window(self) -> Tuple[Optional[str], Optional[str]]:
        """Return (window_title, process_name) for the currently focused window."""
        pass
