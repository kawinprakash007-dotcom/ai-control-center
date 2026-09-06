import time
from typing import Optional, Tuple, Dict, Any, List

from core.interfaces.computer_interface import ComputerBackendInterface
from core.models.computer import (
    ScreenDimensions,
    ComputerObservation,
    ALLOWED_SPECIAL_KEYS,
)


class MockComputerBackend(ComputerBackendInterface):
    """
    Deterministic mock computer backend for testing.
    Validates screen bounds, tracks action history, and never interacts with the real desktop.
    """

    def __init__(
        self,
        screen_dimensions: Optional[ScreenDimensions] = None,
        active_window_title: str = "Test Application Window",
        process_name: str = "test_app.exe",
    ):
        self.screen = screen_dimensions or ScreenDimensions(width=1920, height=1080)
        self.active_window_title = active_window_title
        self.process_name = process_name
        self.cursor_x = 0
        self.cursor_y = 0
        self.history: List[Dict[str, Any]] = []
        self.should_fail = False
        self.failure_message = "Mock device failure"
        self.screenshot_count = 0

    @property
    def click_count(self) -> int:
        return len([h for h in self.history if h.get("action") in ("click", "double_click")])

    @property
    def cursor_pos(self) -> Tuple[int, int]:
        return (self.cursor_x, self.cursor_y)

    def get_screen_dimensions(self) -> ScreenDimensions:
        return self.screen

    def capture_screenshot(self) -> ComputerObservation:
        if self.should_fail:
            raise RuntimeError(self.failure_message)

        self.screenshot_count += 1
        obs = ComputerObservation(
            timestamp=time.time(),
            screen_dimensions=self.screen,
            screenshot_path=f"/tmp/mock_screenshot_{self.screenshot_count}.png",
            screenshot_base64="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
            active_window_title=self.active_window_title,
            process_name=self.process_name,
            metadata={"mock": True, "screenshot_index": self.screenshot_count},
        )
        self.history.append({"action": "screenshot", "timestamp": obs.timestamp})
        return obs

    def move_cursor(self, x: int, y: int) -> None:
        if self.should_fail:
            raise RuntimeError(self.failure_message)
        if not self.screen.contains(x, y):
            raise ValueError(f"Coordinates ({x}, {y}) out of screen bounds ({self.screen.width}x{self.screen.height})")
        self.cursor_x = x
        self.cursor_y = y
        self.history.append({"action": "move", "x": x, "y": y})

    def click(self, x: int, y: int, button: str = "left") -> None:
        if self.should_fail:
            raise RuntimeError(self.failure_message)
        if not self.screen.contains(x, y):
            raise ValueError(f"Click coordinates ({x}, {y}) out of screen bounds ({self.screen.width}x{self.screen.height})")
        self.cursor_x = x
        self.cursor_y = y
        self.history.append({"action": "click", "x": x, "y": y, "button": button})

    def double_click(self, x: int, y: int, button: str = "left") -> None:
        if self.should_fail:
            raise RuntimeError(self.failure_message)
        if not self.screen.contains(x, y):
            raise ValueError(f"Double-click coordinates ({x}, {y}) out of screen bounds ({self.screen.width}x{self.screen.height})")
        self.cursor_x = x
        self.cursor_y = y
        self.history.append({"action": "double_click", "x": x, "y": y, "button": button})

    def type_text(self, text: str) -> None:
        if self.should_fail:
            raise RuntimeError(self.failure_message)
        if not isinstance(text, str):
            raise TypeError(f"Text must be string, got {type(text).__name__}")
        self.history.append({"action": "type", "text_length": len(text), "text": text})

    def press_key(self, key: str) -> None:
        if self.should_fail:
            raise RuntimeError(self.failure_message)
        k = str(key).strip().lower()
        if len(k) != 1 and k not in ALLOWED_SPECIAL_KEYS:
            raise ValueError(f"Key '{key}' is not an authorized or recognized keyboard key.")
        self.history.append({"action": "press_key", "key": k})

    def scroll(self, amount: int, direction: str = "down") -> None:
        if self.should_fail:
            raise RuntimeError(self.failure_message)
        if not isinstance(amount, int):
            raise TypeError("Scroll amount must be an integer.")
        self.history.append({"action": "scroll", "amount": amount, "direction": direction})

    def wait(self, seconds: float) -> None:
        if self.should_fail:
            raise RuntimeError(self.failure_message)
        if seconds < 0:
            raise ValueError("Wait duration cannot be negative.")
        self.history.append({"action": "wait", "seconds": seconds})

    def get_active_window(self) -> Tuple[Optional[str], Optional[str]]:
        return self.active_window_title, self.process_name
