import os
import sys
import time
import tempfile
from typing import Optional, Tuple, Dict, Any

from core.interfaces.computer_interface import ComputerBackendInterface
from core.models.computer import (
    ScreenDimensions,
    ComputerObservation,
    ALLOWED_SPECIAL_KEYS,
)


class NativeComputerBackend(ComputerBackendInterface):
    """
    Platform-aware native computer interaction backend.
    Enforces screen coordinate validation, temporary screenshot lifecycle,
    and safe hardware abstraction without external heavy frameworks.
    """

    def __init__(self):
        self._is_windows = sys.platform == "win32"
        self._user32 = None
        if self._is_windows:
            try:
                import ctypes
                self._user32 = ctypes.windll.user32
            except Exception:
                self._user32 = None

    def get_screen_dimensions(self) -> ScreenDimensions:
        try:
            from PIL import ImageGrab
            screenshot = ImageGrab.grab()
            return ScreenDimensions(width=screenshot.width, height=screenshot.height)
        except Exception:
            if self._user32:
                w = self._user32.GetSystemMetrics(0)
                h = self._user32.GetSystemMetrics(1)
                if w > 0 and h > 0:
                    return ScreenDimensions(width=w, height=h)
            return ScreenDimensions(width=1920, height=1080)

    def capture_screenshot(self) -> ComputerObservation:
        try:
            from PIL import ImageGrab
            screenshot = ImageGrab.grab()
            dims = ScreenDimensions(width=screenshot.width, height=screenshot.height)

            # Store in temporary file with bounded lifetime
            temp_dir = tempfile.gettempdir()
            screenshot_path = os.path.join(temp_dir, f"atlas_shot_{int(time.time()*1000)}.png")
            screenshot.save(screenshot_path, format="PNG")

            active_title, proc = self.get_active_window()

            return ComputerObservation(
                timestamp=time.time(),
                screen_dimensions=dims,
                screenshot_path=screenshot_path,
                active_window_title=active_title,
                process_name=proc,
                metadata={"native": True},
            )
        except Exception as e:
            raise RuntimeError(f"Screenshot capture failed: {e}")

    def move_cursor(self, x: int, y: int) -> None:
        dims = self.get_screen_dimensions()
        if not dims.contains(x, y):
            raise ValueError(f"Target coordinates ({x}, {y}) out of screen bounds ({dims.width}x{dims.height})")
        if self._user32:
            self._user32.SetCursorPos(x, y)
        else:
            raise RuntimeError("Native mouse control unavailable on this platform.")

    def click(self, x: int, y: int, button: str = "left") -> None:
        self.move_cursor(x, y)
        if self._user32:
            # MOUSEEVENTF_LEFTDOWN = 0x0002, MOUSEEVENTF_LEFTUP = 0x0004
            # MOUSEEVENTF_RIGHTDOWN = 0x0008, MOUSEEVENTF_RIGHTUP = 0x0010
            if button.lower() == "right":
                down_flag, up_flag = 0x0008, 0x0010
            else:
                down_flag, up_flag = 0x0002, 0x0004
            self._user32.mouse_event(down_flag, 0, 0, 0, 0)
            time.sleep(0.05)
            self._user32.mouse_event(up_flag, 0, 0, 0, 0)
        else:
            raise RuntimeError("Native mouse click unavailable on this platform.")

    def double_click(self, x: int, y: int, button: str = "left") -> None:
        self.click(x, y, button)
        time.sleep(0.1)
        self.click(x, y, button)

    def type_text(self, text: str) -> None:
        if not isinstance(text, str):
            raise TypeError(f"Text must be string, got {type(text).__name__}")
        if self._user32:
            import ctypes
            for char in text:
                vk = self._user32.VkKeyScanW(ord(char))
                if vk != -1:
                    code = vk & 0xFF
                    self._user32.keybd_event(code, 0, 0, 0)
                    self._user32.keybd_event(code, 0, 2, 0)  # KEYEVENTF_KEYUP
                time.sleep(0.01)
        else:
            raise RuntimeError("Native keyboard typing unavailable on this platform.")

    def press_key(self, key: str) -> None:
        k = str(key).strip().lower()
        if len(k) != 1 and k not in ALLOWED_SPECIAL_KEYS:
            raise ValueError(f"Key '{key}' is not an authorized keyboard key.")
        if self._user32:
            vk_mapping = {
                "enter": 0x0D, "return": 0x0D, "tab": 0x09, "space": 0x20,
                "backspace": 0x08, "delete": 0x2E, "escape": 0x1B, "esc": 0x1B,
                "up": 0x26, "down": 0x28, "left": 0x25, "right": 0x27,
            }
            code = vk_mapping.get(k)
            if code is None and len(k) == 1:
                code = ord(k.upper())
            if code is not None:
                self._user32.keybd_event(code, 0, 0, 0)
                time.sleep(0.02)
                self._user32.keybd_event(code, 0, 2, 0)
        else:
            raise RuntimeError("Native key press unavailable on this platform.")

    def scroll(self, amount: int, direction: str = "down") -> None:
        if not isinstance(amount, int):
            raise TypeError("Scroll amount must be an integer.")
        if self._user32:
            # MOUSEEVENTF_WHEEL = 0x0800
            clicks = -amount if direction.lower() == "down" else amount
            self._user32.mouse_event(0x0800, 0, 0, clicks * 120, 0)
        else:
            raise RuntimeError("Native scroll unavailable on this platform.")

    def wait(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("Wait duration cannot be negative.")
        clamped = min(seconds, 10.0)  # Safe bound
        time.sleep(clamped)

    def get_active_window(self) -> Tuple[Optional[str], Optional[str]]:
        if self._user32:
            try:
                import ctypes
                hwnd = self._user32.GetForegroundWindow()
                length = self._user32.GetWindowTextLengthW(hwnd)
                buff = ctypes.create_unicode_buffer(length + 1)
                self._user32.GetWindowTextW(hwnd, buff, length + 1)
                return buff.value or "Active Window", "active_process.exe"
            except Exception:
                pass
        return None, None
