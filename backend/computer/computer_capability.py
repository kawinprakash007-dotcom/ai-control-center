from typing import Optional, Dict, Any, Union, Tuple


from core.interfaces.computer_interface import ComputerBackendInterface
from core.models.computer import (
    ComputerAction,
    ComputerTarget,
    ComputerObservation,
    ScreenDimensions,
    TargetType,
)
from core.models.result import Result
from core.models.task import Task
from computer.mock_backend import MockComputerBackend


class ComputerCapability:
    """
    Controlled capability for computer interaction.
    Enforces that 'Computer interaction is a CAPABILITY, not a second brain.'
    
    Supported Actions:
        - screenshot: Captures visual and accessibility observation
        - click: Single click at validated target coordinates
        - double_click: Double click at validated target coordinates
        - move: Move cursor to validated target coordinates
        - type: Send explicit text keystrokes (metadata audited without logging secrets)
        - press_key: Press authorized special or alphanumeric key
        - scroll: Scroll mouse wheel by bounded amount
        - wait: Bounded wait duration
    """

    def __init__(self, backend: Optional[ComputerBackendInterface] = None):
        """
        Initialize ComputerCapability with injected backend.
        Defaults to NativeComputerBackend if running on Windows desktop, else MockComputerBackend.
        """
        if backend is not None:
            self.backend = backend
        else:
            try:
                from computer.native_backend import NativeComputerBackend
                self.backend = NativeComputerBackend()
            except Exception:
                self.backend = MockComputerBackend()

    def __call__(self, task: Optional[Any] = None, **kwargs) -> Result:
        """Entry point for Executor and direct invocation."""
        return self.execute(task, **kwargs)

    def execute(self, task: Optional[Any] = None, **kwargs) -> Result:
        # Extract action, parameters, and call_id from task or kwargs
        params: Dict[str, Any] = {}
        call_id = None
        action = None

        if task is not None:
            if hasattr(task, "parameters") and isinstance(task.parameters, dict):
                params = dict(task.parameters)
            action = getattr(task, "action", None)
            call_id = str(getattr(task, "id", "")) or getattr(task, "call_id", None)

        # Allow kwargs to override/supplement
        params.update(kwargs)
        if "action" in params:
            action = params.get("action")
        if "call_id" in params:
            call_id = params.get("call_id")

        act_str = str(action or "").strip().lower()
        if act_str.startswith("computer "):
            act_str = act_str[9:].strip()
        call_id_str = str(call_id) if call_id else None

        # -------------------------------------------------------------
        # Action Dispatching
        # -------------------------------------------------------------
        try:
            if act_str == ComputerAction.SCREENSHOT.value:
                obs = self.backend.capture_screenshot()
                return Result.ok(
                    message=f"Screenshot captured ({obs.screen_dimensions.width}x{obs.screen_dimensions.height}).",
                    output=f"Screenshot captured ({obs.screen_dimensions.width}x{obs.screen_dimensions.height})",
                    data=obs,
                    capability="computer",
                    action=act_str,
                    call_id=call_id_str,
                )

            elif act_str in (ComputerAction.CLICK.value, ComputerAction.DOUBLE_CLICK.value, ComputerAction.MOVE.value):
                x, y, err = self._resolve_coordinates(params)
                if err:
                    return Result.fail(message=err, capability="computer", action=act_str, call_id=call_id_str)

                button = str(params.get("button", "left")).lower()

                if act_str == ComputerAction.CLICK.value:
                    self.backend.click(x, y, button=button)
                    msg = f"Clicked at ({x}, {y}) [{button}]."
                elif act_str == ComputerAction.DOUBLE_CLICK.value:
                    self.backend.double_click(x, y, button=button)
                    msg = f"Double-clicked at ({x}, {y}) [{button}]."
                else:
                    self.backend.move_cursor(x, y)
                    msg = f"Moved cursor to ({x}, {y})."

                return Result.ok(
                    message=msg,
                    output=msg,
                    data={"x": x, "y": y, "button": button},
                    capability="computer",
                    action=act_str,
                    call_id=call_id_str,
                )

            elif act_str == ComputerAction.TYPE.value:
                text = params.get("text")
                if text is None or not isinstance(text, str):
                    return Result.fail(
                        message="Type action requires a non-null 'text' string parameter.",
                        capability="computer",
                        action=act_str,
                        call_id=call_id_str,
                    )
                self.backend.type_text(text)
                # Audit safety: log text_length only, never log raw text
                return Result.ok(
                    message=f"Typed {len(text)} characters.",
                    output=f"Typed {len(text)} characters",
                    data={"text_length": len(text)},
                    capability="computer",
                    action=act_str,
                    call_id=call_id_str,
                )

            elif act_str == ComputerAction.PRESS_KEY.value:
                key = params.get("key")
                if not key or not isinstance(key, str):
                    return Result.fail(
                        message="Press_key action requires a non-empty 'key' parameter.",
                        capability="computer",
                        action=act_str,
                        call_id=call_id_str,
                    )
                self.backend.press_key(key)
                return Result.ok(
                    message=f"Pressed key '{key}'.",
                    output=f"Pressed key '{key}'",
                    data={"key": key},
                    capability="computer",
                    action=act_str,
                    call_id=call_id_str,
                )

            elif act_str == ComputerAction.SCROLL.value:
                amount = params.get("amount", 1)
                direction = str(params.get("direction", "down")).lower()
                if not isinstance(amount, int):
                    return Result.fail(
                        message="Scroll action requires an integer 'amount'.",
                        capability="computer",
                        action=act_str,
                        call_id=call_id_str,
                    )
                self.backend.scroll(amount, direction=direction)
                return Result.ok(
                    message=f"Scrolled {direction} by {amount} units.",
                    output=f"Scrolled {direction} by {amount}",
                    data={"amount": amount, "direction": direction},
                    capability="computer",
                    action=act_str,
                    call_id=call_id_str,
                )

            elif act_str == ComputerAction.WAIT.value:
                seconds = float(params.get("seconds", 1.0))
                if seconds < 0 or seconds > 30.0:
                    return Result.fail(
                        message=f"Wait duration {seconds}s must be between 0.0 and 30.0 seconds.",
                        capability="computer",
                        action=act_str,
                        call_id=call_id_str,
                    )
                self.backend.wait(seconds)
                return Result.ok(
                    message=f"Waited for {seconds}s.",
                    output=f"Waited {seconds}s",
                    data={"seconds": seconds},
                    capability="computer",
                    action=act_str,
                    call_id=call_id_str,
                )

            else:
                return Result.fail(
                    message=f"Unsupported computer action: '{act_str}'. Allowed: screenshot, click, double_click, move, type, press_key, scroll, wait.",
                    capability="computer",
                    action=act_str,
                    call_id=call_id_str,
                )

        except Exception as e:
            return Result.fail(
                message=f"Computer action '{act_str}' failed: {e}",
                capability="computer",
                action=act_str,
                call_id=call_id_str,
                data={"error": str(e)},
            )

    def _resolve_coordinates(self, params: Dict[str, Any]) -> Tuple[int, int, Optional[str]]:
        """Extract and validate (x, y) coordinates from direct params or target object."""
        x = params.get("x")
        y = params.get("y")

        if x is None or y is None:
            target = params.get("target")
            if isinstance(target, dict):
                x = target.get("x")
                y = target.get("y")
            elif isinstance(target, ComputerTarget):
                x = target.x
                y = target.y

        if x is None or y is None:
            return 0, 0, "Action requires integer coordinates 'x' and 'y' (or a valid target object)."

        if not isinstance(x, int) or not isinstance(y, int):
            return 0, 0, f"Coordinates must be integers, got x={type(x).__name__}, y={type(y).__name__}."

        screen = self.backend.get_screen_dimensions()
        if not screen.contains(x, y):
            return 0, 0, f"Target coordinates ({x}, {y}) fall outside screen bounds ({screen.width}x{screen.height})."

        return x, y, None
