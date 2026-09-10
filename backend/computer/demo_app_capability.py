import os
import sys
import time
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple

from core.models.result import Result


@dataclass(frozen=True)
class ApplicationRegistryEntry:
    """
    Immutable specification of an authorized application in the demonstration registry.
    Ensures that executable paths and arguments are fixed and cannot be model-constructed.
    """
    app_id: str
    display_name: str
    executable_path: str
    launch_args: List[str] = field(default_factory=list)
    process_names: List[str] = field(default_factory=list)
    window_title_patterns: List[str] = field(default_factory=list)


def _resolve_vscode_path() -> str:
    local_app_data = os.environ.get("LOCALAPPDATA", "")
    program_files = os.environ.get("ProgramFiles", r"C:\Program Files")
    candidates = [
        os.path.join(local_app_data, "Programs", "Microsoft VS Code", "Code.exe"),
        os.path.join(program_files, "Microsoft VS Code", "Code.exe"),
        r"C:\Program Files\Microsoft VS Code\Code.exe",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return candidates[0]


def _resolve_chrome_path() -> str:
    program_files = os.environ.get("ProgramFiles", r"C:\Program Files")
    program_files_x86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
    local_app_data = os.environ.get("LOCALAPPDATA", "")
    candidates = [
        os.path.join(program_files, "Google", "Chrome", "Application", "chrome.exe"),
        os.path.join(program_files_x86, "Google", "Chrome", "Application", "chrome.exe"),
        os.path.join(local_app_data, "Google", "Chrome", "Application", "chrome.exe"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return candidates[0]


def _resolve_notepad_path() -> str:
    system_root = os.environ.get("SystemRoot", r"C:\Windows")
    candidates = [
        os.path.join(system_root, "System32", "notepad.exe"),
        os.path.join(system_root, "notepad.exe"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return candidates[0]


# Fixed, immutable registry of allowed demonstration applications
FIXED_APPLICATION_REGISTRY: Dict[str, ApplicationRegistryEntry] = {
    "vscode": ApplicationRegistryEntry(
        app_id="vscode",
        display_name="Visual Studio Code",
        executable_path=_resolve_vscode_path(),
        launch_args=[],
        process_names=["Code.exe", "code.exe"],
        window_title_patterns=["Visual Studio Code"],
    ),
    "chrome": ApplicationRegistryEntry(
        app_id="chrome",
        display_name="Google Chrome",
        executable_path=_resolve_chrome_path(),
        launch_args=[],
        process_names=["chrome.exe"],
        window_title_patterns=["Google Chrome", "Chrome"],
    ),
    "notepad": ApplicationRegistryEntry(
        app_id="notepad",
        display_name="Notepad",
        executable_path=_resolve_notepad_path(),
        launch_args=[],
        process_names=["notepad.exe"],
        window_title_patterns=["Notepad"],
    ),
}


class DemoAppBackendInterface(ABC):
    """Abstract execution backend for bounded demo application lifecycle."""

    @abstractmethod
    def launch(self, entry: ApplicationRegistryEntry) -> Tuple[bool, str, Dict[str, Any]]:
        pass

    @abstractmethod
    def close(self, entry: ApplicationRegistryEntry) -> Tuple[bool, str, Dict[str, Any]]:
        pass

    @abstractmethod
    def focus(self, entry: ApplicationRegistryEntry) -> Tuple[bool, str, Dict[str, Any]]:
        pass


class NativeDemoAppBackend(DemoAppBackendInterface):
    """
    Platform-aware native backend using direct OS process APIs without generic shell.
    Enforces shell=False at all times and operates exclusively on fixed registry entries.
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

    def launch(self, entry: ApplicationRegistryEntry) -> Tuple[bool, str, Dict[str, Any]]:
        if not os.path.exists(entry.executable_path):
            return False, f"Application '{entry.display_name}' executable not found at configured path: {entry.executable_path}", {}

        try:
            # Strictly shell=False; arguments passed as discrete argv array
            cmd_args = [entry.executable_path] + list(entry.launch_args)
            proc = subprocess.Popen(cmd_args, shell=False)
            return True, f"Launched {entry.display_name} (PID: {proc.pid}).", {"pid": proc.pid, "executable": entry.executable_path}
        except Exception as e:
            return False, f"Failed to launch application '{entry.display_name}': {str(e)}", {}

    def _find_hwnd(self, entry: ApplicationRegistryEntry) -> Optional[int]:
        if not self._user32:
            return None
        import ctypes
        found_hwnd = None

        def enum_windows_callback(hwnd, extra):
            nonlocal found_hwnd
            if not self._user32.IsWindowVisible(hwnd):
                return True
            length = self._user32.GetWindowTextLengthW(hwnd)
            if length > 0:
                buff = ctypes.create_unicode_buffer(length + 1)
                self._user32.GetWindowTextW(hwnd, buff, length + 1)
                title = buff.value
                for pattern in entry.window_title_patterns:
                    if pattern.lower() in title.lower():
                        found_hwnd = hwnd
                        return False
            return True

        WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_int, ctypes.c_int)
        cb = WNDENUMPROC(enum_windows_callback)
        self._user32.EnumWindows(cb, 0)
        return found_hwnd

    def focus(self, entry: ApplicationRegistryEntry) -> Tuple[bool, str, Dict[str, Any]]:
        if not self._is_windows or not self._user32:
            return False, "Window focus is not supported on this platform.", {}

        hwnd = self._find_hwnd(entry)
        if hwnd:
            # SW_RESTORE = 9
            self._user32.ShowWindow(hwnd, 9)
            self._user32.SetForegroundWindow(hwnd)
            return True, f"Focused window for {entry.display_name}.", {"hwnd": hwnd}
        return False, f"No running window found for {entry.display_name} to focus.", {}

    def close(self, entry: ApplicationRegistryEntry) -> Tuple[bool, str, Dict[str, Any]]:
        if not self._is_windows or not self._user32:
            return False, "Window close is not supported on this platform.", {}

        hwnd = self._find_hwnd(entry)
        if hwnd:
            # WM_CLOSE = 0x0010
            self._user32.PostMessageW(hwnd, 0x0010, 0, 0)
            return True, f"Sent close signal to {entry.display_name}.", {"hwnd": hwnd}
        return False, f"No running window found for {entry.display_name} to close.", {}


class MockDemoAppBackend(DemoAppBackendInterface):
    """
    Deterministic mock backend for safe, headless unit and integration testing.
    Records all operations without interacting with the OS desktop.
    """

    def __init__(self):
        self.history: List[Dict[str, Any]] = []
        self.should_fail = False
        self.failure_reason = "Mock execution failure"
        self.missing_apps: set[str] = set()

    def launch(self, entry: ApplicationRegistryEntry) -> Tuple[bool, str, Dict[str, Any]]:
        if self.should_fail:
            return False, self.failure_reason, {}
        if entry.app_id in self.missing_apps:
            return False, f"Application '{entry.display_name}' executable not found at configured path: {entry.executable_path}", {}

        record = {"action": "launch", "app_id": entry.app_id, "timestamp": time.time()}
        self.history.append(record)
        return True, f"[MOCK] Launched {entry.display_name}.", {"pid": 9999, "mock": True}

    def focus(self, entry: ApplicationRegistryEntry) -> Tuple[bool, str, Dict[str, Any]]:
        if self.should_fail:
            return False, self.failure_reason, {}
        record = {"action": "focus", "app_id": entry.app_id, "timestamp": time.time()}
        self.history.append(record)
        return True, f"[MOCK] Focused {entry.display_name}.", {"mock": True}

    def close(self, entry: ApplicationRegistryEntry) -> Tuple[bool, str, Dict[str, Any]]:
        if self.should_fail:
            return False, self.failure_reason, {}
        record = {"action": "close", "app_id": entry.app_id, "timestamp": time.time()}
        self.history.append(record)
        return True, f"[MOCK] Closed {entry.display_name}.", {"mock": True}


class DemoAppCapability:
    """
    Controlled capability for bounded demonstration application execution.
    Registered as capability 'computer_app' when DEMO_MODE=true.
    
    Supported Actions:
      - launch: Launch fixed executable for registered app_id
      - close: Request graceful close of registered app_id window
      - focus: Bring registered app_id window to the foreground
      
    Security Invariants:
      1. Model can only supply app_id matching FIXED_APPLICATION_REGISTRY.
      2. Caller cannot supply executable path, command line string, or shell arguments.
      3. No shell is ever invoked (shell=False).
    """

    FORBIDDEN_PARAMETER_KEYS = {
        "executable", "path", "command", "cmd", "shell", "exec", "script",
        "args", "arguments", "cli", "binary", "filename",
    }

    def __init__(
        self,
        backend: Optional[DemoAppBackendInterface] = None,
        registry: Optional[Dict[str, ApplicationRegistryEntry]] = None,
    ):
        self.registry = registry if registry is not None else FIXED_APPLICATION_REGISTRY
        if backend is not None:
            self.backend = backend
        else:
            if sys.platform == "win32":
                self.backend = NativeDemoAppBackend()
            else:
                self.backend = MockDemoAppBackend()

    def __call__(self, task: Optional[Any] = None, **kwargs) -> Result:
        return self.execute(task, **kwargs)

    def execute(self, task: Optional[Any] = None, **kwargs) -> Result:
        params: Dict[str, Any] = {}
        call_id = None
        action = None

        if task is not None:
            if hasattr(task, "parameters") and isinstance(task.parameters, dict):
                params = dict(task.parameters)
            action = getattr(task, "action", None)
            call_id = str(getattr(task, "id", "")) or getattr(task, "call_id", None)

        params.update(kwargs)
        if "action" in params:
            action = params.get("action")
        if "call_id" in params:
            call_id = params.get("call_id")

        act_str = str(action or "").strip().lower()
        if act_str.startswith("computer app "):
            act_str = act_str[13:].strip()
        elif act_str.startswith("computer_app "):
            act_str = act_str[13:].strip()
        elif act_str.startswith("computer "):
            act_str = act_str[9:].strip()

        call_id_str = str(call_id) if call_id else None

        # 1. Action validation
        if act_str not in ("launch", "close", "focus"):
            return Result.fail(
                message=f"Unsupported computer_app action: '{act_str}'. Permitted actions: launch, close, focus.",
                capability="computer_app",
                action=act_str,
                call_id=call_id_str,
            )

        # 2. Strict parameter safety checks: detect parameter injection
        for forbidden in self.FORBIDDEN_PARAMETER_KEYS:
            if forbidden in params:
                return Result.fail(
                    message=f"Parameter '{forbidden}' is forbidden. The model cannot supply arbitrary executable paths or command lines.",
                    capability="computer_app",
                    action=act_str,
                    call_id=call_id_str,
                    data={"error_type": "security_violation", "forbidden_key": forbidden},
                )

        # 3. Resolve app_id
        app_id_raw = params.get("app_id") or params.get("app") or params.get("application")
        if not app_id_raw or not isinstance(app_id_raw, str):
            return Result.fail(
                message="computer_app requires a valid string 'app_id' parameter.",
                capability="computer_app",
                action=act_str,
                call_id=call_id_str,
                data={"error_type": "missing_app_id"},
            )

        app_id = app_id_raw.strip().lower()
        if app_id not in self.registry:
            valid_apps = ", ".join(sorted(self.registry.keys()))
            return Result.fail(
                message=f"Unknown application '{app_id}'. Registered demonstration applications: {valid_apps}.",
                capability="computer_app",
                action=act_str,
                call_id=call_id_str,
                data={"error_type": "unregistered_application", "app_id": app_id},
            )

        entry = self.registry[app_id]
        timestamp = time.time()

        # 4. Controlled dispatch
        try:
            if act_str == "launch":
                success, msg, data = self.backend.launch(entry)
            elif act_str == "focus":
                success, msg, data = self.backend.focus(entry)
            elif act_str == "close":
                success, msg, data = self.backend.close(entry)
            else:
                success, msg, data = False, f"Unhandled action: {act_str}", {}

            audit_data = {
                "application_id": app_id,
                "display_name": entry.display_name,
                "action": act_str,
                "result": "success" if success else "failure",
                "timestamp": timestamp,
                "demo_mode": True,
                **data,
            }

            if success:
                return Result.ok(
                    message=msg,
                    output=msg,
                    data=audit_data,
                    capability="computer_app",
                    action=act_str,
                    call_id=call_id_str,
                )
            else:
                return Result.fail(
                    message=msg,
                    capability="computer_app",
                    action=act_str,
                    call_id=call_id_str,
                    data=audit_data,
                )

        except Exception as ex:
            # Never expose raw OS exception or traceback to caller
            return Result.fail(
                message=f"Execution failure for '{app_id}.{act_str}': {str(ex)}",
                capability="computer_app",
                action=act_str,
                call_id=call_id_str,
                data={
                    "application_id": app_id,
                    "action": act_str,
                    "result": "failure",
                    "timestamp": timestamp,
                    "demo_mode": True,
                    "error": str(ex),
                },
            )
