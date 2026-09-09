from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class Result:
    """
    Standard result model representing the outcome of an execution step or tool operation.

    Attributes:
        success: Whether the operation completed successfully.
        message: Descriptive summary or error message.
        output: Human-readable string output or response text.
        data: Structured domain data (e.g. EvidenceSet, ResearchResult, dict).
        capability: Optional target capability name (for tool execution tracing).
        action: Optional capability action executed.
        call_id: Optional correlation identifier for observability.
    """
    success: bool
    message: str
    output: Optional[str] = None
    data: Any = None
    capability: Optional[str] = None
    action: Optional[str] = None
    call_id: Optional[str] = None

    @property
    def error_code(self) -> Optional[str]:
        if isinstance(self.data, dict):
            return self.data.get("error_code")
        return None

    @classmethod
    def ok(
        cls,
        message: str = "Task completed.",
        output: Optional[str] = None,
        data: Any = None,
        capability: Optional[str] = None,
        action: Optional[str] = None,
        call_id: Optional[str] = None,
    ) -> "Result":
        return cls(
            success=True,
            message=message,
            output=output,
            data=data,
            capability=capability,
            action=action,
            call_id=call_id,
        )

    @classmethod
    def fail(
        cls,
        message: str,
        output: Optional[str] = None,
        data: Any = None,
        capability: Optional[str] = None,
        action: Optional[str] = None,
        call_id: Optional[str] = None,
        error_code: Optional[str] = None,
    ) -> "Result":
        payload = data or {}
        if error_code and isinstance(payload, dict):
            payload["error_code"] = error_code
        return cls(
            success=False,
            message=message,
            output=output,
            data=payload if payload else None,
            capability=capability,
            action=action,
            call_id=call_id,
        )

    @classmethod
    def failure(
        cls,
        message: str,
        output: Optional[str] = None,
        data: Any = None,
        capability: Optional[str] = None,
        action: Optional[str] = None,
        call_id: Optional[str] = None,
        error_code: Optional[str] = None,
    ) -> "Result":
        return cls.fail(
            message=message,
            output=output,
            data=data,
            capability=capability,
            action=action,
            call_id=call_id,
            error_code=error_code,
        )