from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass(frozen=True)
class ToolCall:
    """
    Structured model-proposed action for general capability execution.
    Establishes the clean contract: 'The model proposes. The runtime controls.'

    Attributes:
        capability: Target capability name ('web', 'memory', 'knowledge').
        action: Specific action to perform on the capability (e.g. 'search', 'read', 'query').
        parameters: Action-specific arguments.
        reason: Optional justification or reasoning for issuing this call.
        call_id: Optional correlation identifier for tracing and observability.
    """
    capability: str
    action: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    call_id: Optional[str] = None

    def __post_init__(self):
        # Normalize capability and action strings
        object.__setattr__(self, "capability", str(self.capability or "").strip().lower())
        object.__setattr__(self, "action", str(self.action or "").strip().lower())
        if self.call_id is not None:
            object.__setattr__(self, "call_id", str(self.call_id).strip())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "capability": self.capability,
            "action": self.action,
            "parameters": dict(self.parameters),
            "reason": self.reason,
            "call_id": self.call_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolCall":
        if not isinstance(data, dict):
            raise TypeError(f"ToolCall.from_dict expects dict, got {type(data).__name__}")
        return cls(
            capability=str(data.get("capability", "")),
            action=str(data.get("action", "")),
            parameters=dict(data.get("parameters", {})),
            reason=str(data.get("reason", "")),
            call_id=data.get("call_id"),
        )

    @classmethod
    def from_task(cls, task: Any) -> "ToolCall":
        """
        Convert a legacy Task instance into a structured ToolCall.
        """
        if not hasattr(task, "tool"):
            raise TypeError(f"Cannot convert {type(task).__name__} to ToolCall: missing 'tool' attribute")

        tool_name = str(task.tool or getattr(task, "type", "") or "").strip().lower()
        params = dict(task.parameters) if hasattr(task, "parameters") and isinstance(task.parameters, dict) else {}
        raw_action = str(params.get("action") or params.get("memory_action") or getattr(task, "action", "") or "").strip().lower()

        # Normalize common task action phrases to canonical capability actions
        action = raw_action
        if tool_name == "knowledge":
            if not action or any(w in action for w in ("retrieve", "query", "search", "knowledge")):
                action = "retrieve"
        elif tool_name == "memory":
            if "save" in action or "remember" in action or "value" in params:
                action = "save"
            elif "forget" in action:
                action = "forget"
            elif not action or any(w in action for w in ("read", "recall", "manage", "access", "memory")):
                action = "read"
        elif tool_name == "web":
            if "research" in action or params.get("action") == "research":
                action = "research"
            elif "fetch" in action or params.get("action") == "fetch":
                action = "fetch"
            elif not action or any(w in action for w in ("search", "web")):
                action = "search"
        elif tool_name == "computer":
            if "screenshot" in action:
                action = "screenshot"
            elif "double" in action:
                action = "double_click"
            elif "click" in action:
                action = "click"
            elif "move" in action:
                action = "move"
            elif "type" in action:
                action = "type"
            elif "press" in action or "key" in action:
                action = "press_key"
            elif "scroll" in action:
                action = "scroll"
            elif "wait" in action:
                action = "wait"
            elif not action:
                action = "screenshot"


        call_id = str(task.id) if hasattr(task, "id") and task.id is not None else None

        return cls(
            capability=tool_name,
            action=action,
            parameters=params,
            call_id=call_id,
        )
