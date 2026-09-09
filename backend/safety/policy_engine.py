import uuid
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Set, Union

from core.models.tool_call import ToolCall
from core.models.policy import (
    PolicyDecision,
    PolicyContext,
    PolicyResult,
    AutonomyLevel,
    RiskLevel,
)
from core.interfaces.policy_interface import PolicyEngineInterface


PROHIBITED_CAPABILITIES_AND_ACTIONS: Set[str] = {
    "shell", "bash", "sh", "cmd", "powershell", "subprocess", "exec", "eval",
    "system", "filesystem", "rm", "delete_file", "write_file", "device",
    "browser", "computer_use", "spawn", "sudo", "chmod", "curl", "wget",
}

LEGACY_DESKTOP_TOOLS: Set[str] = {
    "calculator", "chrome", "vscode", "notepad", "explorer", "time",
}

SENSITIVE_KEY_PATTERNS: Set[str] = {
    "password", "secret", "token", "credential", "api_key", "private_key",
    "auth", "ssn", "credit_card",
}


class PolicyRule(ABC):
    """
    Composable, deterministic evaluation rule for capability authorization.
    """
    rule_id: str
    description: str

    @abstractmethod
    def matches(self, tool_call: ToolCall, context: PolicyContext) -> bool:
        """Determine if this rule applies to the proposed tool_call and context."""
        pass

    @abstractmethod
    def evaluate(self, tool_call: ToolCall, context: PolicyContext) -> PolicyResult:
        """Produce a deterministic PolicyResult for the matched call."""
        pass


class ProhibitedCapabilityRule(PolicyRule):
    """
    Strictly forbids dangerous, arbitrary execution, shell, device, or OS access.
    """
    rule_id = "RULE_FORBIDDEN_CAPABILITY"
    description = "Denies all prohibited, shell, OS, filesystem, or arbitrary execution capabilities."

    def matches(self, tool_call: ToolCall, context: PolicyContext) -> bool:
        cap = tool_call.capability.lower()
        act = tool_call.action.lower()
        if cap in ("device_gateway", "device_dispatch"):
            return False
        cap_parts = set(cap.replace("-", "_").split("_"))
        act_parts = set(act.replace("-", "_").split("_"))
        for bad in PROHIBITED_CAPABILITIES_AND_ACTIONS:
            if len(bad) <= 2:
                if bad in (cap, act) or bad in cap_parts or bad in act_parts:
                    return True
            else:
                if bad in cap or bad in act:
                    return True
        return False

    def evaluate(self, tool_call: ToolCall, context: PolicyContext) -> PolicyResult:
        return PolicyResult.deny(
            rule_id=self.rule_id,
            reason=f"Capability or action '{tool_call.capability}.{tool_call.action}' is strictly forbidden by safety policy.",
            explanation="This operation involves restricted OS, shell, or device-level access.",
            metadata={"risk_level": RiskLevel.CRITICAL.value},
        )


class LegacyToolRestrictionRule(PolicyRule):
    """
    Restricts legacy desktop tools from being invoked through the generic model orchestrator.
    """
    rule_id = "RULE_LEGACY_TOOL_RESTRICTED"
    description = "Restricts desktop tools (calculator, chrome, etc.) from generic model invocation."

    def matches(self, tool_call: ToolCall, context: PolicyContext) -> bool:
        return tool_call.capability.lower() in LEGACY_DESKTOP_TOOLS

    def evaluate(self, tool_call: ToolCall, context: PolicyContext) -> PolicyResult:
        return PolicyResult.deny(
            rule_id=self.rule_id,
            reason=f"Legacy desktop capability '{tool_call.capability}' cannot be invoked through the general Tool Orchestrator.",
            explanation=f"'{tool_call.capability}' is a legacy desktop tool and not an authorized model capability.",
            metadata={"risk_level": RiskLevel.HIGH.value},
        )


class DestructiveMemoryRule(PolicyRule):
    """
    Enforces user confirmation for destructive memory operations (e.g. forget).
    """
    rule_id = "RULE_DESTRUCTIVE_MEMORY_FORGET"
    description = "Requires explicit user confirmation before permanently deleting stored memory."

    def matches(self, tool_call: ToolCall, context: PolicyContext) -> bool:
        return tool_call.capability.lower() == "memory" and tool_call.action.lower() in ("forget", "delete")

    def evaluate(self, tool_call: ToolCall, context: PolicyContext) -> PolicyResult:
        key = tool_call.parameters.get("key", "specified memory")

        # Allow if explicitly confirmed, autonomous autonomy level, or trusted pipeline source
        if (
            tool_call.parameters.get("confirmed") is True
            or context.metadata.get("confirmed") is True
            or context.autonomy_level == AutonomyLevel.AUTONOMOUS
            or context.source in ("pipeline", "user_direct")
        ):
            return PolicyResult.allow(
                rule_id="RULE_CONFIRMED_MEMORY_FORGET",
                reason=f"Forgetting memory preference '{key}' was confirmed or explicitly requested by user.",
                metadata={"risk_level": RiskLevel.LOW.value, "key": key},
            )

        return PolicyResult.require_confirmation(
            rule_id=self.rule_id,
            reason=f"Permanently forgetting memory preference '{key}' is destructive and requires confirmation.",
            explanation=f"Deleting stored preference '{key}' cannot be undone. Please confirm.",
            metadata={"risk_level": RiskLevel.MEDIUM.value, "key": key},
        )


class SensitiveMemorySaveRule(PolicyRule):
    """
    Governs memory save operations:
    - Asks permission if key contains sensitive credential patterns.
    - Asks permission if running in MANUAL autonomy mode.
    - Allows general preference saves in ASSISTED/AUTONOMOUS mode.
    """
    rule_id = "RULE_SENSITIVE_MEMORY_SAVE"
    description = "Governs memory save operations based on sensitivity and autonomy level."

    def matches(self, tool_call: ToolCall, context: PolicyContext) -> bool:
        return tool_call.capability.lower() == "memory" and tool_call.action.lower() in ("save", "remember")

    def evaluate(self, tool_call: ToolCall, context: PolicyContext) -> PolicyResult:
        key = str(tool_call.parameters.get("key", "")).lower()

        # Check for credential / sensitive pattern in key
        if any(pat in key for pat in SENSITIVE_KEY_PATTERNS):
            return PolicyResult.ask_permission(
                rule_id="RULE_MEMORY_SAVE_CREDENTIAL_KEY",
                reason=f"Storing sensitive or credential-related key '{key}' requires explicit permission.",
                explanation=f"Saving '{key}' involves potentially sensitive credentials or tokens.",
                metadata={"risk_level": RiskLevel.HIGH.value, "key": key},
            )

        # Check for MANUAL autonomy mode
        if context.autonomy_level == AutonomyLevel.MANUAL:
            return PolicyResult.ask_permission(
                rule_id="RULE_MANUAL_AUTONOMY_MEMORY_SAVE",
                reason="Operating in MANUAL autonomy level requires explicit permission to save memory.",
                explanation=f"Saving memory '{key}' requires your approval under manual mode.",
                metadata={"risk_level": RiskLevel.MEDIUM.value, "key": key},
            )

        # Standard safe preference saving
        return PolicyResult.allow(
            rule_id="RULE_SAFE_MEMORY_SAVE",
            reason=f"Saving general user preference '{key}' is permitted.",
            metadata={"risk_level": RiskLevel.LOW.value, "key": key},
        )


class SafeReadAndSearchRule(PolicyRule):
    """
    Authorizes verified safe read-only capabilities:
    - web: search, fetch, research
    - knowledge: query, retrieve, search
    - memory: read, recall
    """
    rule_id = "RULE_SAFE_READ_AND_SEARCH"
    description = "Permits whitelisted read-only search, retrieval, and research operations."

    SAFE_MATRIX: Dict[str, Set[str]] = {
        "web": {"search", "fetch", "research"},
        "knowledge": {"query", "retrieve", "search", "retrieve knowledge", "retrieve_knowledge"},
        "memory": {"read", "recall"},
    }

    def matches(self, tool_call: ToolCall, context: PolicyContext) -> bool:
        cap = tool_call.capability.lower()
        act = tool_call.action.lower()
        return cap in self.SAFE_MATRIX and act in self.SAFE_MATRIX[cap]

    def evaluate(self, tool_call: ToolCall, context: PolicyContext) -> PolicyResult:
        return PolicyResult.allow(
            rule_id=f"RULE_SAFE_{tool_call.capability.upper()}_{tool_call.action.upper()}",
            reason=f"Read-only operation '{tool_call.capability}.{tool_call.action}' is permitted by safety policy.",
            metadata={"risk_level": RiskLevel.SAFE.value},
        )


class ComputerObservationRule(PolicyRule):
    """
    Authorizes safe visual observation and bounded pause operations:
    - computer: screenshot, wait
    """
    rule_id = "RULE_COMPUTER_OBSERVATION"
    description = "Authorizes safe, read-only computer screenshot and wait operations."

    def matches(self, tool_call: ToolCall, context: PolicyContext) -> bool:
        cap = tool_call.capability.lower()
        act = tool_call.action.lower()
        return cap == "computer" and act in ("screenshot", "wait")

    def evaluate(self, tool_call: ToolCall, context: PolicyContext) -> PolicyResult:
        return PolicyResult.allow(
            rule_id=f"RULE_COMPUTER_{tool_call.action.upper()}",
            reason=f"Computer observation operation '{tool_call.action}' is permitted by safety policy.",
            metadata={"risk_level": RiskLevel.SAFE.value},
        )


class ComputerLowRiskActionRule(PolicyRule):
    """
    Governs low-risk computer interaction actions:
    - computer: move, scroll, click, double_click
    Requires permission in MANUAL autonomy mode; allows in ASSISTED/AUTONOMOUS.
    """
    rule_id = "RULE_COMPUTER_LOW_RISK_ACTION"
    description = "Governs mouse cursor, scroll, and click interactions based on autonomy level."

    def matches(self, tool_call: ToolCall, context: PolicyContext) -> bool:
        cap = tool_call.capability.lower()
        act = tool_call.action.lower()
        return cap == "computer" and act in ("move", "scroll", "click", "double_click")

    def evaluate(self, tool_call: ToolCall, context: PolicyContext) -> PolicyResult:
        if context.autonomy_level == AutonomyLevel.MANUAL:
            return PolicyResult.ask_permission(
                rule_id="RULE_MANUAL_AUTONOMY_COMPUTER_ACTION",
                reason=f"Computer action '{tool_call.action}' requires user approval under MANUAL autonomy mode.",
                explanation=f"Atlas wants to perform computer {tool_call.action}. Please approve to proceed.",
                metadata={"risk_level": RiskLevel.MEDIUM.value},
            )

        return PolicyResult.allow(
            rule_id=f"RULE_COMPUTER_{tool_call.action.upper()}",
            reason=f"Computer action '{tool_call.action}' is authorized.",
            metadata={"risk_level": RiskLevel.LOW.value},
        )


class ComputerSensitiveActionRule(PolicyRule):
    """
    Governs sensitive keyboard interactions:
    - computer: type, press_key
    Enforces permission/confirmation for sensitive text, credential keywords, dangerous shortcuts,
    or MANUAL autonomy mode.
    """
    rule_id = "RULE_COMPUTER_SENSITIVE_KEYBOARD"
    description = "Governs keyboard typing and shortcuts based on content sensitivity and autonomy level."

    DANGEROUS_SHORTCUTS: Set[str] = {
        "ctrl+alt+del", "ctrl+shift+esc", "alt+f4", "win+r", "win+x", "cmd", "powershell",
    }

    SENSITIVE_TEXT_PATTERNS: Set[str] = {
        "password", "secret", "api_key", "token", "sudo", "rm -rf", "curl ", "wget ",
    }

    def matches(self, tool_call: ToolCall, context: PolicyContext) -> bool:
        cap = tool_call.capability.lower()
        act = tool_call.action.lower()
        return cap == "computer" and act in ("type", "press_key")

    def evaluate(self, tool_call: ToolCall, context: PolicyContext) -> PolicyResult:
        act = tool_call.action.lower()
        params = tool_call.parameters

        if act == "press_key":
            key = str(params.get("key", "")).lower().replace(" ", "")
            if any(danger in key for danger in self.DANGEROUS_SHORTCUTS):
                return PolicyResult.require_confirmation(
                    rule_id="RULE_DANGEROUS_KEY_COMBINATION",
                    reason=f"Pressing high-impact shortcut '{key}' requires explicit confirmation.",
                    explanation=f"Key shortcut '{key}' can alter system state. Please confirm.",
                    metadata={"risk_level": RiskLevel.HIGH.value, "key": key},
                )

        if act == "type":
            text = str(params.get("text", "")).lower()
            if any(pat in text for pat in self.SENSITIVE_TEXT_PATTERNS):
                return PolicyResult.ask_permission(
                    rule_id="RULE_SENSITIVE_TEXT_TYPING",
                    reason="Typing content containing sensitive or command patterns requires permission.",
                    explanation="The proposed text contains potential credential or command keywords.",
                    metadata={"risk_level": RiskLevel.HIGH.value, "text_length": len(text)},
                )

        if context.autonomy_level == AutonomyLevel.MANUAL:
            return PolicyResult.ask_permission(
                rule_id="RULE_MANUAL_AUTONOMY_KEYBOARD",
                reason=f"Keyboard operation '{act}' requires permission in MANUAL autonomy mode.",
                explanation=f"Atlas wants to perform keyboard action '{act}'. Please approve.",
                metadata={"risk_level": RiskLevel.MEDIUM.value},
            )

        return PolicyResult.allow(
            rule_id=f"RULE_COMPUTER_{act.upper()}",
            reason=f"Computer keyboard operation '{act}' is authorized.",
            metadata={"risk_level": RiskLevel.LOW.value},
        )


class DeviceGatewayOperationRule(PolicyRule):
    """
    Authorizes governed DeviceGateway dispatch and query operations:
    - device_gateway: dispatch_capability, query_status, list_devices
    """
    rule_id = "RULE_DEVICE_GATEWAY_OPERATION"
    description = "Authorizes governed edge participant command dispatch and status queries."

    def matches(self, tool_call: ToolCall, context: PolicyContext) -> bool:
        cap = tool_call.capability.lower()
        act = tool_call.action.lower()
        return cap in ("device_gateway", "device_dispatch") and act in ("dispatch_capability", "query_status", "list_devices")

    def evaluate(self, tool_call: ToolCall, context: PolicyContext) -> PolicyResult:
        return PolicyResult.allow(
            rule_id="RULE_DEVICE_GATEWAY_ALLOWED",
            reason=f"DeviceGateway operation '{tool_call.action}' is authorized by safety policy.",
            metadata={"risk_level": RiskLevel.LOW.value},
        )


class GoalLifecycleOperationRule(PolicyRule):
    """
    Authorizes governed Goal lifecycle operations:
    - goal: create_goal, update_goal, pause_goal, resume_goal, cancel_goal
    """
    rule_id = "RULE_GOAL_LIFECYCLE_OPERATION"
    description = "Authorizes governed goal lifecycle management operations."

    def matches(self, tool_call: ToolCall, context: PolicyContext) -> bool:
        cap = tool_call.capability.lower()
        act = tool_call.action.lower()
        return cap == "goal" and act in (
            "create_goal",
            "update_goal",
            "pause_goal",
            "resume_goal",
            "cancel_goal",
        )

    def evaluate(self, tool_call: ToolCall, context: PolicyContext) -> PolicyResult:
        return PolicyResult.allow(
            rule_id="RULE_GOAL_LIFECYCLE_ALLOWED",
            reason=f"Goal lifecycle operation '{tool_call.action}' is authorized by safety policy.",
            metadata={"risk_level": RiskLevel.LOW.value},
        )


class DefaultDenyRule(PolicyRule):
    """
    Catch-all default-deny rule enforcing that no unknown capability or action ever fails open.
    """
    rule_id = "RULE_DEFAULT_DENY"
    description = "Default-deny policy for any capability or action not explicitly permitted."

    def matches(self, tool_call: ToolCall, context: PolicyContext) -> bool:
        return True

    def evaluate(self, tool_call: ToolCall, context: PolicyContext) -> PolicyResult:
        return PolicyResult.deny(
            rule_id=self.rule_id,
            reason=f"No safety policy rule permits capability '{tool_call.capability}' with action '{tool_call.action}'. Default deny enforced.",
            explanation=f"The operation '{tool_call.capability}.{tool_call.action}' is not authorized.",
            metadata={"risk_level": RiskLevel.HIGH.value},
        )


class StandardPolicyEngine(PolicyEngineInterface):
    """
    Deterministic, model-neutral policy engine enforcing authorization for capability execution.
    Evaluates ordered PolicyRule implementations with an absolute default-deny guarantee.
    """

    def __init__(self, rules: Optional[List[PolicyRule]] = None):
        """
        Initialize StandardPolicyEngine with an ordered sequence of rules.

        Args:
            rules: Optional custom rules sequence for dependency injection / testing.
        """
        if rules is not None:
            self.rules: List[PolicyRule] = list(rules)
        else:
            self.rules = [
                ProhibitedCapabilityRule(),
                LegacyToolRestrictionRule(),
                DestructiveMemoryRule(),
                SensitiveMemorySaveRule(),
                SafeReadAndSearchRule(),
                ComputerObservationRule(),
                ComputerLowRiskActionRule(),
                ComputerSensitiveActionRule(),
                GoalLifecycleOperationRule(),
                DeviceGatewayOperationRule(),
                DefaultDenyRule(),
            ]


    def evaluate(
        self,
        tool_call: Union[ToolCall, PolicyContext],
        context: Optional[PolicyContext] = None,
    ) -> PolicyResult:
        """
        Evaluate proposed ToolCall or PolicyContext against deterministic rules.
        Never fails open.
        """
        # Support PolicyContext passed directly as first parameter
        if isinstance(tool_call, PolicyContext):
            effective_context = tool_call
            tool_call = ToolCall(
                capability=effective_context.capability,
                action=effective_context.action,
                parameters=dict(effective_context.parameters),
                call_id=effective_context.call_id or f"call_{uuid.uuid4().hex[:8]}",
            )
        elif not isinstance(tool_call, ToolCall):
            return PolicyResult.deny(
                rule_id="RULE_INVALID_TOOL_CALL",
                reason=f"Expected ToolCall instance, got {type(tool_call).__name__}.",
                metadata={"risk_level": RiskLevel.CRITICAL.value},
            )
        else:
            effective_context = context
            if effective_context is None:
                try:
                    effective_context = PolicyContext.from_tool_call(tool_call)
                except Exception as e:
                    return PolicyResult.deny(
                        rule_id="RULE_INVALID_CONTEXT",
                        reason=f"Failed to resolve PolicyContext: {e}. Default deny enforced.",
                        metadata={"risk_level": RiskLevel.CRITICAL.value},
                    )

        if not isinstance(effective_context, PolicyContext):
            return PolicyResult.deny(
                rule_id="RULE_INVALID_CONTEXT",
                reason=f"Invalid PolicyContext type: {type(effective_context).__name__}.",
                metadata={"risk_level": RiskLevel.CRITICAL.value},
            )

        # Evaluate rules in sequence
        try:
            for rule in self.rules:
                if rule.matches(tool_call, effective_context):
                    return rule.evaluate(tool_call, effective_context)

            # Fallback if no rule matched (safety guarantee)
            return PolicyResult.deny(
                rule_id="RULE_DEFAULT_DENY",
                reason=f"Unmatched tool call '{tool_call.capability}.{tool_call.action}'. Default deny enforced.",
                metadata={"risk_level": RiskLevel.HIGH.value},
            )
        except Exception as e:
            # Absolute default-deny on any unexpected internal error
            return PolicyResult.deny(
                rule_id="RULE_POLICY_ENGINE_ERROR",
                reason=f"Internal policy evaluation error: {e}. Default deny enforced.",
                metadata={"risk_level": RiskLevel.CRITICAL.value},
            )
