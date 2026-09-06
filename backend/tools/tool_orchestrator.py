import time
from typing import Optional, Dict, Any, Set, Tuple, Union

from core.models.tool_call import ToolCall
from core.models.result import Result
from core.models.task import Task
from core.models.policy import (
    PolicyDecision,
    PolicyContext,
    PolicyResult,
    AutonomyLevel,
)
from core.interfaces.policy_interface import PolicyEngineInterface
from tools.capability_registry import CapabilityRegistry
from tools.executor import Executor


# Whitelisted standard capabilities and their permitted actions.
# Desktop, shell, device, and filesystem tools are strictly excluded.
DEFAULT_ALLOWED_CAPABILITIES: Dict[str, Set[str]] = {
    "web": {"search", "fetch", "research"},
    "memory": {"save", "read", "forget", "remember", "recall"},
    "knowledge": {"query", "retrieve", "search", "retrieve knowledge", "retrieve_knowledge"},
}

PROHIBITED_STRINGS: Set[str] = {
    "shell", "bash", "sh", "cmd", "powershell", "subprocess", "exec", "eval",
    "system", "filesystem", "rm", "delete_file", "write_file", "device",
    "browser", "computer_use", "spawn",
}


class ToolOrchestrator:
    """
    Central orchestration layer governing capability execution for AI Control Center.
    Enforces the core principle: 'The model proposes. The runtime controls.'

    Execution Pipeline:
        ToolCall
            ↓
        Action & Parameter Validation
            ↓
        Policy & Permission Evaluation (StandardPolicyEngine)
            ↓
        Capability Resolution (CapabilityRegistry)
            ↓
        Capability Execution via Executor
            ↓
        Structured Result with Provenance & Observability
    """

    def __init__(
        self,
        registry: Optional[CapabilityRegistry] = None,
        executor: Optional[Executor] = None,
        allowed_capabilities: Optional[Dict[str, Set[str]]] = None,
        policy_engine: Optional[PolicyEngineInterface] = None,
    ):
        """
        Initialize ToolOrchestrator.

        Args:
            registry: Optional CapabilityRegistry instance (defaults to standard CapabilityRegistry).
            executor: Optional Executor instance (defaults to standard Executor).
            allowed_capabilities: Optional whitelist mapping capability -> allowed actions set.
            policy_engine: Optional PolicyEngineInterface implementation for authorization checks.
        """
        self.registry = registry if registry is not None else CapabilityRegistry()
        self.executor = executor if executor is not None else Executor()
        self.allowed_capabilities: Dict[str, Set[str]] = (
            {k.lower(): {a.lower() for a in v} for k, v in allowed_capabilities.items()}
            if allowed_capabilities is not None
            else {k: set(v) for k, v in DEFAULT_ALLOWED_CAPABILITIES.items()}
        )

        if policy_engine is not None:
            self.policy_engine = policy_engine
        else:
            try:
                from safety.policy_engine import StandardPolicyEngine
                self.policy_engine = StandardPolicyEngine()
            except ImportError:
                self.policy_engine = None

    def get_allowed_capabilities(self) -> Set[str]:
        """Return the set of currently allowed capability names."""
        return set(self.allowed_capabilities.keys())

    def get_allowed_actions(self, capability: str) -> Set[str]:
        """Return allowed actions for a given capability."""
        return set(self.allowed_capabilities.get(capability.lower(), set()))

    def validate_call(self, tool_call: Any) -> Tuple[bool, Optional[str]]:
        """
        Validate a proposed ToolCall against security policies, action whitelists,
        and parameter schemas.

        Returns:
            (is_valid, error_message_if_invalid)
        """
        if not isinstance(tool_call, ToolCall):
            return False, f"Expected ToolCall instance, got {type(tool_call).__name__}."

        cap = tool_call.capability
        act = tool_call.action
        params = tool_call.parameters

        # 1. Reject prohibited names
        if any(bad in cap or bad in act for bad in PROHIBITED_STRINGS):
            return False, f"Prohibited capability or action name: '{cap}:{act}'."

        # 2. Capability whitelist check
        if cap not in self.allowed_capabilities:
            allowed = ", ".join(sorted(self.allowed_capabilities.keys()))
            return (
                False,
                f"Unknown or unauthorized capability: '{cap}'. Allowed capabilities: {allowed}.",
            )

        # 3. Action whitelist check
        allowed_actions = self.allowed_capabilities[cap]
        if act not in allowed_actions:
            allowed_act_str = ", ".join(sorted(allowed_actions))
            return (
                False,
                f"Action '{act}' is not permitted for capability '{cap}'. Allowed actions: {allowed_act_str}.",
            )

        # 4. Parameters type check
        if not isinstance(params, dict):
            return False, f"Parameters must be a dictionary, got {type(params).__name__}."

        # 5. Parameter constraints per capability and action
        if cap == "web":
            if act == "search":
                q = params.get("query")
                if not q or not isinstance(q, str) or not q.strip():
                    return False, "Web search action requires a non-empty 'query' string parameter."
            elif act == "fetch":
                url = params.get("url")
                if not url or not isinstance(url, str) or not url.strip():
                    return False, "Web fetch action requires a non-empty 'url' parameter."
                clean_url = url.strip().lower()
                if not (clean_url.startswith("http://") or clean_url.startswith("https://")):
                    return False, f"Web fetch requires a valid HTTP/HTTPS URL, got '{url}'."
            elif act == "research":
                obj = params.get("query") or params.get("objective")
                if not obj or not isinstance(obj, str) or not obj.strip():
                    return False, "Web research action requires a non-empty 'query' or 'objective' parameter."

        elif cap == "memory":
            if act in ("save", "remember"):
                key = params.get("key")
                val = params.get("value")
                if not key or not isinstance(key, str) or not key.strip():
                    return False, "Memory save requires a non-empty 'key' parameter."
                if val is None or (isinstance(val, str) and not val.strip()):
                    return False, "Memory save requires a non-empty 'value' parameter."
            elif act == "forget":
                key = params.get("key")
                if not key or not isinstance(key, str) or not key.strip():
                    return False, "Memory forget requires a non-empty 'key' parameter."
            elif act in ("read", "recall"):
                # key is optional (if empty/omitted, lists all preferences)
                pass

        elif cap == "knowledge":
            if act in ("query", "retrieve", "search", "retrieve knowledge", "retrieve_knowledge"):
                q = params.get("query")
                if not q or not isinstance(q, str) or not q.strip():
                    return False, "Knowledge query requires a non-empty 'query' string parameter."

        return True, None

    def _check_authorization(
        self,
        tool_call: ToolCall,
        context: Optional[PolicyContext] = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        Authorization check helper using the policy engine.
        Returns:
            (is_authorized, reason_if_not)
        """
        if self.policy_engine is None:
            return True, None

        effective_context = context or PolicyContext.from_tool_call(tool_call)
        try:
            res = self.policy_engine.evaluate(tool_call, effective_context)
            if res.is_allowed:
                return True, None
            return False, f"[{res.rule_id}] {res.reason}"
        except Exception as e:
            return False, f"Policy evaluation error: {e}. Default deny enforced."

    def execute(
        self,
        tool_call: Union[ToolCall, Dict[str, Any]],
        context: Optional[PolicyContext] = None,
    ) -> Result:
        """
        Orchestrate validation, policy evaluation, resolution, and execution of a ToolCall.

        Args:
            tool_call: ToolCall instance or dict representation.
            context: Optional PolicyContext for authorization and autonomy checks.

        Returns:
            Result object carrying success status, output, structured data,
            and correlation metadata (call_id, capability, action).
        """
        start_time = time.perf_counter()

        # Handle dict input safely
        if isinstance(tool_call, dict):
            try:
                tool_call = ToolCall.from_dict(tool_call)
            except Exception as e:
                return Result.fail(
                    message=f"Malformed ToolCall dictionary: {e}",
                    call_id=tool_call.get("call_id") if isinstance(tool_call, dict) else None,
                )

        if not isinstance(tool_call, ToolCall):
            return Result.fail(
                message=f"Expected ToolCall or dict, got {type(tool_call).__name__}.",
            )

        call_id = tool_call.call_id
        cap = tool_call.capability
        act = tool_call.action

        # Step 1: Validation
        is_valid, val_err = self.validate_call(tool_call)
        if not is_valid:
            return Result.fail(
                message=val_err or "Validation failed.",
                capability=cap,
                action=act,
                call_id=call_id,
            )

        # Step 2: Policy & Permission Evaluation
        if self.policy_engine is not None:
            effective_context = context
            if effective_context is None:
                effective_context = PolicyContext.from_tool_call(tool_call)

            try:
                policy_result = self.policy_engine.evaluate(tool_call, effective_context)
            except Exception as e:
                # Default-deny guarantee on policy engine failure
                return Result.fail(
                    message=f"Policy evaluation error: {e}. Execution denied.",
                    capability=cap,
                    action=act,
                    call_id=call_id,
                    data={"policy_error": str(e), "decision": PolicyDecision.DENY.value},
                )

            if policy_result.decision == PolicyDecision.DENY:
                return Result.fail(
                    message=f"Execution denied by policy [{policy_result.rule_id}]: {policy_result.reason}",
                    capability=cap,
                    action=act,
                    call_id=call_id,
                    data={"policy_result": policy_result.to_dict()},
                )

            if policy_result.decision == PolicyDecision.ASK_PERMISSION:
                return Result(
                    success=False,
                    message=f"Permission required for '{cap}.{act}' [{policy_result.rule_id}]: {policy_result.reason}",
                    output=None,
                    data={
                        "policy_result": policy_result.to_dict(),
                        "requires_permission": True,
                        "explanation": policy_result.explanation,
                    },
                    capability=cap,
                    action=act,
                    call_id=call_id,
                )

            if policy_result.decision == PolicyDecision.REQUIRE_CONFIRMATION:
                return Result(
                    success=False,
                    message=f"Confirmation required for '{cap}.{act}' [{policy_result.rule_id}]: {policy_result.reason}",
                    output=None,
                    data={
                        "policy_result": policy_result.to_dict(),
                        "requires_confirmation": True,
                        "explanation": policy_result.explanation,
                    },
                    capability=cap,
                    action=act,
                    call_id=call_id,
                )

        # Step 3: Capability Resolution
        capability_callable = self.registry.get_executor(cap) or self.registry.get(cap)
        if capability_callable is None:
            return Result.fail(
                message=f"Capability '{cap}' is permitted but not registered in CapabilityRegistry.",
                capability=cap,
                action=act,
                call_id=call_id,
            )

        # Step 4: Parameter normalization and execution
        task_params = dict(tool_call.parameters)
        task_params["action"] = act
        if call_id:
            task_params["call_id"] = call_id
        if tool_call.reason:
            task_params["reason"] = tool_call.reason

        # Wrap as standard Task for capability protocol compatibility
        task_id = 1
        if call_id and call_id.isdigit():
            task_id = int(call_id)

        task = Task(
            id=task_id,
            type=cap,
            action=act,
            tool=cap,
            parameters=task_params,
            status="pending",
        )

        try:
            raw_result = self.executor.execute(capability_callable, task)

            return Result(
                success=raw_result.success,
                message=raw_result.message,
                output=raw_result.output,
                data=raw_result.data,
                capability=cap,
                action=act,
                call_id=call_id,
            )

        except Exception as e:
            return Result.fail(
                message=f"Capability execution failure in '{cap}.{act}': {e}",
                capability=cap,
                action=act,
                call_id=call_id,
            )

    def execute_task(self, task: Task, context: Optional[PolicyContext] = None) -> Result:
        """
        Execute a legacy Task object through the orchestrator.
        Translates Task to ToolCall and delegates to execute().
        """
        try:
            tool_call = ToolCall.from_task(task)
            return self.execute(tool_call, context=context)
        except Exception as e:
            return Result.fail(
                message=f"Failed to translate Task to ToolCall: {e}",
            )
