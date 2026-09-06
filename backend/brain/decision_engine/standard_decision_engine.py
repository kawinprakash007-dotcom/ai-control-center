import re
from typing import Any, Dict, List, Optional

from core.interfaces.decision_engine_interface import DecisionEngineInterface
from core.models.request import Request
from core.models.decision import (
    Decision,
    CapabilityType,
    ExecutionMode,
)


class StandardDecisionEngine(DecisionEngineInterface):
    """
    Deterministic implementation of DecisionEngineInterface.
    Evaluates a Request and determines canonical primary_goal, required capabilities,
    execution mode, deterministic confidence score, reasoning, and routing hints.
    """

    KNOWN_TOOLS = {
        "calculator",
        "notepad",
        "chrome",
        "vscode",
        "explorer",
        "terminal",
        "cmd",
        "powershell",
        "time",
    }

    def decide(self, request: Request) -> Decision:
        if not isinstance(request, Request):
            raise TypeError(
                f"StandardDecisionEngine.decide expects a Request instance, got {type(request).__name__}"
            )

        # 1. Quality & Ambiguity Evaluation
        if request.parameters.get("is_empty"):
            return Decision(
                request_id=request.id,
                primary_goal="prompt_user_input",
                required_capabilities=[CapabilityType.CHAT],
                execution_mode=ExecutionMode.DIRECT,
                confidence=1.0,
                reasoning="Empty input requires prompting user for instructions.",
                routing_hints=self._build_routing_hints(request, {"is_empty": True}),
            )

        if request.parameters.get("is_ambiguous"):
            ambiguity_reason = request.parameters.get(
                "ambiguity_reason", "Request is ambiguous or incomplete"
            )
            return Decision(
                request_id=request.id,
                primary_goal="clarify_request",
                required_capabilities=[CapabilityType.CHAT],
                execution_mode=ExecutionMode.DIRECT,
                confidence=0.95,
                reasoning=f"Ambiguous input requires clarification: {ambiguity_reason}",
                routing_hints=self._build_routing_hints(
                    request, {"ambiguity_reason": ambiguity_reason}
                ),
            )

        # 2. Capability Detection
        text = request.normalized_text
        detected_caps: List[CapabilityType] = []
        extra_hints: Dict[str, Any] = {}

        # Memory detection (evaluated FIRST so personal recall precedes generic knowledge)
        is_memory, mem_hints = self._detect_memory(request)
        if is_memory:
            detected_caps.append(CapabilityType.MEMORY)
            if mem_hints:
                extra_hints.update(mem_hints)

        # Tool detection
        is_tool, tool_hint = self._detect_tool(request)
        if is_tool:
            detected_caps.append(CapabilityType.TOOL)
            if tool_hint:
                extra_hints["tool_hint"] = tool_hint

        # Knowledge detection
        if not is_memory:
            is_knowledge, query_hint = self._detect_knowledge(request)
            if is_knowledge:
                detected_caps.append(CapabilityType.KNOWLEDGE)
                if query_hint:
                    extra_hints["query"] = query_hint
        else:
            if re.search(r"\b(search for|find|retrieve|look up|read docs|documentation|documents?)\b", text, re.IGNORECASE):
                is_knowledge, query_hint = self._detect_knowledge(request)
                if is_knowledge and CapabilityType.KNOWLEDGE not in detected_caps:
                    detected_caps.append(CapabilityType.KNOWLEDGE)
                    if query_hint:
                        extra_hints["query"] = query_hint

        # Vision detection
        if re.search(r"\b(screenshot|camera|image|picture|screen\s*capture)\b", text, re.IGNORECASE):
            detected_caps.append(CapabilityType.VISION)

        # Web detection
        if re.search(r"\b(browse|website|web\s*search|google|url|http|https)\b", text, re.IGNORECASE):
            detected_caps.append(CapabilityType.WEB)

        # 3. Strategy & Goal Determination
        routing_hints = self._build_routing_hints(request, extra_hints)

        # Composite / Multi-step detection
        if len(detected_caps) > 1 or re.search(r"\b(and\s+then|after\s+that|followed\s+by)\b", text, re.IGNORECASE):
            if not detected_caps:
                detected_caps = [CapabilityType.CHAT]
            return Decision(
                request_id=request.id,
                primary_goal="execute_workflow",
                required_capabilities=detected_caps,
                execution_mode=ExecutionMode.MULTI_STEP,
                confidence=0.90,
                reasoning=f"Composite request requires multi-step workflow across capabilities: {[c.value for c in detected_caps]}.",
                routing_hints=routing_hints,
            )

        # Single-capability mapping
        if len(detected_caps) == 1:
            cap = detected_caps[0]
            if cap == CapabilityType.KNOWLEDGE:
                return Decision(
                    request_id=request.id,
                    primary_goal="retrieve_knowledge",
                    required_capabilities=[CapabilityType.KNOWLEDGE],
                    execution_mode=ExecutionMode.SINGLE_STEP,
                    confidence=0.95,
                    reasoning="Knowledge retrieval query identified from parameters and terminology.",
                    routing_hints=routing_hints,
                )
            if cap == CapabilityType.TOOL:
                return Decision(
                    request_id=request.id,
                    primary_goal="execute_tool",
                    required_capabilities=[CapabilityType.TOOL],
                    execution_mode=ExecutionMode.SINGLE_STEP,
                    confidence=0.95,
                    reasoning=f"Tool execution action identified (tool: '{extra_hints.get('tool_hint', 'unspecified')}').",
                    routing_hints=routing_hints,
                )
            if cap == CapabilityType.MEMORY:
                return Decision(
                    request_id=request.id,
                    primary_goal="manage_memory",
                    required_capabilities=[CapabilityType.MEMORY],
                    execution_mode=ExecutionMode.SINGLE_STEP,
                    confidence=0.95,
                    reasoning="Memory access or storage directive identified.",
                    routing_hints=routing_hints,
                )
            if cap == CapabilityType.VISION:
                return Decision(
                    request_id=request.id,
                    primary_goal="analyze_visual",
                    required_capabilities=[CapabilityType.VISION],
                    execution_mode=ExecutionMode.SINGLE_STEP,
                    confidence=0.90,
                    reasoning="Visual inspection or screenshot request identified.",
                    routing_hints=routing_hints,
                )
            if cap == CapabilityType.WEB:
                return Decision(
                    request_id=request.id,
                    primary_goal="web_search",
                    required_capabilities=[CapabilityType.WEB],
                    execution_mode=ExecutionMode.SINGLE_STEP,
                    confidence=0.90,
                    reasoning="Web exploration query identified.",
                    routing_hints=routing_hints,
                )

        # Default fallback to general chat
        return Decision(
            request_id=request.id,
            primary_goal="answer_chat",
            required_capabilities=[CapabilityType.CHAT],
            execution_mode=ExecutionMode.DIRECT,
            confidence=0.85,
            reasoning="General conversational dialogue or conversational fallback.",
            routing_hints=routing_hints,
        )

    def _detect_knowledge(self, request: Request) -> tuple[bool, Optional[str]]:
        # Explicit query parameter takes precedence
        if "query" in request.parameters:
            return True, str(request.parameters["query"])

        text = request.normalized_text
        knowledge_patterns = [
            r"\b(search for|find|retrieve|look up|read docs|documentation|documents?)\b",
            r"\b(what is|who is|how does|tell me about|explain|kernel|linux|architecture)\b",
        ]
        for pat in knowledge_patterns:
            if re.search(pat, text, re.IGNORECASE):
                return True, text

        return False, None

    def _detect_tool(self, request: Request) -> tuple[bool, Optional[str]]:
        text = request.normalized_text

        # 1. Action verb followed by target: "open calculator", "launch notepad"
        action_match = re.search(
            r"\b(?:open|launch|start|run|close|execute)\s+([a-zA-Z0-9_\-\.]+)",
            text,
            re.IGNORECASE,
        )
        if action_match:
            candidate = action_match.group(1).lower().strip()
            return True, candidate

        # 2. Standalone tool command (e.g. "calculator", "notepad")
        cleaned = text.strip().lower()
        if cleaned in self.KNOWN_TOOLS:
            return True, cleaned

        # 3. Explicit path parameter present
        if "path" in request.parameters:
            return True, str(request.parameters["path"])

        return False, None

    def _detect_memory(self, request: Request) -> tuple[bool, Dict[str, Any]]:
        text = request.normalized_text.strip()
        hints: Dict[str, Any] = {}

        # 1. SAVE / REMEMBER directives
        save_match = re.match(
            r"^(?:remember|save|store\s+preference|store)[:\s]+(?:that\s+)?(?:the\s+preference\s+that\s+)?(?:my\s+)?(.+?)\s+(?:is\s+called|is\s+named|is\s+set\s+to|is|was|to\s+be|as)\s+(.+?)[.!?]?$",
            text,
            re.IGNORECASE,
        )
        if save_match:
            raw_key = save_match.group(1).strip()
            val = save_match.group(2).strip()
            k = self._normalize_memory_key(raw_key)
            hints["action"] = "save"
            hints["memory_action"] = "save"
            hints["key"] = k
            hints["value"] = val
            hints["user_id"] = "default_user"
            return True, hints

        # Pattern 1b: explicit preference directives without linking verb: e.g. "store preference dark mode"
        pref_match = re.match(
            r"^(?:store\s+preference|save\s+preference|set\s+preference)[:\s]+(?:for\s+)?(.+?)[.!?]?$",
            text,
            re.IGNORECASE,
        )
        if pref_match:
            raw_key = pref_match.group(1).strip()
            k = self._normalize_memory_key(raw_key)
            hints["action"] = "save"
            hints["memory_action"] = "save"
            hints["key"] = k
            hints["value"] = raw_key
            hints["user_id"] = "default_user"
            return True, hints

        # 2. FORGET directives
        forget_match1 = re.match(
            r"^(?:forget|delete|remove|clear)\s+(?:that\s+)?(?:my\s+)?(.+?)\s+(?:is\s+called|is\s+named|is|was|to\s+be|as)\s+(.+?)[.!?]?$",
            text,
            re.IGNORECASE,
        )
        if forget_match1:
            raw_key = forget_match1.group(1).strip()
            k = self._normalize_memory_key(raw_key)
            hints["action"] = "forget"
            hints["memory_action"] = "forget"
            hints["key"] = k
            hints["user_id"] = "default_user"
            return True, hints

        forget_match2 = re.match(
            r"^(?:delete|remove|clear|forget)\s+(?:my\s+)?(?:saved\s+)?preference\s+for\s+(.+?)[.!?]?$",
            text,
            re.IGNORECASE,
        )
        if forget_match2:
            raw_key = forget_match2.group(1).strip()
            k = self._normalize_memory_key(raw_key)
            hints["action"] = "forget"
            hints["memory_action"] = "forget"
            hints["key"] = k
            hints["user_id"] = "default_user"
            return True, hints

        forget_match3 = re.match(
            r"^(?:forget|delete|remove|clear)\s+(?:my\s+)?(.+?)[.!?]?$",
            text,
            re.IGNORECASE,
        )
        if forget_match3:
            raw_key = forget_match3.group(1).strip()
            if raw_key.lower() not in self.KNOWN_TOOLS:
                k = self._normalize_memory_key(raw_key)
                hints["action"] = "forget"
                hints["memory_action"] = "forget"
                hints["key"] = k
                hints["user_id"] = "default_user"
                return True, hints

        # 3. READ directives
        if re.search(
            r"\b(?:what\s+do\s+you\s+(?:remember|know)\s+about\s+me|what\s+do\s+you\s+remember|what\s+memories\s+do\s+you\s+have|list\s+my\s+(?:memories|preferences)|show\s+my\s+(?:memories|preferences))\b",
            text,
            re.IGNORECASE,
        ):
            hints["action"] = "read"
            hints["memory_action"] = "read"
            hints["key"] = "all"
            hints["user_id"] = "default_user"
            return True, hints

        read_match1 = re.match(
            r"^(?:what\s+is|what\'s)\s+my\s+(.+?)[?.!]?$",
            text,
            re.IGNORECASE,
        )
        if read_match1:
            raw_key = read_match1.group(1).strip()
            k = self._normalize_memory_key(raw_key)
            hints["action"] = "read"
            hints["memory_action"] = "read"
            hints["key"] = k
            hints["user_id"] = "default_user"
            return True, hints

        read_match2 = re.match(
            r"^what\s+did\s+i\s+tell\s+you\s+(?:that\s+)?my\s+(.+?)\s+was[?.!]?$",
            text,
            re.IGNORECASE,
        )
        if read_match2:
            raw_key = read_match2.group(1).strip()
            k = self._normalize_memory_key(raw_key)
            hints["action"] = "read"
            hints["memory_action"] = "read"
            hints["key"] = k
            hints["user_id"] = "default_user"
            return True, hints

        read_match3 = re.match(
            r"^(?:recall|tell\s+me)\s+(?:my\s+)?(.+?)[?.!]?$",
            text,
            re.IGNORECASE,
        )
        if read_match3:
            raw_key = read_match3.group(1).strip()
            k = self._normalize_memory_key(raw_key)
            hints["action"] = "read"
            hints["memory_action"] = "read"
            hints["key"] = k
            hints["user_id"] = "default_user"
            return True, hints

        return False, hints

    def _normalize_memory_key(self, raw_key: str) -> str:
        k = raw_key.strip()
        k = re.sub(r"^(?:my|the|preference\s+for|saved\s+preference\s+for)\s+", "", k, flags=re.IGNORECASE)
        k = re.sub(r"[^\w\s-]", "", k).strip().lower()
        k = re.sub(r"[\s-]+", "_", k)
        return k

    def _build_routing_hints(
        self, request: Request, extra: Dict[str, Any]
    ) -> Dict[str, Any]:
        hints: Dict[str, Any] = {}

        # Safe parameter propagation
        for key in ("query", "path", "collection", "author"):
            if key in request.parameters:
                hints[key] = request.parameters[key]

        # Safe constraint propagation
        for key in (
            "max_results",
            "timeout_seconds",
            "privacy_level",
            "confirmation_required",
        ):
            if key in request.constraints:
                hints[key] = request.constraints[key]

        # Merge additional hints without mutating request
        if extra:
            hints.update(extra)

        return hints
