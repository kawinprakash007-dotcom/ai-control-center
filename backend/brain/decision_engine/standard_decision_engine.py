import re
from datetime import datetime
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
        "explorer",
        "time",
    }

    def __init__(self, demo_mode: Optional[bool] = None):
        """
        Initialize StandardDecisionEngine.

        Args:
            demo_mode: Optional boolean flag for demo mode execution. If None, checks settings.
        """
        if demo_mode is None:
            try:
                from config.settings import get_settings
                self.demo_mode = get_settings().demo_mode
            except Exception:
                self.demo_mode = False
        else:
            self.demo_mode = bool(demo_mode)

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

        # 1b. Live State Authority Precedence (Phase 6.6)
        # Device, WorldState, Situation, Mission queries take precedence over RAG and Web.
        classification_raw = request.parameters.get("classification")
        if not classification_raw:
            from brain.request_understanding.standard_understanding import StandardRequestUnderstanding
            u = StandardRequestUnderstanding()
            cls_enum, cls_hints = u._classify_request(request.normalized_text, request.parameters)
            classification_raw = cls_enum.value
            for k, v in cls_hints.items():
                if k not in request.parameters:
                    request.parameters[k] = v

        if classification_raw == "CURRENT_DEVICE_STATE":
            target_device = request.parameters.get("target_device") or "drone"
            query_type = request.parameters.get("query_type") or "status"
            routing_hints = self._build_routing_hints(
                request,
                {
                    "target_device": target_device,
                    "query_type": query_type,
                    "live_query": True,
                },
            )
            return Decision(
                request_id=request.id,
                primary_goal="query_device_state",
                required_capabilities=[CapabilityType.DEVICE],
                execution_mode=ExecutionMode.SINGLE_STEP,
                confidence=0.98,
                reasoning=f"Authoritative live device state query for '{target_device}' ({query_type}).",
                routing_hints=routing_hints,
            )

        if classification_raw == "CURRENT_WORLD_STATE":
            routing_hints = self._build_routing_hints(
                request,
                {
                    "query_type": "world_state",
                    "live_query": True,
                },
            )
            return Decision(
                request_id=request.id,
                primary_goal="query_world_state",
                required_capabilities=[CapabilityType.DEVICE],
                execution_mode=ExecutionMode.SINGLE_STEP,
                confidence=0.98,
                reasoning="Authoritative ATLAS WorldState query.",
                routing_hints=routing_hints,
            )

        if classification_raw == "CURRENT_SITUATION":
            routing_hints = self._build_routing_hints(
                request,
                {
                    "query_type": "situation",
                    "live_query": True,
                },
            )
            return Decision(
                request_id=request.id,
                primary_goal="query_situation_state",
                required_capabilities=[CapabilityType.DEVICE],
                execution_mode=ExecutionMode.SINGLE_STEP,
                confidence=0.98,
                reasoning="Authoritative active situation intelligence query.",
                routing_hints=routing_hints,
            )

        if classification_raw == "CURRENT_MISSION":
            routing_hints = self._build_routing_hints(
                request,
                {
                    "query_type": "mission",
                    "live_query": True,
                },
            )
            return Decision(
                request_id=request.id,
                primary_goal="query_mission_state",
                required_capabilities=[CapabilityType.DEVICE],
                execution_mode=ExecutionMode.SINGLE_STEP,
                confidence=0.98,
                reasoning="Authoritative active mission and goal query.",
                routing_hints=routing_hints,
            )

        # 2. Capability Detection
        text = request.normalized_text
        detected_caps: List[CapabilityType] = []
        extra_hints: Dict[str, Any] = {}

        # Memory detection (evaluated FIRST so personal recall precedes generic knowledge and web)
        is_memory, mem_hints = self._detect_memory(request)
        if is_memory:
            detected_caps.append(CapabilityType.MEMORY)
            if mem_hints:
                extra_hints.update(mem_hints)

        # Web detection (evaluated AFTER memory and BEFORE knowledge/tool so current/external queries route to WEB)
        is_web, web_hints = self._detect_web(request)
        if not is_memory and is_web:
            detected_caps.append(CapabilityType.WEB)
            if web_hints:
                extra_hints.update(web_hints)

        # Computer detection (evaluated BEFORE generic tool so computer-use and desktop requests route safely)
        is_computer, comp_hints = self._detect_computer(request)
        if is_computer:
            detected_caps.append(CapabilityType.TOOL)
            if comp_hints:
                extra_hints.update(comp_hints)

        # Tool detection
        if not is_computer:
            is_tool, tool_hint = self._detect_tool(request)
            if is_tool:
                detected_caps.append(CapabilityType.TOOL)
                if tool_hint:
                    extra_hints["tool_hint"] = tool_hint

        # Knowledge detection
        if not is_memory and not is_web:
            is_knowledge, query_hint = self._detect_knowledge(request)
            if is_knowledge:
                detected_caps.append(CapabilityType.KNOWLEDGE)
                if query_hint:
                    extra_hints["query"] = query_hint
        elif is_memory:
            if re.search(r"\b(search for|find|retrieve|look up|read docs|documentation|documents?)\b", text, re.IGNORECASE):
                is_knowledge, query_hint = self._detect_knowledge(request)
                if is_knowledge and CapabilityType.KNOWLEDGE not in detected_caps:
                    detected_caps.append(CapabilityType.KNOWLEDGE)
                    if query_hint:
                        extra_hints["query"] = query_hint

        # Vision detection
        if not (is_computer and comp_hints.get("action") == "screenshot"):
            if re.search(r"\b(screenshot|camera|image|picture|screen\s*capture)\b", text, re.IGNORECASE):
                detected_caps.append(CapabilityType.VISION)

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
                if extra_hints.get("tool_hint") == "computer_app":
                    app_id = extra_hints.get("app_id", "vscode")
                    action = extra_hints.get("action", "launch")
                    return Decision(
                        request_id=request.id,
                        primary_goal="computer_app",
                        required_capabilities=[CapabilityType.TOOL],
                        execution_mode=ExecutionMode.SINGLE_STEP,
                        confidence=0.98,
                        reasoning=f"Controlled demonstration application execution for '{app_id}' ({action}).",
                        routing_hints=routing_hints,
                    )
                if extra_hints.get("tool_hint") == "computer":
                    return Decision(
                        request_id=request.id,
                        primary_goal="computer_action",
                        required_capabilities=[CapabilityType.TOOL],
                        execution_mode=ExecutionMode.SINGLE_STEP,
                        confidence=0.95,
                        reasoning=f"Computer interaction request identified (action: '{extra_hints.get('action', 'unspecified')}').",
                        routing_hints=routing_hints,
                    )
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
                action = extra_hints.get("action") or routing_hints.get("action", "search")
                primary_goal = "web_research" if action == "research" else "web_search"
                reasoning = (
                    "Web research workflow identified."
                    if action == "research"
                    else "Web exploration query identified."
                )
                return Decision(
                    request_id=request.id,
                    primary_goal=primary_goal,
                    required_capabilities=[CapabilityType.WEB],
                    execution_mode=ExecutionMode.SINGLE_STEP,
                    confidence=0.90,
                    reasoning=reasoning,
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
        text = request.normalized_text

        # ATLAS live state queries must NEVER route to Knowledge/RAG
        if re.search(
            r"\b(?:drone|rover|vision|glass|devices?|battery|telemetry|world\s*state|situations?\s+are\s+active|missions?\s+are\s+active|goals?\s+are\s+running)\b",
            text,
            re.IGNORECASE,
        ) and not re.search(r"\b(?:documentation|docs?|manual|specification|spec|paper)\b", text, re.IGNORECASE):
            return False, None

        # Explicit query parameter takes precedence
        if "query" in request.parameters:
            return True, str(request.parameters["query"])

        knowledge_patterns = [
            r"\b(search for|find|retrieve|look up|read docs|documentation|documents?|paper|uploaded)\b",
            r"\b(what is|who is|what does|how does|tell me about|explain|kernel|linux|architecture)\b",
        ]
        for pat in knowledge_patterns:
            if re.search(pat, text, re.IGNORECASE):
                return True, text

        return False, None

    def _detect_computer(self, request: Request) -> tuple[bool, Dict[str, Any]]:
        text = request.normalized_text.strip()
        hints: Dict[str, Any] = {}

        # 0. Bounded Demo Application Execution (DEMO_MODE only)
        if self.demo_mode:
            app_id = request.parameters.get("demo_app_id")
            app_act = request.parameters.get("demo_app_action", "launch")

            if not app_id:
                app_match = re.search(
                    r"\b(?:open|launch|start|run|close|focus|switch\s+to)\s+(?:the\s+|teh\s+)?(vs\s*code|vscode|visual\s+studio\s+code|chrome|google\s+chrome|notepad)\b",
                    text,
                    re.IGNORECASE,
                )
                if app_match:
                    raw_app = app_match.group(1).lower()
                    if any(v in raw_app for v in ("vscode", "visual studio", "vs code")):
                        app_id = "vscode"
                    elif "chrome" in raw_app:
                        app_id = "chrome"
                    elif "notepad" in raw_app:
                        app_id = "notepad"

                    if re.search(r"\bclose\b", text, re.IGNORECASE):
                        app_act = "close"
                    elif re.search(r"\b(?:focus|switch\s+to)\b", text, re.IGNORECASE):
                        app_act = "focus"
                    else:
                        app_act = "launch"

            cleaned = text.strip().lower()
            if not app_id and cleaned in ("vs code", "vscode", "visual studio code"):
                app_id = "vscode"
                app_act = "launch"
            elif not app_id and cleaned in ("chrome", "google chrome"):
                app_id = "chrome"
                app_act = "launch"
            elif not app_id and cleaned in ("notepad",):
                app_id = "notepad"
                app_act = "launch"

            if app_id in ("vscode", "chrome", "notepad"):
                display_names = {
                    "vscode": "Visual Studio Code",
                    "chrome": "Google Chrome",
                    "notepad": "Notepad",
                }
                hints["tool_hint"] = "computer_app"
                hints["action"] = app_act
                hints["app_id"] = app_id
                hints["target"] = app_id
                hints["app_name"] = display_names.get(app_id, app_id)
                return True, hints

        # 1. Desktop Application Requests (e.g. "open VS Code", "open the vs code in my pc", "start Visual Studio Code")
        vscode_app_match = re.search(
            r"\b(?:open|launch|start|run|close|switch\s+to)\s+(?:the\s+|teh\s+)?(?:vs\s*code|vscode|visual\s+studio\s+code)\b",
            text,
            re.IGNORECASE,
        )
        if vscode_app_match:
            hints["tool_hint"] = "computer"
            hints["action"] = "open_app"
            hints["target"] = "vscode"
            hints["app_name"] = "Visual Studio Code"
            return True, hints

        # PC/Computer context with open/launch/start verbs (e.g. "open ... on my pc", "launch ... on desktop")
        pc_context_match = re.search(
            r"\b(?:open|launch|start|run)\s+(?:the\s+|teh\s+)?([a-zA-Z0-9_\-\.\s]+?)\s+(?:in|on)\s+(?:my\s+)?(?:pc|computer|desktop|machine|laptop)\b",
            text,
            re.IGNORECASE,
        )
        if pc_context_match:
            target_raw = pc_context_match.group(1).strip().lower()
            target = "vscode" if any(v in target_raw for v in ("vs code", "vscode", "visual studio")) else target_raw
            hints["tool_hint"] = "computer"
            hints["action"] = "open_app"
            hints["target"] = target
            hints["app_name"] = target
            return True, hints

        # Standalone VS Code command
        cleaned = text.strip().lower()
        if cleaned in ("vs code", "vscode", "visual studio code"):
            hints["tool_hint"] = "computer"
            hints["action"] = "open_app"
            hints["target"] = "vscode"
            hints["app_name"] = "Visual Studio Code"
            return True, hints

        # 2. Direct GUI Actions matching ComputerAction
        if re.search(r"\bdouble[\s\-_]click\b", text, re.IGNORECASE):
            hints["tool_hint"] = "computer"
            hints["action"] = "double_click"
            return True, hints
        if re.search(r"\b(?:left\s+|right\s+)?click\b", text, re.IGNORECASE):
            hints["tool_hint"] = "computer"
            hints["action"] = "click"
            return True, hints
        if re.search(r"\bmove\s+(?:cursor|mouse)\b", text, re.IGNORECASE):
            hints["tool_hint"] = "computer"
            hints["action"] = "move"
            return True, hints
        if re.search(r"\btype\s+(?:text|keystroke)", text, re.IGNORECASE):
            hints["tool_hint"] = "computer"
            hints["action"] = "type"
            return True, hints
        if re.search(r"\bpress\s+(?:key\s+)?(?:enter|esc|escape|tab|space|backspace|delete)\b", text, re.IGNORECASE):
            hints["tool_hint"] = "computer"
            hints["action"] = "press_key"
            return True, hints
        if re.search(r"\bscroll\s+(?:up|down|left|right)\b", text, re.IGNORECASE):
            hints["tool_hint"] = "computer"
            hints["action"] = "scroll"
            return True, hints
        if re.search(r"\bwait\s+(?:for\s+)?\d+\s*(?:s|sec|seconds?)\b", text, re.IGNORECASE):
            hints["tool_hint"] = "computer"
            hints["action"] = "wait"
            return True, hints
        comp_action_match = re.search(
            r"\bcomputer\s+(screenshot|click|double_click|move|type|press_key|scroll|wait)\b",
            text,
            re.IGNORECASE,
        )
        if comp_action_match:
            hints["tool_hint"] = "computer"
            hints["action"] = comp_action_match.group(1).lower()
            return True, hints

        return False, hints

    def _detect_tool(self, request: Request) -> tuple[bool, Optional[str]]:
        text = request.normalized_text

        # 1. Action verb followed by target in KNOWN_TOOLS: "open calculator", "launch notepad"
        action_match = re.search(
            r"\b(?:open|launch|start|run|close|execute)\s+([a-zA-Z0-9_\-\.]+)",
            text,
            re.IGNORECASE,
        )
        if action_match:
            candidate = action_match.group(1).lower().strip()
            if candidate in self.KNOWN_TOOLS:
                return True, candidate

        # 2. Standalone tool command (e.g. "calculator", "notepad")
        cleaned = text.strip().lower()
        if cleaned in self.KNOWN_TOOLS:
            return True, cleaned

        # 3. Explicit path parameter present (ensure it is not derived from a URL)
        if "path" in request.parameters:
            path_val = str(request.parameters["path"])
            if not re.search(r"https?://", text, re.IGNORECASE) and not path_val.startswith("http"):
                return True, path_val

        return False, None

    def _detect_web(self, request: Request) -> tuple[bool, Dict[str, Any]]:
        text = request.normalized_text.strip()
        orig = request.original_text.strip()
        hints: Dict[str, Any] = {}

        # 0. Conversational greeting check: pure greetings should stay CHAT
        if re.search(
            r"^(?:hello|hi|hey|good\s+morning|good\s+afternoon|good\s+evening|how\s+are\s+you)\b",
            text,
            re.IGNORECASE,
        ) and not re.search(
            r"\b(?:weather|news|price|prices|stock|stocks|score|scores|search|browse|latest|release|version|online|web|match|results?)\b",
            text,
            re.IGNORECASE,
        ):
            return False, hints

        # 1. Explicit request parameters / routing hints
        if "url" in request.parameters:
            hints["action"] = "fetch"
            hints["url"] = str(request.parameters["url"])
            return True, hints

        if request.parameters.get("action") in ("search", "fetch", "research"):
            action = str(request.parameters["action"]).strip().lower()
            hints["action"] = action
            if action == "fetch" and "url" in request.parameters:
                hints["url"] = str(request.parameters["url"])
            elif action in ("search", "research"):
                hints["query"] = str(request.parameters.get("query") or request.parameters.get("objective") or orig)
                if action == "research":
                    hints["objective"] = hints["query"]
            return True, hints

        if request.parameters.get("capability") == "web":
            hints["action"] = "search"
            hints["query"] = str(request.parameters.get("query") or orig)
            return True, hints

        # 2. URL detection -> action = "fetch"
        url_match = re.search(r"https?://[^\s<>\"']+", orig)
        if url_match:
            hints["action"] = "fetch"
            hints["url"] = url_match.group(0).rstrip(".,;!?")
            return True, hints

        # 3. Knowledge / Local Document indicators take precedence over current-info signals
        # (e.g. "What does my uploaded paper say about the latest research?" -> KNOWLEDGE)
        has_local_doc_signal = bool(
            re.search(
                r"\b(?:uploaded|paper|my\s+doc(?:s|umentation)?|my\s+notes?|my\s+file(?:s)?|local\s+file(?:s)?|internal\s+docs?)\b",
                text,
                re.IGNORECASE,
            )
        )

        # 3.5 Research Directives
        research_patterns = [
            r"\bresearch\b",
            r"\bcompare\s+(?:several\s+)?sources\b",
            r"\bdeep\s+search\b",
            r"\binvestigate\s+online\b",
            r"\bmulti[- ]source\s+(?:search|research)\b",
        ]
        has_research_directive = any(
            re.search(pat, text, re.IGNORECASE) for pat in research_patterns
        )
        if has_research_directive and not has_local_doc_signal:
            hints["action"] = "research"
            extracted = self._extract_web_query(orig)
            hints["query"] = extracted
            hints["objective"] = extracted
            return True, hints

        # 4. Explicit Web / Online Directives
        explicit_web_patterns = [
            r"\bsearch\s+(?:the\s+)?(?:web|internet|online)\b",
            r"\b(?:search|look|find|check)\s+(?:it\s+|this\s+)?up\s+online\b",
            r"\b(?:look|check)\s+online\b",
            r"\bfind\s+online\b",
            r"\blook\s+for\s+current\s+information\b",
            r"\bcheck\s+the\s+latest\b",
            r"\bweb\s*search\b",
            r"\b(?:browse\s+the\s+web|browse\s+online)\b",
            r"\b(?:look\s+up\s+online|search\s+online)\b",
            r"\b(?:look\s+up|search\s+for|find|check)\s+.+?\s+(?:on\s+the\s+web|online|on\s+the\s+internet)\b",
            r"\bgoogle(?:\s+for)?\b",
        ]
        has_explicit_web_directive = any(
            re.search(pat, text, re.IGNORECASE) for pat in explicit_web_patterns
        )

        if has_explicit_web_directive:
            hints["action"] = "search"
            hints["query"] = self._extract_web_query(orig)
            return True, hints

        # If it has local document signals and no explicit web directive, let Knowledge handle it
        if has_local_doc_signal:
            return False, hints

        # ATLAS live state queries must NEVER route to Web search
        if re.search(
            r"\b(?:world\s*state|what\s+does\s+atlas|entities\s+present|where\s+are\s+the\s+devices|drone|rover|vision|glass|devices|situations?\s+are\s+active|missions?\s+are\s+active|goals?\s+are\s+running)\b",
            text,
            re.IGNORECASE,
        ):
            return False, hints

        # 5. Current-Information Signals
        current_year = str(datetime.now().year)
        current_signals = [
            r"\blatest\b",
            r"\bcurrent\b",
            r"\btoday\b",
            r"\btonight\b",
            r"\byesterday\b",
            r"\brecent\b",
            r"\brecently\b",
            r"\bthis\s+week\b",
            r"\bthis\s+month\b",
            rf"\b{current_year}\b",
            r"\b202[0-9]\b",
            r"\bprices?\b",
            r"\bnews\b",
            r"\bweather\b",
            r"\bscores?\b",
            r"\bresults?\b",
            r"\bmarket\b",
            r"\bstocks?\b",
            r"\bavailability\b",
            r"\brelease\b",
            r"\bversions?\b",
            r"\bupdated\b",
            r"\bupdate\b",
            r"\bwhat\s+happened\b",
        ]
        has_current_signal = any(
            re.search(pat, text, re.IGNORECASE) for pat in current_signals
        )

        if has_current_signal:
            hints["action"] = "search"
            hints["query"] = self._extract_web_query(orig)
            return True, hints

        return False, hints

    def _extract_web_query(self, orig: str) -> str:
        q = orig.strip()

        # Remove leading command / question wrappers
        leading_wrappers = [
            r"^(?:please\s+)?research\s+(?:the\s+)?(?:latest\s+)?(?:about\s+|on\s+)?",
            r"^(?:please\s+)?research\s+",
            r"^(?:please\s+)?investigate\s+(?:the\s+)?(?:latest\s+)?(?:about\s+|on\s+)?",
            r"^(?:please\s+)?investigate\s+",
            r"^(?:please\s+)?compare\s+(?:several\s+)?sources\s+(?:for|about|on)\s+",
            r"^(?:please\s+)?deep\s+search\s+(?:for|about|on)\s+",
            r"^(?:please\s+)?search\s+(?:the\s+)?(?:web|internet|online)\s+(?:for|about)\s+",
            r"^(?:please\s+)?search\s+(?:the\s+)?(?:web|internet|online)\s+",
            r"^(?:please\s+)?search\s+for\s+",
            r"^(?:please\s+)?search\s+online\s+(?:for|about)\s+",
            r"^(?:please\s+)?look\s+(?:this|it)\s+up\s+online\s+(?:for|about)?\s*",
            r"^(?:please\s+)?look\s+up\s+online\s+(?:for|about)\s+",
            r"^(?:please\s+)?look\s+up\s+",
            r"^(?:please\s+)?look\s+for\s+current\s+information\s+(?:about|on|for)\s+",
            r"^(?:please\s+)?look\s+for\s+",
            r"^(?:please\s+)?find\s+online\s+(?:for|about)\s+",
            r"^(?:please\s+)?find\s+",
            r"^(?:please\s+)?check\s+online\s+(?:for|about)\s+",
            r"^(?:please\s+)?check\s+the\s+latest\s+(?:on|for|about)\s+",
            r"^(?:please\s+)?check\s+",
            r"^(?:please\s+)?browse\s+(?:the\s+web\s+for\s+|online\s+for\s+)?",
            r"^(?:please\s+)?google\s+(?:for\s+)?",
            r"^(?:what\s+is\s+the\s+|what\'s\s+the\s+|what\s+are\s+the\s+|what\s+is\s+|what\'s\s+|what\s+are\s+)",
        ]
        for pat in leading_wrappers:
            m = re.match(pat, q, re.IGNORECASE)
            if m:
                q = q[m.end():].strip()
                break

        # Remove trailing online / web markers
        trailing_wrappers = [
            r"\s+and\s+compare\s+(?:several\s+)?sources[.!?]?$",
            r"\s+comparing\s+(?:several\s+)?sources[.!?]?$",
            r"\s+across\s+(?:several|multiple)\s+sources[.!?]?$",
            r"\s+from\s+(?:several|multiple)\s+sources[.!?]?$",
            r"\s+(?:on\s+the\s+web|online|on\s+the\s+internet)[.!?]?$",
        ]
        for pat in trailing_wrappers:
            q = re.sub(pat, "", q, flags=re.IGNORECASE).strip()

        # Remove trailing punctuation (e.g. '?', '.', '!')
        q = q.rstrip("?.!").strip()

        # Fallback if query was stripped completely
        if not q:
            return orig.rstrip("?.!").strip()

        return q

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
        for key in (
            "query",
            "url",
            "path",
            "collection",
            "author",
            "action",
            "objective",
            "max_iterations",
            "max_searches",
            "max_fetches",
            "min_evidence",
            "queries",
        ):
            if key in request.parameters:
                if key == "path" and (re.search(r"https?://", request.normalized_text, re.IGNORECASE) or "url" in extra):
                    continue
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
