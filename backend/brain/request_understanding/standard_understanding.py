import re
import uuid
from datetime import datetime
from typing import Any, Dict, Tuple, Optional

from core.interfaces.request_understanding_interface import RequestUnderstandingInterface
from core.models.request import Request
from brain.request_understanding.normalizer import normalize_text
from core.models.live_context import ChatQueryClassification


class RequestUnderstandingError(ValueError):
    """Raised when input to RequestUnderstanding is malformed or invalid."""
    pass


class StandardRequestUnderstanding(RequestUnderstandingInterface):
    """
    Deterministic implementation of RequestUnderstandingInterface.
    Normalizes input text, extracts metadata, parameters, constraints, and quality flags.
    """

    def understand(self, input_data: Any) -> Request:
        if isinstance(input_data, str):
            original_text = input_data
            source = "chat"
            session_id = "default"
            explicit_params: Dict[str, Any] = {}
            explicit_constraints: Dict[str, Any] = {}
        elif isinstance(input_data, dict):
            # Text can be provided via 'text' or 'original_text'
            text = input_data.get("text")
            if text is None:
                text = input_data.get("original_text")

            if text is None or not isinstance(text, str):
                raise RequestUnderstandingError(
                    "Dictionary payload must contain 'text' or 'original_text' of type str."
                )

            original_text = text
            source = str(input_data.get("source", "chat"))
            session_id = str(input_data.get("session_id", "default"))

            raw_params = input_data.get("parameters", {})
            if not isinstance(raw_params, dict):
                raise RequestUnderstandingError("Payload 'parameters' must be a dictionary.")
            explicit_params = dict(raw_params)

            raw_constraints = input_data.get("constraints", {})
            if not isinstance(raw_constraints, dict):
                raise RequestUnderstandingError("Payload 'constraints' must be a dictionary.")
            explicit_constraints = dict(raw_constraints)
        else:
            raise RequestUnderstandingError(
                f"Unsupported input type '{type(input_data).__name__}'. Expected str or dict."
            )

        # 1. Normalization
        normalized_text = normalize_text(original_text)

        # 2. Parameters & Constraints Setup
        parameters = dict(explicit_params)
        constraints = dict(explicit_constraints)

        # 3. Assess Empty / Whitespace-only input
        if not normalized_text:
            parameters["is_empty"] = True
            parameters["valid"] = False
            parameters["is_ambiguous"] = True
            parameters["ambiguity_reason"] = "Input is empty or contains only whitespace"
            parameters["clarification_needed"] = True
        else:
            parameters["is_empty"] = False
            parameters["valid"] = True

            # 4. Detect obvious ambiguity or incompleteness without solving it
            is_ambiguous, reason = self._detect_ambiguity(normalized_text)
            if "is_ambiguous" not in parameters:
                parameters["is_ambiguous"] = is_ambiguous
            if "ambiguity_reason" not in parameters:
                parameters["ambiguity_reason"] = reason
            if "clarification_needed" not in parameters:
                parameters["clarification_needed"] = is_ambiguous

            # 5. Simple deterministic extraction
            self._extract_deterministic_attributes(normalized_text, parameters, constraints)

            # 5b. Deterministic Query Classification (Phase 6.6)
            classification, class_hints = self._classify_request(normalized_text, parameters)
            parameters["classification"] = classification.value
            for k, v in class_hints.items():
                if k not in parameters:
                    parameters[k] = v

        # 6. Request Construction
        request_id = f"req-{uuid.uuid4().hex[:12]}"
        timestamp = datetime.now()

        return Request(
            id=request_id,
            original_text=original_text,
            normalized_text=normalized_text,
            session_id=session_id,
            timestamp=timestamp,
            source=source,
            parameters=parameters,
            constraints=constraints,
        )

    def _detect_ambiguity(self, text: str) -> Tuple[bool, Optional[str]]:
        # Only punctuation or ellipsis
        if re.fullmatch(r"[\?\.\!\,\;\:\-\_\s]+", text):
            return True, "Input contains only punctuation"

        # Incomplete command prefixes where argument is missing
        incomplete_patterns = [
            r"^(search for|find|look up|show me|open|run|delete|read|display)\s*$",
            r"^(what is|who is|where is|how to)\s*$",
        ]
        for pattern in incomplete_patterns:
            if re.match(pattern, text, re.IGNORECASE):
                return True, "Incomplete query missing subject or argument"

        # Highly ambiguous single-word deictic or vague references
        vague_tokens = {"it", "that", "this", "do it", "run that", "again"}
        if text.lower() in vague_tokens:
            return True, f"Vague or deictic reference without context: '{text}'"

        return False, None

    def _extract_deterministic_attributes(
        self,
        text: str,
        parameters: Dict[str, Any],
        constraints: Dict[str, Any],
    ) -> None:
        # File / Directory paths (Windows C:\... or Unix /... or relative ./...)
        if "path" not in parameters:
            path_match = re.search(
                r'(?:[a-zA-Z]:\\(?:[^\\/:*?"<>|\r\n\s]+\\)*[^\\/:*?"<>|\r\n\s]+|(?:\.{1,2})?/(?:[^/:\s]+/)*[^/:\s]+)',
                text
            )
            if path_match:
                candidate = path_match.group(0).strip()
                if len(candidate) > 2:
                    parameters["path"] = candidate

        # Query extraction (e.g. 'search for <X>' or 'query: <X>' or 'find <X>')
        if "query" not in parameters:
            query_match = re.search(
                r'(?:search for|query:|find)\s+["\']?([^"\']+)["\']?',
                text,
                re.IGNORECASE
            )
            if query_match:
                q = query_match.group(1).strip()
                if q:
                    parameters["query"] = q

        # max_results / top_k extraction (e.g. "limit 10", "top 5", "max 20")
        if "max_results" not in constraints:
            limit_match = re.search(
                r'\b(?:top|limit|max(?:_results)?)\s*[:=]?\s*(\d+)\b',
                text,
                re.IGNORECASE
            )
            if limit_match:
                constraints["max_results"] = int(limit_match.group(1))

        # timeout_seconds extraction (e.g. "timeout 30s", "timeout: 15 seconds")
        if "timeout_seconds" not in constraints:
            timeout_match = re.search(
                r'\btimeout\s*[:=]?\s*(\d+(?:\.\d+)?)\s*(?:s|sec|seconds)?\b',
                text,
                re.IGNORECASE
            )
            if timeout_match:
                constraints["timeout_seconds"] = float(timeout_match.group(1))

        # privacy_level extraction (e.g. "privacy: local_only", "local only")
        if "privacy_level" not in constraints:
            if re.search(r'\b(?:local\s*only|private)\b', text, re.IGNORECASE):
                constraints["privacy_level"] = "local_only"

        # confirmation_required extraction
        if "confirmation_required" not in constraints:
            if re.search(r'\b(?:confirm\s*first|ask\s*before|with\s*confirmation)\b', text, re.IGNORECASE):
                constraints["confirmation_required"] = True

        # Demo application extraction
        app_match = re.search(
            r'\b(?:open|launch|start|run|close|focus|switch\s+to)\s+(?:the\s+|teh\s+)?(vs\s*code|vscode|visual\s+studio\s+code|chrome|google\s+chrome|notepad)\b',
            text,
            re.IGNORECASE,
        )
        if app_match:
            raw_app = app_match.group(1).lower()
            if any(v in raw_app for v in ("vscode", "visual studio", "vs code")):
                parameters["demo_app_id"] = "vscode"
            elif "chrome" in raw_app:
                parameters["demo_app_id"] = "chrome"
            elif "notepad" in raw_app:
                parameters["demo_app_id"] = "notepad"

            if re.search(r'\bclose\b', text, re.IGNORECASE):
                parameters["demo_app_action"] = "close"
            elif re.search(r'\b(?:focus|switch\s+to)\b', text, re.IGNORECASE):
                parameters["demo_app_action"] = "focus"
            else:
                parameters["demo_app_action"] = "launch"

    def _classify_request(
        self,
        text: str,
        parameters: Dict[str, Any],
    ) -> Tuple[ChatQueryClassification, Dict[str, Any]]:
        """
        Deterministic query classification distinguishing:
        CURRENT_DEVICE_STATE, CURRENT_WORLD_STATE, CURRENT_SITUATION, CURRENT_MISSION,
        GENERAL_KNOWLEDGE, WEB_RESEARCH, ACTION_REQUEST, CONVERSATION.
        """
        norm = text.strip().lower()
        hints: Dict[str, Any] = {}

        # 1. Obvious Action commands (Take off, land, arm, open vs code, etc.)
        action_verb_patterns = [
            r"\b(?:take\s*off|takeoff|land|fly\s+to|navigate\s+to|go\s+to\s+waypoint|hover\s+at)\b",
            r"\b(?:arm|disarm|emergency\s+stop|abort\s+mission)\b",
            r"\b(?:move\s+forward|move\s+backward|turn\s+left|turn\s+right|stop\s+rover)\b",
            r"\b(?:open|launch|start|run|close)\s+(?:the\s+|teh\s+)?(?:vs\s*code|vscode|visual\s+studio\s+code|calculator|notepad|chrome|explorer)\b",
            r"\b(?:double[\s\-_]click|click|type|press\s+key|scroll)\b",
        ]
        is_read_only_question = bool(
            re.search(r"^(?:is|are|what|where|show|check|can|how|tell\s+me|who|do|which)\b", norm)
        )
        if not is_read_only_question:
            for pat in action_verb_patterns:
                if re.search(pat, norm, re.IGNORECASE):
                    hints["is_action"] = True
                    return ChatQueryClassification.ACTION_REQUEST, hints

        # 2. CURRENT_DEVICE_STATE
        device_keywords = {
            "drone": ["drone", "aerial", "uav", "atlas_drone_01"],
            "rover": ["rover", "ground robot", "ugv", "robot", "atlas_rover_01"],
            "vision": ["vision", "camera", "sensor camera", "atlas_vision_01"],
            "glass": ["glass", "glasses", "smart glasses", "hud", "atlas_glass_01"],
            "all": ["devices", "all devices", "registered devices", "any device"],
        }
        matched_target: Optional[str] = None
        for dev_cat, aliases in device_keywords.items():
            for alias in aliases:
                if re.search(r"\b" + re.escape(alias) + r"\b", norm, re.IGNORECASE):
                    matched_target = dev_cat
                    break
            if matched_target:
                break

        device_state_signals = [
            r"\b(?:online|offline|connected|disconnected|reachable|active)\b",
            r"\b(?:status|health|state|telemetry|altitude|position|speed|battery|charge|level|power)\b",
            r"\bwhat\s+devices\b",
            r"\blist\s+(?:all\s+)?devices\b",
            r"\bshow\s+(?:the\s+)?(?:drone|rover|vision|glass|devices)\b",
            r"\bcheck\s+(?:the\s+)?(?:drone|rover|vision|glass|device)\b",
        ]
        has_device_state_signal = any(re.search(pat, norm, re.IGNORECASE) for pat in device_state_signals)

        if matched_target and has_device_state_signal:
            hints["target_device"] = matched_target
            if re.search(r"\b(?:battery|charge|power|level)\b", norm, re.IGNORECASE):
                hints["query_type"] = "battery"
            elif re.search(r"\b(?:telemetry|altitude|position|speed|where)\b", norm, re.IGNORECASE):
                hints["query_type"] = "telemetry"
            elif re.search(r"\b(?:online|offline|connected|disconnected|reachable)\b", norm, re.IGNORECASE):
                hints["query_type"] = "connectivity"
            elif matched_target == "all" or re.search(r"\bwhat\s+devices\b", norm, re.IGNORECASE):
                hints["query_type"] = "list"
            else:
                hints["query_type"] = "status"
            return ChatQueryClassification.CURRENT_DEVICE_STATE, hints

        if matched_target == "all" or re.search(
            r"\bwhat\s+devices\s+(?:are\s+)?(?:online|connected|registered)\b", norm, re.IGNORECASE
        ):
            hints["target_device"] = "all"
            hints["query_type"] = "list"
            return ChatQueryClassification.CURRENT_DEVICE_STATE, hints

        # 3. CURRENT_WORLD_STATE
        if re.search(
            r"\b(?:world\s*state|what\s+does\s+atlas\s+(?:currently\s+)?know|entities\s+(?:currently\s+)?present|where\s+are\s+the\s+devices\s+located|current\s+world)\b",
            norm,
            re.IGNORECASE,
        ):
            hints["query_type"] = "world_state"
            return ChatQueryClassification.CURRENT_WORLD_STATE, hints

        # 4. CURRENT_SITUATION
        if re.search(
            r"\b(?:situations?\s+(?:are\s+)?active|what\s+is\s+happening\s+right\s+now|active\s+situations?|current\s+situations?|active\s+alerts?)\b",
            norm,
            re.IGNORECASE,
        ):
            hints["query_type"] = "situation"
            return ChatQueryClassification.CURRENT_SITUATION, hints

        # 5. CURRENT_MISSION
        if re.search(
            r"\b(?:missions?\s+(?:are\s+)?active|goals?\s+(?:are\s+)?running|active\s+missions?|running\s+goals?|current\s+missions?|current\s+goals?)\b",
            norm,
            re.IGNORECASE,
        ):
            hints["query_type"] = "mission"
            return ChatQueryClassification.CURRENT_MISSION, hints

        # 6. WEB_RESEARCH
        explicit_web = [
            r"\bsearch\s+(?:the\s+)?(?:web|internet|online)\b",
            r"\b(?:search|look|find|check)\s+(?:it\s+|this\s+)?up\s+online\b",
            r"\b(?:look|check|find)\s+online\b",
            r"\bweb\s*search\b",
            r"\bgoogle(?:\s+for)?\b",
            r"\bresearch\b",
        ]
        if any(re.search(pat, norm, re.IGNORECASE) for pat in explicit_web):
            hints["query_type"] = "web"
            return ChatQueryClassification.WEB_RESEARCH, hints

        # 7. GENERAL_KNOWLEDGE
        general_knowledge_patterns = [
            r"\b(?:what\s+is|what\s+are|who\s+is|explain|how\s+does|tell\s+me\s+about)\s+(?:tcp|udp|ip|kalman|filter|linux|kernel|pid|algorithm|os|process|thread|protocol|docker|git)\b",
            r"\b(?:documentation|read\s+docs?|manual|spec|architecture)\b",
        ]
        if any(re.search(pat, norm, re.IGNORECASE) for pat in general_knowledge_patterns):
            hints["query_type"] = "knowledge"
            return ChatQueryClassification.GENERAL_KNOWLEDGE, hints

        # 8. GREETING / CONVERSATION
        if re.search(
            r"^(?:hello|hi|hey|good\s+morning|good\s+afternoon|good\s+evening|how\s+are\s+you|who\s+are\s+you|what\s+can\s+you\s+do)\b",
            norm,
            re.IGNORECASE,
        ):
            return ChatQueryClassification.CONVERSATION, hints

        return ChatQueryClassification.CONVERSATION, hints
