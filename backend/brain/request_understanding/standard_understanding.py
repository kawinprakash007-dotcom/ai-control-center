import re
import uuid
from datetime import datetime
from typing import Any, Dict, Tuple, Optional

from core.interfaces.request_understanding_interface import RequestUnderstandingInterface
from core.models.request import Request
from brain.request_understanding.normalizer import normalize_text


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
