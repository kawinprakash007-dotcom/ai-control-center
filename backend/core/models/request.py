from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any


@dataclass(frozen=True)
class Request:
    """
    Immutable representation of an incoming user request to the AI Control Center.

    Attributes:
        id: Unique identifier for the request.
        original_text: Verbatim raw input provided by the user.
        normalized_text: Sanitized and normalized representation of the text.
        session_id: Contextual session or conversation identifier.
        timestamp: Time at which the request was received or created.
        source: Ingestion channel (default: 'chat').
        parameters: Pre-extracted or explicit parameters.
        constraints: Operational constraints (timeouts, privacy, max_tokens, etc.).
    """

    id: str

    original_text: str

    normalized_text: str

    session_id: str

    timestamp: datetime

    source: str = "chat"

    parameters: Dict[str, Any] = field(default_factory=dict)

    constraints: Dict[str, Any] = field(default_factory=dict)
