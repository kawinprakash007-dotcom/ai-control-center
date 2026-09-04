from dataclasses import dataclass


@dataclass
class Intent:

    intent: str

    tool: str | None = None

    confidence: float = 1.0

    reason: str = ""

    original_message: str = ""