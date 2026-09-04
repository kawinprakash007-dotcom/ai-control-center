from dataclasses import dataclass


@dataclass
class Goal:

    goal: str

    priority: str = "normal"

    status: str = "pending"

    confidence: float = 1.0

    reason: str = ""

    query: str = ""