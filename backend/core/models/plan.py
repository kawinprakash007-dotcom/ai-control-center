from dataclasses import dataclass, field


@dataclass
class Plan:

    goal: str

    steps: list = field(default_factory=list)

    status: str = "pending"

    confidence: float = 1.0