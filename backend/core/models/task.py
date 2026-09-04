from dataclasses import dataclass, field


@dataclass
class Task:

    id: int

    type: str

    action: str

    tool: str | None = None

    parameters: dict = field(default_factory=dict)

    status: str = "pending"

    result: str | None = None