from dataclasses import dataclass, field
from typing import Dict


@dataclass
class Query:

    text: str

    collection: str | None = None

    top_k: int = 5

    filters: Dict = field(default_factory=dict)