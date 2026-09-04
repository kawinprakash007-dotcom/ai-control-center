from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class Embedding:

    chunk_id: str

    vector: List[float]

    metadata: Dict = field(default_factory=dict)