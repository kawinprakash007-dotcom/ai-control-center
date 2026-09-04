from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class Chunk:

    id: str

    document_id: str

    text: str

    chunk_index: int

    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def chunk_id(self) -> str:
        """Alias for id consistent with context schema."""
        return self.id

    @property
    def score(self) -> float | None:
        """Retrieve relevance score or distance from metadata if available."""
        if not self.metadata:
            return None
        return self.metadata.get(
            "score",
            self.metadata.get("similarity", self.metadata.get("distance"))
        )