from dataclasses import dataclass, field
from typing import Dict


@dataclass
class Document:

    id: str

    name: str

    path: str

    content: str

    document_type: str

    collection: str

    metadata: Dict = field(default_factory=dict)