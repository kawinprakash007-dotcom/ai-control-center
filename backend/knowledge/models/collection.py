from dataclasses import dataclass


@dataclass
class Collection:

    name: str

    description: str

    document_count: int = 0