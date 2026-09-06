from dataclasses import dataclass
from typing import Any


@dataclass
class Result:

    success: bool

    message: str

    output: str | None = None

    data: Any = None