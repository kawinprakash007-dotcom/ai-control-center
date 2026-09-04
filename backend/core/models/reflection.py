from dataclasses import dataclass


@dataclass
class ReflectionDecision:

    success: bool

    continue_execution: bool

    retry: bool

    remember: bool

    confidence: float

    message: str