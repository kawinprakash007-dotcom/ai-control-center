from typing import List, Optional
from core.interfaces.perception_interface import VisualPerceptionProvider
from core.models.computer import ComputerObservation
from core.models.perception import (
    VisualElement,
    BoundingBox,
    ElementType,
    PerceptionSource,
)


class MockPerceptionProvider(VisualPerceptionProvider):
    """
    Deterministic in-memory mock perception provider for tests.
    Allows simulating OCR, Accessibility, or CV elements with 100% desktop isolation.
    """
    source = PerceptionSource.MANUAL

    def __init__(
        self,
        elements: Optional[List[VisualElement]] = None,
        available: bool = True,
        should_fail: bool = False,
        failure_message: str = "Mock perception failure",
    ):
        self._elements: List[VisualElement] = elements or []
        self._available = available
        self.should_fail = should_fail
        self.failure_message = failure_message
        self.call_count = 0

    def is_available(self) -> bool:
        return self._available

    def set_elements(self, elements: List[VisualElement]) -> None:
        self._elements = list(elements)

    def add_element(
        self,
        element_id: str,
        element_type: ElementType,
        x: int,
        y: int,
        width: int,
        height: int,
        text: Optional[str] = None,
        confidence: float = 1.0,
        source: PerceptionSource = PerceptionSource.MANUAL,
    ) -> VisualElement:
        box = BoundingBox(x=x, y=y, width=width, height=height)
        el = VisualElement(
            element_id=element_id,
            element_type=element_type,
            bounding_box=box,
            text=text,
            confidence=confidence,
            source=source,
        )
        self._elements.append(el)
        return el

    def perceive(self, observation: ComputerObservation) -> List[VisualElement]:
        self.call_count += 1
        if self.should_fail:
            raise RuntimeError(self.failure_message)
        return list(self._elements)
