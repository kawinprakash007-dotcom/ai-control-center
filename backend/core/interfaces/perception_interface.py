from abc import ABC, abstractmethod
from typing import List, Optional

from core.models.computer import ComputerObservation
from core.models.perception import (
    VisualElement,
    VisualScene,
    GroundingRequest,
    GroundingResult,
    PerceptionSource,
)


class VisualPerceptionProvider(ABC):
    """
    Abstract contract for perception providers (OCR, CV, Accessibility, Multimodal).
    Never controls device directly or executes clicks.
    """
    source: PerceptionSource

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if backend dependencies and hardware are accessible."""
        pass

    @abstractmethod
    def perceive(self, observation: ComputerObservation) -> List[VisualElement]:
        """Extract visual elements from a desktop observation."""
        pass


class PerceptionEngineInterface(ABC):
    """
    Contract for perception engine that aggregates and deduplicates multi-source perception.
    """

    @abstractmethod
    def perceive(self, observation: ComputerObservation) -> VisualScene:
        """Process observation into a unified, deduplicated VisualScene."""
        pass


class TargetGrounderInterface(ABC):
    """
    Contract for resolving user intent / grounding requests to visual targets.
    Produces GroundedTarget data; NEVER triggers mouse/keyboard actions.
    """

    @abstractmethod
    def ground(
        self,
        request: GroundingRequest,
        scene: VisualScene,
        current_observation: Optional[ComputerObservation] = None,
    ) -> GroundingResult:
        """Resolve a GroundingRequest against the VisualScene."""
        pass
