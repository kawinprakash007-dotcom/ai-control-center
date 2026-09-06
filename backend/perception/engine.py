import time
from typing import List, Optional

from core.interfaces.perception_interface import (
    PerceptionEngineInterface,
    VisualPerceptionProvider,
)
from core.models.computer import ComputerObservation
from core.models.perception import VisualScene, VisualElement
from perception.deduplicator import deduplicate_elements
from perception.accessibility_provider import AccessibilityPerceptionProvider
from perception.ocr_provider import OCRPerceptionProvider
from perception.cv_provider import CVPerceptionProvider


class VisualPerceptionEngine(PerceptionEngineInterface):
    """
    Central perception engine aggregating multi-source visual and semantic observations.
    Enforces deterministic loop bounds, deduplication, and model neutrality.
    """

    def __init__(
        self,
        providers: Optional[List[VisualPerceptionProvider]] = None,
        max_elements: int = 200,
        max_passes: int = 3,
        deduplicate: bool = True,
    ):
        if providers is not None:
            self.providers = list(providers)
        else:
            self.providers = [
                AccessibilityPerceptionProvider(),
                OCRPerceptionProvider(),
                CVPerceptionProvider(),
            ]
        self.max_elements = max_elements
        self.max_passes = max_passes
        self.deduplicate = deduplicate

    def perceive(self, observation: ComputerObservation) -> VisualScene:
        """
        Aggregate perceptions from all active providers into a unified VisualScene.
        """
        start_time = time.perf_counter()
        collected_elements: List[VisualElement] = []
        passes = 0

        for provider in self.providers:
            if passes >= self.max_passes:
                break
            if not provider.is_available():
                continue

            try:
                detected = provider.perceive(observation)
                collected_elements.extend(detected)
                passes += 1
            except Exception:
                # Perception failure in one provider must not crash the engine
                continue

        # Deduplicate and fuse multi-source elements
        if self.deduplicate:
            unified_elements = deduplicate_elements(collected_elements)
        else:
            unified_elements = collected_elements

        # Enforce maximum element capacity
        if len(unified_elements) > self.max_elements:
            unified_elements = unified_elements[: self.max_elements]

        duration = round(time.perf_counter() - start_time, 4)

        return VisualScene(
            source_observation_id=observation.observation_id or f"obs_{int(observation.timestamp * 1000)}",
            screen_dimensions=observation.screen_dimensions,
            elements=tuple(unified_elements),
            active_window_title=observation.active_window_title,
            process_name=observation.process_name,
            timestamp=observation.timestamp,
            metadata={
                "processing_duration": duration,
                "passes": passes,
                "raw_detected_count": len(collected_elements),
                "fused_count": len(unified_elements),
            },
        )
