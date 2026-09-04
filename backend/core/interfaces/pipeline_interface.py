from abc import ABC, abstractmethod
from typing import Any

from core.models.pipeline import PipelineResult


class PipelineInterface(ABC):
    """
    Abstract interface for end-to-end request processing in AI Control Center.
    """

    @abstractmethod
    def process(self, input_data: Any) -> PipelineResult:
        """
        Execute the complete Phase 2 lifecycle and return structured PipelineResult.

        Args:
            input_data: Raw input data (str, dict, or event payload).

        Returns:
            Structured, immutable PipelineResult instance.
        """
        pass

    def run(self, input_data: Any) -> str:
        """
        Execute the complete Phase 2 lifecycle and return only the final response string.
        Delegates directly to process().

        Args:
            input_data: Raw input data (str, dict, or event payload).

        Returns:
            Final user-facing response string.
        """
        return self.process(input_data).response
