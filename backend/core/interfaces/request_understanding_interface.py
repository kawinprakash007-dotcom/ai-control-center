from abc import ABC, abstractmethod
from typing import Any

from core.models.request import Request


class RequestUnderstandingInterface(ABC):
    """
    Abstract interface for request understanding and normalization.
    """

    @abstractmethod
    def understand(self, input_data: Any) -> Request:
        """
        Analyze and normalize raw input into a structured Request.

        Args:
            input_data: Raw input data (string, dict, or event payload).

        Returns:
            Structured, immutable Request instance.
        """
        pass
