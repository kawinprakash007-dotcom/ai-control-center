from abc import ABC, abstractmethod

from knowledge.models.document import Document


class DocumentLoader(ABC):

    """
    Base class for every document loader.
    """

    @abstractmethod
    def load(
        self,
        path: str
    ) -> Document:
        """
        Convert a file into a Document model.
        """
        pass

    @abstractmethod
    def supported_extensions(
        self
    ) -> list[str]:
        """
        Return supported file extensions.
        """
        pass