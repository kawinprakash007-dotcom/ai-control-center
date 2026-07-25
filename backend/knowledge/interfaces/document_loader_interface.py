from abc import ABC, abstractmethod

from knowledge.models.document import Document


class DocumentLoaderInterface(ABC):

    @abstractmethod
    def load(
        self,
        path: str
    ) -> Document:
        pass