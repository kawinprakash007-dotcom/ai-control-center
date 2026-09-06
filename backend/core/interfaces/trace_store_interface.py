from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from core.models.runtime import CognitiveTrace


class TraceStoreInterface(ABC):
    """
    Interface for persisting, retrieving, listing, and removing bounded CognitiveTrace documents.
    Provides an observable audit trail of historical turns without heavy database dependencies.
    """

    @abstractmethod
    def save_trace(self, trace: CognitiveTrace, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Persist a serialized CognitiveTrace document.
        Returns the identifier or file path of the saved trace.
        """
        pass

    @abstractmethod
    def load_trace(self, turn_id: str) -> Optional[CognitiveTrace]:
        """
        Load a historical CognitiveTrace by its turn_id.
        Returns None if not found.
        """
        pass

    @abstractmethod
    def list_traces(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        List summaries of stored traces up to limit.
        """
        pass

    @abstractmethod
    def delete_trace(self, turn_id: str) -> bool:
        """
        Remove a stored trace by turn_id.
        Returns True if removed, False otherwise.
        """
        pass
