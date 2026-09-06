from abc import ABC, abstractmethod
from typing import List, Optional, Any, Dict
from core.models.memory import ChatMessage, MemoryEntry, MessageRole


class MemoryServiceInterface(ABC):
    """
    Abstract interface defining the memory service contract for AI Control Center.
    Encapsulates session-isolated episodic conversation history and user-scoped preferences.
    """

    @abstractmethod
    def add_message(
        self,
        session_id: str,
        role: MessageRole,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        message_id: Optional[str] = None,
    ) -> ChatMessage:
        """
        Append a message to the specified session and return the created ChatMessage.
        """
        pass

    @abstractmethod
    def get_history(
        self,
        session_id: str,
        limit: Optional[int] = None,
    ) -> List[ChatMessage]:
        """
        Retrieve chronological conversation history for the specified session up to limit.
        If limit is provided, returns the latest N messages in chronological order.
        """
        pass

    @abstractmethod
    def clear_session(self, session_id: str) -> None:
        """
        Remove all conversation messages for the specified session.
        Does not delete persistent user preferences.
        """
        pass

    @abstractmethod
    def save_preference(
        self,
        user_id: str,
        key: str,
        value: Any,
        category: str = "general",
    ) -> MemoryEntry:
        """
        Save or update a user-scoped preference or fact and return the stored MemoryEntry.
        """
        pass

    @abstractmethod
    def get_preference(
        self,
        user_id: str,
        key: str,
    ) -> Optional[MemoryEntry]:
        """
        Retrieve a user-scoped preference or fact, or None if not found.
        """
        pass

    @abstractmethod
    def delete_preference(
        self,
        user_id: str,
        key: str,
    ) -> bool:
        """
        Delete a user-scoped preference or fact.
        Returns True if a preference was deleted, False if not found.
        """
        pass

    @abstractmethod
    def list_preferences(
        self,
        user_id: str,
    ) -> List[MemoryEntry]:
        """
        Retrieve all preferences or facts for the specified user.
        """
        pass


class MemoryInterface(ABC):
    """
    Legacy interface definition preserved for backward compatibility.
    """

    @abstractmethod
    def save(self, key, value):
        pass

    @abstractmethod
    def load(self, key):
        pass