from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Any


class MessageRole(Enum):
    """
    Role of the speaker in a conversation interaction.
    """
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


@dataclass(frozen=True)
class ChatMessage:
    """
    Immutable representation of an individual message within a conversation session.

    Attributes:
        id: Unique identifier for the message.
        session_id: Correlated conversation or session identifier.
        role: Sender role (MessageRole).
        content: Textual content of the message.
        timestamp: Time at which the message was recorded.
        metadata: Optional contextual or technical metadata dictionary.
    """
    id: str
    session_id: str
    role: MessageRole
    content: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.id or not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("ChatMessage id must be a non-empty string.")
        if not self.session_id or not isinstance(self.session_id, str) or not self.session_id.strip():
            raise ValueError("ChatMessage session_id must be a non-empty string.")
        if not isinstance(self.role, MessageRole):
            raise TypeError(f"ChatMessage role must be a MessageRole enum, got {type(self.role).__name__}")
        if not isinstance(self.content, str):
            raise TypeError(f"ChatMessage content must be a string, got {type(self.content).__name__}")
        if not isinstance(self.timestamp, datetime):
            raise TypeError(f"ChatMessage timestamp must be a datetime instance, got {type(self.timestamp).__name__}")
        if not isinstance(self.metadata, dict):
            raise TypeError(f"ChatMessage metadata must be a dictionary, got {type(self.metadata).__name__}")

        # Defensively copy metadata to prevent shared mutation across instances
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class MemoryEntry:
    """
    Immutable representation of a persistent memory fact or preference.

    Attributes:
        key: Semantic lookup key (e.g. 'preferred_browser', 'editor').
        value: Stored value (any JSON-serializable Python structure or scalar).
        category: Broad classification category (e.g. 'preference', 'fact', 'general').
        user_id: Identifier of the owning user profile.
        updated_at: Timestamp of creation or latest update.
    """
    key: str
    value: Any
    category: str = "general"
    user_id: str = "default_user"
    updated_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        if not self.key or not isinstance(self.key, str) or not self.key.strip():
            raise ValueError("MemoryEntry key must be a non-empty string.")
        if not self.category or not isinstance(self.category, str) or not self.category.strip():
            raise ValueError("MemoryEntry category must be a non-empty string.")
        if not self.user_id or not isinstance(self.user_id, str) or not self.user_id.strip():
            raise ValueError("MemoryEntry user_id must be a non-empty string.")
        if not isinstance(self.updated_at, datetime):
            raise TypeError(f"MemoryEntry updated_at must be a datetime instance, got {type(self.updated_at).__name__}")
