from enum import Enum


class IntentType(Enum):

    CHAT = "chat"

    TOOL = "tool"

    MEMORY = "memory"

    AUTOMATION = "automation"

    VISION = "vision"

    CODING = "coding"

    SEARCH = "search"