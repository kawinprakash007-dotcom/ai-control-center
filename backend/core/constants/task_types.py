from enum import Enum


class TaskType(Enum):

    TOOL = "tool"

    VERIFY = "verify"

    CHAT = "chat"

    MEMORY = "memory"

    WAIT = "wait"

    DECISION = "decision"

    KNOWLEDGE = "knowledge"