from tools.tools import (
    open_calculator,
    open_chrome,
    open_vscode,
    open_notepad,
    open_explorer,
    get_time
)

from tools.capabilities import (
    chat,
    automation,
    vision
)

from tools.knowledge_capability import KnowledgeCapability
from tools.memory_capability import MemoryCapability
from tools.web_capability import WebCapability
from computer.computer_capability import ComputerCapability


TOOL_REGISTRY = {

    # -----------------------------
    # Desktop Tools
    # -----------------------------

    "calculator": open_calculator,

    "chrome": open_chrome,

    "vscode": open_vscode,

    "notepad": open_notepad,

    "explorer": open_explorer,

    "time": get_time,

    # -----------------------------
    # AI Capabilities
    # -----------------------------

    "chat": chat,

    "memory": MemoryCapability(),

    "automation": automation,

    "vision": vision,

    "knowledge": KnowledgeCapability(),

    "web": WebCapability(),

    "computer": ComputerCapability(),

}