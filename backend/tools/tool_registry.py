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
    memory,
    automation,
    vision
)

from tools.knowledge_capability import KnowledgeCapability


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

    "memory": memory,

    "automation": automation,

    "vision": vision,

    "knowledge": KnowledgeCapability()

}