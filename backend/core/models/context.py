from dataclasses import dataclass


@dataclass
class Context:

    current_app: str | None = None

    current_goal: str | None = None

    last_tool: str | None = None

    last_user_message: str | None = None

    last_ai_response: str | None = None

    conversation_count: int = 0