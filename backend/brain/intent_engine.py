import json

from core.models.intent import Intent
from llm.ollama_client import ask_ollama


class IntentEngine:

    def detect(self, message: str) -> Intent:

        prompt = f"""
You are Jarvis's Intent Engine.

Understand the user's request.

Choose ONLY one intent.

Available intents:

chat
tool
memory
automation
vision

If the intent is tool,
also choose the tool.

Available tools:

calculator
chrome
vscode
notepad
explorer
time

Return ONLY valid JSON.

Example:

{{
    "intent":"tool",
    "tool":"calculator",
    "confidence":0.98,
    "reason":"User wants to open Calculator."
}}

User:

{message}
"""

        response = ask_ollama(prompt)

        try:

            data = json.loads(response)

            return Intent(

                intent=data.get("intent", "chat"),

                tool=data.get("tool"),

                confidence=float(data.get("confidence", 1.0)),

                reason=data.get("reason", ""),

                original_message=message

            )

        except Exception:

            return Intent(

                intent="chat",

                confidence=0.0,

                reason="Unable to classify.",

                original_message=message

            )