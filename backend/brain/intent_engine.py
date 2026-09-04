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

chat: casual conversation, greetings, general dialogue
knowledge: factual, technical, domain-specific, or document questions where indexed knowledge may be useful
tool: desktop application controls
memory: user preferences or remembering/recalling personal facts
automation: system automation workflows
vision: image and screen analysis

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

Example:

{{
    "intent":"knowledge",
    "tool":null,
    "confidence":0.95,
    "reason":"User is asking a factual question about operating systems or technical concepts."
}}

User:

{message}
"""

        response = ask_ollama(prompt)

        try:

            data = json.loads(response)

            detected_intent = str(data.get("intent", "chat")).lower().strip()
            if detected_intent in ("knowledge", "search"):
                detected_intent = "knowledge"

            return Intent(

                intent=detected_intent,

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