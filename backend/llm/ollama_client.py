from typing import List, Dict, Optional
import requests

from memory.history import get_history
from .prompts import SYSTEM_PROMPT


OLLAMA_URL = "http://localhost:11434/api/chat"


def ask_ollama(user_message: str, history: Optional[List[Dict[str, str]]] = None):

    messages = []

    messages.append({
        "role": "system",
        "content": SYSTEM_PROMPT
    })

    if history is not None:
        messages.extend(history)
    else:
        messages.extend(get_history())

    messages.append({
        "role": "user",
        "content": user_message
    })

    response = requests.post(

        OLLAMA_URL,

        json={

            "model": "qwen3:8b",

            "messages": messages,

            "stream": False
        }
    )

    data = response.json()

    return data["message"]["content"]