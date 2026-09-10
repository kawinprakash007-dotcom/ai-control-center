import sys
from llm.ollama_client import ask_ollama
from memory.history import get_history
from memory.memory import get_memory


def _safe_print(text: str) -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", "utf-8") or "utf-8"
        print(str(text).encode(enc, errors="replace").decode(enc))


def chat(task=None):
    # If task parameters provide explicit query and history (from Phase 2 pipeline)
    if task is not None and hasattr(task, "parameters") and isinstance(task.parameters, dict):
        query = task.parameters.get("query")
        history = task.parameters.get("history")
        if query is not None:
            _safe_print("\n[Chat Capability]")
            _safe_print("Sending to Ollama:")
            _safe_print(query)

            response = ask_ollama(query, history=history)

            _safe_print("\nOllama Response:")
            _safe_print(response)

            return response

    history = get_history()

    # Get the latest USER message

    for msg in reversed(history):

        if msg["role"] == "user":

            _safe_print("\n[Chat Capability]")
            _safe_print("Sending to Ollama:")
            _safe_print(msg["content"])

            response = ask_ollama(msg["content"])

            _safe_print("\nOllama Response:")
            _safe_print(response)

            return response

    return "Hello!"


def memory():

    value = get_memory("name")

    if value:

        return value

    return "Nothing stored."


def automation():

    return "Automation module is not implemented yet."


def vision():

    return "Vision module is not implemented yet."