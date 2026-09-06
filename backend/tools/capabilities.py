from llm.ollama_client import ask_ollama
from memory.history import get_history
from memory.memory import get_memory


def chat(task=None):
    # If task parameters provide explicit query and history (from Phase 2 pipeline)
    if task is not None and hasattr(task, "parameters") and isinstance(task.parameters, dict):
        query = task.parameters.get("query")
        history = task.parameters.get("history")
        if query is not None:
            print("\n[Chat Capability]")
            print("Sending to Ollama:")
            print(query)

            response = ask_ollama(query, history=history)

            print("\nOllama Response:")
            print(response)

            return response

    history = get_history()

    # Get the latest USER message

    for msg in reversed(history):

        if msg["role"] == "user":

            print("\n[Chat Capability]")
            print("Sending to Ollama:")
            print(msg["content"])

            response = ask_ollama(msg["content"])

            print("\nOllama Response:")
            print(response)

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