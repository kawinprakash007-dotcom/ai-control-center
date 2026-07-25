from llm.ollama_client import ask_ollama
from tools.tool_registry import list_tools


def choose_tool(message):

    tools = "\n".join(list_tools())

    prompt = f"""
Available tools:

{tools}

Choose the BEST tool.

Return ONLY the tool name.

User:

{message}
"""

    return ask_ollama(prompt).strip().lower()