conversation = []

MAX_HISTORY = 20


def add_message(role, content):

    conversation.append({
        "role": role,
        "content": content
    })

    if len(conversation) > MAX_HISTORY:
        conversation.pop(0)


def get_history():
    return conversation


def clear_history():
    conversation.clear()