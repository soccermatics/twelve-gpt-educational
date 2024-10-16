"""
Utilities for more easily convert OpenAI API calls to Gemini API calls.
"""


def convert_messages_format(messages):
    new_messages = []
    system_prompt = None
    if len(messages) > 0 and messages[0]["role"] == "system":
        # If the first message is a system message, store it and return it.
        # Gemini requires the system prompt to be passed in separately.
        system_prompt = messages[0]["content"]
        messages = messages[1:]
    for message in messages:
        role = "model" if message["role"] == "assistant" else "user"
        new_message = {
            "role": role,
            "parts": message["content"],
        }
        new_messages.append(new_message)

    user_query = ""
    if new_messages[-1]["role"] == "user":
        user_query = new_messages.pop()
    return {"system_instruction": system_prompt, "history": new_messages, "content": user_query}
