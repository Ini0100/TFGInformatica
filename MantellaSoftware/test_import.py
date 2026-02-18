
try:
    from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall
    print("Found in openai.types.chat.chat_completion_message_tool_call")
except ImportError:
    print("Not found in openai.types.chat.chat_completion_message_tool_call")

try:
    from openai.types.chat import ChatCompletionMessageToolCall
    print("Found in openai.types.chat")
except ImportError:
    print("Not found in openai.types.chat")

import openai
print(f"OpenAI version: {openai.__version__}")
