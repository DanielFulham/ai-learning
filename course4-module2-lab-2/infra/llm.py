from langchain_anthropic import ChatAnthropic

def initialize_llm(api_key):
    return ChatAnthropic(
        model="claude-haiku-4-5",
        api_key=api_key,
        max_tokens=1000
    )