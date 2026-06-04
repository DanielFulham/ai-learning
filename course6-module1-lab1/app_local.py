"""Course 6 Module 1 — local Ollama variant.

Subset of the lab run against a local Ollama model rather than OpenAI. The
purpose is the capability-gradient comparison: same code, same prompts, same
tool contract — different model class, different failure profile.

The lab's interesting local-vs-hosted findings concentrate in the first few
sections (direct tool tests + single-tool agent calls), so this file does not
mirror app.py's full section structure. For sections 4-7, run against OpenAI.

Tool definitions and print_trace are imported from the shared modules to keep
the contract identical to app.py — drift between the two files is the easiest
way to get a misleading capability comparison.

Model notes (Ollama, tool-calling capability):
- Verified-good: llama3.1:8b, qwen2.5:7b, qwen2.5:14b, mistral-small
- Flaky/broken for tools: gemma3:*, llama3.2:3b, phi3:*
"""
from langchain_ollama import ChatOllama
from langchain.agents import create_agent

from tools import add_numbers, add_numbers_with_options
from tracing import print_trace


MODEL = "mistral:latest"


def run() -> None:
    llm = ChatOllama(model=MODEL, temperature=0)
    print(f"=== Using local model: {MODEL} ===")

    print("\n=== Direct tool tests (no LLM involved) ===")
    print(add_numbers_with_options.invoke({"numbers": [-1.1, -2.1, -3.0], "absolute": False}))
    print(add_numbers_with_options.invoke({"numbers": [-1.1, -2.1, -3.0], "absolute": True}))

    agent = create_agent(llm, [add_numbers])

    print("\n=== Agent call 1: GDP sum ===")
    response = agent.invoke({
        "messages": [{
            "role": "user",
            "content": (
                "In 2023, the US GDP was approximately $27.72 trillion, while "
                "Canada's was around $2.14 trillion and Mexico's was about "
                "$1.79 trillion. What is the total?"
            ),
        }]
    })
    print_trace(response)
    print(f"\nFinal answer: {response['messages'][-1].content}")

    print("\n=== Agent call 2: 'two' as a word ===")
    response = agent.invoke({
        "messages": [{"role": "user", "content": "Add 10, 20, two and 30"}]
    })
    print_trace(response)
    print(f"\nFinal answer: {response['messages'][-1].content}")

    print("\n=== Agent call 3: Negative numbers (regex strips minus signs) ===")
    response = agent.invoke({
        "messages": [{"role": "user", "content": "Add the numbers -10, -20, -30"}]
    })
    print_trace(response)
    print(f"\nFinal answer: {response['messages'][-1].content}")


if __name__ == "__main__":
    run()