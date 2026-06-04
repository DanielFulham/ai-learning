"""Section 2 — Single-tool agent with add_numbers.

Three calls against the same agent (one tool: add_numbers, the string-input
variant with the verbose docstring example). The three queries are chosen to
exercise the silent-acceptance hazard:

- GDP query — decimals masquerading as integers; tool's regex extracts wrong numbers.
- "two and 30" — natural-language number word the regex can't parse.
- Negative numbers — minus signs the regex strips silently.

Each produces a different failure or override pattern in the trace.
"""
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

from tools import add_numbers
from tracing import print_trace

load_dotenv()


def run() -> None:
    llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0)
    agent = create_agent(llm, [add_numbers])

    print("=== Section 2a: GDP sum (decimals stripped by regex) ===")
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

    print("\n=== Section 2b: 'two' as a word ===")
    response = agent.invoke({
        "messages": [{"role": "user", "content": "Add 10, 20, two and 30"}]
    })
    print_trace(response)
    print(f"\nFinal answer: {response['messages'][-1].content}")

    print("\n=== Section 2c: Negative numbers (minus signs stripped by regex) ===")
    response = agent.invoke({
        "messages": [{"role": "user", "content": "Add the numbers -10, -20, -30"}]
    })
    print_trace(response)
    print(f"\nFinal answer: {response['messages'][-1].content}")