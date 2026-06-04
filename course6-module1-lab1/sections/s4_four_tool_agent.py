"""Section 4 — Four-tool math agent (with deliberate-bug subtract).

The mathematical toolkit: add, subtract (with the lab's deliberate bug),
multiply, divide. Four calls exercise the multi-tool routing and surface
distinct failure modes:

- 4a: 25 / 4 — unambiguous, single right tool. Happy-path baseline.
- 4b: Subtract 100, 20, 10 — invokes the buggy subtract_numbers. Tool returns
      -130 (because the first input is negated). LLM accepts the structurally-
      plausible negative result. Canonical silent-wrong-answer trace.
- 4c: Multiply 2, 3, and "four" — LLM normalises "four" → "4" before calling.
      Demonstrates docstring-example-shape priming (multiply's docstring uses
      clean integers: "2, 3, 4").
- 4d: Divide 100 by 5 and then by 2 — sequential operation. Tool returns
      20.0 from a single call; LLM does the second division internally.
      Override / tool-bypass pattern.
"""
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

from tools import add_numbers, subtract_numbers, multiply_numbers, divide_numbers
from tracing import print_trace

load_dotenv()


def run() -> None:
    llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0)
    tools = [add_numbers, subtract_numbers, multiply_numbers, divide_numbers]

    math_agent = create_agent(
        llm,
        tools,
        system_prompt=(
            "You are a helpful mathematical assistant that can perform various "
            "operations. Use the tools precisely and explain your reasoning clearly."
        ),
    )

    print("=== Section 4a: 25 / 4 (unambiguous routing) ===")
    response = math_agent.invoke({
        "messages": [{"role": "user", "content": "What is 25 divided by 4?"}]
    })
    print_trace(response)
    print(f"\nFinal answer: {response['messages'][-1].content}")

    print("\n=== Section 4b: Subtract 100, 20, 10 (deliberate bug) ===")
    response = math_agent.invoke({
        "messages": [{"role": "user", "content": "Subtract 100, 20, and 10."}]
    })
    print_trace(response)
    print(f"\nFinal answer: {response['messages'][-1].content}")

    print("\n=== Section 4c: Multiply 2, 3, and four (word-number normalisation) ===")
    response = math_agent.invoke({
        "messages": [{"role": "user", "content": "Multiply 2, 3, and four."}]
    })
    print_trace(response)
    print(f"\nFinal answer: {response['messages'][-1].content}")

    print("\n=== Section 4d: Divide 100 by 5 and then by 2 (sequential ops) ===")
    response = math_agent.invoke({
        "messages": [{"role": "user", "content": "Divide 100 by 5 and then by 2."}]
    })
    print_trace(response)
    print(f"\nFinal answer: {response['messages'][-1].content}")