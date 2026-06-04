"""Section 7 — Power tool (typed-args, modernised final exercise).

The IBM lab's final exercise wraps a regex-parsing function with `Tool()` and
`initialize_agent` — three deprecated patterns in one cell. The modern
equivalent is a five-line @tool with typed args: the LLM constructs the right
floats from natural language ("5 to the power of 2"), the framework validates
them, and the tool body is the actual computation.

Two checks:
- 7a: Direct invocation with explicit base/exponent dict — confirms the tool
      behaves as a plain typed function.
- 7b: Agent call with natural-language query — confirms the LLM maps English
      ordering ("5 to the power of 2") to parameter names (base=5, exponent=2)
      without any prompting.
"""
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

from tools import (
    add_numbers,
    new_subtract_numbers,
    multiply_numbers,
    divide_numbers,
    calculate_power,
    search_wikipedia,
)
from tracing import print_trace

load_dotenv()


def run() -> None:
    print("=== Section 7a: Direct power tool test ===")
    print(calculate_power.invoke({"base": 5, "exponent": 2}))
    print(calculate_power.invoke({"base": 2, "exponent": 10}))

    llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0)
    power_tools = [
        add_numbers,
        new_subtract_numbers,
        multiply_numbers,
        divide_numbers,
        calculate_power,
        search_wikipedia,
    ]
    power_agent = create_agent(
        llm,
        power_tools,
        system_prompt=(
            "You are a helpful mathematical assistant that can perform various "
            "operations and look up information. Use the tools precisely and "
            "explain your reasoning clearly."
        ),
    )

    print("\n=== Section 7b: Agent call — 'Calculate 5 to the power of 2' ===")
    response = power_agent.invoke({
        "messages": [{"role": "user", "content": "Calculate 5 to the power of 2."}]
    })
    print_trace(response)
    print(f"\nFinal answer: {response['messages'][-1].content}")