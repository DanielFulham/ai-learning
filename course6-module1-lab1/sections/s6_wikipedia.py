"""Section 6 — Wikipedia integration and multi-step agent.

Two pieces:

- 6a: Direct test of search_wikipedia (no LLM). Confirms the wikipedia package
      returns a sensible summary with the User-Agent set in tools.py.
- 6b: Multi-step agent combining math tools + Wikipedia. The query "What is the
      population of Canada? Multiply it by 0.75" requires Wikipedia first, then
      multiplication.

The population call demonstrates the ineffective-tool-call pattern: the LLM
passes only one operand to multiply_numbers (since the second is the literal
0.75 in the prompt), the tool returns the input unchanged, and the LLM does
the 0.75 multiplication itself in the final message. Final answer is correct;
the trace shows multiply was "used"; nothing flags that the tool's output was
not load-bearing.

This is structurally identical to the canonical RAG fallback failure
(retriever returns low-relevance chunks, LLM falls back to parametric
knowledge, final answer is fluent and grounded only in the LLM's prior).
"""
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

from tools import (
    add_numbers,
    new_subtract_numbers,
    multiply_numbers,
    divide_numbers,
    search_wikipedia,
)
from tracing import print_trace

load_dotenv()


def run() -> None:
    print("=== Section 6a: Direct Wikipedia tool test (no LLM) ===")
    print(search_wikipedia.invoke("Population of Canada"))

    llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0)
    tools_updated = [
        add_numbers,
        new_subtract_numbers,
        multiply_numbers,
        divide_numbers,
        search_wikipedia,
    ]
    math_agent_updated = create_agent(
        llm,
        tools_updated,
        system_prompt=(
            "You are a helpful assistant that can perform various mathematical "
            "operations and look up information. Use the tools precisely and "
            "explain your reasoning clearly."
        ),
    )

    print("\n=== Section 6b: Population × 0.75 (multi-step agent) ===")
    response = math_agent_updated.invoke({
        "messages": [{"role": "user", "content": "What is the population of Canada? Multiply it by 0.75"}]
    })
    print_trace(response)
    print(f"\nFinal answer: {response['messages'][-1].content}")