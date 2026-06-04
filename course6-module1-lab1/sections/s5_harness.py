"""Section 5 — Four-case test harness with corrected subtract tool.

Modernised version of the IBM lab's test loop. Cases are data, agent is
invoked per case, the first matching ToolMessage is parsed, and the result
is compared to the expected value.

This is contract testing dressed up as eval — it validates the *tool's* output
against an expected value, not the *agent's* answer against user intent.
Three categories of false signal:

1. Silent-wrong-answer: if tool returns expected-but-wrong value, test PASSes.
2. Silent-right-answer-through-LLM-arithmetic: if tool returns wrong value but
   LLM corrects in the final answer, test FAILs despite correct user-visible
   outcome.
3. JSON parse failure: if a tool ever returns non-JSON content (e.g., the
   error-string branch of sum_numbers_with_complex_output), the harness
   throws and the result is indistinguishable from a wrong-answer failure.
"""
import json

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

from tools import add_numbers, new_subtract_numbers, multiply_numbers, divide_numbers
from tracing import print_trace

load_dotenv()

TEST_CASES = [
    {
        "query": "Subtract 100, 20, and 10.",
        "expected": {"result": 70},
        "description": "Sequential subtraction with corrected tool.",
    },
    {
        "query": "Multiply 2, 3, and 4.",
        "expected": {"result": 24},
        "description": "Multiplication of integer list.",
    },
    {
        "query": "Divide 100 by 5 and then by 2.",
        "expected": {"result": 10.0},
        "description": "Sequential division (note: tool only handles single call).",
    },
    {
        "query": "Subtract 50 from 20.",
        "expected": {"result": -30},
        "description": "Subtraction producing negative result.",
    },
]

TOOL_NAMES = {"add_numbers", "new_subtract_numbers", "multiply_numbers", "divide_numbers"}


def run() -> None:
    llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0)
    tools_updated = [add_numbers, new_subtract_numbers, multiply_numbers, divide_numbers]

    math_agent_new = create_agent(
        llm,
        tools_updated,
        system_prompt=(
            "You are a helpful mathematical assistant that can perform various "
            "operations. Use the tools precisely and explain your reasoning clearly."
        ),
    )

    print("=== Section 5: Test harness — math_agent_new vs four cases ===")
    correct_tasks = []

    for index, test in enumerate(TEST_CASES, start=1):
        query = test["query"]
        expected_result = test["expected"]["result"]

        print(f"\n--- Test Case {index}: {test['description']} ---")
        print(f"Query: {query}")

        response = math_agent_new.invoke({
            "messages": [{"role": "user", "content": query}]
        })

        tool_message = next(
            (
                msg for msg in response["messages"]
                if hasattr(msg, "name") and msg.name in TOOL_NAMES
            ),
            None,
        )

        if tool_message:
            try:
                tool_result = json.loads(tool_message.content)["result"]
                print(f"Tool Result:     {tool_result}")
                print(f"Expected Result: {expected_result}")

                if tool_result == expected_result:
                    print(f"PASS: {test['description']}")
                    correct_tasks.append(test["description"])
                else:
                    print(f"FAIL: {test['description']}")
                    print(f"Final agent answer: {response['messages'][-1].content}")
            except (json.JSONDecodeError, KeyError) as e:
                print(f"FAIL: Harness parse error ({type(e).__name__}) — {e}")
                print(f"Tool content: {tool_message.content!r}")
                print(f"Final agent answer: {response['messages'][-1].content}")
        else:
            print("FAIL: No matching tool was called by the agent")
            print(f"Final agent answer: {response['messages'][-1].content}")

    print(f"\nCorrectly passed tests ({len(correct_tasks)}/{len(TEST_CASES)}):")
    for desc in correct_tasks:
        print(f"  - {desc}")