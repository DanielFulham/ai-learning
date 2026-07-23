"""Custom tool via Tool subclass - the low-level authoring path.

Framework's preferred path for simple tools is @tool decorator (see
frameworks own docs). Tool subclass shape is the escape hatch when you
need lifecycle hooks (emitter, options, schema customisation).

The lab's original safe_calculate used eval with a character-set gate,
which is not a security control - runaway expressions (10*10*10*...)
and deep parenthesisation cause DoS despite passing the char check.
Replaced with ast-based evaluator that only permits literal arithmetic
node types.
"""

import ast
import asyncio
import operator
import os
from typing import Any

from beeai_framework.agents.requirement import RequirementAgent
from beeai_framework.backend import ChatModel
from beeai_framework.context import RunContext
from beeai_framework.emitter import Emitter
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.middleware.trajectory import GlobalTrajectoryMiddleware
from beeai_framework.tools import StringToolOutput, Tool, ToolRunOptions
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from helpers.metrics import print_run_metrics

if not load_dotenv():
    raise RuntimeError(".env not found - check cwd or run from lab root")
if not os.environ.get("ANTHROPIC_API_KEY"):
    raise RuntimeError("ANTHROPIC_API_KEY missing from .env")


_ALLOWED_OPS: dict[type[ast.operator] | type[ast.unaryop], Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_arithmetic(expression: str) -> float:
    """AST-based arithmetic evaluator.

    Only Constant, BinOp, and UnaryOp with add/sub/mul/div are allowed.
    Rejects everything else (calls, names, attribute access, comprehensions)
    so runaway expressions and code injection are structurally impossible.
    """
    tree = ast.parse(expression, mode="eval")

    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)):
                raise ValueError(f"Non-numeric constant: {node.value!r}")
            return float(node.value)
        if isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in _ALLOWED_OPS:
                raise ValueError(f"Operator {op_type.__name__} not permitted")
            return _ALLOWED_OPS[op_type](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in _ALLOWED_OPS:
                raise ValueError(f"Unary operator {op_type.__name__} not permitted")
            return _ALLOWED_OPS[op_type](_eval(node.operand))
        raise ValueError(f"AST node {type(node).__name__} not permitted")

    return _eval(tree)


class CalculatorInput(BaseModel):
    """Input model for basic mathematical calculations."""

    expression: str = Field(
        description="Mathematical expression using +, -, *, / (e.g., '10 + 5', '(3+4)*2')"
    )


class SimpleCalculatorTool(Tool[CalculatorInput, ToolRunOptions, StringToolOutput]):
    """Basic arithmetic tool: add, subtract, multiply, divide."""

    @property
    def name(self) -> str:
        return "SimpleCalculator"

    @property
    def description(self) -> str:
        return "Performs basic arithmetic: +, -, *, /, and parenthesised sub-expressions."

    @property
    def input_schema(self) -> type[CalculatorInput]:
        return CalculatorInput

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(
            namespace=["tool", "calculator", "basic"],
            creator=self,
        )

    async def _run(
        self,
        input: CalculatorInput,
        options: ToolRunOptions | None,
        context: RunContext,
    ) -> StringToolOutput:
        try:
            result = _safe_arithmetic(input.expression.strip())
        except (ValueError, ZeroDivisionError, SyntaxError) as exc:
            return StringToolOutput(f"Calculation error: {exc}")
        return StringToolOutput(f"{input.expression} = {result}")


MATH_QUERIES = [
    "What is 15 + 27?",
    "Calculate 144 divided by 12",
    "I need to know what 8 times 9 equals",
    "What's (10 + 5) * 3 - 7?",
]


async def calculator_agent_example() -> None:
    llm = ChatModel.from_name("anthropic:claude-haiku-4-5")

    agent = RequirementAgent(
        llm=llm,
        tools=[SimpleCalculatorTool()],
        memory=UnconstrainedMemory(),
        instructions=(
            "You are a helpful math assistant. When users ask for calculations, "
            "use the SimpleCalculator tool. Show both the expression and result."
        ),
        middlewares=[GlobalTrajectoryMiddleware(included=[Tool])],
    )

    for query in MATH_QUERIES:
        print(f"\nHuman: {query}")
        result = await agent.run(query)
        print(f"Agent: {result.output[-1].text}")

    print_run_metrics(result.state)


if __name__ == "__main__":
    asyncio.run(calculator_agent_example())
