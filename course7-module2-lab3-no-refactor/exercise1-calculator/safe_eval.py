"""Safe mathematical expression evaluation via AST walking.

Rejects everything not in the operator/function/name allowlists. The
inverse of eval(): eval() allows arbitrary code by default, this allows
nothing by default. New capabilities require adding to the allowlists
explicitly.

Same architectural category as Lab 26's read-only SQLite URI: a
deterministic boundary at the tool layer, not probabilistic instruction
to the LLM. Prompt-injected `__import__(...)` fails at the Name
allowlist check without ever reaching a function call.
"""

import ast
import math
import operator
from typing import Callable


BINOPS: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.FloorDiv: operator.floordiv,
}

UNARYOPS: dict[type[ast.unaryop], Callable[[float], float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _abs(x: float) -> float:
    return abs(x)


def _floor(x: float) -> float:
    return float(math.floor(x))


def _ceil(x: float) -> float:
    return float(math.ceil(x))


FUNCS: dict[str, Callable[..., float]] = {
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log2": math.log2,
    "log10": math.log10,
    "exp": math.exp,
    "pow": math.pow,
    "floor": _floor,
    "ceil": _ceil,
    "abs": _abs,
}

NAMES: dict[str, float] = {
    "pi": math.pi,
    "e": math.e,
}


def _reject(kind: str, detail: str) -> ValueError:
    return ValueError(f"{kind} not permitted: {detail}")


def _walk(node: ast.expr) -> float:
    """Evaluate one AST node against the allowlists. Recurses on children."""
    match node:
        case ast.Constant(value=v) if isinstance(v, (int, float)):
            return v
        case ast.Constant(value=v):
            raise _reject("constant type", f"{type(v).__name__} ({v!r})")

        case ast.BinOp(left=l, op=op, right=r) if type(op) in BINOPS:
            return BINOPS[type(op)](_walk(l), _walk(r))
        case ast.BinOp(op=op):
            raise _reject("operator", type(op).__name__)

        case ast.UnaryOp(op=op, operand=x) if type(op) in UNARYOPS:
            return UNARYOPS[type(op)](_walk(x))
        case ast.UnaryOp(op=op):
            raise _reject("unary operator", type(op).__name__)

        case ast.Call(func=ast.Name(id=name), args=args) if name in FUNCS:
            return FUNCS[name](*[_walk(a) for a in args])
        case ast.Call(func=ast.Name(id=name)):
            raise _reject("function", name)
        case ast.Call():
            raise _reject("call", "only bare-name function calls allowed")

        case ast.Name(id=name) if name in NAMES:
            return NAMES[name]
        case ast.Name(id=name):
            raise _reject("name", name)

        case _:
            raise _reject("node type", type(node).__name__)


def safe_eval(expression: str) -> float:
    """Parse and evaluate a math expression against the allowlists.

    Raises ValueError, SyntaxError, or ZeroDivisionError. Callers wrap
    for their error surface (e.g. LangChain tools returning a string).
    """
    tree = ast.parse(expression, mode="eval")
    return _walk(tree.body)