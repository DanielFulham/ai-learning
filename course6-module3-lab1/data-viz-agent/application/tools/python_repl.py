"""Python REPL tool factory.

Produces a LangChain tool that executes agent-generated Python code against
a pre-bound pandas DataFrame. The tool's closure carries the DataFrame and
the figure store, exposing them to agent code via `df`, `pd`, `plt`, and
`save_figure` — a function the agent calls to persist a matplotlib Figure
through the configured store.

Each tool invocation gets a fresh namespace seeded with the pre-bound
names. Variables defined in agent code do not persist between calls.

Same factory pattern Lab 24 used for the YouTube tools (make_fetch_transcript,
make_get_full_metadata) — tools wrap their dependencies via closure, the
application's composition root wires the dependencies in, the tools never
touch infra directly.
"""

import ast
import contextlib
import io

import matplotlib.pyplot as plt
import pandas as pd
from langchain_core.tools import BaseTool, tool
from matplotlib.figure import Figure

from interfaces.figure_store_interface import FigureStoreInterface


def make_python_repl(
    df: pd.DataFrame,
    figure_store: FigureStoreInterface,
) -> BaseTool:
    """Construct a python_repl tool bound to the given DataFrame and figure store.

    The returned tool exposes a single `code: str` parameter to the LLM. Code
    executes against four pre-bound names:

    - `df`     — the pandas DataFrame
    - `pd`     — the pandas module
    - `plt`    — matplotlib.pyplot
    - `save_figure(fig)` — persists a matplotlib Figure via the figure store
                           and returns the saved reference as a string
    """

    def save_figure(figure: Figure) -> str:
        """Persist a matplotlib Figure through the figure store.

        Called by the agent's code when it wants a chart saved. Returns the
        reference string (path, URL, etc.) the store provides, which the
        agent can include in its final answer.
        """
        return figure_store.save(figure)

    # Pre-bound names available to every tool invocation.
    base_namespace: dict[str, object] = {
        "df": df,
        "pd": pd,
        "plt": plt,
        "save_figure": save_figure,
    }

    @tool
    def python_repl(code: str) -> str:
        """Execute Python against the pandas DataFrame `df`.

        Available in the execution namespace:
        - `df`     — the pandas DataFrame
        - `pd`     — pandas
        - `plt`    — matplotlib.pyplot
        - `save_figure(fig)` — saves a matplotlib Figure and returns the
          saved path. You MUST call save_figure(fig) for every chart you
          create, or the chart will be lost.

        Do not call plt.show() — it has no effect in this environment.

        Returns the value of the final expression, or stdout if the code
        only prints, or '(no output)' if neither.
        """
        # Fresh namespace per call, seeded from the closure-captured base.
        repl_locals = dict(base_namespace)

        try:
            tree = ast.parse(code, mode="exec")
            buf = io.StringIO()
            result: object = None

            with contextlib.redirect_stdout(buf):
                if tree.body and isinstance(tree.body[-1], ast.Expr):
                    last = tree.body[-1]
                    body = tree.body[:-1]
                    if body:
                        exec(
                            compile(ast.Module(body=body, type_ignores=[]), "<repl>", "exec"),
                            repl_locals,
                        )
                    result = eval(
                        compile(ast.Expression(last.value), "<repl>", "eval"),
                        repl_locals,
                    )
                else:
                    exec(compile(tree, "<repl>", "exec"), repl_locals)

            stdout = buf.getvalue()
            if result is not None:
                return str(result) + (f"\nStdout:\n{stdout}" if stdout else "")
            return stdout or "(no output)"

        except Exception as e:
            return f"Error: {type(e).__name__}: {e}"

    return python_repl