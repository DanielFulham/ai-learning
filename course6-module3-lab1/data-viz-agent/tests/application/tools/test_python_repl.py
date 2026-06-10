"""Tests for application.tools.python_repl.

The python_repl tool executes agent-generated Python against a pre-bound
DataFrame. It exposes four pre-bound names: `df`, `pd`, `plt`, and
`save_figure(fig)`. The agent calls save_figure explicitly when it wants
a Figure persisted; the tool does not inspect matplotlib's global state.

Tests inject a mock FigureStoreInterface so no filesystem I/O happens.
Matplotlib uses the Agg backend so no GUI windows open during test runs.
"""

from unittest.mock import MagicMock

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from matplotlib.figure import Figure

from application.tools.python_repl import make_python_repl
from interfaces.figure_store_interface import FigureStoreInterface


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Small DataFrame for the tool to operate on."""
    return pd.DataFrame({
        "name": ["Alice", "Bob", "Carol"],
        "age": [30, 25, 35],
        "score": [85, 72, 91],
    })


@pytest.fixture
def figure_store() -> MagicMock:
    """A mock FigureStoreInterface that returns a deterministic path."""
    store = MagicMock(spec=FigureStoreInterface)
    store.save.return_value = "fake/path/figure_01.png"
    return store


@pytest.fixture(autouse=True)
def clean_matplotlib_state():
    """Each test starts and ends with no open matplotlib figures.

    autouse=True so the fixture runs for every test without explicit
    request. Matplotlib's figure stack is global process state; without
    this hygiene, figures from one test would leak into the next.

    Note: the tool itself no longer manages matplotlib state — this
    hygiene is purely for test isolation, not for production behaviour.
    """
    plt.close("all")
    yield
    plt.close("all")


# --- Expression evaluation ---


def test_simple_expression_returns_value(sample_df: pd.DataFrame, figure_store: MagicMock) -> None:
    tool = make_python_repl(sample_df, figure_store)
    result = tool.invoke({"code": "len(df)"})
    assert result == "3"


def test_dataframe_expression_returns_stringified_value(
    sample_df: pd.DataFrame, figure_store: MagicMock
) -> None:
    tool = make_python_repl(sample_df, figure_store)
    result = tool.invoke({"code": "df['age'].mean()"})
    assert "30" in result  # mean of [30, 25, 35] is 30.0


def test_multiline_code_with_final_expression(
    sample_df: pd.DataFrame, figure_store: MagicMock
) -> None:
    """Statements run first, then the final expression's value is returned."""
    tool = make_python_repl(sample_df, figure_store)
    result = tool.invoke({"code": "x = df['age'].max()\nx * 2"})
    assert result == "70"

def test_variables_do_not_persist_across_invocations(
    sample_df: pd.DataFrame, figure_store: MagicMock
) -> None:
    """Each tool invocation gets a fresh namespace. Variables defined in
    agent code in one call are gone by the next."""
    tool = make_python_repl(sample_df, figure_store)

    tool.invoke({"code": "x = 42"})
    result = tool.invoke({"code": "x"})

    assert result.startswith("Error: NameError")


def test_shadowing_prebound_name_does_not_persist(
    sample_df: pd.DataFrame, figure_store: MagicMock
) -> None:
    """If agent code reassigns df, pd, plt, or save_figure, the change is
    scoped to that single invocation. The next call sees the original
    pre-bound values."""
    tool = make_python_repl(sample_df, figure_store)

    tool.invoke({"code": "df = 'corrupted'"})
    result = tool.invoke({"code": "len(df)"})

    assert result == "3"

# --- Statement-only execution ---


def test_statements_only_returns_no_output_marker(
    sample_df: pd.DataFrame, figure_store: MagicMock
) -> None:
    """When code is purely statements with no print or final expression, the
    tool returns a 'no output' marker so the LLM knows execution succeeded
    silently."""
    tool = make_python_repl(sample_df, figure_store)
    result = tool.invoke({"code": "x = 1 + 1"})
    assert result == "(no output)"


def test_print_output_captured(sample_df: pd.DataFrame, figure_store: MagicMock) -> None:
    tool = make_python_repl(sample_df, figure_store)
    result = tool.invoke({"code": "print('hello')"})
    assert "hello" in result


# --- Pre-bound names ---


def test_df_is_prebound(sample_df: pd.DataFrame, figure_store: MagicMock) -> None:
    """The DataFrame passed at construction is accessible as `df` in the REPL."""
    tool = make_python_repl(sample_df, figure_store)
    result = tool.invoke({"code": "df.shape[0]"})
    assert result == "3"


def test_pd_is_prebound(sample_df: pd.DataFrame, figure_store: MagicMock) -> None:
    """The pandas module is accessible as `pd` without import."""
    tool = make_python_repl(sample_df, figure_store)
    result = tool.invoke({"code": "pd.Series([1, 2, 3]).sum()"})
    assert result == "6"


def test_plt_is_prebound(sample_df: pd.DataFrame, figure_store: MagicMock) -> None:
    """The pyplot module is accessible as `plt` without import."""
    tool = make_python_repl(sample_df, figure_store)
    result = tool.invoke({"code": "type(plt).__name__"})
    assert result == "module"


def test_save_figure_is_prebound(sample_df: pd.DataFrame, figure_store: MagicMock) -> None:
    """save_figure is callable in the REPL namespace, alongside df/pd/plt."""
    tool = make_python_repl(sample_df, figure_store)
    result = tool.invoke({"code": "callable(save_figure)"})
    assert result == "True"


# --- save_figure behaviour ---


def test_save_figure_passes_figure_to_store(
    sample_df: pd.DataFrame, figure_store: MagicMock
) -> None:
    """When the agent calls save_figure(fig), the figure is passed verbatim
    to the figure store. No magic, no inference — just delegation."""
    tool = make_python_repl(sample_df, figure_store)
    code = (
        "fig, ax = plt.subplots()\n"
        "ax.bar(['a', 'b'], [1, 2])\n"
        "save_figure(fig)"
    )
    tool.invoke({"code": code})

    figure_store.save.assert_called_once()
    saved_figure = figure_store.save.call_args.args[0]
    assert isinstance(saved_figure, Figure)


def test_save_figure_returns_store_reference(
    sample_df: pd.DataFrame, figure_store: MagicMock
) -> None:
    """save_figure(fig) returns the reference string the store provides,
    so the agent can use it in its prose answer."""
    figure_store.save.return_value = "fake/path/figure_42.png"
    tool = make_python_repl(sample_df, figure_store)
    code = (
        "fig, ax = plt.subplots()\n"
        "ax.bar(['a', 'b'], [1, 2])\n"
        "save_figure(fig)"
    )
    result = tool.invoke({"code": code})
    assert "fake/path/figure_42.png" in result


def test_save_figure_can_be_called_multiple_times(
    sample_df: pd.DataFrame, figure_store: MagicMock
) -> None:
    """The agent can save multiple figures in a single tool call.
    Each call hits the store independently — no batching, no dedup."""
    figure_store.save.side_effect = ["path/one.png", "path/two.png"]
    tool = make_python_repl(sample_df, figure_store)
    code = (
        "fig1, ax1 = plt.subplots()\n"
        "ax1.bar(['a'], [1])\n"
        "p1 = save_figure(fig1)\n"
        "fig2, ax2 = plt.subplots()\n"
        "ax2.bar(['b'], [2])\n"
        "p2 = save_figure(fig2)\n"
        "(p1, p2)"
    )
    result = tool.invoke({"code": code})

    assert figure_store.save.call_count == 2
    assert "path/one.png" in result
    assert "path/two.png" in result


def test_no_save_figure_call_means_no_store_call(
    sample_df: pd.DataFrame, figure_store: MagicMock
) -> None:
    """If the agent doesn't call save_figure, the store is not invoked.
    This is the load-bearing property of explicit capture: the tool does
    not save figures speculatively based on matplotlib state."""
    tool = make_python_repl(sample_df, figure_store)
    # Code that creates a figure but never calls save_figure.
    code = (
        "fig, ax = plt.subplots()\n"
        "ax.bar(['a', 'b'], [1, 2])"
    )
    tool.invoke({"code": code})
    figure_store.save.assert_not_called()


def test_non_plotting_code_never_calls_store(
    sample_df: pd.DataFrame, figure_store: MagicMock
) -> None:
    """Pure calculation code (no plotting, no save_figure) must not touch
    the store. Regression guard for any future implicit-capture drift."""
    tool = make_python_repl(sample_df, figure_store)
    tool.invoke({"code": "df['age'].mean()"})
    tool.invoke({"code": "df.shape"})
    tool.invoke({"code": "x = 1 + 1"})
    figure_store.save.assert_not_called()


# --- Error handling ---


def test_syntax_error_returns_error_string(
    sample_df: pd.DataFrame, figure_store: MagicMock
) -> None:
    tool = make_python_repl(sample_df, figure_store)
    result = tool.invoke({"code": "this is not valid python !!!"})
    assert result.startswith("Error: SyntaxError")


def test_runtime_error_returns_error_string(
    sample_df: pd.DataFrame, figure_store: MagicMock
) -> None:
    tool = make_python_repl(sample_df, figure_store)
    result = tool.invoke({"code": "df['nonexistent_column']"})
    assert result.startswith("Error: KeyError")


def test_indentation_error_returns_error_string(
    sample_df: pd.DataFrame, figure_store: MagicMock
) -> None:
    """Regression guard: the agent has been observed emitting code with stray
    leading indentation. The tool must report it cleanly, not crash."""
    tool = make_python_repl(sample_df, figure_store)
    result = tool.invoke({"code": " df.head()"})
    assert result.startswith("Error: IndentationError")


def test_tool_never_raises(sample_df: pd.DataFrame, figure_store: MagicMock) -> None:
    """Whatever the input, the tool returns a string. It must never raise —
    if it did, the agent loop would crash on bad code."""
    tool = make_python_repl(sample_df, figure_store)
    for bad_code in ["1/0", "import nonexistent_module", "raise RuntimeError('boom')"]:
        result = tool.invoke({"code": bad_code})
        assert isinstance(result, str)
        assert result.startswith("Error:")


def test_save_figure_error_propagates_as_string(
    sample_df: pd.DataFrame, figure_store: MagicMock
) -> None:
    """If the figure store raises (disk full, S3 timeout, etc.), the error
    surfaces to the agent as a string just like any other tool error."""
    figure_store.save.side_effect = RuntimeError("disk full")
    tool = make_python_repl(sample_df, figure_store)
    code = (
        "fig, ax = plt.subplots()\n"
        "ax.bar(['a'], [1])\n"
        "save_figure(fig)"
    )
    result = tool.invoke({"code": code})
    assert "Error: RuntimeError" in result
    assert "disk full" in result


# --- Factory independence ---


def test_separate_factories_have_independent_dataframes() -> None:
    """make_python_repl returns a fresh tool with its own closure. Two
    factory calls must not share DataFrame state."""
    df_a = pd.DataFrame({"x": [1, 2, 3]})
    df_b = pd.DataFrame({"x": [10, 20, 30, 40]})
    store = MagicMock(spec=FigureStoreInterface)

    tool_a = make_python_repl(df_a, store)
    tool_b = make_python_repl(df_b, store)

    assert tool_a.invoke({"code": "len(df)"}) == "3"
    assert tool_b.invoke({"code": "len(df)"}) == "4"


def test_separate_factories_have_independent_figure_stores() -> None:
    """Each factory's save_figure closes over its own store. Calling
    save_figure in one tool's code must not hit another tool's store."""
    store_a = MagicMock(spec=FigureStoreInterface)
    store_a.save.return_value = "store_a/figure.png"
    store_b = MagicMock(spec=FigureStoreInterface)
    store_b.save.return_value = "store_b/figure.png"

    df = pd.DataFrame({"x": [1, 2, 3]})
    tool_a = make_python_repl(df, store_a)
    tool_b = make_python_repl(df, store_b)

    code = (
        "fig, ax = plt.subplots()\n"
        "ax.bar(['a'], [1])\n"
        "save_figure(fig)"
    )
    tool_a.invoke({"code": code})

    store_a.save.assert_called_once()
    store_b.save.assert_not_called()