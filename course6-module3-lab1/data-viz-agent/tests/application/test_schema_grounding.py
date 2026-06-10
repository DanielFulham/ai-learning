"""Tests for application.schema_grounding.

build_system_prompt is a pure function over a DataFrame — no I/O, no LLM.
Tests verify it injects the right schema info and includes the behavioural
rules that govern agent behaviour at runtime.

These are unit tests. They check the prompt's structure and content without
calling an LLM. Behavioural tests against a real LLM (verifying that the
prompt actually achieves correct code generation) belong in a separate
integration suite.
"""

import pandas as pd
import pytest

from application.schema_grounding import build_system_prompt


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Small DataFrame standing in for any tabular data."""
    return pd.DataFrame({
        "name": ["Alice", "Bob", "Carol", "Dan"],
        "age": [30, 25, 35, 28],
        "active": [True, False, True, True],
    })


def test_prompt_states_row_and_column_counts(sample_df: pd.DataFrame) -> None:
    prompt = build_system_prompt(sample_df)
    assert "4 rows" in prompt
    assert "3 columns" in prompt


def test_prompt_includes_all_column_names(sample_df: pd.DataFrame) -> None:
    prompt = build_system_prompt(sample_df)
    for column in sample_df.columns:
        assert column in prompt


def test_prompt_includes_dtype_section(sample_df: pd.DataFrame) -> None:
    """The schema must surface column types in some form. The exact dtype
    label (e.g. 'object' vs 'str', 'int64' vs 'Int64') is a pandas-version
    detail and is not pinned here — only that the section exists."""
    prompt = build_system_prompt(sample_df)
    assert "Column types:" in prompt


def test_prompt_distinguishes_numeric_from_other_columns(sample_df: pd.DataFrame) -> None:
    """The LLM needs to distinguish numeric columns from string/bool columns
    so it generates correct code (arithmetic on numbers, quotes on strings,
    masks on booleans). We test that the dtype label on each column's line
    contains a recognisable token of the right family — without pinning the
    exact dtype string, which varies by pandas version."""
    prompt = build_system_prompt(sample_df)
    age_line = _dtype_line_for(prompt, "age")
    active_line = _dtype_line_for(prompt, "active")

    assert "int" in age_line.lower()
    assert "bool" in active_line.lower()


def test_prompt_includes_sample_rows(sample_df: pd.DataFrame) -> None:
    prompt = build_system_prompt(sample_df)
    assert "Alice" in prompt
    assert "Bob" in prompt
    assert "Carol" in prompt
    # Only 3 sample rows are shown — Dan should not appear.
    assert "Dan" not in prompt


def test_prompt_describes_the_dataframe_variable(sample_df: pd.DataFrame) -> None:
    """The LLM needs to know the DataFrame is named `df`."""
    prompt = build_system_prompt(sample_df)
    assert "`df`" in prompt


def test_prompt_forbids_plt_show(sample_df: pd.DataFrame) -> None:
    """Regression guard: the plt.show() ban must survive future edits to the prompt."""
    prompt = build_system_prompt(sample_df)
    assert "plt.show" in prompt


def test_prompt_includes_pie_chart_rule(sample_df: pd.DataFrame) -> None:
    """Regression guard: pie-chart-for-averages must remain explicitly forbidden."""
    prompt = build_system_prompt(sample_df)
    assert "pie chart" in prompt.lower()
    assert "bar chart" in prompt.lower()


def test_prompt_requests_numerical_values_in_answers(sample_df: pd.DataFrame) -> None:
    """Regression guard: the instruction to surface numbers in answers must persist."""
    prompt = build_system_prompt(sample_df)
    assert "numerical values" in prompt.lower()


def test_empty_dataframe_produces_valid_prompt() -> None:
    """Edge case: prompt builder must not crash on an empty DataFrame."""
    empty_df = pd.DataFrame({"a": [], "b": []})
    prompt = build_system_prompt(empty_df)
    assert "0 rows" in prompt
    assert "2 columns" in prompt


def _dtype_line_for(prompt: str, column_name: str) -> str:
    """Return the schema line for a given column (excludes the header line)."""
    for line in prompt.splitlines():
        if column_name in line and "Column types" not in line:
            return line
    raise AssertionError(f"No schema line found for column {column_name!r}")