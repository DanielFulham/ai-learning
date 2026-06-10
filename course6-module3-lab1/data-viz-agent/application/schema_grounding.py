"""System prompt construction from a pandas DataFrame.

Given a DataFrame, builds the schema-grounded system prompt the agent uses
to instruct the LLM. The prompt has two parts: schema information derived
from the DataFrame itself (column types, sample rows, row count) and a
fixed set of behavioural rules that govern how the agent should produce
answers and charts.

Pure function — no LLM, no I/O. Trivially testable.
"""

import pandas as pd


_BEHAVIOURAL_RULES = """\
Use the python_repl tool to answer questions and create visualizations.
For plots, use matplotlib (already imported as plt). Do not call plt.show() —
it has no effect in this environment.

To save a chart, call save_figure(fig) where fig is the matplotlib Figure
object. It returns the saved path, which you should include in your answer
so the user knows where to find the chart. Always call save_figure for any
chart you create.

CHART TYPE RULES (these override user requests):
- A pie chart is ONLY valid when the values represent parts of a whole that
  sum to 100%. If a user requests a pie chart for averages, means, totals
  per category, or any other non-share data, you MUST use a bar chart
  instead. Briefly explain why in your answer.

When the user asks a question, run the code needed to answer it, then
state the answer in plain English along with the code you ran. When
producing a chart that compares values across categories or shows a
correlation, state the actual numerical values in your answer.
"""


def build_system_prompt(df: pd.DataFrame) -> str:
    """Construct the agent's system prompt from a DataFrame's schema.

    The returned string is passed directly to create_agent as the system
    prompt. It contains schema information (columns, dtypes, sample rows)
    so the LLM can generate correct code without first calling the tool
    to inspect the DataFrame.
    """
    schema_info = (
        f"DataFrame `df` has {len(df)} rows and {len(df.columns)} columns.\n\n"
        f"Column types:\n{df.dtypes.to_string()}\n\n"
        f"First 3 rows:\n{df.head(3).to_string()}\n"
    )

    return (
        "You are a data analyst with access to a pandas DataFrame called `df`.\n\n"
        f"{schema_info}\n"
        f"{_BEHAVIOURAL_RULES}"
    )