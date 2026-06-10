"""Demo: run the lab's seven queries against the modular DataVizAgent.

Demonstrates that the onion-architected agent produces the same outputs as
the script in app.py, with all the seams in place: LLM provider, figure
store, tool, system prompt, and agent are independently swappable.

Run with:

    python demo.py

Outputs land in ./output/figure_NN.png, numbered across runs.
"""

from dotenv import load_dotenv
import matplotlib

# Non-interactive matplotlib backend — must precede pyplot import.
matplotlib.use("Agg")

import pandas as pd

from application.container import initialise
from domain.agent_trace import AgentTrace


QUERIES = [
    "How many rows of data are in this file?",
    "Give me all the data where student's age is over 18 years old.",
    "Generate a bar chart to plot the gender count.",
    "Generate a pie chart to display average value of Walc for each Gender.",
    "Create box plots to analyze the relationship between 'freetime' (amount of free time) "
        "and 'G3' (final grade) across different levels of free time.",
    "Generate scatter plots to examine the correlation between 'Dalc' (daily alcohol "
        "consumption) and 'G3', and between 'Walc' (weekend alcohol consumption) and 'G3'.",
    "Plot a scatter plot showing the correlation between the number of absences "
        "('absences') and final grades ('G3') of students.",
]


def format_trace(trace: AgentTrace) -> str:
    """Pretty-print an AgentTrace as human-readable text."""
    lines: list[str] = []
    lines.append(f"\n👤 User: {trace.query}")
    for call in trace.tool_calls:
        code = call.args.get("code", call.args)
        lines.append(f"\n🤖 Tool call → {call.name}:")
        lines.append(f"```python\n{code}\n```")
        lines.append(f"\n🔧 Tool result: {call.result}")
    lines.append(f"\n🤖 Answer: {trace.final_answer}")
    return "\n".join(lines)


def main() -> None:
    load_dotenv()
    df = pd.read_csv("data/student-mat.csv")

    agent = initialise(df=df)

    for query in QUERIES:
        print(f"\n{'=' * 80}")
        trace = agent.run_with_trace(query)
        print(format_trace(trace))


if __name__ == "__main__":
    main()