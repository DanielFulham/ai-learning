"""Debug the Dalc/Walc figure-capture bug in isolation."""

from dotenv import load_dotenv
import matplotlib

matplotlib.use("Agg")

import pandas as pd

from application.container import initialise


def main() -> None:
    load_dotenv()
    df = pd.read_csv("data/student-mat.csv")

    agent = initialise(df=df)

    query = (
        "Generate scatter plots to examine the correlation between 'Dalc' "
        "(daily alcohol consumption) and 'G3', and between 'Walc' "
        "(weekend alcohol consumption) and 'G3'."
    )

    trace = agent.run_with_trace(query)
    for i, call in enumerate(trace.tool_calls):
        print(f"\n--- Tool call {i + 1} ---")
        print(f"Code:\n{call.args.get('code')}")
        print(f"\nResult:\n{call.result}")

    print(f"\n--- Final answer ---\n{trace.final_answer}")


if __name__ == "__main__":
    main()