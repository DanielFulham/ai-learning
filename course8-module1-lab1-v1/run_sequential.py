from pathlib import Path

from sequential import ChainState, app
from shared import describe_graph, render_graph_png


def main() -> None:
    describe_graph(app)
    render_graph_png(app, str(Path(__file__).parent / "graph_sequential.png"))

    input_state: ChainState = {
        "job_description": (
            "We are looking for a data scientist with experience in machine "
            "learning, NLP, and Python. Prior work with large datasets and "
            "experience deploying models into production is required."
        ),
        "resume_summary": "",
        "cover_letter": "",
    }

    result = app.invoke(input_state, config={"configurable": {"thread_id": "sequential-demo-1"}})

    print("\nRESUME SUMMARY")
    print("==============")
    print(result["resume_summary"])
    print("\nCOVER LETTER")
    print("============")
    print(result["cover_letter"])


if __name__ == "__main__":
    main()