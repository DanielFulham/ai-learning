from pathlib import Path
from routing import RouterState, app
from shared import describe_graph, render_graph_png


def main() -> None:
    describe_graph(app)
    render_graph_png(app, str(Path(__file__).parent / "graph_routing.png"))

    demos: list[RouterState] = [
        {
            "user_input": "Can you translate this sentence: I love programming?",
            "task_type": "",
            "output": "",
        },
        {
            "user_input": (
                "Can you summarize this sentence: I love programming so much "
                "it is the best thing ever. All I want to do is programming?"
            ),
            "task_type": "",
            "output": "",
        },
    ]

    for i, router_state in enumerate(demos, start=1):
        print(f"\n--- Demo {i} ---")
        result = app.invoke(
            router_state,
            config={"configurable": {"thread_id": f"routing-demo-{i}"}},
        )
        print(f"Task type: {result['task_type']}")
        print(f"Output: {result['output']}")


if __name__ == "__main__":
    main()