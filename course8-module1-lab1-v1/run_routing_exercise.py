from pathlib import Path
from routing_exercise import ExerciseRouterState, app
from shared import describe_graph, render_graph_png



def main() -> None:
    describe_graph(app)
    render_graph_png(app, str(Path(__file__).parent / "graph_routing_exercise.png"))


    test_cases: list[ExerciseRouterState] = [
        {
            "user_input": "I need a ride from downtown to the airport at 3pm",
            "task_type": "",
            "output": "",
        },
        {
            "user_input": "I want to order 2 large pepperoni pizzas for delivery",
            "task_type": "",
            "output": "",
        },
        {
            "user_input": "I need milk, bread, eggs, and vegetables for the week",
            "task_type": "",
            "output": "",
        },
        {
            "user_input": "What's the weather like today?",
            "task_type": "",
            "output": "",
        },
    ]

    for i, test_input in enumerate(test_cases, start=1):
        print(f"\n--- Demo {i} ---")
        print(f"User input: {test_input['user_input']}")
        result = app.invoke(
            test_input,
            config={"configurable": {"thread_id": f"routing-exercise-{i}"}},
        )
        print(f"Task type: {result['task_type']}")
        print(f"Output: {result['output']}")
        print("-" * 40)


if __name__ == "__main__":
    main()