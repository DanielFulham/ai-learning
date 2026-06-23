import argparse
import sys

from dotenv import load_dotenv

from application.container import initialise
from application.lab_app import LabApp

load_dotenv()


def run_auth(app: LabApp) -> None:
    """Run the Auth workflow — uses whichever InputProvider the container
    was built with (Console interactively, Scripted for the `all` flow)."""
    print("\n[Auth] " + "-" * 60)
    print("Please authenticate to continue...\n")
    credentials = app.auth.run()
    print(f"\nResult: {credentials.message}")


def run_qa(app: LabApp) -> None:
    """Run the QA workflow against three canned questions.

    The third question deliberately triggers V1's hallucination case —
    keyword match on 'guided project' returns LangGraph context for a
    question whose actual subject is project recommendations."""
    print("\n[QA] " + "-" * 60)
    questions = [
        "What is the weather today?",
        "What is LangGraph?",
        "What is the best guided project?",
    ]
    for question in questions:
        print(f"\nQ: {question}")
        exchange = app.qa.run(question)
        print(f"A: {exchange.answer}")


def run_counter(app: LabApp) -> None:
    """Run the Counter workflow — 13-iteration cycle to termination."""
    print("\n[Counter] " + "-" * 60)
    tick = app.counter.run()
    print(f"\nFinal tick: n={tick.n}, letter={tick.letter!r}")


def run_all(app: LabApp) -> None:
    """Run all three workflows back to back.

    Auth runs first as the integrated greeter. Its verdict is logged but
    does NOT gate the others. Counter and QA always run. Coupling between
    workloads belongs here at the entry point, not inside any graph.
    """
    run_auth(app)
    run_counter(app)
    run_qa(app)


def main(argv: list[str] | None = None) -> int:
    """Parse argv, build the LabApp, dispatch to the chosen workflow."""
    parser = argparse.ArgumentParser(
        prog="demo",
        description="Course 7 Lab 27 V2 — LangGraph 101 onion-architected demo",
    )
    parser.add_argument(
        "workflow",
        choices=["auth", "qa", "counter", "all"],
        help="Which workflow to run",
    )
    parser.add_argument(
        "--provider",
        choices=["ollama", "openai"],
        default="ollama",
        help="Chat model provider (default: ollama)",
    )
    args = parser.parse_args(argv)

    use_scripted_auth_input = args.workflow == "all"
    use_openai = args.provider == "openai"

    app = initialise(
        use_openai=use_openai,
        use_scripted_auth_input=use_scripted_auth_input,
    )

    dispatch = {
        "auth": run_auth,
        "qa": run_qa,
        "counter": run_counter,
        "all": run_all,
    }
    dispatch[args.workflow](app)
    return 0


if __name__ == "__main__":
    sys.exit(main())