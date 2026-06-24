"""V3a QA demo.

Runs three canned QA questions through the event-sourced QA workflow,
then prints the RunSummary for each by replaying the event log.

Flags:
    --provider {ollama,openai}     LLM provider (default: ollama)
    --persistence {memory,sqlite}  Event store backend (default: memory)
    --db-path PATH                 SQLite DB path (default: ./events.db
                                    when persistence=sqlite)
    --quiet                        Suppress node-update console output;
                                    only the summaries print

Usage:
    python demo.py                                # in-memory default, Ollama
    python demo.py --persistence sqlite           # writes to ./events.db
    python demo.py --persistence sqlite --db-path runs.db --quiet
    python demo.py --provider openai              # requires OPENAI_API_KEY in .env

    # Flags compose freely:
    python demo.py --provider openai --persistence sqlite
    python demo.py --provider openai --persistence sqlite --db-path openai-runs.db --quiet

"""

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

from application.container import initialise
from application.lab_app import LabApp
from application.projections.run_summary_projection import summarise_run
from domain.run_result import RunResult
from domain.run_summary import RunSummary


_CANNED_QUESTIONS = [
    "What is LangGraph?",
    "How do conditional edges work in LangGraph?",
    "What is the best guided project?",
]
"""Three canned QA prompts. The third is the V2-preserved hallucination
demo — the question's subject is not about LangGraph, but
context_provider_node's keyword match on 'guided project' returns
LangGraph context regardless. The summaries make the mismatch visible.
"""


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="V3a QA demo — runs three canned questions and "
        "prints run summaries from the event log.",
    )
    parser.add_argument(
        "--provider",
        choices=["ollama", "openai"],
        default="ollama",
        help="LLM provider (default: ollama)",
    )
    parser.add_argument(
        "--persistence",
        choices=["memory", "sqlite"],
        default="memory",
        help="Event store backend (default: memory)",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=None,
        help="SQLite DB path (defaults to ./events.db when "
        "persistence=sqlite)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress node-update console output; only summaries print",
    )
    return parser.parse_args(argv)


def _format_summary(index: int, result: RunResult, summary: RunSummary) -> str:
    """Format one run's summary as a human-readable block."""
    completed = (
        summary.completed_at.isoformat() if summary.completed_at else "—"
    )
    answer_or_error = (
        summary.answer
        if summary.answer is not None
        else f"({summary.error_info.exception_type}) "
        f"{summary.error_info.exception_message}"
        if summary.error_info is not None
        else "(no terminal event)"
    )
    return (
        f"\nRun {index} ({result.run_id}):\n"
        f"  Question:    {summary.question}\n"
        f"  Status:      {summary.final_status}\n"
        f"  Started:     {summary.started_at.isoformat()}\n"
        f"  Completed:   {completed}\n"
        f"  Events:      {summary.event_count}\n"
        f"  Answer:      {answer_or_error}"
    )


def _run_demo(app: LabApp) -> list[RunResult]:
    results: list[RunResult] = []
    for index, question in enumerate(_CANNED_QUESTIONS, start=1):
        print(f"\n{'=' * 60}")
        print(f"Question {index}: {question}")
        print(f"{'=' * 60}")
        result = app.qa.run(question)
        results.append(result)
        print(f"\nAnswer: {result.exchange.answer}")
    return results


def _print_summaries(app: LabApp, results: list[RunResult]) -> None:
    print(f"\n{'=' * 60}")
    print("RUN SUMMARIES")
    print(f"{'=' * 60}")
    for index, result in enumerate(results, start=1):
        events = app.event_store.events_for_run(result.run_id)
        summary = summarise_run(events)
        if summary is None:
            print(f"\nRun {index} ({result.run_id}): no events recorded")
            continue
        print(_format_summary(index, result, summary))


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    load_dotenv()

    db_path = args.db_path
    if args.persistence == "sqlite" and db_path is None:
        db_path = Path("events.db")

    app = initialise(
        use_openai=(args.provider == "openai"),
        use_sqlite_persistence=(args.persistence == "sqlite"),
        db_path=db_path,
        use_console_consumer=not args.quiet,
    )

    print(f"V3a QA demo")
    print(f"Provider:    {args.provider}")
    print(f"Persistence: {args.persistence}")
    if args.persistence == "sqlite":
        print(f"DB path:     {db_path}")
    print(f"Console:     {'quiet' if args.quiet else 'enabled'}")

    results = _run_demo(app)
    _print_summaries(app, results)

    return 0


if __name__ == "__main__":
    sys.exit(main())