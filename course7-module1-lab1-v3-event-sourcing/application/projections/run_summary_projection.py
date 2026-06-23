from datetime import datetime
from typing import Sequence

from domain.error_info import ErrorInfo
from domain.events.base import BaseAgentEvent
from domain.events.qa_events import (
    AnswerGenerated,
    ModelInvocationFailed,
    QuestionReceived,
)
from domain.run_summary import RunStatus, RunSummary


def summarise_run(events: Sequence[BaseAgentEvent]) -> RunSummary | None:
    """Derive a RunSummary from one run's events.

    Pure function. No I/O. Caller is responsible for passing events for
    exactly one run (one aggregate_id); a mixed-aggregate input raises.

    Returns None if events is empty — the caller distinguishes "no events
    for this run" from "events exist but summary is incomplete." Empty
    is a no-run, not a degenerate run.

    For non-empty input:
    - run_id, started_at, event_count derive from the event sequence
    - question, answer, error_info populate from the corresponding event
      types when present
    - completed_at is the occurred_at of the terminating event
      (AnswerGenerated or ModelInvocationFailed), or None for incomplete
    - final_status branches: error_info wins over answer when both somehow
      coexist (matches the translator's failure-wins dispatch)
    """
    if not events:
        return None

    run_id = events[0].aggregate_id
    if any(e.aggregate_id != run_id for e in events):
        raise ValueError(
            "summarise_run received events from multiple aggregate_ids; "
            "pass only events for a single run"
        )

    started_at = events[0].occurred_at
    event_count = len(events)

    question: str | None = None
    answer: str | None = None
    error_info: ErrorInfo | None = None
    completed_at: datetime | None = None

    for event in events:
        if isinstance(event, QuestionReceived):
            question = event.question
        elif isinstance(event, AnswerGenerated):
            answer = event.answer
            completed_at = event.occurred_at
        elif isinstance(event, ModelInvocationFailed):
            error_info = event.error_info
            completed_at = event.occurred_at

    final_status: RunStatus
    if error_info is not None:
        final_status = "failed"
    elif answer is not None:
        final_status = "success"
    else:
        final_status = "incomplete"

    return RunSummary(
        run_id=run_id,
        started_at=started_at,
        completed_at=completed_at,
        question=question,
        final_status=final_status,
        answer=answer,
        error_info=error_info,
        event_count=event_count,
    )