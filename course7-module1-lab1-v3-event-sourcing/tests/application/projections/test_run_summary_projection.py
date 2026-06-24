from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID, uuid4

import pytest

from application.projections.run_summary_projection import summarise_run
from domain.error_info import ErrorInfo
from domain.events.base import BaseAgentEvent
from domain.events.qa_events import (
    AnswerGenerated,
    ContextRetrieved,
    ModelInvocationFailed,
    QuestionReceived,
)


_T0 = datetime(2026, 6, 23, 12, 0, tzinfo=timezone.utc)


def _at(seconds: int) -> datetime:
    return _T0 + timedelta(seconds=seconds)


def _common(run_id: UUID, seconds: int) -> dict[str, Any]:
    return {
        "event_id": uuid4(),
        "aggregate_id": run_id,
        "occurred_at": _at(seconds),
    }


class TestSummariseRunEmpty:

    def test_empty_events_returns_none(self) -> None:
        """Pinned: empty is no-run, not degenerate run. Caller distinguishes
        absent from incomplete by checking against None vs status='incomplete'."""
        assert summarise_run([]) is None


class TestSummariseRunMultiAggregateGuard:

    def test_events_from_multiple_runs_raises(self) -> None:
        """Pinned: input contract is one aggregate_id. The projection
        does not silently merge or split across runs."""
        run_a = uuid4()
        run_b = uuid4()
        events: list[BaseAgentEvent] = [
            QuestionReceived(**_common(run_a, 0), question="Q1"),
            QuestionReceived(**_common(run_b, 1), question="Q2"),
        ]

        with pytest.raises(ValueError, match="multiple aggregate_ids"):
            summarise_run(events)


class TestSummariseRunSuccessPath:

    def test_question_context_answer_yields_success(self) -> None:
        run_id = uuid4()
        info_q = QuestionReceived(**_common(run_id, 0), question="Q?")
        info_c = ContextRetrieved(**_common(run_id, 1), context="C")
        info_a = AnswerGenerated(**_common(run_id, 2), answer="A")

        summary = summarise_run([info_q, info_c, info_a])

        assert summary is not None
        assert summary.run_id == run_id
        assert summary.started_at == _at(0)
        assert summary.completed_at == _at(2)
        assert summary.question == "Q?"
        assert summary.final_status == "success"
        assert summary.answer == "A"
        assert summary.error_info is None
        assert summary.event_count == 3


class TestSummariseRunFailurePath:

    def test_question_context_model_failed_yields_failed(self) -> None:
        run_id = uuid4()
        info = ErrorInfo(
            exception_type="OllamaConnectionError",
            exception_message="Connection refused",
        )
        events: list[BaseAgentEvent] = [
            QuestionReceived(**_common(run_id, 0), question="Q?"),
            ContextRetrieved(**_common(run_id, 1), context="C"),
            ModelInvocationFailed(**_common(run_id, 2), error_info=info),
        ]

        summary = summarise_run(events)

        assert summary is not None
        assert summary.final_status == "failed"
        assert summary.error_info == info
        assert summary.answer is None
        assert summary.completed_at == _at(2)
        assert summary.event_count == 3

    def test_failure_wins_over_success_if_both_somehow_present(self) -> None:
        """Pinned: if AnswerGenerated and ModelInvocationFailed both appear
        for the same run (shouldn't happen by graph topology, but defensive),
        the summary reports 'failed'. Matches the translator's
        failure-precedence dispatch."""
        run_id = uuid4()
        info = ErrorInfo(exception_type="X", exception_message="Y")
        events: list[BaseAgentEvent] = [
            QuestionReceived(**_common(run_id, 0), question="Q?"),
            AnswerGenerated(**_common(run_id, 1), answer="A"),
            ModelInvocationFailed(**_common(run_id, 2), error_info=info),
        ]

        summary = summarise_run(events)

        assert summary is not None
        assert summary.final_status == "failed"
        assert summary.error_info == info


class TestSummariseRunIncompletePath:

    def test_question_only_yields_incomplete(self) -> None:
        """Pinned: the QuestionReceived event alone means the run started
        but didn't reach a terminating node. Status incomplete; answer
        and error_info are None; completed_at is None."""
        run_id = uuid4()
        events: list[BaseAgentEvent] = [
            QuestionReceived(**_common(run_id, 0), question="Q?"),
        ]

        summary = summarise_run(events)

        assert summary is not None
        assert summary.final_status == "incomplete"
        assert summary.answer is None
        assert summary.error_info is None
        assert summary.completed_at is None
        assert summary.event_count == 1
        assert summary.question == "Q?"

    def test_question_and_context_yields_incomplete(self) -> None:
        """Pinned: graph stopped after ContextNode but before QANode emitted.
        Context was retrieved but no answer or failure event fired."""
        run_id = uuid4()
        events: list[BaseAgentEvent] = [
            QuestionReceived(**_common(run_id, 0), question="Q?"),
            ContextRetrieved(**_common(run_id, 1), context="C"),
        ]

        summary = summarise_run(events)

        assert summary is not None
        assert summary.final_status == "incomplete"
        assert summary.completed_at is None
        assert summary.event_count == 2


class TestSummariseRunDerivedFields:

    def test_started_at_is_first_events_occurred_at(self) -> None:
        run_id = uuid4()
        events: list[BaseAgentEvent] = [
            QuestionReceived(**_common(run_id, 10), question="Q?"),
            AnswerGenerated(**_common(run_id, 20), answer="A"),
        ]

        summary = summarise_run(events)
        assert summary is not None
        assert summary.started_at == _at(10)

    def test_run_id_is_first_events_aggregate_id(self) -> None:
        run_id = uuid4()
        events: list[BaseAgentEvent] = [
            QuestionReceived(**_common(run_id, 0), question="Q?"),
        ]

        summary = summarise_run(events)
        assert summary is not None
        assert summary.run_id == run_id

    def test_event_count_matches_input_length(self) -> None:
        run_id = uuid4()
        events: list[BaseAgentEvent] = [
            QuestionReceived(**_common(run_id, 0), question="Q?"),
            ContextRetrieved(**_common(run_id, 1), context="C"),
            ContextRetrieved(**_common(run_id, 2), context=None),  # duplicate
            AnswerGenerated(**_common(run_id, 3), answer="A"),
        ]

        summary = summarise_run(events)
        assert summary is not None
        assert summary.event_count == 4