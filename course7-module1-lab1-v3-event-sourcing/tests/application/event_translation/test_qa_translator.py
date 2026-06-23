from datetime import datetime, timezone
from typing import Any, Callable
from uuid import UUID, uuid4

import pytest

from application.event_translation.qa_translator import translate_qa_update
from domain.error_info import ErrorInfo
from domain.events.qa_events import (
    AnswerGenerated,
    ContextRetrieved,
    ModelInvocationFailed,
)
from domain.qa_exchange import QAExchange


_FIXED_TIME = datetime(2026, 6, 23, 12, 0, tzinfo=timezone.utc)


def _fixed_clock() -> Callable[[], datetime]:
    return lambda: _FIXED_TIME


def _counting_clock() -> tuple[Callable[[], datetime], dict[str, int]]:
    counter = {"calls": 0}

    def clock() -> datetime:
        counter["calls"] += 1
        return _FIXED_TIME

    return clock, counter


def _make_exchange(
    question: str = "What is LangGraph?",
    context: str | None = None,
    answer: str | None = None,
    error_info: ErrorInfo | None = None,
) -> QAExchange:
    return QAExchange(
        question=question,
        context=context,
        answer=answer,
        error_info=error_info,
    )


class TestTranslateContextNodeUpdate:

    def test_context_present_emits_context_retrieved(self) -> None:
        run_id = uuid4()
        exchange = _make_exchange(context="Some retrieved context.")

        events = translate_qa_update(
            node_name="ContextNode",
            state_delta={"exchange": exchange},
            run_id=run_id,
            clock=_fixed_clock(),
        )

        assert len(events) == 1
        assert isinstance(events[0], ContextRetrieved)
        assert events[0].context == "Some retrieved context."
        assert events[0].aggregate_id == run_id
        assert events[0].occurred_at == _FIXED_TIME

    def test_context_none_emits_context_retrieved_with_none(self) -> None:
        """Pinned: the observability-consistency lift. A missed keyword
        match is recorded as ContextRetrieved(context=None), not as a
        silent absence of the event. ThreadHistoryProjection in V3c joins
        on this."""
        run_id = uuid4()
        exchange = _make_exchange(context=None)

        events = translate_qa_update(
            node_name="ContextNode",
            state_delta={"exchange": exchange},
            run_id=run_id,
            clock=_fixed_clock(),
        )

        assert len(events) == 1
        assert isinstance(events[0], ContextRetrieved)
        assert events[0].context is None


class TestTranslateQANodeUpdateSuccess:

    def test_answer_without_error_info_emits_answer_generated(self) -> None:
        run_id = uuid4()
        exchange = _make_exchange(
            context="Some context.",
            answer="An answer.",
        )

        events = translate_qa_update(
            node_name="QANode",
            state_delta={"exchange": exchange},
            run_id=run_id,
            clock=_fixed_clock(),
        )

        assert len(events) == 1
        assert isinstance(events[0], AnswerGenerated)
        assert events[0].answer == "An answer."
        assert events[0].aggregate_id == run_id
        assert events[0].occurred_at == _FIXED_TIME


class TestTranslateQANodeUpdateFailure:

    def test_error_info_present_emits_model_invocation_failed(self) -> None:
        run_id = uuid4()
        info = ErrorInfo(
            exception_type="OllamaConnectionError",
            exception_message="Connection refused",
        )
        exchange = _make_exchange(
            answer="I couldn't reach the model.",
            error_info=info,
        )

        events = translate_qa_update(
            node_name="QANode",
            state_delta={"exchange": exchange},
            run_id=run_id,
            clock=_fixed_clock(),
        )

        assert len(events) == 1
        assert isinstance(events[0], ModelInvocationFailed)
        assert events[0].error_info == info
        assert events[0].aggregate_id == run_id

    def test_error_info_present_with_answer_takes_failure_branch(self) -> None:
        """Pinned: when both `answer` and `error_info` are present on the
        exchange (the V3a failure-path contract — qa_node sets a user-safe
        answer AND populates error_info), the translator emits
        ModelInvocationFailed, NOT AnswerGenerated. Diagnostic wins;
        the user-safe answer is for the UI, the event is for the log."""
        run_id = uuid4()
        info = ErrorInfo(exception_type="X", exception_message="Y")
        exchange = _make_exchange(
            answer="A user-safe message",
            error_info=info,
        )

        events = translate_qa_update(
            node_name="QANode",
            state_delta={"exchange": exchange},
            run_id=run_id,
            clock=_fixed_clock(),
        )

        assert len(events) == 1
        assert isinstance(events[0], ModelInvocationFailed)


class TestTranslateQANodeInvariants:

    def test_qa_node_without_answer_or_error_info_raises(self) -> None:
        """Pinned: QANode must populate either answer or error_info before
        yielding its update. Neither populated is a node implementation
        bug, not a runtime condition the translator silently handles."""
        run_id = uuid4()
        exchange = _make_exchange(context="C")

        with pytest.raises(ValueError, match="neither answer nor error_info"):
            translate_qa_update(
                node_name="QANode",
                state_delta={"exchange": exchange},
                run_id=run_id,
                clock=_fixed_clock(),
            )


class TestTranslateInvalidInputs:

    def test_unknown_node_name_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown QA node"):
            translate_qa_update(
                node_name="MysteryNode",
                state_delta={"exchange": _make_exchange()},
                run_id=uuid4(),
                clock=_fixed_clock(),
            )

    def test_missing_exchange_in_delta_raises(self) -> None:
        with pytest.raises(ValueError, match="missing or malformed"):
            translate_qa_update(
                node_name="ContextNode",
                state_delta={},
                run_id=uuid4(),
                clock=_fixed_clock(),
            )

    def test_non_qaexchange_in_delta_raises(self) -> None:
        """Pinned: the second narrowing stage. The streaming consumer
        gives us `dict[str, Any]`; the translator narrows the specific
        key against the domain type. A dict value of the wrong type is
        a framework-boundary surprise, raised loudly."""
        wrong_payload: dict[str, Any] = {"exchange": {"not": "a QAExchange"}}
        with pytest.raises(ValueError, match="missing or malformed"):
            translate_qa_update(
                node_name="ContextNode",
                state_delta=wrong_payload,
                run_id=uuid4(),
                clock=_fixed_clock(),
            )


class TestTranslatePropagation:

    def test_run_id_propagated_as_aggregate_id(self) -> None:
        """Pinned: aggregate_id on every emitted event is the run_id passed
        in. This is the link the event store uses for replay-by-run."""
        run_id = uuid4()
        exchange = _make_exchange(answer="A")

        events = translate_qa_update(
            node_name="QANode",
            state_delta={"exchange": exchange},
            run_id=run_id,
            clock=_fixed_clock(),
        )

        assert all(e.aggregate_id == run_id for e in events)

    def test_clock_called_exactly_once_per_emitted_event(self) -> None:
        """Pinned: the clock injection is one call per event. The
        translator does not cache the timestamp across events; if a
        future update emits two events, each gets its own clock reading."""
        clock, counter = _counting_clock()
        exchange = _make_exchange(answer="A")

        events = translate_qa_update(
            node_name="QANode",
            state_delta={"exchange": exchange},
            run_id=uuid4(),
            clock=clock,
        )

        assert len(events) == 1
        assert counter["calls"] == 1

    def test_event_ids_are_unique_across_calls(self) -> None:
        """Pinned: event_id is uuid4 at translation time. Two calls with
        identical inputs produce events with different event_ids — pins
        the 'translator is the canonical source of event IDs' contract."""
        run_id = uuid4()
        exchange = _make_exchange(answer="A")
        delta: dict[str, Any] = {"exchange": exchange}

        first = translate_qa_update("QANode", delta, run_id, _fixed_clock())
        second = translate_qa_update("QANode", delta, run_id, _fixed_clock())

        assert first[0].event_id != second[0].event_id

    def test_schema_version_defaults_to_1_on_emitted_events(self) -> None:
        """Pinned: V3a emits schema_version=1 across all event types.
        Bumping the default is a deliberate cross-cutting decision."""
        run_id = uuid4()
        events = translate_qa_update(
            node_name="QANode",
            state_delta={"exchange": _make_exchange(answer="A")},
            run_id=run_id,
            clock=_fixed_clock(),
        )

        assert all(e.schema_version == 1 for e in events)