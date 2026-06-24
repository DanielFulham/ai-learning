from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
from typing import Any, assert_never
from uuid import UUID, uuid4

import pytest

from domain.error_info import ErrorInfo
from domain.events.base import BaseAgentEvent
from domain.events.qa_events import (
    AnswerGenerated,
    ContextRetrieved,
    ModelInvocationFailed,
    QAEvent,
    QuestionReceived,
)


def _common_kwargs(
    event_id: UUID | None = None,
    aggregate_id: UUID | None = None,
    occurred_at: datetime | None = None,
) -> dict[str, Any]:
    return {
        "event_id": event_id or uuid4(),
        "aggregate_id": aggregate_id or uuid4(),
        "occurred_at": occurred_at or datetime(2026, 6, 23, 12, 0, tzinfo=timezone.utc),
    }


def _make_question_received(question: str = "What is LangGraph?") -> QuestionReceived:
    return QuestionReceived(**_common_kwargs(), question=question)


def _make_context_retrieved(context: str | None = "Some context.") -> ContextRetrieved:
    return ContextRetrieved(**_common_kwargs(), context=context)


def _make_answer_generated(answer: str = "An answer.") -> AnswerGenerated:
    return AnswerGenerated(**_common_kwargs(), answer=answer)


def _make_model_invocation_failed(
    info: ErrorInfo | None = None,
) -> ModelInvocationFailed:
    return ModelInvocationFailed(
        **_common_kwargs(),
        error_info=info or ErrorInfo(
            exception_type="OllamaConnectionError",
            exception_message="Connection refused",
        ),
    )


class TestQuestionReceived:

    def test_inherits_base_event(self) -> None:
        """Pinned: BaseAgentEvent inheritance is structural — `event_id`,
        `aggregate_id`, `occurred_at`, `schema_version` must be reachable
        on every event type the translator emits."""
        event = _make_question_received()
        assert isinstance(event, BaseAgentEvent)

    def test_carries_question_payload(self) -> None:
        event = _make_question_received(question="What is event sourcing?")
        assert event.question == "What is event sourcing?"

    def test_is_frozen(self) -> None:
        event = _make_question_received()
        with pytest.raises(FrozenInstanceError):
            setattr(event, "question", "different")


class TestContextRetrieved:

    def test_inherits_base_event(self) -> None:
        event = _make_context_retrieved()
        assert isinstance(event, BaseAgentEvent)

    def test_carries_context_when_matched(self) -> None:
        event = _make_context_retrieved(context="Some retrieved context.")
        assert event.context == "Some retrieved context."

    def test_carries_none_when_no_match(self) -> None:
        """Pinned: None records a keyword-match miss explicitly. The translator
        always emits the event; the event payload distinguishes hit from miss.
        ThreadHistoryProjection joins on this against QuestionReceived."""
        event = _make_context_retrieved(context=None)
        assert event.context is None

    def test_is_frozen(self) -> None:
        event = _make_context_retrieved()
        with pytest.raises(FrozenInstanceError):
            setattr(event, "context", "different")


class TestAnswerGenerated:

    def test_inherits_base_event(self) -> None:
        event = _make_answer_generated()
        assert isinstance(event, BaseAgentEvent)

    def test_carries_answer_payload(self) -> None:
        event = _make_answer_generated(answer="The answer is 42.")
        assert event.answer == "The answer is 42."

    def test_is_frozen(self) -> None:
        event = _make_answer_generated()
        with pytest.raises(FrozenInstanceError):
            setattr(event, "answer", "different")


class TestModelInvocationFailed:

    def test_inherits_base_event(self) -> None:
        event = _make_model_invocation_failed()
        assert isinstance(event, BaseAgentEvent)

    def test_carries_error_info_payload(self) -> None:
        info = ErrorInfo(
            exception_type="httpx.HTTPError",
            exception_message="503 Service Unavailable",
        )
        event = _make_model_invocation_failed(info=info)
        assert event.error_info == info
        assert event.error_info.exception_type == "httpx.HTTPError"
        assert event.error_info.exception_message == "503 Service Unavailable"

    def test_is_frozen(self) -> None:
        event = _make_model_invocation_failed()
        replacement = ErrorInfo(exception_type="X", exception_message="Y")
        with pytest.raises(FrozenInstanceError):
            setattr(event, "error_info", replacement)


class TestQAEventUnion:

    def test_exhaustive_match_covers_every_member(self) -> None:
        """Pinned: QAEvent is closed within the QA service. The translator
        boundary uses match-case + typing.assert_never to enforce
        exhaustiveness at type-check time. This test instantiates one of
        each variant and routes through a placeholder match — if a new
        member is added to QAEvent without updating this match, pyright
        flags assert_never as reachable with an unhandled type.

        The placeholder handler is intentionally minimal — V3a's actual
        translator-output dispatch (event store append, projection update)
        lands in the EventTranslatingStreamConsumer commit. This test pins
        the union shape, not the dispatch behaviour.
        """

        def handle(event: QAEvent) -> str:
            match event:
                case QuestionReceived():
                    return "question_received"
                case ContextRetrieved():
                    return "context_retrieved"
                case AnswerGenerated():
                    return "answer_generated"
                case ModelInvocationFailed():
                    return "model_invocation_failed"
                case _:
                    assert_never(event)

        assert handle(_make_question_received()) == "question_received"
        assert handle(_make_context_retrieved()) == "context_retrieved"
        assert handle(_make_answer_generated()) == "answer_generated"
        assert handle(_make_model_invocation_failed()) == "model_invocation_failed"