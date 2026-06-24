from datetime import datetime, timezone
from typing import Any, Callable
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest

from application.event_translating_stream_consumer import (
    EventTranslatingStreamConsumer,
    TranslatorFunction,
)
from domain.events.base import BaseAgentEvent
from domain.events.qa_events import AnswerGenerated, ContextRetrieved
from interfaces.agent_event_store_interface import AgentEventStoreInterface
from interfaces.stream_consumer_interface import StreamConsumerInterface


_FIXED_TIME = datetime(2026, 6, 23, 12, 0, tzinfo=timezone.utc)


def _fixed_clock() -> Callable[[], datetime]:
    return lambda: _FIXED_TIME


def _make_translator(
    events_to_return: list[BaseAgentEvent] | None = None,
) -> MagicMock:
    """A MagicMock typed against the translator function shape, with a
    spec-equivalent calling pattern. Returns the given events list (or
    empty by default)."""
    translator = MagicMock(return_value=events_to_return or [])
    return translator


def _make_context_retrieved_event(run_id: UUID) -> ContextRetrieved:
    return ContextRetrieved(
        event_id=uuid4(),
        aggregate_id=run_id,
        occurred_at=_FIXED_TIME,
        context="C",
    )


def _make_answer_generated_event(run_id: UUID) -> AnswerGenerated:
    return AnswerGenerated(
        event_id=uuid4(),
        aggregate_id=run_id,
        occurred_at=_FIXED_TIME,
        answer="A",
    )


def _make_consumer(
    run_id: UUID | None = None,
    event_store: AgentEventStoreInterface | None = None,
    inner_consumer: StreamConsumerInterface | None = None,
    translator: TranslatorFunction | None = None,
) -> EventTranslatingStreamConsumer:
    return EventTranslatingStreamConsumer(
        run_id=run_id or uuid4(),
        event_store=event_store or MagicMock(spec=AgentEventStoreInterface),
        inner_consumer=inner_consumer or MagicMock(spec=StreamConsumerInterface),
        translator=translator or _make_translator(),
        clock=_fixed_clock(),
    )


def _accepts_consumer(consumer: StreamConsumerInterface) -> None:
    """Type-guard helper."""


class TestEventTranslatingStreamConsumerInterfaceSatisfaction:

    def test_satisfies_stream_consumer_interface(self) -> None:
        """Pinned: the decorator concrete satisfies StreamConsumerInterface
        structurally. Service code typed against the interface can hold
        either V2's ConsoleStreamConsumer or V3's EventTranslatingStreamConsumer
        without change."""
        _accepts_consumer(_make_consumer())


class TestEventTranslatingStreamConsumerFanOut:

    def test_calls_inner_consumer_with_same_args(self) -> None:
        """Pinned: the inner consumer sees the same (node_name, state_delta)
        the decorator received. Wrapping is transparent to the inner."""
        inner = MagicMock(spec=StreamConsumerInterface)
        consumer = _make_consumer(inner_consumer=inner)

        delta = {"exchange": "anything"}
        consumer.on_update("SomeNode", delta)

        inner.on_update.assert_called_once_with("SomeNode", delta)

    def test_calls_translator_with_node_name_delta_run_id_and_clock(self) -> None:
        run_id = uuid4()
        translator = _make_translator()
        consumer = _make_consumer(run_id=run_id, translator=translator)

        delta: dict[str, Any] = {"exchange": "anything"}
        consumer.on_update("SomeNode", delta)

        assert translator.call_count == 1
        args = translator.call_args.args
        assert args[0] == "SomeNode"
        assert args[1] == delta
        assert args[2] == run_id
        # args[3] is the clock callable — verified by behaviour, not identity
        clock_arg = args[3]
        assert clock_arg() == _FIXED_TIME

    def test_appends_every_translator_event_to_the_store(self) -> None:
        """Pinned: every event returned by the translator is appended in
        order. Multi-event updates fan out cleanly."""
        run_id = uuid4()
        event_a = _make_context_retrieved_event(run_id)
        event_b = _make_answer_generated_event(run_id)
        store = MagicMock(spec=AgentEventStoreInterface)
        translator = _make_translator(events_to_return=[event_a, event_b])
        consumer = _make_consumer(
            run_id=run_id, event_store=store, translator=translator
        )

        consumer.on_update("SomeNode", {})

        # Two appends, in order
        assert store.append.call_count == 2
        first_append_event = store.append.call_args_list[0].args[0]
        second_append_event = store.append.call_args_list[1].args[0]
        assert first_append_event is event_a
        assert second_append_event is event_b

    def test_empty_translator_output_appends_nothing(self) -> None:
        """Pinned: a translator returning an empty list is valid. Nothing
        is appended to the store. Currently every V3a translator returns
        exactly one event per update, but the empty case is the contract
        for translators that choose to discard certain node updates
        (e.g. synthetic framework nodes a future translator might filter)."""
        store = MagicMock(spec=AgentEventStoreInterface)
        consumer = _make_consumer(
            event_store=store, translator=_make_translator(events_to_return=[])
        )

        consumer.on_update("SomeNode", {})

        store.append.assert_not_called()


class TestEventTranslatingStreamConsumerOrdering:

    def test_inner_called_before_translation(self) -> None:
        """Pinned: inner consumer fires before translation. Dev visibility
        is preserved even if translation raises — the print trail is
        intact when reading a crash report."""
        call_order: list[str] = []

        inner = MagicMock(spec=StreamConsumerInterface)
        inner.on_update.side_effect = lambda *_: call_order.append("inner")

        translator = MagicMock(
            side_effect=lambda *_: call_order.append("translator") or []
        )

        consumer = _make_consumer(inner_consumer=inner, translator=translator)
        consumer.on_update("SomeNode", {})

        assert call_order == ["inner", "translator"]


class TestEventTranslatingStreamConsumerFailurePropagation:

    def test_translator_exception_propagates(self) -> None:
        """Pinned: translator failures crash the run. No try/except —
        unknown nodes and malformed deltas are bugs to fix, not runtime
        conditions to swallow."""
        translator = MagicMock(side_effect=ValueError("translator bug"))
        consumer = _make_consumer(translator=translator)

        with pytest.raises(ValueError, match="translator bug"):
            consumer.on_update("SomeNode", {})

    def test_translator_exception_still_calls_inner(self) -> None:
        """Corollary of inner-first ordering: if translation raises, the
        inner consumer has already run. Print output survives the crash."""
        inner = MagicMock(spec=StreamConsumerInterface)
        translator = MagicMock(side_effect=ValueError("translator bug"))
        consumer = _make_consumer(inner_consumer=inner, translator=translator)

        with pytest.raises(ValueError):
            consumer.on_update("SomeNode", {})

        inner.on_update.assert_called_once()

    def test_event_store_append_exception_propagates(self) -> None:
        """Pinned: event store failures also propagate. Persistence
        problems are not silently masked."""
        run_id = uuid4()
        event = _make_context_retrieved_event(run_id)
        store = MagicMock(spec=AgentEventStoreInterface)
        store.append.side_effect = RuntimeError("store full")
        translator = _make_translator(events_to_return=[event])
        consumer = _make_consumer(
            run_id=run_id, event_store=store, translator=translator
        )

        with pytest.raises(RuntimeError, match="store full"):
            consumer.on_update("SomeNode", {})


class TestEventTranslatingStreamConsumerPerRunLifecycle:

    def test_run_id_is_stable_across_multiple_on_update_calls(self) -> None:
        """Pinned: the consumer is per-run. The same run_id is passed to
        the translator on every on_update call for the consumer's
        lifetime. Constructing a new consumer is the only way to change
        the run_id — that's the V3a lifecycle shift from V2's
        per-process consumer."""
        run_id = uuid4()
        translator = _make_translator()
        consumer = _make_consumer(run_id=run_id, translator=translator)

        consumer.on_update("NodeA", {})
        consumer.on_update("NodeB", {})
        consumer.on_update("NodeC", {})

        for call in translator.call_args_list:
            assert call.args[2] == run_id