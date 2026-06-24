from datetime import datetime
from typing import Any, Callable, Sequence
from uuid import UUID

from domain.events.base import BaseAgentEvent
from interfaces.agent_event_store_interface import AgentEventStoreInterface
from interfaces.stream_consumer_interface import StreamConsumerInterface


TranslatorFunction = Callable[
    [str, dict[str, Any], UUID, Callable[[], datetime]],
    Sequence[BaseAgentEvent],
]


class EventTranslatingStreamConsumer:
    """Per-run stream consumer that wraps an inner consumer and writes
    events to the store.

    Decorator pattern over `StreamConsumerInterface`. Each `on_update`:
    1. Calls the inner consumer first — dev visibility is preserved
       even if translation fails. The print trail is intact when a
       crash report needs reading.
    2. Translates the update via the injected translator function.
    3. Appends every returned event to the store in order.

    Constructed per-run, not per-process: `run_id` is fixed for the
    lifetime of this instance. This is the V3a lifecycle shift —
    V2's ConsoleStreamConsumer was per-process (no run identifier),
    V3a's translating consumer is per-run (event IDs and aggregate IDs
    are per-run concepts). Each `service.run()` call constructs a fresh
    consumer with a fresh run_id.

    The translator is injected, not hardcoded. Each agent service binds
    its own translator (`translate_qa_update` for QA, `translate_auth_update`
    for Auth in V3b, `translate_counter_update` for Counter in V3c). The
    consumer is generic; the translator is the per-service concern.

    Translation failures propagate. Bugs (unknown node, malformed delta)
    crash the run loudly rather than silently dropping events.
    """

    def __init__(
        self,
        run_id: UUID,
        event_store: AgentEventStoreInterface,
        inner_consumer: StreamConsumerInterface,
        translator: TranslatorFunction,
        clock: Callable[[], datetime],
    ) -> None:
        self._run_id = run_id
        self._event_store = event_store
        self._inner = inner_consumer
        self._translator = translator
        self._clock = clock

    def on_update(self, node_name: str, state_delta: dict[str, Any]) -> None:
        self._inner.on_update(node_name, state_delta)

        events = self._translator(node_name, state_delta, self._run_id, self._clock)
        for event in events:
            self._event_store.append(event)