from datetime import datetime
from typing import Callable
from uuid import uuid4

from langgraph.graph.state import CompiledStateGraph

from application.event_translating_stream_consumer import (
    EventTranslatingStreamConsumer,
)
from application.event_translation.qa_translator import translate_qa_update
from domain.events.qa_events import QuestionReceived
from domain.qa_exchange import QAExchange
from domain.run_result import RunResult
from domain.state_schemas import QAState
from interfaces.agent_event_store_interface import AgentEventStoreInterface
from interfaces.stream_consumer_interface import StreamConsumerInterface


class QAAgentService:
    """Application service wrapping the compiled QA graph.

    V3a shape: constructor takes the graph plus the V3a dependencies
    (event store, inner stream consumer, clock). The translator is
    hardcoded to `translate_qa_update` because the QA service IS bound
    to its own per-service translator.

    Each `.run()` call generates a fresh run_id, fires a `QuestionReceived`
    event directly (run-lifecycle events bypass the translator), and
    constructs a per-run `EventTranslatingStreamConsumer`. Returns a
    `RunResult` carrying both the final exchange and the run_id — the
    run_id lets callers replay-by-run via the event store.
    """

    def __init__(
        self,
        graph: CompiledStateGraph,
        event_store: AgentEventStoreInterface,
        inner_consumer: StreamConsumerInterface,
        clock: Callable[[], datetime],
    ) -> None:
        self._graph = graph
        self._event_store = event_store
        self._inner_consumer = inner_consumer
        self._clock = clock

    def run(self, question: str) -> RunResult:
        run_id = uuid4()
        initial_exchange = QAExchange(question=question)

        self._event_store.append(
            QuestionReceived(
                event_id=uuid4(),
                aggregate_id=run_id,
                occurred_at=self._clock(),
                question=question,
            )
        )

        consumer = EventTranslatingStreamConsumer(
            run_id=run_id,
            event_store=self._event_store,
            inner_consumer=self._inner_consumer,
            translator=translate_qa_update,
            clock=self._clock,
        )

        initial_state: QAState = {"exchange": initial_exchange}
        final_exchange: QAExchange | None = None

        for mode, chunk in self._graph.stream(
            initial_state,
            stream_mode=["updates", "values"],
        ):
            if mode == "updates" and isinstance(chunk, dict):
                for node_name, state_delta in chunk.items():
                    consumer.on_update(node_name, state_delta)
            elif mode == "values" and isinstance(chunk, dict):
                candidate = chunk.get("exchange")
                if isinstance(candidate, QAExchange):
                    final_exchange = candidate

        if final_exchange is None:
            raise RuntimeError(
                "QA workflow completed without producing an exchange — "
                "graph topology is broken"
            )
        return RunResult(exchange=final_exchange, run_id=run_id)