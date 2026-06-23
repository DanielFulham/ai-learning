from langgraph.graph.state import CompiledStateGraph

from domain.qa_exchange import QAExchange
from domain.state_schemas import QAState
from interfaces.stream_consumer_interface import StreamConsumerInterface


class QAAgentService:
    """Wraps the compiled QA graph."""

    def __init__(
        self,
        graph: CompiledStateGraph,
        stream_consumer: StreamConsumerInterface,
    ) -> None:
        self._graph = graph
        self._consumer = stream_consumer

    def run(self, question: str) -> QAExchange:
        initial_state: QAState = {"exchange": QAExchange(question=question)}
        final_exchange: QAExchange | None = None

        for mode, chunk in self._graph.stream(
            initial_state,
            stream_mode=["updates", "values"],
        ):
            if mode == "updates" and isinstance(chunk, dict):
                for node_name, state_delta in chunk.items():
                    self._consumer.on_update(node_name, state_delta)
            elif mode == "values" and isinstance(chunk, dict):
                candidate = chunk.get("exchange")
                if isinstance(candidate, QAExchange):
                    final_exchange = candidate

        if final_exchange is None:
            raise RuntimeError(
                "QA workflow completed without producing an exchange — "
                "graph topology is broken"
            )
        return final_exchange