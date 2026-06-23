from langgraph.graph.state import CompiledStateGraph

from domain.counter_tick import CounterTick
from interfaces.stream_consumer_interface import StreamConsumerInterface


class CounterAgentService:
    """Wraps the compiled Counter graph."""

    def __init__(
        self,
        graph: CompiledStateGraph,
        stream_consumer: StreamConsumerInterface,
    ) -> None:
        self._graph = graph
        self._consumer = stream_consumer

    def run(self) -> CounterTick:
        final_tick: CounterTick | None = None

        for mode, chunk in self._graph.stream(
            {},
            stream_mode=["updates", "values"],
        ):
            if mode == "updates" and isinstance(chunk, dict):
                for node_name, state_delta in chunk.items():
                    self._consumer.on_update(node_name, state_delta)
            elif mode == "values" and isinstance(chunk, dict):
                candidate = chunk.get("tick")
                if isinstance(candidate, CounterTick):
                    final_tick = candidate

        if final_tick is None:
            raise RuntimeError(
                "Counter workflow completed without producing a tick — "
                "graph topology is broken"
            )
        return final_tick