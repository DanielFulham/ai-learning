from langgraph.graph.state import CompiledStateGraph

from domain.auth_credentials import AuthCredentials
from domain.state_schemas import AuthState
from interfaces.stream_consumer_interface import StreamConsumerInterface


class AuthAgentService:
    """Wraps the compiled Auth graph."""

    def __init__(
        self,
        graph: CompiledStateGraph,
        stream_consumer: StreamConsumerInterface,
    ) -> None:
        self._graph = graph
        self._consumer = stream_consumer

    def run(self, initial_state: AuthState | None = None) -> AuthCredentials:
        final_credentials: AuthCredentials | None = None

        for mode, chunk in self._graph.stream(
            initial_state or {},
            stream_mode=["updates", "values"],
        ):
            if mode == "updates" and isinstance(chunk, dict):
                for node_name, state_delta in chunk.items():
                    self._consumer.on_update(node_name, state_delta)
            elif mode == "values" and isinstance(chunk, dict):
                candidate = chunk.get("credentials")
                if isinstance(candidate, AuthCredentials):
                    final_credentials = candidate

        if final_credentials is None:
            raise RuntimeError(
                "Auth workflow completed without producing credentials — "
                "graph topology is broken"
            )
        return final_credentials