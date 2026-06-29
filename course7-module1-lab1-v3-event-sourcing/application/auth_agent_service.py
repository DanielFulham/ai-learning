from datetime import datetime
from typing import Callable
from uuid import uuid4

from langgraph.graph.state import CompiledStateGraph

from application.event_translating_stream_consumer import (
    EventTranslatingStreamConsumer,
)
from application.event_translation.auth_translator import translate_auth_update
from domain.auth_credentials import AuthCredentials
from domain.auth_result import AuthResult
from domain.state_schemas import AuthState
from interfaces.agent_event_store_interface import AgentEventStoreInterface
from interfaces.stream_consumer_interface import StreamConsumerInterface


class AuthAgentService:
    """Application service wrapping the compiled Auth graph.

    Constructor mirrors `QAAgentService` exactly: the graph plus the V3a
    dependencies (event store, inner stream consumer, clock). The
    translator is hardcoded to `translate_auth_update` because the Auth
    service IS bound to its own per-service translator.

    `run()` differs from QA's in five ways the auth flow demands:
    - It is parameterless. The auth flow takes its input interactively
      through the input provider inside the graph; there is no question to
      pass.
    - It seeds `{"credentials": AuthCredentials()}` as the initial state —
      `input_node` requires the carrier to be present before streaming.
    - It fires no direct run-lifecycle event. QA appends `QuestionReceived`
      directly; auth's first event (`LoginAttempted`) comes from the
      translator, so the service appends nothing directly.
    - It binds `translate_auth_update` to the per-run consumer.
    - It captures the final `AuthCredentials` from the `values`-mode
      chunk's `credentials` key, and returns an `AuthResult`.
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

    def run(self) -> AuthResult:
        run_id = uuid4()

        consumer = EventTranslatingStreamConsumer(
            run_id=run_id,
            event_store=self._event_store,
            inner_consumer=self._inner_consumer,
            translator=translate_auth_update,
            clock=self._clock,
        )

        initial_state: AuthState = {"credentials": AuthCredentials()}
        final_credentials: AuthCredentials | None = None

        for mode, chunk in self._graph.stream(
            initial_state,
            stream_mode=["updates", "values"],
        ):
            if mode == "updates" and isinstance(chunk, dict):
                for node_name, state_delta in chunk.items():
                    consumer.on_update(node_name, state_delta)
            elif mode == "values" and isinstance(chunk, dict):
                candidate = chunk.get("credentials")
                if isinstance(candidate, AuthCredentials):
                    final_credentials = candidate

        if final_credentials is None:
            raise RuntimeError(
                "Auth workflow completed without producing credentials — "
                "graph topology is broken"
            )
        return AuthResult(credentials=final_credentials, run_id=run_id)
