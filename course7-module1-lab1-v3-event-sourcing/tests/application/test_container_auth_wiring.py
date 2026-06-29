from unittest.mock import MagicMock, patch

from langgraph.graph.state import CompiledStateGraph

from application.auth_agent_service import AuthAgentService
from application.container import initialise
from application.qa_agent_service import QAAgentService
from domain.events.auth_events import LoginAttempted, LoginFailed, LoginSucceeded
from interfaces.input_provider_interface import InputProviderInterface

from _helpers.scripted_input_provider import ScriptedInputProvider


def _mock_graph() -> MagicMock:
    return MagicMock(spec=CompiledStateGraph)


class TestContainerAuthService:

    def test_auth_service_constructed_when_not_injected(self) -> None:
        """Pinned: the container builds an AuthAgentService and the service
        holds the same event_store instance the bundle exposes."""
        app = initialise(qa_graph=_mock_graph(), auth_graph=_mock_graph())

        assert isinstance(app.auth, AuthAgentService)
        assert app.auth._event_store is app.event_store

    def test_input_provider_defaults_to_console(self) -> None:
        """Pinned: when input_provider is not injected, the container
        constructs ConsoleInputProvider() and builds the auth graph with it.
        Patched at the import site in application.container, the same way C5
        patched SqliteCheckpointer."""
        console_instance = MagicMock(spec=InputProviderInterface)
        with patch(
            "application.container.ConsoleInputProvider",
            return_value=console_instance,
        ) as console_ctor, patch(
            "application.container.build_auth_graph"
        ) as mock_build:
            mock_build.return_value = _mock_graph()
            initialise(qa_graph=_mock_graph())

        console_ctor.assert_called_once_with()
        mock_build.assert_called_once_with(console_instance)

    def test_explicit_input_provider_overrides_default(self) -> None:
        """Pinned: an injected input_provider wins — ConsoleInputProvider is
        never constructed, and the auth graph is built with the injected
        instance. Mirrors the C5 checkpointer-override test."""
        injected = MagicMock(spec=InputProviderInterface)
        with patch(
            "application.container.ConsoleInputProvider"
        ) as console_ctor, patch(
            "application.container.build_auth_graph"
        ) as mock_build:
            mock_build.return_value = _mock_graph()
            initialise(qa_graph=_mock_graph(), input_provider=injected)

        console_ctor.assert_not_called()
        mock_build.assert_called_once_with(injected)

    def test_explicit_auth_graph_overrides_build(self) -> None:
        """Pinned: an injected auth_graph wins — build_auth_graph is never
        called, and the auth service is constructed with the injected
        graph."""
        injected_graph = MagicMock(spec=CompiledStateGraph)
        with patch("application.container.build_auth_graph") as mock_build:
            app = initialise(qa_graph=_mock_graph(), auth_graph=injected_graph)

        mock_build.assert_not_called()
        assert isinstance(app.auth, AuthAgentService)
        assert app.auth._graph is injected_graph


class TestContainerF17Singleton:

    def test_event_store_shared_across_qa_and_auth(self) -> None:
        """F17 release gate: the one event_store instance is shared across
        LabApp, the QA service, and the Auth service. The Counter assertion
        lands in V3c."""
        app = initialise(qa_graph=_mock_graph(), auth_graph=_mock_graph())

        assert isinstance(app.qa, QAAgentService)
        assert isinstance(app.auth, AuthAgentService)
        assert app.event_store is app.qa._event_store
        assert app.event_store is app.auth._event_store
        assert app.qa._event_store is app.auth._event_store


class TestContainerAuthFlowEndToEnd:

    def test_auth_success_flow_appends_expected_events(self) -> None:
        """Proves the whole C6 wiring: the container constructs the scripted
        input provider's graph, builds the service from it, exposes it on
        LabApp, and running the service appends the success-path event
        sequence to the shared store. The scripted provider feeds the
        canonical valid pair (username then password, one shared queue)."""
        provider = ScriptedInputProvider(["test_user", "secure_password"])
        app = initialise(
            qa_graph=_mock_graph(),
            input_provider=provider,
            use_console_consumer=False,
        )

        result = app.auth.run()

        events = app.event_store.events_for_run(result.run_id)
        assert [type(e).__name__ for e in events] == [
            "LoginAttempted",
            "LoginSucceeded",
        ]
        assert isinstance(events[0], LoginAttempted)
        assert isinstance(events[1], LoginSucceeded)
        assert events[0].username == "test_user"

    def test_auth_failure_then_retry_success_appends_expected_events(self) -> None:
        """Captures the loop-on-failure topology as architectural truth. The
        auth graph's failure branch loops FailureNode -> InputNode (no edge
        to END from failure); SuccessNode -> END is the only terminus. So the
        minimal terminating run that includes a failure is attempt -> fail ->
        loop -> attempt -> succeed, which the translator renders as the
        four-event sequence below. failure_node clears username AND password,
        so the scripted provider must supply both fields for both attempts —
        four strings, one shared queue. The username field on each
        LoginAttempted pins that the attempt reflects its own scripted input,
        not a cached value from the prior attempt."""
        provider = ScriptedInputProvider(
            ["wrong_user", "wrong_password", "test_user", "secure_password"]
        )
        app = initialise(
            qa_graph=_mock_graph(),
            input_provider=provider,
            use_console_consumer=False,
        )

        result = app.auth.run()

        events = app.event_store.events_for_run(result.run_id)
        assert [type(e).__name__ for e in events] == [
            "LoginAttempted",
            "LoginFailed",
            "LoginAttempted",
            "LoginSucceeded",
        ]
        assert isinstance(events[0], LoginAttempted)
        assert isinstance(events[1], LoginFailed)
        assert isinstance(events[2], LoginAttempted)
        assert isinstance(events[3], LoginSucceeded)
        assert events[0].username == "wrong_user"
        assert events[2].username == "test_user"
