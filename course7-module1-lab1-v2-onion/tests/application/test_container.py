from unittest.mock import MagicMock, patch

from langchain_core.language_models import BaseChatModel

from application.auth_agent_service import AuthAgentService
from application.container import initialise
from application.counter_agent_service import CounterAgentService
from application.lab_app import LabApp
from application.qa_agent_service import QAAgentService
from application.scripted_input_provider import ScriptedInputProvider
from infra.console_input_provider import ConsoleInputProvider
from interfaces.chat_model_provider_interface import ChatModelProviderInterface
from interfaces.input_provider_interface import InputProviderInterface
from interfaces.stream_consumer_interface import StreamConsumerInterface


def _make_provider() -> MagicMock:
    provider = MagicMock(spec=ChatModelProviderInterface)
    provider.create.return_value = MagicMock(spec=BaseChatModel)
    return provider


def test_initialise_returns_lab_app_with_all_three_services() -> None:
    app = initialise(
        chat_model_provider=_make_provider(),
        input_provider=MagicMock(spec=InputProviderInterface),
        stream_consumer=MagicMock(spec=StreamConsumerInterface),
    )
    assert isinstance(app, LabApp)
    assert isinstance(app.auth, AuthAgentService)
    assert isinstance(app.qa, QAAgentService)
    assert isinstance(app.counter, CounterAgentService)


def test_initialise_calls_provider_create_once() -> None:
    provider = _make_provider()
    initialise(
        chat_model_provider=provider,
        input_provider=MagicMock(spec=InputProviderInterface),
        stream_consumer=MagicMock(spec=StreamConsumerInterface),
    )
    provider.create.assert_called_once()


def test_initialise_is_stateless_returns_fresh_instances() -> None:
    """Container creates new services on each call. Pinning the stateless
    contract — caching is the entry point's job, not the container's."""
    provider = _make_provider()
    input_provider = MagicMock(spec=InputProviderInterface)
    stream_consumer = MagicMock(spec=StreamConsumerInterface)

    app_one = initialise(
        chat_model_provider=provider,
        input_provider=input_provider,
        stream_consumer=stream_consumer,
    )
    app_two = initialise(
        chat_model_provider=provider,
        input_provider=input_provider,
        stream_consumer=stream_consumer,
    )

    assert app_one is not app_two
    assert app_one.auth is not app_two.auth
    assert app_one.qa is not app_two.qa
    assert app_one.counter is not app_two.counter


def test_initialise_defaults_to_ollama_provider() -> None:
    """Default is Ollama — pinned so changing the default requires updating
    this test, surfacing the decision."""
    with patch("application.container.OllamaChatModelProvider") as mock_ollama_class:
        mock_ollama_class.return_value = _make_provider()
        initialise(
            input_provider=MagicMock(spec=InputProviderInterface),
            stream_consumer=MagicMock(spec=StreamConsumerInterface),
        )
        mock_ollama_class.assert_called_once_with()


def test_initialise_with_use_openai_true_uses_openai_provider() -> None:
    """When no provider is explicitly injected and `use_openai=True`,
    the container constructs an OpenAIChatModelProvider."""
    with patch("application.container.OpenAIChatModelProvider") as mock_openai_class:
        mock_openai_class.return_value = _make_provider()
        initialise(
            input_provider=MagicMock(spec=InputProviderInterface),
            stream_consumer=MagicMock(spec=StreamConsumerInterface),
            use_openai=True,
        )
        mock_openai_class.assert_called_once_with()


def test_initialise_with_use_openai_false_uses_ollama_provider() -> None:
    """When `use_openai=False` (default), the container constructs Ollama."""
    with patch("application.container.OllamaChatModelProvider") as mock_ollama_class:
        mock_ollama_class.return_value = _make_provider()
        initialise(
            input_provider=MagicMock(spec=InputProviderInterface),
            stream_consumer=MagicMock(spec=StreamConsumerInterface),
            use_openai=False,
        )
        mock_ollama_class.assert_called_once_with()


def test_explicit_provider_overrides_use_openai_flag() -> None:
    """When chat_model_provider is passed explicitly, `use_openai` is
    ignored entirely — neither Ollama nor OpenAI constructors are called."""
    explicit_provider = _make_provider()

    with patch("application.container.OpenAIChatModelProvider") as mock_openai_class, patch(
        "application.container.OllamaChatModelProvider"
    ) as mock_ollama_class:
        initialise(
            chat_model_provider=explicit_provider,
            input_provider=MagicMock(spec=InputProviderInterface),
            stream_consumer=MagicMock(spec=StreamConsumerInterface),
            use_openai=True,
        )
        mock_openai_class.assert_not_called()
        mock_ollama_class.assert_not_called()
        explicit_provider.create.assert_called_once()


def test_initialise_defaults_to_console_input_provider() -> None:
    """Default input provider is ConsoleInputProvider — interactive runs."""
    app = initialise(
        chat_model_provider=_make_provider(),
        stream_consumer=MagicMock(spec=StreamConsumerInterface),
    )
    assert isinstance(app, LabApp)


def test_initialise_with_use_scripted_auth_input_constructs_scripted_provider() -> None:
    """When `use_scripted_auth_input=True`, the container constructs a
    ScriptedInputProvider — the actual concrete used by `demo.py all`."""
    with patch(
        "application.container.ScriptedInputProvider"
    ) as mock_scripted_class:
        mock_scripted_class.return_value = MagicMock(spec=InputProviderInterface)
        initialise(
            chat_model_provider=_make_provider(),
            stream_consumer=MagicMock(spec=StreamConsumerInterface),
            use_scripted_auth_input=True,
        )
        mock_scripted_class.assert_called_once()


def test_initialise_with_use_scripted_auth_input_false_constructs_console_provider() -> None:
    """When `use_scripted_auth_input=False` (default), ConsoleInputProvider
    is constructed."""
    with patch(
        "application.container.ConsoleInputProvider"
    ) as mock_console_class:
        mock_console_class.return_value = MagicMock(spec=InputProviderInterface)
        initialise(
            chat_model_provider=_make_provider(),
            stream_consumer=MagicMock(spec=StreamConsumerInterface),
            use_scripted_auth_input=False,
        )
        mock_console_class.assert_called_once_with()


def test_explicit_input_provider_overrides_use_scripted_flag() -> None:
    """Explicit input_provider injection wins over the boolean flag."""
    explicit_input = MagicMock(spec=InputProviderInterface)

    with patch(
        "application.container.ScriptedInputProvider"
    ) as mock_scripted_class, patch(
        "application.container.ConsoleInputProvider"
    ) as mock_console_class:
        initialise(
            chat_model_provider=_make_provider(),
            input_provider=explicit_input,
            stream_consumer=MagicMock(spec=StreamConsumerInterface),
            use_scripted_auth_input=True,
        )
        mock_scripted_class.assert_not_called()
        mock_console_class.assert_not_called()