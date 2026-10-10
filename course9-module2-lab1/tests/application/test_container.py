from unittest.mock import MagicMock, patch

from fastmcp import FastMCP
from langchain_core.language_models import BaseChatModel

from application.agent_service import AgentService
from application.container import initialise, initialise_http_host, initialise_server
from infra.in_process_http_server_host import InProcessHttpServerHost
from interfaces.chat_model_provider_interface import ChatModelProviderInterface
from interfaces.document_store_interface import DocumentStoreInterface
from interfaces.tool_source_interface import ToolSourceInterface


def make_provider() -> MagicMock:
    provider = MagicMock(spec=ChatModelProviderInterface)
    provider.create.return_value = MagicMock(spec=BaseChatModel)
    return provider


def test_initialise_returns_an_agent_service() -> None:
    service = initialise(
        chat_model_provider=make_provider(),
        tool_source=MagicMock(spec=ToolSourceInterface),
    )
    assert isinstance(service, AgentService)


def test_initialise_calls_provider_create_once() -> None:
    provider = make_provider()
    initialise(chat_model_provider=provider, tool_source=MagicMock(spec=ToolSourceInterface))
    provider.create.assert_called_once()


def test_initialise_is_stateless_returns_fresh_instances() -> None:
    """The container creates a new service on each call; caching is the entry point's job."""
    provider = make_provider()
    source = MagicMock(spec=ToolSourceInterface)
    one = initialise(chat_model_provider=provider, tool_source=source)
    two = initialise(chat_model_provider=provider, tool_source=source)
    assert one is not two


def test_initialise_defaults_to_the_anthropic_provider() -> None:
    with patch("application.container.AnthropicChatModelProvider") as provider_class:
        provider_class.return_value = make_provider()
        initialise(tool_source=MagicMock(spec=ToolSourceInterface))
        provider_class.assert_called_once_with()


def test_explicit_provider_means_the_default_is_never_built() -> None:
    with patch("application.container.AnthropicChatModelProvider") as provider_class:
        initialise(
            chat_model_provider=make_provider(),
            tool_source=MagicMock(spec=ToolSourceInterface),
        )
        provider_class.assert_not_called()


def test_transport_picks_the_matching_tool_source() -> None:
    with (
        patch("application.container.HttpToolSource") as http_class,
        patch("application.container.StdioToolSource") as stdio_class,
        patch("application.container.MultiServerToolSource") as multi_class,
    ):
        initialise(chat_model_provider=make_provider(), transport="http")
        initialise(chat_model_provider=make_provider(), transport="stdio")
        initialise(chat_model_provider=make_provider(), transport="multi")
        http_class.assert_called_once()
        stdio_class.assert_called_once()
        multi_class.assert_called_once()


def test_explicit_tool_source_overrides_transport() -> None:
    with (
        patch("application.container.HttpToolSource") as http_class,
        patch("application.container.StdioToolSource") as stdio_class,
        patch("application.container.MultiServerToolSource") as multi_class,
    ):
        initialise(
            chat_model_provider=make_provider(),
            tool_source=MagicMock(spec=ToolSourceInterface),
            transport="multi",
        )
        http_class.assert_not_called()
        stdio_class.assert_not_called()
        multi_class.assert_not_called()


def test_initialise_server_returns_a_server_using_the_injected_store() -> None:
    store = MagicMock(spec=DocumentStoreInterface)
    with patch("application.container.build_server") as build:
        build.return_value = MagicMock(spec=FastMCP)
        initialise_server(document_store=store)
        build.assert_called_once_with(store)


def test_initialise_server_defaults_to_a_file_document_store() -> None:
    with (
        patch("application.container.FileDocumentStore") as store_class,
        patch("application.container.build_server") as build,
    ):
        initialise_server()
        store_class.assert_called_once()
        build.assert_called_once_with(store_class.return_value)


def test_initialise_server_reads_documents_from_the_configured_directory() -> None:
    with (
        patch("application.container.FileDocumentStore") as store_class,
        patch("application.container.build_server"),
        patch("application.container.DOCUMENTS_DIR", "TEST_DIR"),
    ):
        initialise_server()
        store_class.assert_called_once_with("TEST_DIR")


def test_initialise_http_host_uses_the_configured_port() -> None:
    with patch("application.container.PORT", 12345):
        host = initialise_http_host()
    assert isinstance(host, InProcessHttpServerHost)
    assert host._port == 12345
