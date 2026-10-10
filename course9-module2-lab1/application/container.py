from typing import Literal

from fastmcp import FastMCP

from application.agent_service import AgentService
from application.calculator_server import build_server
from config import DOCUMENTS_DIR, HTTP_URL, PORT, SERVER_ARGS, SERVER_COMMAND
from infra.anthropic_chat_model_provider import AnthropicChatModelProvider
from infra.file_document_store import FileDocumentStore
from infra.http_tool_source import HttpToolSource
from infra.in_process_http_server_host import InProcessHttpServerHost
from infra.multi_server_tool_source import MultiServerToolSource
from infra.stdio_tool_source import StdioToolSource
from interfaces.chat_model_provider_interface import ChatModelProviderInterface
from interfaces.document_store_interface import DocumentStoreInterface
from interfaces.tool_source_interface import ToolSourceInterface

Transport = Literal["http", "stdio", "multi"]


def _make_tool_source(transport: Transport) -> ToolSourceInterface:
    if transport == "http":
        return HttpToolSource(HTTP_URL)
    if transport == "stdio":
        return StdioToolSource(SERVER_COMMAND, SERVER_ARGS)
    return MultiServerToolSource(SERVER_COMMAND, SERVER_ARGS, HTTP_URL)


def initialise(
    chat_model_provider: ChatModelProviderInterface | None = None,
    tool_source: ToolSourceInterface | None = None,
    transport: Transport = "stdio",
) -> AgentService:
    """Composition root. Wires concretes to interfaces and returns AgentService.

    Stateless: creates fresh instances every call. Optional injection
    parameters default to production concretes. `transport` only picks the
    default tool source; an explicitly injected `tool_source` wins.
    """
    if chat_model_provider is None:
        chat_model_provider = AnthropicChatModelProvider()

    if tool_source is None:
        tool_source = _make_tool_source(transport)

    return AgentService(chat_model_provider.create(), tool_source)


def initialise_server(document_store: DocumentStoreInterface | None = None) -> FastMCP:
    """Composition root for the MCP server: wires a document store and builds it."""
    if document_store is None:
        document_store = FileDocumentStore(DOCUMENTS_DIR)

    return build_server(document_store)


def initialise_http_host() -> InProcessHttpServerHost:
    """Composition root for the HTTP host used to serve the MCP server in-process."""
    return InProcessHttpServerHost(PORT)
