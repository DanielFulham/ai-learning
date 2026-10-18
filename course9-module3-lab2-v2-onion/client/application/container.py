from application.gui_service import GuiService
from application.host_service import HostService
from application.interfaces.gui_service_interface import GuiServiceInterface
from application.interfaces.host_service_interface import HostServiceInterface
from config import DEFAULT_SERVER, MCP_ENDPOINT, MODEL_ID, SERVER_COMMAND, STDIO_SERVER_ARGS
from infra.http_transport import HttpTransport
from infra.mcp_sdk_session import McpSdkSession
from infra.openai_chat_model import OpenAiChatModel
from infra.stdio_transport import StdioTransport
from interfaces.llm_chat_model_interface import LlmChatModelInterface
from interfaces.mcp_session_interface import McpSessionInterface
from interfaces.mcp_transport_interface import McpTransportInterface


def _build_session(server: str) -> McpSessionInterface:
    """An http(s) address is a running server, and /mcp is added if it is missing.

    Anything else is a server script, launched over stdio.
    """
    transport: McpTransportInterface
    if server.startswith(("http://", "https://")):
        base = server.rstrip("/")
        transport = HttpTransport(base if base.endswith(MCP_ENDPOINT) else base + MCP_ENDPOINT)
    else:
        transport = StdioTransport(SERVER_COMMAND, [server, *STDIO_SERVER_ARGS])
    return McpSdkSession(transport)


def initialise_gui(
    server: str = DEFAULT_SERVER, session: McpSessionInterface | None = None
) -> GuiServiceInterface:
    """Composition root for the GUI app. Stateless: creates fresh instances every call.

    `server` only picks the default session; an injected `session` wins. No model is built, so
    the GUI needs no API key.
    """
    if session is None:
        session = _build_session(server)

    return GuiService(session)


def initialise_host(
    server: str = DEFAULT_SERVER,
    session: McpSessionInterface | None = None,
    chat_model: LlmChatModelInterface | None = None,
) -> HostServiceInterface:
    """Composition root for the AI host app. Stateless: creates fresh instances every call.

    The OpenAI model reads its key on the first request, so building this needs no key.
    """
    if session is None:
        session = _build_session(server)

    if chat_model is None:
        chat_model = OpenAiChatModel(MODEL_ID)

    return HostService(session, chat_model)
