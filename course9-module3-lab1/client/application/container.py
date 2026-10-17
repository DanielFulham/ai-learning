from application.client_service import ClientService
from application.interfaces.client_service_interface import ClientServiceInterface
from config import DEFAULT_SERVER_SCRIPT, SERVER_COMMAND
from infra.console_user_input import ConsoleUserInput
from infra.http_transport import HttpTransport
from infra.official_mcp_session import OfficialMcpSession
from infra.stdio_transport import StdioTransport
from interfaces.mcp_session_interface import McpSessionInterface
from interfaces.mcp_transport_interface import McpTransportInterface
from interfaces.user_input_interface import UserInputInterface


def initialise(
    server: str = DEFAULT_SERVER_SCRIPT,
    session: McpSessionInterface | None = None,
    user_input: UserInputInterface | None = None,
) -> ClientServiceInterface:
    """Composition root. Wires concretes to interfaces and returns the client service.

    Stateless: creates fresh instances every call. Optional injection parameters default to
    production concretes. `server` is either the path of a server script (launched as a
    subprocess over stdio) or an http(s) URL of a server that is already running. It only
    picks the default transport; an explicitly injected `session` wins.
    """
    if user_input is None:
        user_input = ConsoleUserInput()

    if session is None:
        transport: McpTransportInterface
        if server.startswith(("http://", "https://")):
            transport = HttpTransport(server)
        else:
            transport = StdioTransport(SERVER_COMMAND, [server])
        session = OfficialMcpSession(transport)

    return ClientService(session, user_input)
