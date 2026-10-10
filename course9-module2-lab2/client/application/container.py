from application.chat_service import ChatService
from application.client_service import ClientService
from application.handlers import (
    make_elicitation_handler,
    make_notification_handler,
    make_progress_handler,
)
from application.interfaces.client_service_interface import ClientServiceInterface
from config import DEFAULT_SERVER_SCRIPT, MAX_TOKENS, MODEL_ID, SERVER_COMMAND
from infra.anthropic_chat_model import AnthropicChatModel
from infra.console_user_input import ConsoleUserInput
from infra.fastmcp_session import FastmcpSession
from interfaces.chat_model_interface import ChatModelInterface
from interfaces.mcp_session_interface import McpSessionInterface
from interfaces.user_input_interface import UserInputInterface


def initialise(
    server: str = DEFAULT_SERVER_SCRIPT,
    chat_model: ChatModelInterface | None = None,
    session: McpSessionInterface | None = None,
    user_input: UserInputInterface | None = None,
) -> ClientServiceInterface:
    """Composition root. Wires concretes to interfaces and returns the client service.

    Stateless: creates fresh instances every call. Optional injection
    parameters default to production concretes. `server` is either the path of a
    server script (launched as a subprocess over stdio) or an http(s) URL of a
    server that is already running. It only picks the default session; an
    explicitly injected `session` wins.
    """
    if user_input is None:
        user_input = ConsoleUserInput()

    if chat_model is None:
        chat_model = AnthropicChatModel(MODEL_ID, MAX_TOKENS)

    if session is None:
        handlers = (
            make_elicitation_handler(user_input),
            make_progress_handler(user_input),
            make_notification_handler(user_input),
        )
        if server.startswith(("http://", "https://")):
            session = FastmcpSession.for_http(server, *handlers)
        else:
            session = FastmcpSession.for_stdio(SERVER_COMMAND, [server], *handlers)

    chat_service = ChatService(chat_model, session, user_input)
    return ClientService(chat_service, session, user_input)
