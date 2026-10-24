from collections.abc import Callable
from dataclasses import dataclass

import config
from application.approval_desks import ApprovalDesks
from application.gui_service import GuiService
from application.host_service import HostService
from application.interfaces.gui_service_interface import GuiServiceInterface
from application.interfaces.host_service_interface import HostServiceInterface
from domain.server_launch import ServerLaunch
from infra.mcp_sdk_session import McpSdkSession
from infra.openai_chat_model import OpenAiChatModel
from interfaces.llm_chat_model_interface import LlmChatModelInterface
from interfaces.mcp_session_interface import McpSessionInterface


@dataclass(frozen=True)
class GuiApp:
    """What the GUI entry point needs: the shared service, the per-session approval desks, and
    `shutdown`, which closes the server connection when the app exits."""

    service: GuiServiceInterface
    desks: ApprovalDesks
    shutdown: Callable[[], None]


@dataclass(frozen=True)
class HostApp:
    """What the host entry point needs: the shared service, the per-session approval desks, and
    `shutdown`, which closes the server connection when the app exits."""

    service: HostServiceInterface
    desks: ApprovalDesks
    shutdown: Callable[[], None]


def _default_launch() -> ServerLaunch:
    return ServerLaunch(config.DEFAULT_SERVER_SCRIPT, config.DEFAULT_SERVER_PYTHON)


def _build_session(launch: ServerLaunch) -> McpSessionInterface:
    """The server's script, launched over stdio with the server's own interpreter."""
    return McpSdkSession.for_stdio(
        str(launch.python), [str(launch.script)], cwd=str(launch.script.parent)
    )


def initialise_gui(
    launch: ServerLaunch | None = None, session: McpSessionInterface | None = None
) -> GuiApp:
    """Composition root for the GUI app. Stateless: creates fresh instances every call.

    `launch` only picks the default session; an injected `session` wins. No model is built, so
    the GUI needs no API key.
    """
    if session is None:
        session = _build_session(launch or _default_launch())
    service = GuiService(session, config.POLICY_RESOURCE_URI, config.AUDIT_LOG_RESOURCE_URI)
    return GuiApp(service, ApprovalDesks(config.APPROVAL_TIMEOUT_SECONDS), session.shutdown)


def initialise_host(
    launch: ServerLaunch | None = None,
    session: McpSessionInterface | None = None,
    chat_model: LlmChatModelInterface | None = None,
) -> HostApp:
    """Composition root for the AI host app. Stateless: creates fresh instances every call.

    The OpenAI model reads its key on the first request, so building this needs no key.
    """
    if session is None:
        session = _build_session(launch or _default_launch())
    if chat_model is None:
        chat_model = OpenAiChatModel(config.MODEL_ID)
    return HostApp(
        HostService(session, chat_model),
        ApprovalDesks(config.APPROVAL_TIMEOUT_SECONDS),
        session.shutdown,
    )
