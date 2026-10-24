import gradio as gr

from application.approval_desks import ApprovalDesks

# Used when a request carries no session hash, so every such call still shares one desk.
_FALLBACK_SESSION = "NO_SESSION_HASH"


def session_id(request: gr.Request) -> str:
    """The id of the browser session that made a request, which picks its approval desk."""
    return request.session_hash or _FALLBACK_SESSION


def forget_session(desks: ApprovalDesks, request: gr.Request) -> None:
    """Called when a page closes: its desk is dropped and its waiting questions are cancelled."""
    desks.forget(session_id(request))
