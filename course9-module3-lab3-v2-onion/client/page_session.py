import gradio as gr

from application.approval_desks import ApprovalDesks


class MissingSessionError(ValueError):
    """A request carried no session hash, so there is no approval desk it could safely use."""


def session_id(request: gr.Request) -> str:
    """The id of the browser session that made a request, which picks its approval desk.

    A request with no session hash is refused. Sharing one desk between such requests would let
    one caller see and answer another's question.
    """
    if not request.session_hash:
        raise MissingSessionError("The request has no session, so it cannot be given a desk")
    return request.session_hash


def forget_session(desks: ApprovalDesks, request: gr.Request) -> None:
    """Called when a page closes: its desk is dropped and its waiting questions are cancelled.

    A request with no session hash never had a desk, so there is nothing to drop."""
    if request.session_hash:
        desks.forget(request.session_hash)
