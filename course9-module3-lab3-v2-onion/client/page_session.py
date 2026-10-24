import gradio as gr

# Used when a request carries no session hash, so every such call still shares one desk.
_FALLBACK_SESSION = "NO_SESSION_HASH"


def session_id(request: gr.Request) -> str:
    """The id of the browser session that made a request, which picks its approval desk."""
    return request.session_hash or _FALLBACK_SESSION
