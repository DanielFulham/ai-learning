"""Workflow completion signal.

`ContextEquals` reads `WorkflowState.context_vars`, which only
`EV_CONTEXT_SET` envelopes mutate. Two things that look like they would
write it and don't: `response_schema` (the adapter encodes `reply.body`
into the packet, so a schema only shapes text) and the plugin's `context`
tool (read-only search/quote over the WAL, unrelated despite the name).
`workflow_helpers.set_context` is the write path.

Only `consultation` holds this tool. `set_context` has loose participant
semantics — any participant may call it regardless of turn order — so
whoever holds it can end the consultation.
"""

from typing import Literal

from ag2 import tool
from ag2.network import ChannelInject
from ag2.network.workflow_helpers import set_context

CONSULTATION_STATUS_KEY = "consultation_status"

ConsultationStatus = Literal["complete", "escalate"]

# Never None: ContextEquals treats a missing key as None, so a transition
# on None fires on turn one against an empty context.
STATUS_COMPLETE: ConsultationStatus = "complete"
STATUS_ESCALATE: ConsultationStatus = "escalate"


@tool
async def record_consultation_status(
    status: ConsultationStatus,
    channel: ChannelInject = None,
) -> str:
    """Record the outcome of the consultation and end the session.

    Args:
        status: "escalate" if the patient should seek in-person medical
            care, "complete" if the consultation can close without it.
    """
    if channel is None:
        raise RuntimeError(
            "record_consultation_status requires an active workflow channel; "
            "ChannelInject was not populated."
        )
    await set_context(channel, CONSULTATION_STATUS_KEY, status)
    # This return value is prompt content, not a status line. The write is
    # invisible to the caller — `fold` treats EV_CONTEXT_SET as auxiliary and
    # doesn't re-run `_select` — so a bare acknowledgement leaves the model
    # with no evidence anything happened, and it calls the tool again.
    return (
        f"Status recorded as {status!r}. Do not call this tool again. "
        "Now write your final message to the patient; the consultation closes "
        "when that message is sent."
    )
