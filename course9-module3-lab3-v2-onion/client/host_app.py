import asyncio
import sys
from collections.abc import AsyncIterator

import gradio as gr
from dotenv import load_dotenv

import config
from application.approval_desks import ApprovalDesks
from application.container import initialise_host
from application.formatters import format_chat_reply
from application.interfaces.approval_desk_interface import ApprovalDeskInterface
from application.interfaces.host_service_interface import HostServiceInterface
from cli_arguments import parse_arguments
from domain.conversation import Message
from page_lifecycle import serve
from page_session import forget_session, session_id

Shown = list[dict[str, str | None]]


async def chat_turn(
    service: HostServiceInterface,
    desk: ApprovalDeskInterface,
    message: str,
    shown: Shown,
    conversation: tuple[Message, ...],
    poll_seconds: float = config.POLL_SECONDS,
) -> AsyncIterator[tuple[Shown, tuple[Message, ...]]]:
    """One chat turn as a stream of page updates.

    The message shows up at once. If the server asks a question while the model's turn is
    running, the exact question is added to the chat, and the user's next message ("yes" or
    "no") answers it. If a question is already waiting when the message arrives, the message is
    its answer and goes straight to the service.
    """
    if not message.strip():
        yield shown, conversation
        return

    shown = [*shown, {"role": "user", "content": message}]

    if desk.pending_message() is not None:
        reply = await service.chat(message, conversation, desk)
        yield [*shown, {"role": "assistant", "content": format_chat_reply(reply)}], reply.history
        return

    yield shown, conversation
    turn = asyncio.create_task(service.chat(message, conversation, desk))
    announced: str | None = None
    while not turn.done():
        waiting = desk.pending_message()
        if waiting is None:
            announced = None
        elif waiting != announced:
            announced = waiting
            prompt = f"Approval needed:\n{waiting}\n\nReply yes or no."
            shown = [*shown, {"role": "assistant", "content": prompt}]
            yield shown, conversation
        await asyncio.wait({turn}, timeout=poll_seconds)

    reply = turn.result()
    yield [*shown, {"role": "assistant", "content": format_chat_reply(reply)}], reply.history


async def guarded_chat_turn(
    service: HostServiceInterface,
    desk: ApprovalDeskInterface,
    message: str,
    shown: Shown,
    conversation: tuple[Message, ...],
) -> AsyncIterator[tuple[Shown, tuple[Message, ...]] | None]:
    """`chat_turn`, with one ordinary turn at a time per session.

    Two turns at once would each start from the same history and one would overwrite the other.
    A message that arrives while a turn is running is not started: this yields None once. The
    exception is the reply to a question that is waiting, which must get through while the turn
    that asked is still running.
    """
    if desk.pending_message() is not None:
        async for step in chat_turn(service, desk, message, shown, conversation):
            yield step
        return
    if not desk.begin_turn():
        yield None
        return
    try:
        async for step in chat_turn(service, desk, message, shown, conversation):
            yield step
    finally:
        desk.end_turn()


def build_interface(service: HostServiceInterface, desks: ApprovalDesks) -> gr.Blocks:
    """The chat page. Gradio lives here and nowhere deeper."""

    async def chat_wrapper(
        message: str,
        shown: Shown,
        conversation: tuple[Message, ...],
        request: gr.Request,
    ) -> AsyncIterator[tuple[object, object]]:
        desk = desks.desk_for(session_id(request))
        async for step in guarded_chat_turn(service, desk, message, shown, conversation):
            if step is None:
                gr.Warning("Still working on your last message. Wait for the reply, then send it.")
                yield gr.skip(), gr.skip()
            else:
                yield step

    async def reset_conversation() -> tuple[Shown, tuple[Message, ...]]:
        return [], ()

    def close_page(request: gr.Request) -> None:
        forget_session(desks, request)

    with gr.Blocks(title="MCP Permission AI Host") as interface:
        gr.Markdown(f"""
# MCP Permission AI Host
Chat with {config.MODEL_ID} using the MCP server's tools.

The server decides what runs. When a tool needs your approval, the exact question is shown in
the chat: reply **yes** or **no**.
""")

        chatbot = gr.Chatbot(label="Conversation", height=500)
        # Each browser session gets its own conversation for the model.
        conversation = gr.State(())

        with gr.Row():
            msg = gr.Textbox(
                label="Your message", placeholder="Ask me to use MCP tools...", scale=4
            )
            clear = gr.Button("Clear", scale=1)

        # No concurrency limit: the reply to a waiting question arrives while the turn that raised
        # it is still running, and it must not queue behind that turn.
        msg.submit(
            fn=chat_wrapper,
            inputs=[msg, chatbot, conversation],
            outputs=[chatbot, conversation],
            concurrency_limit=None,
            api_name="chat",
        ).then(lambda: "", outputs=msg)

        clear.click(fn=reset_conversation, outputs=[chatbot, conversation])

    interface.unload(close_page)
    return interface


def main() -> None:
    load_dotenv()
    app = initialise_host(parse_arguments(sys.argv))
    serve(
        build_interface(app.service, app.desks),
        app.shutdown,
        config.UI_HOST,
        config.HOST_PORT,
    )


if __name__ == "__main__":
    main()
