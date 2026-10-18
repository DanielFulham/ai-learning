import sys

import gradio as gr
from dotenv import load_dotenv

from application.container import initialise_host
from application.formatters import format_chat_reply
from application.interfaces.host_service_interface import HostServiceInterface
from cli_arguments import parse_arguments
from config import HOST_PORT, MODEL_ID, UI_HOST
from domain.conversation import Message


async def chat_turn(
    service: HostServiceInterface,
    message: str,
    shown: list[dict[str, str | None]],
    conversation: tuple[Message, ...],
) -> tuple[list[dict[str, str | None]], tuple[Message, ...]]:
    """The conversation shown on the page and the one kept for the model, each with this message
    and the answer added."""
    if not message.strip():
        return shown, conversation

    reply = await service.chat(message, conversation)
    shown = shown + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": format_chat_reply(reply)},
    ]
    return shown, reply.history


def build_interface(service: HostServiceInterface, server: str, roots: str) -> gr.Blocks:
    """The lab's chat page. Gradio lives here and nowhere deeper."""

    async def chat_wrapper(
        message: str, shown: list[dict[str, str | None]], conversation: tuple[Message, ...]
    ) -> tuple[list[dict[str, str | None]], tuple[Message, ...]]:
        return await chat_turn(service, message, shown, conversation)

    async def reset_conversation() -> tuple[list[dict[str, str | None]], tuple[Message, ...]]:
        return [], ()

    with gr.Blocks(title="MCP HTTP AI Host") as interface:
        gr.Markdown(f"""
# MCP HTTP AI Host
Chat with GPT-4o-mini using tools from the MCP HTTP server.

**Server:** {server}
**Workspace Roots:** {roots}
**Model:** {MODEL_ID}

The AI can use all available MCP tools, resources, and prompts during the conversation.
""")

        chatbot = gr.Chatbot(label="Conversation", height=500)
        # Each browser session gets its own conversation for the model.
        conversation = gr.State(())

        with gr.Row():
            msg = gr.Textbox(
                label="Your message", placeholder="Ask me to use MCP tools...", scale=4
            )
            clear = gr.Button("Clear", scale=1)

        msg.submit(
            fn=chat_wrapper, inputs=[msg, chatbot, conversation], outputs=[chatbot, conversation]
        ).then(lambda: "", outputs=msg)

        clear.click(fn=reset_conversation, outputs=[chatbot, conversation])

    return interface


def main() -> None:
    load_dotenv()
    server, roots = parse_arguments(sys.argv)
    service = initialise_host(server)
    build_interface(service, server, roots).queue().launch(
        server_name=UI_HOST, server_port=HOST_PORT
    )


if __name__ == "__main__":
    main()
