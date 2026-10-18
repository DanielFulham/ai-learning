import sys

import gradio as gr
from dotenv import load_dotenv

from application.container import initialise_host
from application.formatters import format_chat_reply
from application.interfaces.host_service_interface import HostServiceInterface
from cli_arguments import parse_arguments
from config import HOST_PORT, MODEL_ID, UI_HOST


async def chat_turn(
    service: HostServiceInterface, message: str, history: list[dict[str, str | None]]
) -> list[dict[str, str | None]]:
    """The conversation shown on the page, with this message and the model's answer added."""
    if not message.strip():
        return history

    reply = await service.chat(message)
    return history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": format_chat_reply(reply)},
    ]


def build_interface(service: HostServiceInterface, server: str, roots: str) -> gr.Blocks:
    """The lab's chat page. Gradio lives here and nowhere deeper."""

    async def chat_wrapper(
        message: str, history: list[dict[str, str | None]]
    ) -> list[dict[str, str | None]]:
        return await chat_turn(service, message, history)

    async def reset_conversation() -> list[dict[str, str | None]]:
        service.reset()
        return []

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

        with gr.Row():
            msg = gr.Textbox(
                label="Your message", placeholder="Ask me to use MCP tools...", scale=4
            )
            clear = gr.Button("Clear", scale=1)

        msg.submit(fn=chat_wrapper, inputs=[msg, chatbot], outputs=chatbot).then(
            lambda: "", outputs=msg
        )

        clear.click(fn=reset_conversation, outputs=chatbot)

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
