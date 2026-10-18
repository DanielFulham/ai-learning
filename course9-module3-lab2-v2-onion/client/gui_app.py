import sys

import gradio as gr

from application.container import initialise_gui
from application.interfaces.gui_service_interface import GuiServiceInterface
from cli_arguments import parse_arguments
from config import GUI_PORT, UI_HOST


def with_choices(output: str, choices: list[str]) -> tuple[str, dict[str, object]]:
    """The text for a textbox, and the update that fills the dropdown beside it."""
    return output, gr.update(choices=choices)


def build_interface(service: GuiServiceInterface, server: str, roots: str) -> gr.Blocks:
    """The lab's three-tab Gradio page. Gradio lives here and nowhere deeper."""

    async def list_tools() -> tuple[str, dict[str, object]]:
        return with_choices(*await service.list_tools())

    async def list_prompts() -> tuple[str, dict[str, object]]:
        return with_choices(*await service.list_prompts())

    with gr.Blocks(title="MCP HTTP Client") as interface:
        gr.Markdown("# MCP HTTP Client - Remote Server Access")
        gr.Markdown(f"""
**Server:** {server}
**Workspace Roots:** {roots}

This client connects to a remote MCP server via HTTP transport.
All file operations are restricted to the workspace roots directory.
""")

        with gr.Tabs():
            with gr.Tab("Tools"):
                gr.Markdown("### Discover and Execute Server Tools")
                with gr.Row():
                    with gr.Column():
                        list_tools_btn = gr.Button("List Tools", variant="primary")
                        tools_output = gr.Textbox(label="Available Tools", lines=5)

                    with gr.Column():
                        tool_dropdown = gr.Dropdown(
                            label="Select Tool", choices=[], interactive=True
                        )
                        tool_args = gr.Textbox(
                            label="Arguments (JSON)",
                            placeholder='{"filepath": "test.txt"}',
                            lines=3,
                        )
                        call_tool_btn = gr.Button("Call Tool", variant="primary")
                        tool_result = gr.Textbox(label="Tool Result", lines=8)

                list_tools_btn.click(fn=list_tools, outputs=[tools_output, tool_dropdown])
                call_tool_btn.click(
                    fn=service.call_tool, inputs=[tool_dropdown, tool_args], outputs=tool_result
                )

            with gr.Tab("Resources"):
                gr.Markdown("### Access Server Resources")
                with gr.Row():
                    with gr.Column():
                        list_resources_btn = gr.Button("List Resource Templates", variant="primary")
                        resources_output = gr.Textbox(label="Available Resources", lines=5)

                    with gr.Column():
                        resource_uri = gr.Textbox(
                            label="Resource URI", placeholder="file://workspace/README.md", lines=1
                        )
                        read_resource_btn = gr.Button("Read Resource", variant="primary")
                        resource_content = gr.Textbox(label="Resource Content", lines=10)

                list_resources_btn.click(fn=service.list_resources, outputs=resources_output)
                read_resource_btn.click(
                    fn=service.read_resource, inputs=resource_uri, outputs=resource_content
                )

            with gr.Tab("Prompts"):
                gr.Markdown("### List and Get Prompts")
                with gr.Row():
                    with gr.Column():
                        list_prompts_btn = gr.Button("List Prompts", variant="primary")
                        prompts_output = gr.Textbox(label="Available Prompts", lines=5)

                    with gr.Column():
                        prompt_dropdown = gr.Dropdown(
                            label="Select Prompt", choices=[], interactive=True
                        )
                        prompt_args = gr.Textbox(
                            label="Arguments (JSON)",
                            placeholder='{"filename": "example.py"}',
                            lines=2,
                        )
                        get_prompt_btn = gr.Button("Get Prompt", variant="primary")
                        prompt_result = gr.Textbox(label="Prompt Messages", lines=10)

                list_prompts_btn.click(fn=list_prompts, outputs=[prompts_output, prompt_dropdown])
                get_prompt_btn.click(
                    fn=service.get_prompt,
                    inputs=[prompt_dropdown, prompt_args],
                    outputs=prompt_result,
                )

    return interface


def main() -> None:
    server, roots = parse_arguments(sys.argv)
    service = initialise_gui(server)
    build_interface(service, server, roots).queue().launch(
        server_name=UI_HOST, server_port=GUI_PORT
    )


if __name__ == "__main__":
    main()
