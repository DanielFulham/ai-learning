import sys

import gradio as gr

import config
from application.approval_desks import ApprovalDesks
from application.container import initialise_gui
from application.interfaces.approval_desk_interface import ApprovalDeskInterface
from application.interfaces.gui_service_interface import GuiServiceInterface
from cli_arguments import parse_arguments
from domain.approval import ApprovalChoice
from page_session import forget_session, session_id

_NO_QUESTION = "No question is waiting."


def with_choices(output: str, choices: list[str]) -> tuple[str, dict[str, object]]:
    """The text for a textbox, and the update that fills the dropdown beside it."""
    return output, gr.update(choices=choices)


def waiting_question_text(desk: ApprovalDeskInterface) -> str:
    return desk.pending_message() or _NO_QUESTION


def answer_waiting_question(desk: ApprovalDeskInterface, choice: ApprovalChoice) -> str:
    if desk.respond(choice):
        return f"Answer sent: {choice.value}"
    return _NO_QUESTION


def build_interface(service: GuiServiceInterface, desks: ApprovalDesks) -> gr.Blocks:
    """The GUI page. Gradio lives here and nowhere deeper.

    The server can ask a question before it runs a tool. The question appears in the panel at
    the top, and Approve or Decline answers it. Each browser session has its own desk, so one
    session's click can only answer that session's question.
    """

    def desk_for(request: gr.Request) -> ApprovalDeskInterface:
        return desks.desk_for(session_id(request))

    async def list_tools() -> tuple[str, dict[str, object]]:
        return with_choices(*await service.list_tools())

    async def list_prompts() -> tuple[str, dict[str, object]]:
        return with_choices(*await service.list_prompts())

    async def call_tool(tool: str | None, arguments: str, request: gr.Request) -> str:
        return await service.call_tool(tool, arguments, desk_for(request))

    async def read_resource(uri: str, request: gr.Request) -> str:
        return await service.read_resource(uri, desk_for(request))

    async def view_policy(request: gr.Request) -> str:
        return await service.view_policy(desk_for(request))

    async def view_audit_log(request: gr.Request) -> str:
        return await service.view_audit_log(desk_for(request))

    def show_waiting_question(request: gr.Request) -> str:
        return waiting_question_text(desk_for(request))

    def approve(request: gr.Request) -> str:
        return answer_waiting_question(desk_for(request), ApprovalChoice.ACCEPT)

    def decline(request: gr.Request) -> str:
        return answer_waiting_question(desk_for(request), ApprovalChoice.DECLINE)

    def close_page(request: gr.Request) -> None:
        forget_session(desks, request)

    with gr.Blocks(title="MCP Permission Client") as interface:
        gr.Markdown("""
# MCP Permission Client
The server decides what runs. When a tool needs your approval, the question appears here.
""")
        with gr.Group():
            waiting = gr.Textbox(label="Waiting for your approval", value=_NO_QUESTION, lines=3)
            with gr.Row():
                approve_btn = gr.Button("Approve", variant="primary")
                decline_btn = gr.Button("Decline", variant="stop")
            answer_note = gr.Textbox(label="Last answer", lines=1)
        ticker = gr.Timer(config.POLL_SECONDS)
        ticker.tick(fn=show_waiting_question, outputs=waiting, api_name="waiting_question")
        approve_btn.click(fn=approve, outputs=answer_note, api_name="approve")
        decline_btn.click(fn=decline, outputs=answer_note, api_name="decline")

        with gr.Tabs():
            with gr.Tab("Tools"):
                gr.Markdown("### List and Call Tools")
                with gr.Row():
                    with gr.Column():
                        list_tools_btn = gr.Button("List Tools", variant="primary")
                        tools_output = gr.Textbox(label="Available Tools", lines=10)
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
                        tool_result = gr.Textbox(label="Result", lines=10)

                list_tools_btn.click(
                    fn=list_tools, outputs=[tools_output, tool_dropdown], api_name="list_tools"
                )
                call_tool_btn.click(
                    fn=call_tool,
                    inputs=[tool_dropdown, tool_args],
                    outputs=tool_result,
                    api_name="call_tool",
                )

            with gr.Tab("Resources"):
                gr.Markdown("### List and Read Resources")
                with gr.Row():
                    with gr.Column():
                        list_resources_btn = gr.Button("List Resources", variant="primary")
                        resources_output = gr.Textbox(label="Available Resources", lines=10)
                    with gr.Column():
                        resource_uri = gr.Textbox(
                            label="Resource URI", placeholder="file://audit/log"
                        )
                        read_resource_btn = gr.Button("Read Resource", variant="primary")
                        resource_content = gr.Textbox(label="Resource Content", lines=10)

                list_resources_btn.click(
                    fn=service.list_resources, outputs=resources_output, api_name="list_resources"
                )
                read_resource_btn.click(
                    fn=read_resource,
                    inputs=resource_uri,
                    outputs=resource_content,
                    api_name="read_resource",
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
                            placeholder='{"operation": "write_file", "risk_level": "MEDIUM"}',
                            lines=2,
                        )
                        get_prompt_btn = gr.Button("Get Prompt", variant="primary")
                        prompt_result = gr.Textbox(label="Prompt Messages", lines=10)

                list_prompts_btn.click(
                    fn=list_prompts,
                    outputs=[prompts_output, prompt_dropdown],
                    api_name="list_prompts",
                )
                get_prompt_btn.click(
                    fn=service.get_prompt,
                    inputs=[prompt_dropdown, prompt_args],
                    outputs=prompt_result,
                    api_name="get_prompt",
                )

            with gr.Tab("Permissions"):
                gr.Markdown(
                    "### Policy and Audit Log\n"
                    "Read-only. The server owns the policy: change it in the server's "
                    "`state/permissions.json`."
                )
                with gr.Row():
                    with gr.Column():
                        view_policy_btn = gr.Button("View Policy", variant="secondary")
                        policy_output = gr.Textbox(label="Policy", lines=10)
                    with gr.Column():
                        view_audit_btn = gr.Button("View Audit Log", variant="secondary")
                        audit_output = gr.Textbox(label="Audit Log", lines=15)

                view_policy_btn.click(fn=view_policy, outputs=policy_output, api_name="view_policy")
                view_audit_btn.click(fn=view_audit_log, outputs=audit_output, api_name="view_audit")

    interface.unload(close_page)
    return interface


def main() -> None:
    app = initialise_gui(parse_arguments(sys.argv))
    build_interface(app.service, app.desks).queue().launch(
        server_name=config.UI_HOST, server_port=config.GUI_PORT
    )


if __name__ == "__main__":
    main()
