from application.formatters import (
    format_prompt,
    format_rendered_prompt,
    format_resource,
    format_text_parts,
    format_tool,
)
from application.mappers import (
    parse_json_object,
    parse_prompt_arguments,
    to_prompt_info,
    to_rendered_prompt,
    to_resource_info,
    to_tool_call_result,
    to_tool_info,
)
from interfaces.approval_source_interface import ApprovalSourceInterface
from interfaces.mcp_session_interface import McpSessionInterface

_TOOL_ERROR_WITHOUT_TEXT = "the tool reported an error"


class GuiService:
    """What the GUI client does for each button, as text a page can show.

    Every method connects first (the session ignores a second connect). The service keeps no
    state: calls that can make the server ask a question take the approval source of the page
    session that made them, so several sessions can share one service. The policy and the audit
    log are shown, not edited: the server owns the policy.
    """

    def __init__(self, session: McpSessionInterface, policy_uri: str, audit_log_uri: str) -> None:
        self._session = session
        self._policy_uri = policy_uri
        self._audit_log_uri = audit_log_uri

    async def list_tools(self) -> tuple[str, list[str]]:
        """The tools as bullet lines, and their names for the dropdown."""
        await self._session.connect()
        tools = [to_tool_info(raw) for raw in await self._session.list_tools()]
        return "\n".join(format_tool(t) for t in tools), [t.name for t in tools]

    async def call_tool(
        self, tool_name: str | None, arguments_json: str, approvals: ApprovalSourceInterface
    ) -> str:
        await self._session.connect()
        if not tool_name:
            return "Error: Please select a tool from the dropdown first"
        try:
            arguments = parse_json_object(arguments_json)
        except ValueError:
            return "Error: Arguments must be a JSON object"
        try:
            raw = await self._session.call_tool(tool_name, arguments, approvals)
        except Exception as e:
            return f"Error: {e}"
        result = to_tool_call_result(raw)
        if result.is_error:
            return f"Error: {' '.join(result.texts) or _TOOL_ERROR_WITHOUT_TEXT}"
        return format_text_parts(result.texts) or "No response"

    async def list_resources(self) -> str:
        await self._session.connect()
        resources = [to_resource_info(raw) for raw in await self._session.list_resources()]
        if not resources:
            return "No resources available"
        return "\n\n".join(format_resource(r) for r in resources)

    async def read_resource(self, uri: str, approvals: ApprovalSourceInterface) -> str:
        await self._session.connect()
        if not uri:
            return "Error: Please enter a resource URI"
        return await self._read(uri, approvals)

    async def view_policy(self, approvals: ApprovalSourceInterface) -> str:
        await self._session.connect()
        return await self._read(self._policy_uri, approvals)

    async def view_audit_log(self, approvals: ApprovalSourceInterface) -> str:
        await self._session.connect()
        return await self._read(self._audit_log_uri, approvals)

    async def _read(self, uri: str, approvals: ApprovalSourceInterface) -> str:
        try:
            texts = await self._session.read_resource(uri, approvals)
        except Exception as e:
            return f"Error: {e}"
        return format_text_parts(tuple(texts)) or "No content"

    async def list_prompts(self) -> tuple[str, list[str]]:
        """The prompts as bullet lines with their arguments, and their names for the dropdown."""
        await self._session.connect()
        prompts = [to_prompt_info(raw) for raw in await self._session.list_prompts()]
        return "\n".join(format_prompt(p) for p in prompts), [p.name for p in prompts]

    async def get_prompt(self, prompt_name: str | None, arguments_json: str) -> str:
        await self._session.connect()
        if not prompt_name:
            return "Error: Please select a prompt from the dropdown first"
        try:
            arguments = parse_prompt_arguments(arguments_json)
        except ValueError:
            return "Error: Prompt arguments must be a JSON object of text values"
        try:
            rendered = to_rendered_prompt(await self._session.get_prompt(prompt_name, arguments))
        except Exception as e:
            return f"Error: {e}"
        return format_rendered_prompt(rendered)
