import json

from application.formatters import (
    format_prompt,
    format_rendered_prompt,
    format_resource_template,
    format_text_parts,
    format_tool,
)
from application.mappers import (
    to_prompt_info,
    to_rendered_prompt,
    to_resource_template_info,
    to_tool_call_result,
    to_tool_info,
)
from interfaces.mcp_session_interface import McpSessionInterface

_TOOL_ERROR_WITHOUT_TEXT = "the tool reported an error"


class GuiService:
    """What the lab's GUI client does for each button, as text a page can show.

    Every method connects first (the session ignores a second connect), as the lab's do.
    """

    def __init__(self, session: McpSessionInterface) -> None:
        self._session = session

    async def list_tools(self) -> tuple[str, list[str]]:
        """The tools as bullet lines, and their names for the dropdown."""
        await self._session.connect()
        tools = [to_tool_info(raw) for raw in await self._session.list_tools()]
        return "\n".join(format_tool(t) for t in tools), [t.name for t in tools]

    async def call_tool(self, tool_name: str | None, arguments_json: str) -> str:
        await self._session.connect()
        if not tool_name:
            return "Error: Please select a tool from the dropdown first"
        try:
            arguments = json.loads(arguments_json) if arguments_json else {}
            result = to_tool_call_result(await self._session.call_tool(tool_name, arguments))
            if result.is_error:
                return f"Error: {' '.join(result.texts) or _TOOL_ERROR_WITHOUT_TEXT}"
            return format_text_parts(result.texts) or "No response"
        except json.JSONDecodeError:
            return "Error: Invalid JSON format"
        except Exception as e:
            return f"Error: {e}"

    async def list_resources(self) -> str:
        await self._session.connect()
        templates = [
            to_resource_template_info(raw) for raw in await self._session.list_resource_templates()
        ]
        if not templates:
            return "No resources available"
        return "\n\n".join(format_resource_template(t) for t in templates)

    async def read_resource(self, uri: str) -> str:
        await self._session.connect()
        if not uri:
            return "Error: Please enter a resource URI"
        try:
            return format_text_parts(tuple(await self._session.read_resource(uri))) or "No content"
        except Exception as e:
            return f"Error: {e}"

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
            arguments = json.loads(arguments_json) if arguments_json else {}
            rendered = to_rendered_prompt(await self._session.get_prompt(prompt_name, arguments))
            return format_rendered_prompt(rendered)
        except json.JSONDecodeError:
            return "Error: Invalid JSON format"
        except Exception as e:
            return f"Error: {e}"
