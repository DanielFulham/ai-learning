from domain.history_entry import HistoryEntry
from domain.prompt_info import PromptInfo, RenderedPrompt
from domain.resource_template_info import ResourceTemplateInfo
from domain.tool_info import ToolInfo

CONNECTED = "✓ Connected to MCP server"

WELCOME = "\n=== MCP Client ==="

COMMANDS_LINE = (
    "Commands: tools | call | resources | read | prompts | prompt | history | help | quit\n"
)

HELP_TEXT = """
  Available Commands:
  -------------------
  tools       - List available tools
  call        - Invoke a tool
  resources   - List resource templates
  read        - Read a resource by URI
  prompts     - List prompt templates
  prompt      - Get a rendered prompt
  history     - Show the tool call history
  help        - Show this help message
  quit        - Exit the client
"""


def format_tool(tool: ToolInfo) -> str:
    return f"  • {tool.name}: {tool.description}"


def format_resource_template(template: ResourceTemplateInfo) -> str:
    return f"  • {template.name}\n    URI template: {template.uri_template}"


def format_prompt(prompt: PromptInfo) -> str:
    args_info = f" (args: {', '.join(prompt.argument_names)})" if prompt.argument_names else ""
    return f"  • {prompt.name}: {prompt.description}{args_info}"


def format_rendered_prompt(prompt: RenderedPrompt) -> str:
    lines = [f"\n--- Prompt: {prompt.description} ---"]
    lines.extend(f"{message.role}: {message.text}" for message in prompt.messages)
    return "\n".join(lines)


def format_history(entries: list[HistoryEntry]) -> str:
    if not entries:
        return "  No operations yet"
    lines = ["  Operation History:"]
    lines.extend(f"    {i}. {e.operation}: {e.name}" for i, e in enumerate(entries, 1))
    return "\n".join(lines)
