from domain.conversation import ChatReply
from domain.prompt_info import PromptInfo, RenderedPrompt
from domain.resource_info import ResourceInfo
from domain.tool_info import ToolInfo

_MAX_RESULT_CHARS = 200


def format_tool(tool: ToolInfo) -> str:
    return f"• {tool.name}: {tool.description}"


def format_resource(resource: ResourceInfo) -> str:
    return f"• {resource.uri}\n  {resource.name}: {resource.description}"


def format_prompt(prompt: PromptInfo) -> str:
    args_info = f" (args: {', '.join(prompt.argument_names)})" if prompt.argument_names else ""
    return f"• {prompt.name}: {prompt.description}{args_info}"


def format_text_parts(texts: tuple[str, ...]) -> str:
    """Each text part followed by a newline, as the lab's GUI shows tool and resource output."""
    return "".join(f"{text}\n" for text in texts)


def format_rendered_prompt(prompt: RenderedPrompt) -> str:
    output = f"--- Prompt: {prompt.description} ---\n\n"
    for message in prompt.messages:
        output += f"{message.role}: {message.text}\n"
    return output


def format_chat_reply(reply: ChatReply) -> str:
    """The answer, led by the tools run to get it (long results cut short, one line each)."""
    if not reply.tool_activity:
        return reply.text
    lines = ["**Tool calls**"]
    for activity in reply.tool_activity:
        result = " ".join(activity.result.split())
        if len(result) > _MAX_RESULT_CHARS:
            result = result[:_MAX_RESULT_CHARS] + "…"
        lines.append(f"- `{activity.name}({activity.arguments_json})` → {result}")
    return "\n".join(lines) + "\n\n" + reply.text
