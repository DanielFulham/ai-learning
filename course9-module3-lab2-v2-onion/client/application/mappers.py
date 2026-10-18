from domain.prompt_info import PromptInfo, PromptMessage, RenderedPrompt
from domain.resource_template_info import ResourceTemplateInfo
from domain.tool_info import ToolCallResult, ToolInfo
from interfaces.mcp_session_interface import (
    RawPrompt,
    RawPromptResult,
    RawResourceTemplate,
    RawTool,
    RawToolResult,
)


def to_tool_info(raw: RawTool) -> ToolInfo:
    return ToolInfo(
        name=raw["name"], description=raw["description"], input_schema=raw["input_schema"]
    )


def to_tool_call_result(raw: RawToolResult) -> ToolCallResult:
    return ToolCallResult(texts=tuple(raw["texts"]), is_error=raw["is_error"])


def to_resource_template_info(raw: RawResourceTemplate) -> ResourceTemplateInfo:
    return ResourceTemplateInfo(
        name=raw["name"], uri_template=raw["uri_template"], description=raw["description"]
    )


def to_prompt_info(raw: RawPrompt) -> PromptInfo:
    return PromptInfo(
        name=raw["name"],
        description=raw["description"],
        argument_names=tuple(raw["argument_names"]),
    )


def to_rendered_prompt(raw: RawPromptResult) -> RenderedPrompt:
    return RenderedPrompt(
        description=raw["description"],
        messages=tuple(PromptMessage(role=m["role"], text=m["text"]) for m in raw["messages"]),
    )
