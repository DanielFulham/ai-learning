import json
from collections.abc import Mapping

from domain.approval import ApprovalChoice
from domain.prompt_info import PromptInfo, PromptMessage, RenderedPrompt
from domain.resource_info import ResourceInfo
from domain.tool_info import ToolCallResult, ToolInfo
from interfaces.approval_source_interface import ApprovalAnswer
from interfaces.mcp_session_interface import (
    RawPrompt,
    RawPromptResult,
    RawResource,
    RawTool,
    RawToolResult,
)

_ANSWERS: Mapping[ApprovalChoice, ApprovalAnswer] = {
    ApprovalChoice.ACCEPT: "accept",
    ApprovalChoice.DECLINE: "decline",
    ApprovalChoice.CANCEL: "cancel",
}


def to_tool_info(raw: RawTool) -> ToolInfo:
    return ToolInfo(
        name=raw["name"], description=raw["description"], input_schema=raw["input_schema"]
    )


def to_tool_call_result(raw: RawToolResult) -> ToolCallResult:
    return ToolCallResult(texts=tuple(raw["texts"]), is_error=raw["is_error"])


def to_resource_info(raw: RawResource) -> ResourceInfo:
    return ResourceInfo(uri=raw["uri"], name=raw["name"], description=raw["description"])


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


def answer_from_choice(choice: ApprovalChoice) -> ApprovalAnswer:
    """Turn the user's choice into the plain word the session sends back to the server."""
    return _ANSWERS[choice]


_YES_WORDS = frozenset({"yes", "y", "approve", "ok", "confirm"})
_NO_WORDS = frozenset({"no", "n", "deny", "decline"})


def choice_from_reply(text: str) -> ApprovalChoice | None:
    """Read a typed reply as an answer to a waiting question, or None if it is not one.

    Only a bare word counts, so a sentence that happens to contain "yes" never approves.
    """
    word = text.strip().lower()
    if word in _YES_WORDS:
        return ApprovalChoice.ACCEPT
    if word in _NO_WORDS:
        return ApprovalChoice.DECLINE
    if word == "cancel":
        return ApprovalChoice.CANCEL
    return None


def parse_json_object(text: str) -> dict[str, object]:
    """Parse text the user typed as the arguments of a call. Empty text means no arguments.

    Raises ValueError (JSONDecodeError is one) unless the text is a JSON object.
    """
    if not text.strip():
        return {}
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("Arguments must be a JSON object")
    return {str(key): value for key, value in parsed.items()}


def parse_prompt_arguments(text: str) -> dict[str, str]:
    """Like `parse_json_object`, but every value must be text, as the protocol requires."""
    arguments: dict[str, str] = {}
    for key, value in parse_json_object(text).items():
        if not isinstance(value, str):
            raise ValueError(f"The value for {key!r} must be text")
        arguments[key] = value
    return arguments
