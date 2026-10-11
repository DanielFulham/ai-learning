import json
from typing import Any, Literal

from domain.conversation import ToolSpec
from domain.directory_item import DirectoryItem
from domain.prompt_spec import PromptArgument, PromptSpec
from interfaces.mcp_session_interface import RawPrompt, RawTool


def to_tool_spec(raw: RawTool) -> ToolSpec:
    return ToolSpec(
        name=raw["name"],
        description=raw["description"] or "MCP Tool",
        input_schema=raw["input_schema"],
    )


def to_prompt_spec(raw: RawPrompt) -> PromptSpec:
    return PromptSpec(
        name=raw["name"],
        arguments=tuple(PromptArgument(a["name"], a["required"]) for a in raw["arguments"]),
    )


def _load_payload(text: str) -> dict[str, Any]:
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("The server's reply was not a JSON object")
    if "error" in payload:
        raise ValueError(str(payload["error"]))
    return payload


def parse_file_content(text: str) -> str:
    """Read the file text out of the server's file resource reply."""
    return str(_load_payload(text)["file_content"])


def parse_directory_listing(text: str) -> list[DirectoryItem]:
    """Read the rows out of the server's directory resource reply."""
    items: list[DirectoryItem] = []
    for item in _load_payload(text)["items"]:
        kind: Literal["file", "directory"] = "directory" if item["type"] == "directory" else "file"
        items.append(
            DirectoryItem(
                name=item["name"], kind=kind, size=int(item["size"]), modified=item["modified"]
            )
        )
    return items
