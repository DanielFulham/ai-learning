import json

import pytest

from application.mappers import (
    parse_directory_listing,
    parse_file_content,
    to_prompt_spec,
    to_tool_spec,
)
from interfaces.mcp_session_interface import RawPrompt, RawTool


def test_to_tool_spec_maps_fields() -> None:
    raw: RawTool = {"name": "TEST_TOOL", "description": "TEST_DESC", "input_schema": {"a": 1}}
    spec = to_tool_spec(raw)
    assert (spec.name, spec.description, spec.input_schema) == ("TEST_TOOL", "TEST_DESC", {"a": 1})


def test_to_tool_spec_defaults_empty_description() -> None:
    raw: RawTool = {"name": "TEST_TOOL", "description": "", "input_schema": {}}
    assert to_tool_spec(raw).description == "MCP Tool"


def test_to_prompt_spec_maps_arguments() -> None:
    raw: RawPrompt = {
        "name": "TEST_PROMPT",
        "arguments": [{"name": "TEST_ARG", "required": True}],
    }
    spec = to_prompt_spec(raw)
    assert spec.name == "TEST_PROMPT"
    assert [(a.name, a.required) for a in spec.arguments] == [("TEST_ARG", True)]


def test_parse_file_content() -> None:
    assert parse_file_content(json.dumps({"file_content": "TEST_TEXT"})) == "TEST_TEXT"


def test_parse_file_content_raises_the_servers_error() -> None:
    with pytest.raises(ValueError, match="TEST_ERROR"):
        parse_file_content(json.dumps({"error": "TEST_ERROR"}))


def test_parse_directory_listing() -> None:
    text = json.dumps(
        {
            "items": [
                {"name": "a.txt", "type": "file", "size": 3, "modified": "TEST_TIME"},
                {"name": "sub", "type": "directory", "size": 0, "modified": "TEST_TIME"},
            ]
        }
    )
    items = parse_directory_listing(text)
    assert [(i.name, i.kind, i.size) for i in items] == [
        ("a.txt", "file", 3),
        ("sub", "directory", 0),
    ]


def test_parse_directory_listing_raises_the_servers_error() -> None:
    with pytest.raises(ValueError, match="TEST_ERROR"):
        parse_directory_listing(json.dumps({"error": "TEST_ERROR"}))


def test_parse_rejects_non_object() -> None:
    with pytest.raises(ValueError, match="not a JSON object"):
        parse_file_content("[1, 2]")
