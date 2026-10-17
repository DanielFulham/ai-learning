from dataclasses import FrozenInstanceError

import pytest

from domain.history_entry import HistoryEntry
from domain.prompt_info import PromptInfo, PromptMessage, RenderedPrompt
from domain.resource_template_info import ResourceTemplateInfo
from domain.tool_info import ToolCallResult, ToolInfo


def test_tool_info_is_frozen() -> None:
    tool = ToolInfo(name="TEST_TOOL", description="TEST_DESC")
    with pytest.raises(FrozenInstanceError):
        setattr(tool, "name", "TEST_OTHER")


def test_tool_call_result_is_frozen() -> None:
    result = ToolCallResult(texts=("TEST_TEXT",), is_error=False)
    with pytest.raises(FrozenInstanceError):
        setattr(result, "is_error", True)


def test_resource_template_info_is_frozen() -> None:
    template = ResourceTemplateInfo(name="TEST_NAME", uri_template="TEST://{x}")
    with pytest.raises(FrozenInstanceError):
        setattr(template, "name", "TEST_OTHER")


def test_prompt_info_defaults_to_no_arguments_and_is_frozen() -> None:
    prompt = PromptInfo(name="TEST_PROMPT", description="TEST_DESC")
    assert prompt.argument_names == ()
    with pytest.raises(FrozenInstanceError):
        setattr(prompt, "name", "TEST_OTHER")


def test_rendered_prompt_defaults_to_no_messages_and_is_frozen() -> None:
    prompt = RenderedPrompt(description="TEST_DESC")
    assert prompt.messages == ()
    with pytest.raises(FrozenInstanceError):
        setattr(prompt, "description", "TEST_OTHER")


def test_prompt_message_is_frozen() -> None:
    message = PromptMessage(role="user", text="TEST_TEXT")
    with pytest.raises(FrozenInstanceError):
        setattr(message, "text", "TEST_OTHER")


def test_history_entry_is_frozen() -> None:
    entry = HistoryEntry(operation="tool", name="TEST_TOOL")
    with pytest.raises(FrozenInstanceError):
        setattr(entry, "name", "TEST_OTHER")
