from collections.abc import Sequence

from anthropic import AsyncAnthropic
from anthropic.types import (
    MessageParam,
    TextBlockParam,
    ToolParam,
    ToolResultBlockParam,
    ToolUseBlockParam,
)

from domain.conversation import Message, ModelReply, ToolCall, ToolSpec
from interfaces.chat_model_interface import ChatModelInterface


class AnthropicChatModel(ChatModelInterface):
    """Chat model backed by the Anthropic Messages API.

    Requires `ANTHROPIC_API_KEY` in the environment unless a client is passed in.
    """

    def __init__(
        self, model_id: str, max_tokens: int, client: AsyncAnthropic | None = None
    ) -> None:
        self._model_id = model_id
        self._max_tokens = max_tokens
        self._client = client or AsyncAnthropic()

    async def complete(self, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> ModelReply:
        response = await self._client.messages.create(
            model=self._model_id,
            max_tokens=self._max_tokens,
            messages=[_to_message_param(m) for m in messages],
            tools=[_to_tool_param(t) for t in tools],
        )

        texts: list[str] = []
        calls: list[ToolCall] = []
        for block in response.content:
            if block.type == "text":
                texts.append(block.text)
            elif block.type == "tool_use":
                arguments = dict(block.input) if isinstance(block.input, dict) else {}
                calls.append(ToolCall(block.id, block.name, arguments))

        # A tool call in a reply cut off at max_tokens may be incomplete, so it is not run.
        run_tools = response.stop_reason == "tool_use"
        return ModelReply(
            text="\n".join(texts),
            tool_calls=tuple(calls) if run_tools else (),
            truncated=response.stop_reason == "max_tokens",
        )


def _to_tool_param(tool: ToolSpec) -> ToolParam:
    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": tool.input_schema,
    }


def _to_message_param(message: Message) -> MessageParam:
    if message.tool_results:
        results: list[ToolResultBlockParam] = []
        for r in message.tool_results:
            block: ToolResultBlockParam = {
                "type": "tool_result",
                "tool_use_id": r.tool_call_id,
                "content": r.content,
            }
            if r.is_error:
                block["is_error"] = True
            results.append(block)
        return {"role": message.role, "content": results}

    if message.tool_calls:
        content: list[TextBlockParam | ToolUseBlockParam] = []
        if message.text:
            content.append({"type": "text", "text": message.text})
        for call in message.tool_calls:
            content.append(
                {"type": "tool_use", "id": call.id, "name": call.name, "input": call.arguments}
            )
        return {"role": message.role, "content": content}

    return {"role": message.role, "content": message.text}
