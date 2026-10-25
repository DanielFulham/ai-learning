from collections.abc import Callable, Sequence

from openai import AsyncOpenAI, omit
from openai.types.chat import (
    ChatCompletionMessageFunctionToolCall,
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
)

from domain.conversation import Message, ModelReply, ToolCall
from domain.tool_info import ToolInfo
from interfaces.llm_chat_model_interface import LlmChatModelInterface


class OpenAiChatModel(LlmChatModelInterface):
    """Chat model backed by the OpenAI Chat Completions API.

    Needs `OPENAI_API_KEY` in the environment when the first request is made, unless a client
    is passed in. Building the object does not need it: the OpenAI client raises without a
    key, so it is created on first use.
    """

    def __init__(
        self,
        model_id: str,
        client: AsyncOpenAI | None = None,
        client_factory: Callable[[], AsyncOpenAI] = AsyncOpenAI,
    ) -> None:
        self._model_id = model_id
        self._client = client
        self._client_factory = client_factory

    def _get_client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = self._client_factory()
        return self._client

    async def complete(
        self, messages: Sequence[Message], tools: Sequence[ToolInfo]
    ) -> ModelReply | None:
        response = await self._get_client().chat.completions.create(
            model=self._model_id,
            messages=[_to_message_param(message) for message in messages],
            tools=[_to_tool_param(tool) for tool in tools] if tools else omit,
            tool_choice="auto" if tools else omit,
        )
        if not response.choices:
            return None

        message = response.choices[0].message
        calls = [
            ToolCall(id=call.id, name=call.function.name, arguments_json=call.function.arguments)
            for call in message.tool_calls or []
            if isinstance(call, ChatCompletionMessageFunctionToolCall)
        ]
        return ModelReply(text=message.content, tool_calls=tuple(calls))


def _to_tool_param(tool: ToolInfo) -> ChatCompletionToolParam:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.input_schema,
        },
    }


def _to_message_param(message: Message) -> ChatCompletionMessageParam:
    if message.role == "tool":
        return {
            "role": "tool",
            "tool_call_id": message.tool_call_id or "",
            "content": message.text or "",
        }

    if message.role == "assistant":
        if message.tool_calls:
            return {
                "role": "assistant",
                "content": message.text or "",
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {"name": call.name, "arguments": call.arguments_json},
                    }
                    for call in message.tool_calls
                ],
            }
        return {"role": "assistant", "content": message.text}

    return {"role": "user", "content": message.text or ""}
