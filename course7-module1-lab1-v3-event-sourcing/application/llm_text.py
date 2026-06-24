from langchain_core.language_models import BaseChatModel


def invoke_text(model: BaseChatModel, prompt: str) -> str:
    """Invoke the model with a text prompt; return narrowed `str` content.

    `BaseMessage.content` is typed `str | list[content blocks]` to support
    multimodal models. Text-only nodes that read `response.content` without
    a narrowing guard will silently start returning stringified content-
    block lists the day someone swaps in a multimodal model. This helper
    centralises the narrowing in one place — nodes call `invoke_text`
    and get a stripped `str` back.

    Application-layer concern: this lives in `application/`, not `infra/`,
    because the narrowing is an application-layer contract, not a model-
    provider detail.
    """
    response = model.invoke(prompt)
    if not isinstance(response.content, str):
        raise TypeError(
            f"Expected str content from text-only prompt, got "
            f"{type(response.content).__name__}"
        )
    return response.content.strip()