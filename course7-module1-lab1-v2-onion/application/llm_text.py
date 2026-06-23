from langchain_core.language_models import BaseChatModel


def invoke_text(model: BaseChatModel, prompt: str) -> str:
    """Invoke a chat model with a text prompt and return a stripped `str`.

    Owns the narrowing from `BaseMessage.content`'s `str | list` union into
    `str`. Every application node that calls an LLM goes through this
    helper so the narrowing exists in exactly one place. Closes V1
    finding #4 — the narrowing was inlined in the QA node; V2 lifts it to
    a single application-layer location.

    Raises `TypeError` if the model returns a non-`str` content (multimodal
    block list, etc.) — an explicit guard makes the assumption visible
    rather than hiding it behind `cast(str, response.content)`.
    """
    response = model.invoke(prompt)
    if not isinstance(response.content, str):
        raise TypeError(
            f"Expected str content from text-only prompt, got "
            f"{type(response.content).__name__}"
        )
    return response.content.strip()