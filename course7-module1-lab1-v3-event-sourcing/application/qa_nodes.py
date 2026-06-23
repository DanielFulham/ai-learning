from dataclasses import replace

import httpx
from langchain_core.language_models import BaseChatModel

from application.llm_text import invoke_text
from domain.error_info import ErrorInfo
from domain.qa_exchange import QAExchange
from domain.state_schemas import QAState


# Recoverable LLM call failures — transport-layer errors and timeouts.
_RECOVERABLE_LLM_ERRORS: tuple[type[BaseException], ...] = (
    httpx.HTTPError,
    ConnectionError,
    TimeoutError,
)


def context_provider_node(state: QAState) -> QAState:
    """Look up canned context by keyword match on the question.

    V1's keyword-matching context provider, preserved as-is for the
    workflow's behaviour. The hallucination in V1's third demo question
    (`"What is the best guided project?"`) still happens — this matcher
    returns LangGraph context for any question containing 'guided project',
    regardless of subject. The production move is an eval pipeline over
    golden questions, not a smarter matcher inside the node.

    V3a behavioural change: always returns an explicit `{"exchange":
    replace(current, context=...)}` delta, including on the no-match case
    (where context is set to None). V2 returned `state` unchanged on no
    match, which depended on LangGraph emitting `updates` for no-op nodes.
    Being explicit guarantees the translator sees a ContextRetrieved
    event for every ContextNode execution — the observability
    consistency lift. Behaviour of the hallucination is unchanged; the
    event log is what changed.
    """
    current = state.get("exchange")
    if current is None:
        raise ValueError("context_provider_node requires an exchange in state")

    question_lower = current.question.lower()
    if "langgraph" in question_lower or "guided project" in question_lower:
        context = (
            "This guided project is about using LangGraph, a Python library "
            "to design state-based workflows. LangGraph simplifies building "
            "complex applications by connecting modular nodes with conditional edges."
        )
        return {"exchange": replace(current, context=context)}

    return {"exchange": replace(current, context=None)}


def make_qa_node(model: BaseChatModel):
    """Factory closing over the chat model.

    V3a behavioural change: the broad `except Exception` from V2 is
    replaced with a specific catch over `_RECOVERABLE_LLM_ERRORS`.
    On a recoverable error, the node populates `error_info` on the
    QAExchange with the exception's type and message, AND sets a
    user-safe message on `answer`. The translator branches on the
    presence of `error_info` to emit `ModelInvocationFailed`; the
    user-safe `answer` is for the UI only.

    Logic bugs propagate unchanged — `ValueError` from a missing
    exchange, `KeyError` from a malformed state, etc. — and crash the
    run loudly rather than being silently converted into a model
    invocation failure.
    """

    def qa_node(state: QAState) -> QAState:
        current = state.get("exchange")
        if current is None:
            raise ValueError("qa_node requires an exchange in state")

        if current.context is None:
            return {
                "exchange": replace(
                    current,
                    answer="I don't have enough context to answer your question.",
                ),
            }

        prompt = (
            f"Context: {current.context}\n"
            f"Question: {current.question}\n"
            f"Answer the question based on the provided context."
        )

        try:
            answer = invoke_text(model, prompt)
        except _RECOVERABLE_LLM_ERRORS as e:
            return {
                "exchange": replace(
                    current,
                    answer="I couldn't reach the model right now. Please try again.",
                    error_info=ErrorInfo(
                        exception_type=type(e).__name__,
                        exception_message=str(e),
                    ),
                ),
            }

        return {"exchange": replace(current, answer=answer)}

    return qa_node