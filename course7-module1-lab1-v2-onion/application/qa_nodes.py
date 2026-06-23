from dataclasses import replace

from langchain_core.language_models import BaseChatModel

from application.llm_text import invoke_text
from domain.qa_exchange import QAExchange
from domain.state_schemas import QAState


def context_provider_node(state: QAState) -> QAState:
    """Look up canned context by keyword match on the question.

    V1's keyword-matching context provider, preserved as-is. The hallucination
    in V1's third demo question (`"What is the best guided project?"`)
    happens because this matcher returns LangGraph context for any question
    containing 'guided project', regardless of the question's actual subject.
    V2 keeps the bug visible — the production move is an eval pipeline over
    golden questions, not a smarter matcher inside the node. That move
    belongs in V3 work or a future lab.
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

    return state


def make_qa_node(model: BaseChatModel):
    """Factory closing over the chat model.

    Returns a node that builds the QA prompt from the exchange, calls
    `invoke_text` for the narrowed-`str` response, and stamps the answer
    on the exchange.
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
        except Exception as e:
            return {
                "exchange": replace(current, answer=f"An error occurred: {e}"),
            }

        return {"exchange": replace(current, answer=answer)}

    return qa_node