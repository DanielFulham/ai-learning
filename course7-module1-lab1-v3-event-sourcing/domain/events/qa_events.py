from dataclasses import dataclass

from domain.error_info import ErrorInfo
from domain.events.base import BaseAgentEvent


@dataclass(frozen=True, kw_only=True)
class QuestionReceived(BaseAgentEvent):
    """Question entered the QA workflow.

    Fires at the entry point of the run. Carries the raw question text so
    projections can correlate downstream events (ContextRetrieved,
    AnswerGenerated) with the input that produced them via aggregate_id.
    """

    question: str


@dataclass(frozen=True, kw_only=True)
class ContextRetrieved(BaseAgentEvent):
    """Context lookup completed.

    `context` is `str | None` — `None` records a keyword-match miss
    explicitly rather than as an absent event. This is the observability-
    consistency lift on `context_provider_node`: the translator always
    sees an explicit delta and always emits the event, including the
    empty-match case.

    The V2 keyword-match hallucination (matching 'guided project' returns
    LangGraph context regardless of subject) is preserved as behaviour
    and recorded as data — ThreadHistoryProjection in V3c joins this
    event against QuestionReceived to answer 'context did not match
    question subject'.
    """

    context: str | None


@dataclass(frozen=True, kw_only=True)
class AnswerGenerated(BaseAgentEvent):
    """Model invocation succeeded; answer was produced.

    Fires only on the success branch of `qa_node`. The failure branch
    fires `ModelInvocationFailed` instead — the two events are mutually
    exclusive per run.
    """

    answer: str


@dataclass(frozen=True, kw_only=True)
class ModelInvocationFailed(BaseAgentEvent):
    """Model invocation raised before producing an answer.

    The user-facing answer (set on the QAExchange) is the clean message;
    `error_info` here carries the diagnostic the production observability
    surface needs. V2's `except Exception: return f"An error occurred: {e}"`
    leaked exception text into the user-facing answer — V3a separates
    diagnostic-to-log from user-safe-message.
    """

    error_info: ErrorInfo


QAEvent = QuestionReceived | ContextRetrieved | AnswerGenerated | ModelInvocationFailed
"""Closed union over the QA service's event types.

Open across services (AuthEvent and CounterEvent are separate unions),
closed within (the QA translator returns `list[QAEvent]` and pyright
enforces exhaustiveness at the translator boundary via assert_never).

Adding a new QA event type is a one-line union extension and a new
dataclass under this module. Removing one is a breaking schema change
that should be a deliberate version bump on the affected event,
not a silent union shrink.
"""