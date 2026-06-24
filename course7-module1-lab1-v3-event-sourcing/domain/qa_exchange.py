from dataclasses import dataclass

from domain.error_info import ErrorInfo


@dataclass(frozen=True)
class QAExchange:
    """One QA interaction.

    V2 invariant preserved: question must be non-empty after strip.

    V3a adds `error_info: ErrorInfo | None`. Translator branches on its
    presence to choose between `AnswerGenerated` and `ModelInvocationFailed`.
    Domain layer does not enforce mutual exclusion between `answer` and
    `error_info` — `qa_node` sets a user-safe `answer` AND populates
    `error_info` on the failure path, so both are present together in
    the failure case. Translator handles the dispatch.
    """

    question: str
    context: str | None = None
    answer: str | None = None
    error_info: ErrorInfo | None = None

    def __post_init__(self) -> None:
        if not self.question.strip():
            raise ValueError(
                "QAExchange.question must be non-empty after strip()"
            )