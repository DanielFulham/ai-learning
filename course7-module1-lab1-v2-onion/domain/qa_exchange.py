from dataclasses import dataclass


@dataclass(frozen=True)
class QAExchange:
    """Immutable question-context-answer triple.

    Invariant: the question is non-empty after strip. Whitespace-only
    questions are not a valid input shape — the QA workflow has nothing
    to do with them.
    """

    question: str
    context: str | None = None
    answer: str | None = None

    def __post_init__(self) -> None:
        if self.question.strip() == "":
            raise ValueError("QAExchange requires a non-empty question after strip")