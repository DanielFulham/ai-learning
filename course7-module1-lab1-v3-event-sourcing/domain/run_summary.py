from dataclasses import dataclass
from datetime import datetime
from typing import Literal
from uuid import UUID

from domain.error_info import ErrorInfo


RunStatus = Literal["success", "failed", "incomplete"]
"""Outcome of one run.

- success: AnswerGenerated event present, no ModelInvocationFailed
- failed: ModelInvocationFailed event present
- incomplete: neither — run interrupted before the QANode emitted
  (typically because the graph stopped after ContextNode, or the run
  was interrupted entirely)
"""


@dataclass(frozen=True)
class RunSummary:
    """Per-run aggregate projection.

    Derived from the event log via `summarise_run`. Captures the
    high-level outcome of one QA run for observability and reporting.
    The semantic relationship between fields is:

    - status="success" → answer is non-None, error_info is None
    - status="failed"  → error_info is non-None, answer is None
                         (the user-safe message lives on the exchange,
                         not the summary)
    - status="incomplete" → both answer and error_info are None

    Invariants are not enforced by `__post_init__` for V3a — the
    projection function is the only constructor and pyright catches
    the Literal violation at call sites. If a future surface ever
    constructs RunSummary by hand from untyped data, add the runtime
    check.
    """

    run_id: UUID
    started_at: datetime
    completed_at: datetime | None
    question: str | None
    final_status: RunStatus
    answer: str | None
    error_info: ErrorInfo | None
    event_count: int