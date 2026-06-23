from dataclasses import dataclass


@dataclass(frozen=True)
class ErrorInfo:
    """Diagnostic payload for failed model invocations.

    Carries exception type and message for the event log; the user-facing
    answer is set separately on the QAExchange. The translator branches on
    presence of error_info: present → ModelInvocationFailed, absent →
    AnswerGenerated.

    Deliberately minimal for V3a. Traceback and timestamp are explicitly
    held back — the event's own occurred_at field carries timing, and
    traceback expansion is V3b/V3c territory if the event log proves to
    need it.
    """

    exception_type: str
    exception_message: str