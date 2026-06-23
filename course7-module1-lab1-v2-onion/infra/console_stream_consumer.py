from dataclasses import is_dataclass, replace
from typing import Any


def _redact(value: Any) -> Any:
    """Walk the value tree and redact any dataclass field literally named
    `password`. Tactical V2 fix to close the visible-credential surface in
    demo output. V3 replaces this with a domain-declared sensitive-field
    policy consulted by a separate RedactingStreamConsumer concrete —
    see the V2 lab note's deferred-hardening section.
    """
    if isinstance(value, dict):
        return {k: _redact(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact(v) for v in value]
    if is_dataclass(value) and not isinstance(value, type) and hasattr(value, "password"):
        try:
            return replace(value, password="***")
        except TypeError:
            return value
    return value

class ConsoleStreamConsumer:
    """stdout-backed stream consumer.

    Receives `(node_name, state_delta)` tuples from a LangGraph stream and
    prints a structured line per update. Closes V1 finding #6 — the
    `print()` calls that used to live inside node bodies now live in one
    place subscribed to the stream.

    Format is deliberately terse: `[node_name] state_delta`. Production
    observability (structured logging, tracing) is a separate concrete
    against the same interface. Field-level redaction here is tactical;
    proper sensitive-field policy lands in V3.
    """

    def on_update(self, node_name: str, state_delta: dict[str, Any]) -> None:
        print(f"[{node_name}] {_redact(state_delta)}")