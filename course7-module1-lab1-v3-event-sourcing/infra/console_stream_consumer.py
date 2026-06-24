from dataclasses import is_dataclass, replace
from typing import Any


def _redact(value: Any) -> Any:
    """Walk the value tree and redact any dataclass field literally named
    `password`. Tactical V2 fix to close the visible-credential surface in
    demo output. V3a inherits unchanged — the dev console is still the
    dev console; sensitive-field policy at the event-store boundary lives
    in the per-service translator (V3b's auth translator drops password
    from LoginAttempted payload).
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
    prints a structured line per update. Format is deliberately terse:
    `[node_name] state_delta`. The dev observation channel; not the
    canonical record (the event store is).
    """

    def on_update(self, node_name: str, state_delta: dict[str, Any]) -> None:
        print(f"[{node_name}] {_redact(state_delta)}")