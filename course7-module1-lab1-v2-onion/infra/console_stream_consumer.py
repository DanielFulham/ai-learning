from typing import Any


class ConsoleStreamConsumer:
    """stdout-backed stream consumer.

    Receives `(node_name, state_delta)` tuples from a LangGraph stream and
    prints a structured line per update. Closes V1 finding #6 — the
    `print()` calls that used to live inside node bodies now live in one
    place subscribed to the stream.

    Format is deliberately terse: `[node_name] state_delta`. Production
    observability (structured logging, tracing) is a separate concrete
    against the same interface.
    """

    def on_update(self, node_name: str, state_delta: dict[str, Any]) -> None:
        print(f"[{node_name}] {state_delta}")