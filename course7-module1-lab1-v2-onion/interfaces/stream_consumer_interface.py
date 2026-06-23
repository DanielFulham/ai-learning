from typing import Any, Protocol


class StreamConsumerInterface(Protocol):
    """Consumer seam for LangGraph stream updates.

    Concretes receive `(node_name, state_delta)` tuples emitted by
    `graph.stream(stream_mode="updates")` and decide what to do with them
    — print to the console (`ConsoleStreamConsumer`), drop them
    (`SilentStreamConsumer` for tests), or write events to a store
    (V3's event-sourced consumer).

    Closes V1 finding #6 — moving `print()` calls out of node bodies and
    onto a subscriber pattern. Same primitive that drives console output
    in V2 drives the event store in V3.
    """

    def on_update(self, node_name: str, state_delta: dict[str, Any]) -> None: ...