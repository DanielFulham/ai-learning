from typing import Any


class NullStreamConsumer:
    """No-op stream consumer.

    Satisfies `StreamConsumerInterface` without side effects. The default
    inner consumer for tests that don't care about console output, and
    for production deployments where stdout isn't the right observation
    channel (Lambda, headless services).

    The decorator pattern always wraps a concrete; this is the boring
    concrete the decorator wraps when there's nothing else to wrap.
    """

    def on_update(self, node_name: str, state_delta: dict[str, Any]) -> None:
        pass