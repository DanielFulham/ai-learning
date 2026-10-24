from dataclasses import dataclass


@dataclass(frozen=True)
class Completed:
    """The tool ran and returned this text."""

    text: str


@dataclass(frozen=True)
class Failed:
    """The tool was allowed to run and raised an error."""

    reason: str


GateResult = Completed | Failed
