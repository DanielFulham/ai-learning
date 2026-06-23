import string
from dataclasses import dataclass


@dataclass(frozen=True)
class CounterTick:
    """Immutable counter state at one point in the cycle.

    A CounterTick instance represents a tick that has happened — n >= 1 and
    a real letter. The "no tick yet" state is the absence of a tick key in
    the TypedDict, not a CounterTick with n=0.

    Invariants: n >= 1; letter is a single lowercase ASCII character.
    The lowercase-only constraint pins V1's `random.choice(string.ascii_lowercase)`
    behaviour as a domain contract.
    """

    n: int
    letter: str

    def __post_init__(self) -> None:
        if self.n < 1:
            raise ValueError(f"CounterTick.n must be >= 1, got {self.n}")
        if len(self.letter) != 1 or self.letter not in string.ascii_lowercase:
            raise ValueError(
                f"CounterTick.letter must be a single lowercase ASCII character, "
                f"got {self.letter!r}"
            )