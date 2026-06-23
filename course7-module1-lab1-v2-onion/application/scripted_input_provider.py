from collections.abc import Sequence


class ScriptedInputProvider:
    """Canned-response input provider.

    Constructor takes a sequence of responses; `prompt()` returns them in
    order. Used by `demo.py all` so the auth workflow doesn't block on
    stdin during the integrated demo, and by application-layer tests that
    exercise the auth flow end-to-end without mocking.

    Raises if `prompt()` is called more times than there are scripted
    responses — silent fallback to empty string would hide a test or
    demo-script mismatch.
    """

    def __init__(self, responses: Sequence[str]) -> None:
        self._responses = list(responses)
        self._index = 0

    def prompt(self, message: str) -> str:
        if self._index >= len(self._responses):
            raise IndexError(
                f"ScriptedInputProvider exhausted after {len(self._responses)} responses; "
                f"called again with message {message!r}"
            )
        response = self._responses[self._index]
        self._index += 1
        return response