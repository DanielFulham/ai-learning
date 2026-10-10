from typing import Protocol


class DocumentStoreInterface(Protocol):
    """Read-only access to named documents.

    `read` returns the document's text, or None if there is no such document.
    A name that tries to escape the store counts as no such document.
    """

    def read(self, name: str) -> str | None: ...
