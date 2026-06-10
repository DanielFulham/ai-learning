"""Protocol for figure persistence.

The Python REPL tool produces matplotlib figures as a side effect of executing
agent-generated code. This interface defines how those figures get persisted
without coupling the tool to any specific storage backend.

Local development uses LocalFigureStore (writes to ./output). Production
deployments would swap in S3FigureStore, BlobStore, or similar — without
touching the tool, the agent, or the prompt.
"""

from typing import Protocol

from matplotlib.figure import Figure


class FigureStoreInterface(Protocol):
    """A persistence target for matplotlib figures."""

    def save(self, figure: Figure) -> str:
        """Persist the figure and return a stable reference to it.

        The returned string is included verbatim in the tool's response to the
        LLM, so it should be human-readable and meaningful — a file path for
        local stores, a URL for remote stores. The agent will mention it in
        its final answer to the user.
        """
        ...