from collections.abc import Sequence
from contextlib import AbstractAsyncContextManager
from typing import Protocol

from langchain_core.tools import BaseTool


class ToolSourceInterface(Protocol):
    """Where an agent's tools come from.

    `open()` is an async context manager because MCP tools hold a live
    connection to their server: the tools are only usable while the block is
    open, so the agent must run inside it.
    """

    def open(self) -> AbstractAsyncContextManager[Sequence[BaseTool]]: ...
