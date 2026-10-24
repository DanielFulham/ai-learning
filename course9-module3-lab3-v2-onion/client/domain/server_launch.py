from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ServerLaunch:
    """How to start the MCP server: its script, and the interpreter to run it with."""

    script: Path
    python: Path
