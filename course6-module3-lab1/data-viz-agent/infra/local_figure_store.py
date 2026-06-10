"""Local filesystem implementation of FigureStoreInterface.

Writes matplotlib figures to a configurable output directory as numbered
PNG files: figure_01.png, figure_02.png, etc. On construction, the store
scans the output directory and resumes numbering from the highest existing
file, so figures persist across process restarts.

Used in development and demos. For production, swap with a cloud-backed
implementation (S3, GCS, blob storage) — the interface is the same; the
agent and tool are untouched.
"""

import os
import re
from pathlib import Path

from matplotlib.figure import Figure


class LocalFigureStore:
    """Writes figures to a local directory as figure_NN.png files."""

    _FILENAME_PATTERN = re.compile(r"figure_(\d+)\.png$")

    def __init__(self, output_dir: str = "output") -> None:
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._counter = self._highest_existing_number()

    def save(self, figure: Figure) -> str:
        """Persist a figure and return its filesystem path as a string."""
        self._counter += 1
        path = self._output_dir / f"figure_{self._counter:02d}.png"
        figure.savefig(path, bbox_inches="tight", dpi=120)
        return str(path)

    def _highest_existing_number(self) -> int:
        """Scan the output directory and return the highest figure number found.

        Used at construction time so numbering resumes after a process restart
        instead of clobbering previous figures.
        """
        if not self._output_dir.is_dir():
            return 0
        numbers = [
            int(match.group(1))
            for name in os.listdir(self._output_dir)
            if (match := self._FILENAME_PATTERN.match(name))
        ]
        return max(numbers, default=0)