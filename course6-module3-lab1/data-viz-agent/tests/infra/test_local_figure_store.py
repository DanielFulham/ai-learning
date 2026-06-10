"""Tests for infra.local_figure_store.

LocalFigureStore writes matplotlib figures to a numbered sequence of PNG
files in a configurable directory. Tests verify the numbering, the resume
behaviour on construction, and the file output.

Uses pytest's tmp_path fixture so each test gets an isolated directory
that's cleaned up automatically. No global state, no leftover files.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless test runs

import matplotlib.pyplot as plt
import pytest
from matplotlib.figure import Figure

from infra.local_figure_store import LocalFigureStore


@pytest.fixture
def a_figure():
    """A minimal matplotlib Figure for save() to write to disk."""
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 4, 9])
    yield fig
    plt.close(fig)


def test_save_writes_a_png_file(tmp_path: Path, a_figure: Figure) -> None:
    store = LocalFigureStore(output_dir=str(tmp_path))
    reference = store.save(a_figure)

    assert Path(reference).exists()
    assert Path(reference).suffix == ".png"
    assert Path(reference).stat().st_size > 0


def test_first_save_uses_number_01(tmp_path: Path, a_figure: Figure) -> None:
    store = LocalFigureStore(output_dir=str(tmp_path))
    reference = store.save(a_figure)
    assert reference.endswith("figure_01.png")


def test_subsequent_saves_increment(tmp_path: Path, a_figure: Figure) -> None:
    store = LocalFigureStore(output_dir=str(tmp_path))
    first = store.save(a_figure)
    second = store.save(a_figure)
    third = store.save(a_figure)

    assert first.endswith("figure_01.png")
    assert second.endswith("figure_02.png")
    assert third.endswith("figure_03.png")


def test_constructor_creates_output_dir_if_missing(tmp_path: Path) -> None:
    new_dir = tmp_path / "does_not_exist_yet"
    assert not new_dir.exists()

    LocalFigureStore(output_dir=str(new_dir))

    assert new_dir.is_dir()


def test_resumes_numbering_from_existing_files(tmp_path: Path, a_figure: Figure) -> None:
    """If figures from a previous process are present, the next save must
    continue from the highest existing number, not clobber figure_01.png."""
    # Simulate a previous run: write three placeholder figure files.
    (tmp_path / "figure_01.png").write_bytes(b"")
    (tmp_path / "figure_02.png").write_bytes(b"")
    (tmp_path / "figure_03.png").write_bytes(b"")

    store = LocalFigureStore(output_dir=str(tmp_path))
    reference = store.save(a_figure)

    assert reference.endswith("figure_04.png")


def test_ignores_non_figure_files_when_resuming(tmp_path: Path, a_figure: Figure) -> None:
    """Files that don't match the figure_NN.png pattern must not interfere
    with counter resumption."""
    (tmp_path / "notes.txt").write_text("not a figure")
    (tmp_path / "image.jpg").write_bytes(b"")
    (tmp_path / "figure_extra.png").write_bytes(b"")  # extra suffix, not a match

    store = LocalFigureStore(output_dir=str(tmp_path))
    reference = store.save(a_figure)

    # No real figure files exist — counter should start fresh.
    assert reference.endswith("figure_01.png")


def test_resumes_from_highest_not_count(tmp_path: Path, a_figure: Figure) -> None:
    """Counter resumes from MAX existing number, not from the count of files.
    This protects against gaps in numbering (e.g. after manual deletion)."""
    (tmp_path / "figure_01.png").write_bytes(b"")
    (tmp_path / "figure_05.png").write_bytes(b"")  # gap from 01 to 05

    store = LocalFigureStore(output_dir=str(tmp_path))
    reference = store.save(a_figure)

    assert reference.endswith("figure_06.png")


def test_default_output_dir(tmp_path: Path, a_figure: Figure, monkeypatch: pytest.MonkeyPatch) -> None:
    """When no output_dir is specified, the store uses 'output' relative to cwd."""
    monkeypatch.chdir(tmp_path)
    store = LocalFigureStore()
    reference = store.save(a_figure)

    assert (tmp_path / "output" / "figure_01.png").exists()
    assert "output" in reference