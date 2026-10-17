import os
import sys
from pathlib import Path

import pytest

from infra.local_file_store import LocalFileStore
from interfaces.file_store_interface import PathOutsideRootError


def _make_store(tmp_path: Path) -> LocalFileStore:
    return LocalFileStore(tmp_path / "root")


def _outside_paths(tmp_path: Path) -> list[str]:
    return ["../outside.txt", "sub/../../outside.txt", str(tmp_path / "outside.txt")]


def test_constructor_does_not_create_the_root_folder(tmp_path: Path) -> None:
    _make_store(tmp_path)
    assert not (tmp_path / "root").exists()


def test_write_then_read_round_trips(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    store.write_text("a.txt", "TEST_CONTENT")
    assert store.read_text("a.txt") == "TEST_CONTENT"


def test_write_creates_the_root_and_parent_folders(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    store.write_text("sub/deeper/a.txt", "TEST_CONTENT")
    assert (tmp_path / "root" / "sub" / "deeper" / "a.txt").is_file()


def test_write_replaces_existing_content(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    store.write_text("a.txt", "one")
    store.write_text("a.txt", "two")
    assert store.read_text("a.txt") == "two"


def test_write_and_read_use_utf8(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    store.write_text("a.txt", "TEST_é✓")
    assert (tmp_path / "root" / "a.txt").read_bytes() == "TEST_é✓".encode()
    assert store.read_text("a.txt") == "TEST_é✓"


def test_exists_is_true_for_a_file_and_false_for_a_missing_path(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    store.write_text("a.txt", "TEST_CONTENT")
    assert store.exists("a.txt") is True
    assert store.exists("nope.txt") is False


def test_exists_is_true_for_a_folder_as_in_the_lab(tmp_path: Path) -> None:
    # The lab checks Path.exists(), which is true for a folder too. Pinned on purpose.
    store = _make_store(tmp_path)
    store.write_text("sub/a.txt", "TEST_CONTENT")
    assert store.exists("sub") is True


def test_read_of_a_missing_file_raises(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    with pytest.raises(FileNotFoundError):
        store.read_text("nope.txt")


@pytest.mark.parametrize("which", range(3))
def test_every_method_refuses_a_path_outside_the_root(tmp_path: Path, which: int) -> None:
    store = _make_store(tmp_path)
    bad = _outside_paths(tmp_path)[which]
    with pytest.raises(PathOutsideRootError):
        store.exists(bad)
    with pytest.raises(PathOutsideRootError):
        store.read_text(bad)
    with pytest.raises(PathOutsideRootError):
        store.write_text(bad, "TEST_CONTENT")


def test_refused_write_leaves_nothing_outside_the_root(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    with pytest.raises(PathOutsideRootError):
        store.write_text("../outside.txt", "TEST_CONTENT")
    assert not (tmp_path / "outside.txt").exists()


def test_refused_read_does_not_return_the_outside_file(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    (tmp_path / "outside.txt").write_text("TEST_SECRET", encoding="utf-8")
    with pytest.raises(PathOutsideRootError):
        store.read_text("../outside.txt")


def test_path_that_goes_up_and_back_inside_is_allowed(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    store.write_text("a.txt", "TEST_CONTENT")
    assert store.read_text("sub/../a.txt") == "TEST_CONTENT"


@pytest.mark.skipif(sys.platform != "win32", reason="backslash is a separator only on Windows")
def test_backslash_traversal_is_refused_on_windows(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    with pytest.raises(PathOutsideRootError):
        store.read_text("..\\outside.txt")


def test_a_link_that_points_outside_the_root_is_refused(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    (tmp_path / "root").mkdir()
    (tmp_path / "outside.txt").write_text("TEST_SECRET", encoding="utf-8")
    try:
        os.symlink(tmp_path / "outside.txt", tmp_path / "root" / "link.txt")
    except OSError:
        pytest.skip("this machine does not allow creating symbolic links")
    with pytest.raises(PathOutsideRootError):
        store.read_text("link.txt")


def test_a_relative_root_is_resolved_when_the_store_is_built(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The check compares against the resolved root, so it must not move with the working directory.
    monkeypatch.chdir(tmp_path)
    store = LocalFileStore(Path("rel_root"))
    other = tmp_path / "elsewhere"
    other.mkdir()
    monkeypatch.chdir(other)
    store.write_text("a.txt", "TEST_CONTENT")
    assert (tmp_path / "rel_root" / "a.txt").read_text(encoding="utf-8") == "TEST_CONTENT"
