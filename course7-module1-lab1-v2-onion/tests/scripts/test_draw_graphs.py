from pathlib import Path

from scripts.draw_graphs import emit_mermaid_files


def test_emits_three_files_into_target_directory(tmp_path: Path) -> None:
    paths = emit_mermaid_files(output_dir=tmp_path)

    assert set(paths.keys()) == {"auth", "qa", "counter"}
    for name in ("auth", "qa", "counter"):
        assert paths[name].exists()
        assert paths[name].name == f"{name}.mmd"


def test_emitted_files_contain_mermaid_syntax(tmp_path: Path) -> None:
    """Mermaid output starts with a graph declaration. Pinning the format
    so a future LangGraph version that changes the diagram format surfaces
    as a test failure, not a silent corruption of the lab note."""
    paths = emit_mermaid_files(output_dir=tmp_path)

    for path in paths.values():
        content = path.read_text(encoding="utf-8")
        # LangGraph emits "graph TD" or "flowchart TD" — both are valid Mermaid.
        assert "graph" in content.lower() or "flowchart" in content.lower()


def test_emits_default_output_directory_when_none_passed(monkeypatch, tmp_path: Path) -> None:
    """When called with no args, the script writes to the configured
    docs/graphs/ directory. Monkeypatched so the test doesn't pollute
    the real one."""
    import scripts.draw_graphs as draw_graphs

    monkeypatch.setattr(draw_graphs, "OUTPUT_DIR", tmp_path / "docs" / "graphs")
    paths = emit_mermaid_files()

    for path in paths.values():
        assert (tmp_path / "docs" / "graphs") in path.parents