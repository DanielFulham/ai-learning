"""Emit Mermaid diagrams for each compiled graph in the V2 lab.

Run once per commit to keep `docs/graphs/*.mmd` in sync with the code.
The lab note embeds these inline so the diagrams stay live, not snapshot.

Usage (from the lab's root directory):

    python scripts/draw_graphs.py

Produces three files in docs/graphs/:
    auth.mmd
    qa.mmd
    counter.mmd

The script doesn't need a real chat model or input provider — it only
needs compiled graphs to introspect. We wire mocks for the dependencies
the graph topology doesn't depend on, so the script runs offline.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from unittest.mock import MagicMock

from langchain_core.language_models import BaseChatModel

from application.graph_builders import (
    build_auth_graph,
    build_counter_graph,
    build_qa_graph,
)
from interfaces.input_provider_interface import InputProviderInterface


OUTPUT_DIR = Path(__file__).parent.parent / "docs" / "graphs"


def emit_mermaid_files(output_dir: Path | None = None) -> dict[str, Path]:
    """Build the three graphs, emit a Mermaid diagram for each.

    Returns a dict of workflow name -> path of the file written, so
    callers (and tests) can verify what landed where.
    """
    target = output_dir or OUTPUT_DIR
    target.mkdir(parents=True, exist_ok=True)

    auth_graph = build_auth_graph(MagicMock(spec=InputProviderInterface))
    qa_graph = build_qa_graph(MagicMock(spec=BaseChatModel))
    counter_graph = build_counter_graph()

    paths: dict[str, Path] = {}
    for name, graph in [
        ("auth", auth_graph),
        ("qa", qa_graph),
        ("counter", counter_graph),
    ]:
        mermaid = graph.get_graph().draw_mermaid()
        path = target / f"{name}.mmd"
        path.write_text(mermaid, encoding="utf-8")
        paths[name] = path
        print(f"Wrote {path}")

    return paths


if __name__ == "__main__":
    emit_mermaid_files()