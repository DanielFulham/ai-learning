from unittest.mock import MagicMock

from langchain_core.language_models import BaseChatModel
from langgraph.graph.state import CompiledStateGraph

from application.graph_builders import build_qa_graph


class TestBuildQAGraph:

    def test_returns_compiled_state_graph(self) -> None:
        model = MagicMock(spec=BaseChatModel)
        graph = build_qa_graph(model)
        assert isinstance(graph, CompiledStateGraph)

    def test_graph_has_context_and_qa_nodes(self) -> None:
        """Pinned: the QA graph topology is ContextNode + QANode. The
        translator's node-name dispatch hardcodes these names; if the
        builder ever renames them, the translator silently stops emitting
        events for the renamed nodes. This test makes that contract
        visible."""
        model = MagicMock(spec=BaseChatModel)
        graph = build_qa_graph(model)
        node_names = set(graph.get_graph().nodes.keys())
        assert "ContextNode" in node_names
        assert "QANode" in node_names