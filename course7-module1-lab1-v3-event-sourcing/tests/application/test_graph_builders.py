from unittest.mock import MagicMock

from langchain_core.language_models import BaseChatModel
from langgraph.graph.state import CompiledStateGraph

from application.graph_builders import (
    QA_CONTEXT_NODE_NAME,
    QA_QA_NODE_NAME,
    build_qa_graph,
)


class TestQANodeNameConstants:

    def test_qa_context_node_name_value(self) -> None:
        """Pinned: the exported constant is the canonical string. If this
        value ever changes, both the graph builder and the translator
        change together — a rename can no longer be made in one place
        without updating tests."""
        assert QA_CONTEXT_NODE_NAME == "ContextNode"

    def test_qa_qa_node_name_value(self) -> None:
        assert QA_QA_NODE_NAME == "QANode"


class TestBuildQAGraph:

    def test_returns_compiled_state_graph(self) -> None:
        model = MagicMock(spec=BaseChatModel)
        graph = build_qa_graph(model)
        assert isinstance(graph, CompiledStateGraph)

    def test_graph_has_context_and_qa_nodes(self) -> None:
        """Pinned: the QA graph topology is ContextNode + QANode. Uses the
        exported constants so a rename in graph_builders causes this test
        to fail immediately rather than silently passing with stale literals."""
        model = MagicMock(spec=BaseChatModel)
        graph = build_qa_graph(model)
        node_names = set(graph.get_graph().nodes.keys())
        assert QA_CONTEXT_NODE_NAME in node_names
        assert QA_QA_NODE_NAME in node_names