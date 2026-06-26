from unittest.mock import MagicMock

from langchain_core.language_models import BaseChatModel
from langgraph.graph.state import CompiledStateGraph

from application.graph_builders import (
    AUTH_FAILURE_NODE_NAME,
    AUTH_INPUT_NODE_NAME,
    AUTH_SUCCESS_NODE_NAME,
    AUTH_VALIDATE_CREDENTIALS_NODE_NAME,
    QA_CONTEXT_NODE_NAME,
    QA_QA_NODE_NAME,
    build_auth_graph,
    build_qa_graph,
)
from interfaces.input_provider_interface import InputProviderInterface


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


class TestAuthNodeNameConstants:

    def test_auth_input_node_name_value(self) -> None:
        """Pinned: AUTH_ constants are the canonical node-name strings for
        the Auth graph. Rename-safe: a change here breaks the translator
        tests and this test simultaneously."""
        assert AUTH_INPUT_NODE_NAME == "InputNode"

    def test_auth_validate_credentials_node_name_value(self) -> None:
        assert AUTH_VALIDATE_CREDENTIALS_NODE_NAME == "ValidateCredentialsNode"

    def test_auth_success_node_name_value(self) -> None:
        assert AUTH_SUCCESS_NODE_NAME == "SuccessNode"

    def test_auth_failure_node_name_value(self) -> None:
        assert AUTH_FAILURE_NODE_NAME == "FailureNode"


class TestBuildAuthGraph:

    def test_returns_compiled_state_graph(self) -> None:
        provider = MagicMock(spec=InputProviderInterface)
        graph = build_auth_graph(provider)
        assert isinstance(graph, CompiledStateGraph)

    def test_graph_has_all_four_auth_nodes(self) -> None:
        """Pinned: the Auth graph topology includes all four nodes named
        by the exported AUTH_* constants. Rename-safe — uses constants,
        not literals."""
        provider = MagicMock(spec=InputProviderInterface)
        graph = build_auth_graph(provider)
        node_names = set(graph.get_graph().nodes.keys())
        assert AUTH_INPUT_NODE_NAME in node_names
        assert AUTH_VALIDATE_CREDENTIALS_NODE_NAME in node_names
        assert AUTH_SUCCESS_NODE_NAME in node_names
        assert AUTH_FAILURE_NODE_NAME in node_names