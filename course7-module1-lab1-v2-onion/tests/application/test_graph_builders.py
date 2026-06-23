from unittest.mock import MagicMock

from langchain_core.language_models import BaseChatModel
from langgraph.graph.state import CompiledStateGraph

from application.graph_builders import (
    build_auth_graph,
    build_counter_graph,
    build_qa_graph,
)
from interfaces.input_provider_interface import InputProviderInterface


def test_build_auth_graph_returns_compiled_state_graph() -> None:
    provider = MagicMock(spec=InputProviderInterface)
    graph = build_auth_graph(provider)
    assert isinstance(graph, CompiledStateGraph)


def test_auth_graph_has_expected_nodes() -> None:
    provider = MagicMock(spec=InputProviderInterface)
    graph = build_auth_graph(provider)
    node_names = set(graph.get_graph().nodes.keys())
    assert {"InputNode", "ValidateCredential", "Success", "Failure"}.issubset(node_names)


def test_build_qa_graph_returns_compiled_state_graph() -> None:
    model = MagicMock(spec=BaseChatModel)
    graph = build_qa_graph(model)
    assert isinstance(graph, CompiledStateGraph)


def test_qa_graph_has_expected_nodes() -> None:
    model = MagicMock(spec=BaseChatModel)
    graph = build_qa_graph(model)
    node_names = set(graph.get_graph().nodes.keys())
    assert {"ContextNode", "QANode"}.issubset(node_names)


def test_qa_graph_does_not_have_input_validation_node() -> None:
    """V1's InputValidationNode is gone — `QAExchange.__post_init__` enforces
    the non-empty-question invariant. Pinning the absence so the node
    doesn't get re-added by accident."""
    model = MagicMock(spec=BaseChatModel)
    graph = build_qa_graph(model)
    node_names = set(graph.get_graph().nodes.keys())
    assert "InputNode" not in node_names
    assert "InputValidationNode" not in node_names


def test_build_counter_graph_returns_compiled_state_graph() -> None:
    graph = build_counter_graph()
    assert isinstance(graph, CompiledStateGraph)


def test_counter_graph_has_only_add_node() -> None:
    """V1's PrintOutNode is gone — observation is the streaming consumer's
    job. AddNode is the only domain node; START and END are framework
    bookends."""
    graph = build_counter_graph()
    node_names = set(graph.get_graph().nodes.keys()) - {"__start__", "__end__"}
    assert node_names == {"AddNode"}