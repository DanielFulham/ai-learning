from langchain_core.language_models import BaseChatModel
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

from application.auth_nodes import (
    failure_node,
    make_input_node,
    success_node,
    validate_credentials_node,
)
from application.counter_nodes import add_node
from application.qa_nodes import context_provider_node, make_qa_node
from application.routers import auth_router, counter_stop_router
from domain.state_schemas import AuthState, CounterState, QAState
from interfaces.input_provider_interface import InputProviderInterface


def build_auth_graph(input_provider: InputProviderInterface) -> CompiledStateGraph:
    """Conditional + loop topology.

    InputNode → ValidateCredential → router → (Success → END) or (Failure → InputNode)
    """
    workflow = StateGraph(AuthState)
    workflow.add_node("InputNode", make_input_node(input_provider))
    workflow.add_node("ValidateCredential", validate_credentials_node)
    workflow.add_node("Success", success_node)
    workflow.add_node("Failure", failure_node)

    workflow.add_edge(START, "InputNode")
    workflow.add_edge("InputNode", "ValidateCredential")
    workflow.add_conditional_edges(
        "ValidateCredential",
        auth_router,
        {"success": "Success", "failure": "Failure"},
    )
    workflow.add_edge("Success", END)
    workflow.add_edge("Failure", "InputNode")

    return workflow.compile()


def build_qa_graph(model: BaseChatModel) -> CompiledStateGraph:
    """Linear + LLM topology.

    ContextNode → QANode → END

    V1's InputValidationNode is dropped — `QAExchange.__post_init__`
    enforces the non-empty-question invariant at the domain layer, so
    the graph never sees an invalid exchange.
    """
    workflow = StateGraph(QAState)
    workflow.add_node("ContextNode", context_provider_node)
    workflow.add_node("QANode", make_qa_node(model))

    workflow.add_edge(START, "ContextNode")
    workflow.add_edge("ContextNode", "QANode")
    workflow.add_edge("QANode", END)

    return workflow.compile()


def build_counter_graph() -> CompiledStateGraph:
    """Cyclic + termination topology.

    AddNode → router → (continue → AddNode) or (stop → END)

    V1's PrintOutNode is dropped — observation is the streaming consumer's
    job, not the node's. The router fires after every AddNode invocation.
    """
    workflow = StateGraph(CounterState)
    workflow.add_node("AddNode", add_node)

    workflow.add_edge(START, "AddNode")
    workflow.add_conditional_edges(
        "AddNode",
        counter_stop_router,
        {"continue": "AddNode", "stop": END},
    )

    return workflow.compile()