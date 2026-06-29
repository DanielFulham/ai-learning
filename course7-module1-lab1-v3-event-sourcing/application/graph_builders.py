from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from application.auth_nodes import (
    failure_node,
    make_input_node,
    success_node,
    validate_credentials_node,
)
from application.qa_nodes import context_provider_node, make_qa_node
from domain.state_schemas import AuthState, QAState
from interfaces.input_provider_interface import InputProviderInterface


QA_CONTEXT_NODE_NAME = "ContextNode"
QA_QA_NODE_NAME = "QANode"

AUTH_INPUT_NODE_NAME = "InputNode"
AUTH_VALIDATE_CREDENTIALS_NODE_NAME = "ValidateCredentialsNode"
AUTH_SUCCESS_NODE_NAME = "SuccessNode"
AUTH_FAILURE_NODE_NAME = "FailureNode"


def build_qa_graph(model: BaseChatModel) -> CompiledStateGraph:
    """Build the compiled QA graph.

    Linear topology: START → ContextNode → QANode → END. ContextNode
    looks up canned context; QANode invokes the model with that context.
    Both nodes return explicit deltas — the observability consistency
    lift on ContextNode (always emit a ContextRetrieved event, including
    on context=None misses) is the behaviour change from V2.

    `build_auth_graph` landed in V3b alongside the Auth workflow's
    translator. The V3 series is terminal at V3b; the Counter workflow
    was scoped out (see lab note Findings section).
    """
    qa_node = make_qa_node(model)

    builder = StateGraph(QAState)
    builder.add_node(QA_CONTEXT_NODE_NAME, context_provider_node)
    builder.add_node(QA_QA_NODE_NAME, qa_node)

    builder.add_edge(START, QA_CONTEXT_NODE_NAME)
    builder.add_edge(QA_CONTEXT_NODE_NAME, QA_QA_NODE_NAME)
    builder.add_edge(QA_QA_NODE_NAME, END)

    return builder.compile()


def _auth_router(state: AuthState) -> str:
    """Route after credential validation.

    Returns "success" if is_authenticated is True, "failure" otherwise.
    is_authenticated=None (not yet validated) routes to failure — the
    router should only fire after ValidateCredentialsNode has set the
    verdict to a bool.
    """
    credentials = state.get("credentials")
    if credentials is None:
        return "failure"
    return "success" if credentials.is_authenticated is True else "failure"


def build_auth_graph(input_provider: InputProviderInterface) -> CompiledStateGraph:
    """Build the compiled Auth graph.

    Conditional + loop topology:
    START → InputNode → ValidateCredentialsNode → (router) → SuccessNode → END
                              ↑                         ↓
                              └──── FailureNode ←───────┘ (failure loops back)

    The router returns "success" or "failure" based on
    credentials.is_authenticated after ValidateCredentialsNode runs.
    FailureNode clears username/password for a clean retry loop.
    """
    input_node = make_input_node(input_provider)

    builder = StateGraph(AuthState)
    builder.add_node(AUTH_INPUT_NODE_NAME, input_node)
    builder.add_node(AUTH_VALIDATE_CREDENTIALS_NODE_NAME, validate_credentials_node)
    builder.add_node(AUTH_SUCCESS_NODE_NAME, success_node)
    builder.add_node(AUTH_FAILURE_NODE_NAME, failure_node)

    builder.add_edge(START, AUTH_INPUT_NODE_NAME)
    builder.add_edge(AUTH_INPUT_NODE_NAME, AUTH_VALIDATE_CREDENTIALS_NODE_NAME)
    builder.add_conditional_edges(
        AUTH_VALIDATE_CREDENTIALS_NODE_NAME,
        _auth_router,
        {"success": AUTH_SUCCESS_NODE_NAME, "failure": AUTH_FAILURE_NODE_NAME},
    )
    builder.add_edge(AUTH_SUCCESS_NODE_NAME, END)
    builder.add_edge(AUTH_FAILURE_NODE_NAME, AUTH_INPUT_NODE_NAME)

    return builder.compile()