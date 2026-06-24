from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from application.qa_nodes import context_provider_node, make_qa_node
from domain.state_schemas import QAState


def build_qa_graph(model: BaseChatModel) -> CompiledStateGraph:
    """Build the compiled QA graph.

    Linear topology: START → ContextNode → QANode → END. ContextNode
    looks up canned context; QANode invokes the model with that context.
    Both nodes return explicit deltas — the V3a observability consistency
    lift on ContextNode (always emit a ContextRetrieved event, including
    on context=None misses) is the only behaviour change from V2.

    V3a includes only the QA graph builder. `build_auth_graph` and
    `build_counter_graph` land in V3b and V3c alongside their workflows'
    translators and projection joins.
    """
    qa_node = make_qa_node(model)

    builder = StateGraph(QAState)
    builder.add_node("ContextNode", context_provider_node)
    builder.add_node("QANode", qa_node)

    builder.add_edge(START, "ContextNode")
    builder.add_edge("ContextNode", "QANode")
    builder.add_edge("QANode", END)

    return builder.compile()