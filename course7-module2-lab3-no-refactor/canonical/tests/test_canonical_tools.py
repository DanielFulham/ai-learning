"""Smoke tests for the canonical ReAct build.

The tool functions themselves are trivial (a Tavily wrapper and a keyword
ladder); the value here is pinning that (1) the module imports cleanly,
(2) the graph compiles, (3) both tools land in the registry with the
expected names. Full-loop tests would require an LLM and are out of scope.
"""
from unittest.mock import patch


def test_module_imports_without_llm_or_network():
    """Importing app.py should not fire OpenAI or Tavily on module load."""
    with patch("langchain_openai.ChatOpenAI") as mock_llm, \
         patch("langchain_tavily.TavilySearch") as mock_search:
        import react_canonical as app  # noqa: F401

    mock_llm.assert_called_once()
    mock_search.assert_called_once()


def test_graph_compiles_and_has_expected_nodes():
    import react_canonical as app

    compiled = app.graph.get_graph()
    node_names = set(compiled.nodes.keys())
    assert "agent" in node_names
    assert "tools" in node_names


def test_tools_registry_has_both_tools():
    import react_canonical as app

    assert "search_tool" in app.tools_by_name
    assert "recommend_clothing" in app.tools_by_name