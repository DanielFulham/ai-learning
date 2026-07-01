"""Smoke tests for the news-summariser build. The summariser fires the
inner LLM on invocation, so test_news_summarizer_tool mocks it. The
outer agent behaviour is out of scope (would require the full LLM loop).
"""
from unittest.mock import patch, MagicMock


def test_module_imports_without_llm_or_network():
    with patch("langchain_openai.ChatOpenAI") as mock_llm, \
         patch("langchain_tavily.TavilySearch") as mock_search:
        import react_news as app  # noqa: F401

    # Two ChatOpenAI instances: outer model and summariser
    assert mock_llm.call_count == 2
    mock_search.assert_called_once()


def test_graph_compiles_and_has_expected_nodes():
    import react_news as app

    compiled = app.graph.get_graph()
    node_names = set(compiled.nodes.keys())
    assert "agent" in node_names
    assert "tools" in node_names


def test_tools_registry_has_both_tools():
    import react_news as app

    assert "search_tool" in app.tools_by_name
    assert "news_summarizer_tool" in app.tools_by_name


def test_news_summarizer_tool_invokes_inner_model():
    """The summariser should call summariser_model.invoke with a
    SystemMessage + the news content wrapped as a HumanMessage."""
    import react_news as app

    fake_response = MagicMock()
    fake_response.content = "Test summary output"

    with patch.object(app.summariser_model, "invoke", return_value=fake_response) as m:
        result = app.news_summarizer_tool.invoke({"news_content": "some raw news text"})

    assert result == "Test summary output"
    assert m.call_count == 1
    call_messages = m.call_args[0][0]
    assert len(call_messages) == 2
    # First is SystemMessage with the summariser prompt
    assert "summarisation assistant" in call_messages[0].content
    # Second is HumanMessage with the raw news
    assert call_messages[1].content == "some raw news text"