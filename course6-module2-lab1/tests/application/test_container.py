"""Tests for container.initialise() — the composition root.

These tests verify the wiring is correct: the returned agent has the right
concrete types injected at each layer. A mock LLM is passed in, so no API
key is required and no real OpenAI client is constructed.
"""
from typing import cast
from unittest.mock import MagicMock

from langchain_core.language_models import BaseChatModel

from application.container import initialise
from application.data_wizard_agent import DataWizardAgent
from infra.cached_dataset_loader import CachedDatasetLoader
from infra.langchain_tool_agent import LangChainToolAgent
from infra.local_csv_discovery import LocalCsvDiscovery
from infra.local_csv_loader import LocalCsvLoader
from infra.sklearn_classification_evaluator import SklearnClassificationEvaluator
from infra.sklearn_regression_evaluator import SklearnRegressionEvaluator


def _mock_llm() -> MagicMock:
    """Build a MagicMock satisfying BaseChatModel."""
    return MagicMock(spec=BaseChatModel)


def test_initialise_returns_data_wizard_agent():
    result = initialise(llm=_mock_llm())

    assert isinstance(result, DataWizardAgent)


def test_initialise_wires_langchain_tool_agent_as_inner_agent():
    result = initialise(llm=_mock_llm())

    inner = cast(LangChainToolAgent, result._agent)
    assert isinstance(inner, LangChainToolAgent)


def test_initialise_wires_cached_loader_decorating_local_csv_loader():
    result = initialise(llm=_mock_llm())
    inner_agent = cast(LangChainToolAgent, result._agent)

    cache = cast(CachedDatasetLoader, inner_agent._loader)
    assert isinstance(cache, CachedDatasetLoader)

    underlying = cast(LocalCsvLoader, cache._inner)
    assert isinstance(underlying, LocalCsvLoader)


def test_initialise_wires_local_csv_discovery():
    result = initialise(llm=_mock_llm())
    inner_agent = cast(LangChainToolAgent, result._agent)

    assert isinstance(inner_agent._discovery, LocalCsvDiscovery)


def test_initialise_wires_sklearn_classification_evaluator():
    result = initialise(llm=_mock_llm())
    inner_agent = cast(LangChainToolAgent, result._agent)

    assert isinstance(inner_agent._classification_evaluator, SklearnClassificationEvaluator)


def test_initialise_wires_sklearn_regression_evaluator():
    result = initialise(llm=_mock_llm())
    inner_agent = cast(LangChainToolAgent, result._agent)

    assert isinstance(inner_agent._regression_evaluator, SklearnRegressionEvaluator)


def test_initialise_is_stateless_returns_new_instance_each_call():
    first = initialise(llm=_mock_llm())
    second = initialise(llm=_mock_llm())

    assert first is not second


def test_initialise_passes_injected_llm_to_inner_agent():
    """When an LLM is provided, it gets passed through to the tool agent."""
    mock_llm = _mock_llm()
    result = initialise(llm=mock_llm)
    inner_agent = cast(LangChainToolAgent, result._agent)

    assert inner_agent._llm is mock_llm