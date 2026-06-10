"""Tests for application.container.

initialise() is the composition root. Tests verify:
- It returns a DataVizAgent
- It accepts an injected LLM provider, calling provider.create() exactly once
- It accepts an injected figure store, wiring it through to the python_repl tool
- It defaults sensibly when no providers are passed
- It requires a DataFrame (no default)

No API calls happen in any test — we always inject mocks so the real
OpenAIChatModelProvider's create() (which would call init_chat_model)
is never invoked.
"""

from unittest.mock import MagicMock

import pandas as pd
import pytest
from langchain_core.language_models import BaseChatModel

from application.container import initialise
from application.data_viz_agent import DataVizAgent
from interfaces.chat_model_provider_interface import ChatModelProviderInterface
from interfaces.figure_store_interface import FigureStoreInterface


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})


@pytest.fixture
def mock_provider() -> MagicMock:
    """A mock LLM provider that returns a mock BaseChatModel."""
    provider = MagicMock(spec=ChatModelProviderInterface)
    provider.create.return_value = MagicMock(spec=BaseChatModel)
    return provider


@pytest.fixture
def mock_figure_store() -> MagicMock:
    return MagicMock(spec=FigureStoreInterface)


# --- Return type and basic construction ---


def test_returns_data_viz_agent(
    sample_df: pd.DataFrame,
    mock_provider: MagicMock,
    mock_figure_store: MagicMock,
) -> None:
    agent = initialise(
        df=sample_df,
        chat_model_provider=mock_provider,
        figure_store=mock_figure_store,
    )
    assert isinstance(agent, DataVizAgent)


def test_each_call_returns_a_fresh_agent(
    sample_df: pd.DataFrame,
    mock_provider: MagicMock,
    mock_figure_store: MagicMock,
) -> None:
    """initialise() is not memoised — each call builds a new agent. This
    keeps the contract stateless."""
    agent_1 = initialise(
        df=sample_df,
        chat_model_provider=mock_provider,
        figure_store=mock_figure_store,
    )
    agent_2 = initialise(
        df=sample_df,
        chat_model_provider=mock_provider,
        figure_store=mock_figure_store,
    )
    assert agent_1 is not agent_2


# --- Provider injection ---


def test_calls_provider_create_exactly_once(
    sample_df: pd.DataFrame,
    mock_provider: MagicMock,
    mock_figure_store: MagicMock,
) -> None:
    """The container calls provider.create() once during construction and
    hands the result to DataVizAgent. Multiple create() calls would be
    wasteful and signal a leak in the wiring."""
    initialise(
        df=sample_df,
        chat_model_provider=mock_provider,
        figure_store=mock_figure_store,
    )
    assert mock_provider.create.call_count == 1


def test_uses_injected_provider_not_default(
    sample_df: pd.DataFrame,
    mock_provider: MagicMock,
    mock_figure_store: MagicMock,
) -> None:
    """When a provider is passed, the default OpenAIChatModelProvider must
    not be instantiated."""
    fake_llm = MagicMock(spec=BaseChatModel)
    mock_provider.create.return_value = fake_llm

    agent = initialise(
        df=sample_df,
        chat_model_provider=mock_provider,
        figure_store=mock_figure_store,
    )
    # The agent's compiled graph was built with our fake_llm; we don't have
    # a clean way to assert that from outside without leaking internals,
    # so we just verify the provider was used.
    mock_provider.create.assert_called_once()


def test_default_provider_used_when_none_passed(
    sample_df: pd.DataFrame,
    mock_figure_store: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When no provider is passed, the container instantiates
    OpenAIChatModelProvider. We patch the class to verify the default path
    runs, without making a real OpenAI call."""
    instantiations: list[object] = []
    fake_llm = MagicMock(spec=BaseChatModel)

    class FakeOpenAIChatModelProvider:
        def __init__(self) -> None:
            instantiations.append(self)

        def create(self) -> BaseChatModel:
            return fake_llm

    monkeypatch.setattr(
        "application.container.OpenAIChatModelProvider",
        FakeOpenAIChatModelProvider,
    )

    initialise(df=sample_df, figure_store=mock_figure_store)

    assert len(instantiations) == 1


# --- Figure store injection ---

def test_uses_injected_figure_store_in_tool(
    sample_df: pd.DataFrame,
    mock_provider: MagicMock,
    mock_figure_store: MagicMock,
) -> None:
    """The figure store passed to initialise() must be wired through to the
    python_repl tool's save_figure closure. We verify this by exercising
    the tool indirectly — if the wrong store were used, mock_figure_store.save
    would never fire."""
    mock_figure_store.save.return_value = "expected/path/figure_42.png"

    agent = initialise(
        df=sample_df,
        chat_model_provider=mock_provider,
        figure_store=mock_figure_store,
    )

    python_repl_tool = _find_tool(agent, "python_repl")
    python_repl_tool.invoke({
        "code": (
            "import matplotlib.pyplot as plt\n"
            "fig, ax = plt.subplots()\n"
            "ax.bar(['a'], [1])\n"
            "save_figure(fig)"
        )
    })

    mock_figure_store.save.assert_called_once()


def test_default_figure_store_used_when_none_passed(
    sample_df: pd.DataFrame,
    mock_provider: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When no figure store is passed, the container instantiates
    LocalFigureStore. We patch the class to verify the default path runs."""
    instantiations: list[object] = []

    class FakeLocalFigureStore:
        def __init__(self) -> None:
            instantiations.append(self)

        def save(self, figure: object) -> str:
            return "fake"

    monkeypatch.setattr(
        "application.container.LocalFigureStore",
        FakeLocalFigureStore,
    )

    initialise(df=sample_df, chat_model_provider=mock_provider)

    assert len(instantiations) == 1


# --- DataFrame requirement ---


def test_df_has_no_default() -> None:
    """The DataFrame parameter is required — no sensible default exists.
    Calling initialise() without a df must raise TypeError."""
    with pytest.raises(TypeError):
        initialise()  # type: ignore[call-arg]


# --- Helpers ---


def _find_tool(agent: DataVizAgent, name: str):
    """Reach into the compiled agent to find a tool by name.

    This is a deliberate test-only seam — production code should not do this.
    The test uses it to verify the container correctly wired a dependency
    through to a tool, which is otherwise impossible to assert from outside.
    """
    # The compiled graph stores its tools internally. We walk the graph nodes
    # to find the ToolNode and extract the bound tool list.
    for node_name, node in agent._agent.nodes.items():
        bound = getattr(node, "bound", None)
        if bound is not None:
            tools = getattr(bound, "tools_by_name", None)
            if tools and name in tools:
                return tools[name]
    raise AssertionError(f"Tool {name!r} not found in compiled agent")