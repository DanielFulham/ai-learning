"""Composition root.

initialise() is the single function the demos and any future entry points
call. It wires the LLM provider, the figure store, the tool, the system
prompt, and the agent into a working DataVizAgent.

Production callers pass only the DataFrame; defaults supply the OpenAI
provider and local figure store. Tests pass mock providers and mock stores
so the agent can be constructed without an API key, without a filesystem,
and without a network.

This is the only file in the application that imports from infra. The agent,
the tool, and the prompt builder import only from interfaces and standard
libraries.
"""

import pandas as pd

from application.data_viz_agent import DataVizAgent
from application.schema_grounding import build_system_prompt
from application.tools.python_repl import make_python_repl
from infra.local_figure_store import LocalFigureStore
from infra.openai_chat_model import OpenAIChatModelProvider
from interfaces.chat_model_provider_interface import ChatModelProviderInterface
from interfaces.figure_store_interface import FigureStoreInterface


def initialise(
    df: pd.DataFrame,
    chat_model_provider: ChatModelProviderInterface | None = None,
    figure_store: FigureStoreInterface | None = None,
) -> DataVizAgent:
    """Construct a DataVizAgent ready to answer questions about the given DataFrame.

    The DataFrame is required — there is no sensible default. The LLM provider
    and figure store have sensible production defaults (OpenAI, local filesystem)
    that callers override only when they need to (typically in tests, or when
    swapping storage for a remote backend).
    """
    if chat_model_provider is None:
        chat_model_provider = OpenAIChatModelProvider()
    if figure_store is None:
        figure_store = LocalFigureStore()

    llm = chat_model_provider.create()
    python_repl = make_python_repl(df, figure_store)
    system_prompt = build_system_prompt(df)

    return DataVizAgent(
        llm=llm,
        tools=[python_repl],
        system_prompt=system_prompt,
    )