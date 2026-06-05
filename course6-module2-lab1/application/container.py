from pathlib import Path
from typing import Optional

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from application.data_wizard_agent import DataWizardAgent
from infra.cached_dataset_loader import CachedDatasetLoader
from infra.langchain_tool_agent import LangChainToolAgent
from infra.local_csv_discovery import LocalCsvDiscovery
from infra.local_csv_loader import LocalCsvLoader
from infra.sklearn_classification_evaluator import SklearnClassificationEvaluator
from infra.sklearn_regression_evaluator import SklearnRegressionEvaluator


def initialise(llm: Optional[BaseChatModel] = None) -> DataWizardAgent:
    """Wire concrete infrastructure to interfaces and return the agent.

    Args:
        llm: Optional chat model override. Production code calls initialise()
             with no args; the default OpenAI model is constructed. Tests pass
             a MagicMock(spec=BaseChatModel) to avoid needing an API key.

    This is the only file allowed to import from both `infra/` and `application/`.
    Swapping an implementation — local CSVs to S3, Random Forest to XGBoost,
    OpenAI to Ollama — is a single line change here.
    """
    data_dir = Path(__file__).parent.parent / "data"

    if llm is None:
        llm = init_chat_model("gpt-4.1-nano", model_provider="openai")

    discovery = LocalCsvDiscovery(data_dir=data_dir)
    loader = CachedDatasetLoader(inner=LocalCsvLoader(data_dir=data_dir))
    classification_evaluator = SklearnClassificationEvaluator()
    regression_evaluator = SklearnRegressionEvaluator()

    tool_agent = LangChainToolAgent(
        llm=llm,
        discovery=discovery,
        loader=loader,
        classification_evaluator=classification_evaluator,
        regression_evaluator=regression_evaluator,
    )

    return DataWizardAgent(agent=tool_agent)