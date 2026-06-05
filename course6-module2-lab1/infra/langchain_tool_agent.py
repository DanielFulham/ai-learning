from io import StringIO

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool, tool
from typing import Literal, List, Union
from pydantic import BaseModel, Field

from interfaces.classification_evaluator_interface import ClassificationEvaluatorInterface
from interfaces.dataset_discovery_interface import DatasetDiscoveryInterface
from interfaces.dataset_loader_interface import DatasetLoaderInterface
from interfaces.regression_evaluator_interface import RegressionEvaluatorInterface
from interfaces.tool_calling_agent_interface import ToolCallingAgentInterface

from domain.models import (
    ClassificationResult,
    DataFrameMethodResult,
    DatasetSummariesResult,
    DatasetSummary,
    DiscoveryResult,
    RegressionResult,
    ToolError,
)


SYSTEM_PROMPT = (
    "You are a data science assistant. Use the available tools to analyse CSV files "
    "in the lab data directory. For each dataset, determine whether the prediction task "
    "is classification or regression based on the target column's data type and cardinality, "
    "then train and evaluate the appropriate model.\n\n"
    "Workflow:\n"
    "1. Use list_csv_files to discover available datasets.\n"
    "2. Use get_dataset_summaries to inspect columns, dtypes, and row counts.\n"
    "3. Use call_dataframe_method with method='describe' if you need to see value ranges.\n"
    "4. Choose evaluate_classification_dataset for categorical targets (int target with few "
    "unique values), or evaluate_regression_dataset for continuous targets (float target).\n"
    "5. Report the result clearly to the user."
)


class DataFrameMethodInput(BaseModel):
    """Inputs accepted by call_dataframe_method.

    The 'method' field uses a Literal allow-list — the LLM cannot pass arbitrary
    method names, only the ones explicitly enumerated here. This is the load-bearing
    safety perimeter: getattr-on-LLM-strings is treated as untrusted input.
    """
    file_name: str = Field(description="The CSV filename in the lab data directory.")
    method: Literal["head", "tail", "describe", "info", "shape", "dtypes", "columns"] = Field(
        description="The DataFrame method to call. Only these no-argument methods are permitted."
    )


class LangChainToolAgent(ToolCallingAgentInterface):
    """LangChain implementation of the tool-calling agent.

    Receives the LLM via constructor injection — does not construct it. This makes
    the class testable without an API key and consistent with all other infra
    dependencies, which are also injected.
    """

    def __init__(
        self,
        llm: BaseChatModel,
        discovery: DatasetDiscoveryInterface,
        loader: DatasetLoaderInterface,
        classification_evaluator: ClassificationEvaluatorInterface,
        regression_evaluator: RegressionEvaluatorInterface,
    ) -> None:
        self._llm = llm
        self._discovery = discovery
        self._loader = loader
        self._classification_evaluator = classification_evaluator
        self._regression_evaluator = regression_evaluator

        tools = self._build_tools()
        self._agent = create_agent(llm, tools, system_prompt=SYSTEM_PROMPT)

    def ask(self, query: str) -> str:
        result = self._agent.invoke({
            "messages": [{"role": "user", "content": query}]
        })
        final_message = result["messages"][-1]
        content = final_message.content
        return content if isinstance(content, str) else str(content)

    def _build_tools(self) -> list[BaseTool]:
        """Build the @tool functions as closures over the injected interfaces."""

        discovery = self._discovery
        loader = self._loader
        classification_evaluator = self._classification_evaluator
        regression_evaluator = self._regression_evaluator

        @tool
        def list_csv_files() -> Union[DiscoveryResult, ToolError]:
            """List all CSV files in the lab data directory.

            Returns:
                DiscoveryResult with file basenames on success, or ToolError on failure.
            """
            try:
                files = discovery.list_datasets()
                return DiscoveryResult(files=files)
            except Exception as e:
                return ToolError(message=f"Discovery failed: {e}")

        @tool
        def get_dataset_summaries(
            file_names: List[str]
        ) -> Union[DatasetSummariesResult, ToolError]:
            """Analyze multiple CSV files and return metadata summaries for each.

            Args:
                file_names: A list of CSV filenames (basename only) in the lab data directory.

            Returns:
                DatasetSummariesResult on success, ToolError on failure. Check 'status'.
            """
            try:
                summaries = []
                for name in file_names:
                    df = loader.load(name)
                    summaries.append(DatasetSummary(
                        file_name=name,
                        column_names=df.columns.tolist(),
                        data_types={str(k): str(v) for k, v in df.dtypes.items()},
                        row_count=len(df),
                    ))
                return DatasetSummariesResult(summaries=summaries)
            except FileNotFoundError as e:
                return ToolError(message=str(e))
            except Exception as e:
                return ToolError(message=f"Summary failed: {e}")

        @tool(args_schema=DataFrameMethodInput)
        def call_dataframe_method(
            file_name: str,
            method: Literal["head", "tail", "describe", "info", "shape", "dtypes", "columns"],
        ) -> Union[DataFrameMethodResult, ToolError]:
            """Run a safe, no-argument DataFrame method (head, tail, describe, info, shape, dtypes, columns).

            Args:
                file_name: The CSV filename in the lab data directory.
                method: One of: head, tail, describe, info, shape, dtypes, columns.

            Returns:
                DataFrameMethodResult on success, ToolError on failure. Check 'status'.
            """
            try:
                df = loader.load(file_name)
                if method == "info":
                    buf = StringIO()
                    df.info(buf=buf)
                    output = buf.getvalue()
                else:
                    attr = getattr(df, method)
                    result = attr() if callable(attr) else attr
                    output = str(result)
                return DataFrameMethodResult(file_name=file_name, method=method, output=output)
            except FileNotFoundError as e:
                return ToolError(message=str(e))
            except Exception as e:
                return ToolError(message=f"Method '{method}' failed on '{file_name}': {e}")

        @tool
        def evaluate_classification_dataset(
            file_name: str,
            target_column: str,
        ) -> Union[ClassificationResult, ToolError]:
            """Train and evaluate a classifier on the dataset using the specified target column.

            Args:
                file_name: The CSV filename in the lab data directory.
                target_column: The column to use as the classification target.

            Returns:
                ClassificationResult on success, ToolError on failure. Check 'status'.
            """
            try:
                df = loader.load(file_name)
                accuracy = classification_evaluator.evaluate(df, target_column)
                return ClassificationResult(accuracy=accuracy)
            except FileNotFoundError as e:
                return ToolError(message=str(e))
            except ValueError as e:
                return ToolError(message=str(e))
            except Exception as e:
                return ToolError(message=f"Classification evaluation failed: {e}")

        @tool
        def evaluate_regression_dataset(
            file_name: str,
            target_column: str,
        ) -> Union[RegressionResult, ToolError]:
            """Train and evaluate a regression model on the dataset using the specified target column.

            Args:
                file_name: The CSV filename in the lab data directory.
                target_column: The column to use as the regression target.

            Returns:
                RegressionResult on success, ToolError on failure. Check 'status'.
            """
            try:
                df = loader.load(file_name)
                metrics = regression_evaluator.evaluate(df, target_column)
                return RegressionResult(
                    r2_score=metrics.r2,
                    mean_squared_error=metrics.mse,
                )
            except FileNotFoundError as e:
                return ToolError(message=str(e))
            except ValueError as e:
                return ToolError(message=str(e))
            except Exception as e:
                return ToolError(message=f"Regression evaluation failed: {e}")

        return [
            list_csv_files,
            get_dataset_summaries,
            call_dataframe_method,
            evaluate_classification_dataset,
            evaluate_regression_dataset,
        ]