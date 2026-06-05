# Course 6 — Lab 22: DataWizard, AI-Powered Data Analysis with LCEL

Onion-architected tool-calling agent over CSV datasets. An LLM orchestrates five
infrastructure tools through interface contracts to discover, inspect, and
evaluate data. Demonstrates strict-onion layering, dependency injection,
decorator-pattern caching, and a test suite that runs without an API key.

Built as part of the IBM RAG and Agentic AI Professional Certificate — Course 6,
Module 2.

---

## What It Does

The agent takes a natural-language question about CSV datasets and decides which
of five tools to call to answer it. The tools form a small data-science surface:

- `list_csv_files` — discover available datasets in the data directory
- `get_dataset_summaries` — return column names, dtypes, row counts for one or
  more files
- `call_dataframe_method` — run a safe, no-argument pandas method (`head`,
  `tail`, `describe`, `info`, `shape`, `dtypes`, `columns`) against a dataset
- `evaluate_classification_dataset` — train a RandomForestClassifier on an 80/20
  split and return accuracy
- `evaluate_regression_dataset` — train a RandomForestRegressor on an 80/20
  split and return R² and MSE

The LLM chooses which tools to call based on the question, in what order, and
how to interpret the results. The user types "Tell me about the dataset" and the
agent discovers the files, summarises both, and reports back. The user types
"Train both and report the metrics" and the agent runs the appropriate
evaluators on each.

---

## Stack

| Component             | Source Lab                                          | This implementation                                              |
| --------------------- | --------------------------------------------------- | ---------------------------------------------------------------- |
| LLM                   | `gpt-4o-mini` via `init_chat_model`                 | `gpt-4.1-nano` via `init_chat_model`                             |
| Agent construction    | `create_openai_tools_agent` + `AgentExecutor`       | `create_agent` from `langchain.agents` (LangChain 1.x)           |
| Tool decoration       | `@tool` from `langchain_core.tools`                 | Same, plus `args_schema` Pydantic class for the dangerous tool   |
| Data store            | `pd.read_csv` + module-level `DATAFRAME_CACHE` dict | Injected `DatasetLoaderInterface` + `CachedDatasetLoader` decorator |
| Architecture          | Flat script                                         | Strict onion — domain, application, infrastructure, interfaces   |
| Test surface          | None                                                | 55 tests, all passing without an API key                         |
| Entry point           | `while True: input()` chat loop                     | Same CLI shape but behind a dependency-injected agent            |

---

## Setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install langchain langchain-openai langchain-ollama openai pandas numpy scikit-learn python-dotenv pytest
```

Create `.env` next to `datawizard.py`:

```
OPENAI_API_KEY=sk-...
```

Run the agent:

```powershell
python datawizard.py
```

Run the tests (no API key required):

```powershell
python -m pytest tests/ -v
```

---

## File Layout

```
course6-module2-lab1/
├── datawizard.py                          # entry point: singleton + CLI loop
├── domain/
│   ├── __init__.py
│   └── models.py                          # 5 Pydantic response models
├── application/
│   ├── __init__.py
│   ├── data_wizard_agent.py               # thin orchestrator: ask(query) -> str
│   ├── container.py                       # composition root, accepts optional LLM
│   └── interfaces/
│       ├── __init__.py
│       └── data_wizard_agent_interface.py # contract consumed by entry point
├── interfaces/                            # 5 infra-satisfaction contracts
│   ├── __init__.py
│   ├── classification_evaluator_interface.py
│   ├── dataset_discovery_interface.py
│   ├── dataset_loader_interface.py
│   ├── regression_evaluator_interface.py
│   └── tool_calling_agent_interface.py
├── infra/                                 # 6 concrete implementations
│   ├── __init__.py
│   ├── cached_dataset_loader.py           # decorator over DatasetLoaderInterface
│   ├── langchain_tool_agent.py            # owns LLM, 5 tools, agent loop
│   ├── local_csv_discovery.py
│   ├── local_csv_loader.py
│   ├── sklearn_classification_evaluator.py
│   └── sklearn_regression_evaluator.py
├── tests/
│   ├── conftest.py                        # sys.path setup
│   ├── domain/                            # Pydantic field validation
│   ├── application/                       # wizard delegation + container wiring
│   └── infra/                             # 4 infra classes + cache decorator
├── data/
│   ├── classification-dataset.csv         # Wisconsin Breast Cancer
│   └── regression-dataset.csv             # California Housing
├── requirements.txt
└── .env
```

---

## Key Concepts

### Onion Architecture for a Tool-Using Agent

Four layers, dependencies pointing inward:

- **Domain (`domain/models.py`)** — Pydantic models for typed responses that
  cross the LLM boundary. Pydantic rather than frozen dataclass because these
  models cross the LLM boundary and the runtime validation (`Field(ge=0.0,
  le=1.0)` on `accuracy`, `Field(ge=0.0)` on `mean_squared_error`) is
  load-bearing safety, not decoration.
- **Interfaces (`interfaces/`)** — Five Python `Protocol` definitions that
  infrastructure satisfies. Structural typing, no `ABC`, no explicit
  inheritance required.
- **Infrastructure (`infra/`)** — Six concrete classes. Five satisfy interfaces
  directly; one (`CachedDatasetLoader`) is a decorator that satisfies the loader
  interface by wrapping another loader. The `LangChainToolAgent` owns the LLM,
  all five `@tool`-decorated closures, and the LangGraph agent construction.
  This is the single class that knows LangChain exists.
- **Application (`application/`)** — `DataWizardAgent` is the orchestrator.
  Holds a `ToolCallingAgentInterface`, delegates `ask(query)` to it. The named
  layer where future orchestration concerns (logging, retries) would land.
  `container.py` is the composition root — the only file allowed to import from
  both `infra/` and `application/`.

Two interface locations matter. `interfaces/` (root) holds contracts
infrastructure satisfies — multiple concrete classes might compete to satisfy
each one. `application/interfaces/` holds the contract the entry point consumes.
The entry point holds a `DataWizardAgentInterface` and never sees the concrete
`DataWizardAgent`. This means the entry point doesn't change when the
application service is decorated or swapped.

### Discriminated Union Returns as Tool Contracts

Every tool returns `Union[<SuccessModel>, ToolError]`. Each model carries a
`status: Literal["ok"]` or `status: Literal["error"]` discriminator. Consumers
branch on `.status` before reading other fields.

```python
@tool
def evaluate_classification_dataset(
    file_name: str,
    target_column: str,
) -> Union[ClassificationResult, ToolError]:
    try:
        df = loader.load(file_name)
        accuracy = classification_evaluator.evaluate(df, target_column)
        return ClassificationResult(accuracy=accuracy)
    except FileNotFoundError as e:
        return ToolError(message=str(e))
    except ValueError as e:
        return ToolError(message=str(e))
```

The Union pattern makes the failure mode explicit — the caller cannot
accidentally read `.accuracy` on an error response, Python refuses because
`ToolError` has no such field.

The error message is LLM context. `ToolError(message="column not found")` is
useless. `ToolError(message="column 'targt' not found, available: ['target',
'feature_1', ...]")` is recoverable. The LLM literally reads its own error and
self-corrects on the next iteration.

### Pydantic args_schema as Safety Perimeter

The dangerous tool — `call_dataframe_method` — uses an explicit `args_schema`
class with a `Literal` allow-list:

```python
class DataFrameMethodInput(BaseModel):
    file_name: str = Field(description="The CSV filename in the lab data directory.")
    method: Literal["head", "tail", "describe", "info", "shape", "dtypes", "columns"]

@tool(args_schema=DataFrameMethodInput)
def call_dataframe_method(file_name: str, method: str) -> ...:
    ...
```

The `Literal[...]` is sent in the OpenAI function-calling schema as a JSON
Schema `enum`. The model is *told* what values are allowed before it tries.
Pydantic *enforces* the same constraint at the tool boundary — if the LLM
fabricates a `method="to_csv"` call, validation rejects it before the function
body runs. Two layers of defence: model is informed, framework enforces.

The function body still does `getattr(df, method)`. That's safe — not because
the function defends itself, but because the schema rejected unauthorised values
upstream. Take the schema away and the danger returns instantly.

### Decorator Pattern for Cross-Cutting Concerns

`CachedDatasetLoader` is a decorator that satisfies `DatasetLoaderInterface` by
wrapping another loader:

```python
class CachedDatasetLoader(DatasetLoaderInterface):
    def __init__(self, inner: DatasetLoaderInterface) -> None:
        self._inner = inner
        self._cache: dict[str, pd.DataFrame] = {}

    def load(self, name: str) -> pd.DataFrame:
        if name not in self._cache:
            self._cache[name] = self._inner.load(name)
        return self._cache[name]
```

The agent calls `loader.load(name)` and doesn't know how many layers are between
it and the disk. The composition root decides:

```python
# In container.py
loader = CachedDatasetLoader(inner=LocalCsvLoader(data_dir=data_dir))
```

Drop the cache by removing the wrapper. Add it back by adding the wrapper.
Stack more decorators (logging, retry, timeout) and they compose without
touching the agent or the underlying loader. This is the structural answer to
where cross-cutting behaviour lives in onion architecture: each concern is a
decorator that satisfies the underlying interface, and the composition root
decides the stack.

### Dependency Injection for the LLM

The first version of `LangChainToolAgent` called `init_chat_model(...)` inside
its constructor. The tests for `container.initialise()` failed with
`openai.OpenAIError: Missing credentials` because the constructor couldn't be
exercised without an API key. That's a coupling, not a test bug.

The fix is dependency inversion: `LangChainToolAgent` receives a `BaseChatModel`
as a constructor argument. The container builds the LLM and passes it in. The
container's own test-override hook is an optional `llm` parameter:

```python
def initialise(llm: Optional[BaseChatModel] = None) -> DataWizardAgent:
    if llm is None:
        llm = init_chat_model("gpt-4.1-nano", model_provider="openai")
    # ... rest of wiring
```

Production calls `initialise()` with no args, the default LLM is built. Tests
call `initialise(llm=MagicMock(spec=BaseChatModel))`, no API key needed. This is
the testability boundary the onion shape is meant to deliver — the moment a
test can't run without external dependencies, the architecture has a coupling to
surface.

### Tests as Architectural Specification

The test suite reads as the executable specification of what the system
promises. The 55 tests across four layers pin:

- `list_datasets` returns CSV files in a directory
- `list_datasets` ignores non-CSV files
- `list_datasets` returns empty list when directory has no CSVs (not `None`)
- `load` returns DataFrame when file exists, raises `FileNotFoundError` with
  our error format when missing
- Cache hits on second load of same file, returns same DataFrame instance
- Cache exceptions are not cached (failed loads re-attempt on retry)
- `evaluate` returns float between 0 and 1 for classification; raises
  `ValueError` with available columns listed when target column missing
- `RegressionMetrics` is a frozen dataclass — fields cannot be reassigned
- Container wires `CachedDatasetLoader` decorating `LocalCsvLoader`
- Container is stateless — returns new instance each call
- Container accepts optional LLM injection for test isolation

A meaningful subset of these tests pin design decisions, not just behaviour —
empty list vs None, same instance vs copy, frozen dataclass vs mutable, cached
failure vs re-attempt. Each makes a deliberate decision auditable: someone
changing it has to break a test, which forces the conversation about whether
the test or the new code is correct.

---

## Architectural Patterns

The same onion principles apply across two distinct agent topologies:

- **Deterministic pipelines** (e.g. RAG) — orchestration is fixed in code,
  control flow is deterministic
- **Tool-using agents** (this lab) — the LLM decides which tools to call and
  in what order; control flow is decided at runtime by the model

The structural concerns are identical despite the topology difference: domain
at the centre, infrastructure at the edges, composition root as the only swap
surface, tests pinning architectural decisions. The same patterns — dependency
injection, decorator for cross-cutting concerns, Protocol-based interfaces,
two interface locations — apply to both.

---

## Architecture Walkthrough — One Call

When the user types "Tell me about the dataset":

```
datawizard.py
  _agent.ask("Tell me about the dataset")
    → DataWizardAgent.ask()                            [application]
      → ToolCallingAgentInterface.ask()                [interface]
        → LangChainToolAgent.ask()                     [infra]
          → LangGraph agent.invoke(...)
          → LLM emits tool_call: list_csv_files
            → list_csv_files() closure
              → DatasetDiscoveryInterface.list_datasets()  [interface]
              → LocalCsvDiscovery.list_datasets()      [infra]
            → returns DiscoveryResult(...)             [domain]
          → LLM emits tool_call: get_dataset_summaries
            → get_dataset_summaries() closure
              → DatasetLoaderInterface.load() x2       [interface]
              → CachedDatasetLoader.load() x2          [infra decorator]
                → LocalCsvLoader.load() x2 [first time]
                  → pd.read_csv(...)
            → returns DatasetSummariesResult(...)      [domain]
          → LLM emits final message
        → returns final_message.content
    → returns response string
  print(f"\nAgent: {response}")
```

Every cross-layer call goes through a Protocol. Every concrete class is wired
in `container.py`. No global state, no module-level constants for paths, no
shared mutable cache across functions. The cache that exists is
instance-scoped, injectable, and removable in one line of the composition root.

---

## Exercises Completed

| Phase                    | Topic                                         | Notes                                                                                                            |
| ------------------------ | --------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| 1 — Modern agent         | `create_agent` over `AgentExecutor`           | Single line of agent construction, LangGraph loop hidden, message history returned as audit log                  |
| 2 — Chat loop            | While-True CLI surface                        | Per-turn stateless; system prompt encodes default workflow so "Yes" on turn 2 works without conversation memory  |
| 3 — Onion refactor       | Domain / interfaces / infra / app split       | Four interfaces, six infra concretes, two application classes, two interface locations, composition root         |
| 4 — Cache decorator      | `CachedDatasetLoader` wrapping loader         | One-line container change adds caching; consumer doesn't know it's cached                                         |
| 5 — Dependency injection | LLM moved from agent constructor to container | Container tests pass without API key; production behaviour unchanged                                              |
| 6 — Test suite           | 55 tests across all four layers               | No API key, no network, no framework mocking — direct constructor-injected mocks                                  |

---

## Production Insights

**Numpy and Python numeric types don't serialise the same way.**
`np.float64(0.85)` looks identical to `0.85` in a `print()`, but
`json.dumps({"accuracy": np.float64(0.85)})` raises `TypeError`. The moment
your tool's return value crosses a serialisation boundary — HTTP, LangSmith
trace, JSON log, message back to the LLM — numpy types blow up. Cast at the
infra boundary.

**The Pydantic `args_schema` with `Literal` is upstream of execution, not
downstream of failure.** Validation happens before the function body runs. If
the LLM tries `method="to_csv"`, it never reaches `getattr`. The validation
error becomes part of the message stream the LLM sees, so the LLM learns to
retry with a valid method. This is the modern equivalent of the parser loop in
legacy ReAct: same job, native to the tool-calling API.

**Tool call IDs are the correlation primitive for agent observability.** Every
call has an ID, every result echoes it. Production traces (LangSmith, OTel)
use these IDs to build flame graphs. Eval harnesses use them to attribute tool
errors back to specific calls. Single-call paths don't need them; the moment
the model emits two parallel calls in one AIMessage, IDs become load-bearing.

**System prompt vs user query as competing authorities.** The agent balances
system prompt instructions against user-query interpretation. Vague workflow
descriptions are interpreted as suggestions, not contracts. System prompts that
want to override user-query intent need explicit, repeated framing: "Always
perform all steps unless the user explicitly opts out."

**Decorators don't need their own interfaces — they satisfy the interface they
wrap.** That's the whole point of the pattern. If a decorator needs its own
interface, it's a different capability that should be composed alongside the
original, not in front of it. The test: "does the caller need to know it's the
decorated version?" No → decorator, same interface.

---

## What This Doesn't Cover

- **Property-based testing** with `hypothesis` — would generate random inputs
  and assert invariants rather than hand-picked DataFrames.
- **Concurrency** — the cache decorator is documented as not-thread-safe.
  Production would back it with `functools.lru_cache(maxsize=N)` or a
  thread-safe store.
- **LLM integration tests** — no test exercises the round-trip against real
  data, since that would require an API key. Integration testing is manual via
  `datawizard.py` and an eyeball check against reference numbers.
- **Streaming** — `create_agent` returns a graph that supports `.stream()` for
  per-tick events. The CLI uses `.invoke()` for the full loop.
- **Conversation memory across turns** — each `agent.invoke()` is stateless.
  "Yes" on turn 2 works because the system prompt encodes the default workflow,
  not because the agent remembers turn 1.
- **Production deployment** — this is a CLI. A real surface would be Gradio,
  FastAPI, or a Slack bot. Each is a new adapter at the entry-point layer; the
  application and infrastructure layers don't change.

---

## What's Next

The codebase demonstrates onion architecture applied to a tool-using agent.
The folder structure tells the story; the test suite proves the seams; the
composition root makes operational decisions declarative. The same shape
extends naturally to additional capabilities — new tools, new evaluators, new
model providers — through the existing interface contracts.

---

**Completed:** 5 June 2026