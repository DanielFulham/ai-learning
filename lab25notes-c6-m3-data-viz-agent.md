# Course 6 — Lab 25: Data Visualisation Agent — Explicit Capture via Onion Architecture

Onion-architected natural-language data analytics agent. Wraps a pandas
DataFrame in a LangChain agent that answers questions, generates code,
and persists matplotlib figures through a configurable store. Built
from the IBM lab's `create_pandas_dataframe_agent` script as the
starting point, rebuilt against LangChain 1.x canonical (`create_agent`
on LangGraph, explicit tool-calling, OpenAI provider) and re-architected
so figure persistence is an explicit primitive the agent calls, not a
heuristic the tool infers from matplotlib's global state.

Built as part of the IBM RAG and Agentic AI Professional Certificate —
Course 6, Module 3.

---

## What It Does

One tool, one DataFrame, one agent. The LLM writes pandas/matplotlib
code in response to natural-language questions; the tool executes it
against a pre-bound DataFrame and exposes a `save_figure(fig)` primitive
for explicit chart persistence.

Two demos run on top:

- **Seven-query demo** (`demo.py`) — exercises the agent end-to-end
  against the Student Alcohol Consumption dataset (UCI Machine Learning,
  via Kaggle). Counts, filters, bar/pie/box/scatter charts. Mirrors the
  IBM lab's task set so the modernisation comparison is faithful.

The agent's response contains the LLM's natural-language answer; the
trace object exposes every tool call, its code, and its result for
debugging and observability.

---

## Stack

| Component             | Implementation                                                          |
| --------------------- | ----------------------------------------------------------------------- |
| LLM                   | `gpt-4.1-mini` via `init_chat_model`                                    |
| Agent harness         | `langchain.agents.create_agent` (LangGraph runtime)                     |
| Execution primitive   | Custom `python_repl` tool — AST-aware exec/eval, explicit figure save   |
| Figure persistence    | `LocalFigureStore` writes numbered PNGs to `./output/`; resumes from disk |
| LLM provider coupling | `OpenAIChatModelProvider` behind `ChatModelProviderInterface`           |
| Trace surface         | `AgentTrace` domain object with `ToolCallRecord` per tool invocation    |
| Architecture          | Strict onion — domain, interfaces, application, infra                   |
| Test surface          | 51 tests across 6 files, all passing without an API key                 |
| Entry point           | `demo.py` runs `initialise(df=...)` against a CSV                       |

---

## Setup

    python -m venv venv
    .\venv\Scripts\Activate.ps1
    pip install -U langchain langchain-openai langchain-core python-dotenv pandas matplotlib pytest

Create `.env` next to `demo.py`:

    OPENAI_API_KEY=sk-...

Place the dataset at `data/student-mat.csv` (UCI Machine Learning Student
Alcohol Consumption, mathematics course).

Run the demo:

    python demo.py

Run the tests (no API key required):

    pytest -v

---

## File Layout

    data-viz-agent/
    ├── demo.py                                       # entry: seven-query demo
    ├── conftest.py                                   # pytest sys.path setup
    ├── pytest.ini
    ├── requirements.txt
    ├── .env
    │
    ├── application/
    │   ├── __init__.py
    │   ├── data_viz_agent.py                         # DataVizAgent: run, run_with_trace
    │   ├── container.py                              # composition root: initialise()
    │   ├── schema_grounding.py                       # build_system_prompt(df) -> str
    │   └── tools/
    │       ├── __init__.py
    │       └── python_repl.py                        # make_python_repl(df, store)
    │
    ├── domain/
    │   ├── __init__.py
    │   └── agent_trace.py                            # AgentTrace, ToolCallRecord
    │
    ├── interfaces/
    │   ├── __init__.py
    │   ├── chat_model_provider_interface.py          # Protocol — LLM provider
    │   └── figure_store_interface.py                 # Protocol — figure persistence
    │
    ├── infra/
    │   ├── __init__.py
    │   ├── openai_chat_model.py                      # OpenAIChatModelProvider
    │   └── local_figure_store.py                     # LocalFigureStore
    │
    ├── data/
    │   └── student-mat.csv                           # UCI dataset (gitignored)
    │
    ├── output/                                       # generated charts (gitignored)
    │
    └── tests/
        ├── __init__.py
        ├── application/
        │   ├── __init__.py
        │   ├── test_data_viz_agent.py                # LLM and create_agent patched
        │   ├── test_container.py                     # providers mocked
        │   ├── test_schema_grounding.py              # pure function tests
        │   └── tools/
        │       ├── __init__.py
        │       └── test_python_repl.py               # figure_store mocked
        └── infra/
            ├── __init__.py
            ├── test_local_figure_store.py            # tmp_path fixture
            └── test_openai_chat_model.py             # init_chat_model patched

---

## Key Concepts

### From the IBM Lab to LangChain 1.x

The IBM lab uses `create_pandas_dataframe_agent` from
`langchain-experimental` — a 2024-era convenience function that hides
three things behind one call:

1. A `PythonAstREPLTool` with the DataFrame pre-bound
2. A schema-grounded prompt that injects `df.head().to_markdown()` into
   a hidden ReAct prefix
3. The legacy `AgentExecutor` driving a text-parsing ReAct loop

The IBM lab targets LangChain 0.1.x. LangChain 1.x has restructured
agent construction around `create_agent`, the OpenAI tool-calling API,
and structured JSON tool arguments (replacing the ReAct text-parsing
loop that `AgentExecutor` used). This rebuild adopts the 1.x patterns
and unbundles the three responsibilities that `create_pandas_dataframe_agent` combines:

- The Python REPL is a hand-written tool with explicit `df`, `pd`, `plt`,
  and `save_figure` in its closure
- The schema-grounded prompt is a pure function (`build_system_prompt`)
  that returns a string the application controls and the tests pin
- The agent runs on `create_agent` from `langchain.agents`, which uses
  the OpenAI tool-calling API (structured JSON, no regex parsing)

Each piece is independently testable. Each piece is independently
swappable. The convenience function's three responsibilities become
three explicit modules across the onion.

### The Onion Made the Redesign Cheap

Tearing out implicit capture and replacing it with `save_figure` touched
exactly two files: `application/tools/python_repl.py` (the tool body)
and `application/schema_grounding.py` (one new sentence in the prompt).
The `LocalFigureStore`, `FigureStoreInterface`, `DataVizAgent`,
`container.py`, the demo entry point, and every test file outside the
two changed modules — all untouched.

The seams paid out. The figure store interface had nothing to do with
matplotlib internals; it accepted a `Figure` object and returned a
string. When the question changed from "what figure should we save" to
"the agent will tell us what to save," the store's contract was already
correct. The interface absorbed the redesign without changing.

This is the production case for strict onion architecture in agentic
systems: **the bugs in agentic code are not in the LLM, they are in the
seams between the LLM's outputs and the runtime's state**. Every seam
in the onion is a place to either preserve invariants or fail loudly.
When a heuristic over a coupled subsystem turns out to be wrong, you
want exactly one module to change.

### Provider and Store DI Seams

Two injectable dependencies, both following the same pattern. The
container's `initialise()` accepts an optional `chat_model_provider`
and `figure_store`, defaulting to `OpenAIChatModelProvider()` and
`LocalFigureStore()` when not specified. The DataFrame is required —
no sensible default. Tests pass mocks for both and the agent is
constructed without an API key, without a filesystem, and without a
network.

Pattern continuity with Lab 23 (single injectable: LLM provider) and
Lab 24 (single injectable plus a strategy enum). Different domain;
identical seam shape.

---

## Architecture Walkthrough — One Call

When `python demo.py` runs the Dalc/Walc scatter query:

1. `demo.py` calls `initialise(df=df)`
2. The composition root builds the OpenAI chat model, the local figure
   store (scanning `output/` for existing files), the `python_repl` tool
   (closure carries `df`, store, and `save_figure` callable), the
   schema-grounded system prompt, and wraps it all in a `DataVizAgent`
3. `agent.run_with_trace("Generate scatter plots...")` invokes the
   compiled LangGraph
4. First LLM call: HumanMessage + system prompt + tool schemas in →
   AIMessage with a tool call to `python_repl` containing the scatter
   plot code
5. Tool dispatched: `python_repl` parses, exec/evals the code; the
   agent's code calls `save_figure(fig)` which calls `store.save(fig)`,
   writing `output/figure_04.png` and returning the path to the agent's
   code; the tool returns the bare expression value
6. ToolMessage in → second LLM call → AIMessage with a second tool call
   for the second scatter plot
7. Tool dispatched again, same shape, `output/figure_05.png` written
8. ToolMessage in → third LLM call → terminal AIMessage with the
   natural-language summary
9. Agent extracts tool calls by `tool_call_id`, extracts the final
   answer from the last AIMessage, returns an `AgentTrace`

Every transition is explicit. Every boundary is named. The reader can
trace the exact path from natural-language input to natural-language
output, with the tool's two saves and the agent's three LLM calls
visible at obvious instrumentation points.

---

## Production Insights

**LLM behaviour rules belong at the tool boundary, not the prompt.**
The first attempt at preventing `plt.show()` was a sentence in the system
prompt: *"Do not call plt.show()."* GPT-4.1-mini followed it most of the
time and broke it occasionally — the model's training-data prior on
"matplotlib code ends with `plt.show()`" is too strong for a prompt
rule to fully overcome. The actual defence is `matplotlib.use("Agg")`
at module load, which makes `plt.show()` a no-op regardless of what the
LLM emits. Prompts shift probabilities; tool boundaries enforce.

**Heuristics over global state are the wrong shape for agentic tools.**
Implicit figure capture seemed safer than asking the agent to be
explicit — fewer instructions for the LLM to remember. But matplotlib's
pyplot state is a process-global side channel; the tool's correctness
depended on guessing matplotlib's behaviour across pandas-plotting,
`tight_layout`, expression evaluation, and inter-call cleanup.
*Explicit primitives the agent calls* outperform *implicit detection of
agent side effects*, every time. The LLM is a better judge of intent
than any heuristic.

**`tool_call_id` is the correlation primitive in modern agent loops.**
The legacy `AgentExecutor` used positional ordering of intermediate
steps. LangGraph emits messages in an order that's *not* guaranteed
to be positional — `ToolMessage` results arrive interleaved with new
`AIMessage` tool calls, and within a single batch the results can
appear in any order. The agent's trace extractor builds an
`id_to_result` dict first, then walks `AIMessage`s. Same pattern as
Lab 23's manual dispatch, now applied to trace extraction.

**Design decisions:**

- `python_repl` does not call `figure_store.save` unless the agent calls `save_figure` (regression guard against implicit capture)
- `LocalFigureStore` resumes from MAX existing number, not count (protects against gaps after manual deletion)
- Two `make_python_repl` calls produce tools with independent closures

---

## Tests as Architectural Specification

The 51 tests pin both behaviour and design decisions:

**Behaviour:**

- `python_repl` returns expression values, statement stdout, or `(no output)`
- `python_repl` returns errors as strings, never raises
- `save_figure(fig)` calls the figure store with the exact Figure object
- `save_figure(fig)` returns the store's reference string to the agent
- Multiple `save_figure` calls in one tool invocation all hit the store
- Schema-grounded prompts contain the column dtypes, sample rows, and behavioural rules
- `DataVizAgent.run` returns the final answer string
- `DataVizAgent.run_with_trace` returns an `AgentTrace` with paired tool calls
- Tool calls and results are paired by `tool_call_id`, not by position
- Multi-step agent runs (multiple `AIMessage`s with tool calls) all appear in the trace
- `LocalFigureStore` writes PNGs to disk, increments the counter, resumes from disk

---

## What This Doesn't Cover

- **Conversation memory.** Each `agent.run(query)` starts a fresh
  message list. No follow-up queries like "now break that down by
  region." Memory would live in a conversation manager above the agent
  layer; the agent itself stays stateless.
- **Streaming.** The agent uses `.invoke()`. For token-level UX, a
  `DataVizAgent.stream(query)` method would sit alongside `run()` and
  yield events from the LangGraph stream API. Current public surface
  doesn't expose it.
- **Eval harness.** The seven-query demo verifies the loop works by
  inspection. A real eval suite would run golden questions against the
  agent, assert semantic content of the answers ("answer mentions 395",
  "answer contains a correlation coefficient between -0.1 and 0.0"),
  and report pass rates. The architecture exposes the right surface for
  this (`AgentTrace` has every tool call and the final answer), but the
  harness is unwritten.
- **PII redaction at schema grounding.** The schema prompt currently
  includes raw sample rows from the DataFrame. For Supporter 360 and
  similar, supporter names/emails/IDs would need to be redacted before
  the schema hits the LLM context. The right place is a
  `SchemaSanitiserInterface` wrapping `build_system_prompt`; it doesn't
  exist yet.
- **Tool surface for production analytics.** `python_repl` gives the
  agent arbitrary Python execution, including arbitrary `df` mutations
  in principle (the agent could reassign `df`). For a real product, the
  tool surface would be a curated set of safe operations (read-only
  queries, predefined chart templates) — same architectural shape,
  smaller blast radius. The current tool is appropriate for analytics
  exploration; not for direct user-facing deployment.
- **Cost and latency observability.** Token counts are visible in the
  raw LangGraph response but aren't surfaced through `AgentTrace`. The
  trace exposes the right shape for instrumentation; the spans aren't
  wired.
- **Concurrent agent instances against the same DataFrame.** The
  factory pattern guarantees independent closures, but the DataFrame
  itself is mutable; if two agents shared one DataFrame and both ran
  destructive code, results would interleave. No defence against this
  in the current architecture. Production version would either deep-copy
  on initialisation or expose the DataFrame as immutable.

---

## What's Next

The same architectural shape extends naturally:

- **SQL agent (Module 3 Lesson 2)** — replace `df` with a database
  connection, replace `python_repl` with `sql_query`, keep the rest.
  Same composition root, same interface seams, same trace surface.
  The Supporter 360 production target.
- **Multi-DataFrame agent** — `initialise(dfs={"orders": orders_df,
  "customers": customers_df})` exposes multiple DataFrames in the REPL
  namespace. One change to `make_python_repl`'s closure; everything
  else identical.
- **Remote figure store** — `S3FigureStore(bucket=...)` implements
  `FigureStoreInterface`, one-line swap in `container.py`. The agent,
  the tool, the prompt, all untouched.
- **Custom orchestration variant** — if a deployment needs manual
  instrumentation around each tool call (per Lab 24's three-strategy
  pattern), introduce `DataVizAgentInterface` and add a sibling
  implementation. Current single implementation stays as the default;
  the variant is opt-in via the container.

---

**Completed:** 10 June 2026