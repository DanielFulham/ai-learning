# Course 8 — Lab 32: Workflow Patterns with LangGraph

> Code: [`course8-module1-lab1-v1/`](course8-module1-lab1-v1/)

Three canonical workflow patterns — Sequential (Prompt Chaining), Routing (Intent-Based Dispatch), and Parallel (Concurrent Execution with reducer merging) — implemented as separate LangGraph modules with a shared LLM factory, run scripts per pattern, and a cross-provider timing comparison on the parallel pattern (Anthropic vs OpenAI). Modernised against the notebook's 0.6.6 shape: `with_structured_output` over hand-rolled `bind_tools([Pydantic])`, `Annotated` reducers over distinct-keys-per-parallel-writer, `add_edge(START/END, ...)` over the deprecated `set_entry_point`/`set_finish_point`, explicit path_map on `add_conditional_edges` for graph introspection, `InMemorySaver` wired from the start, and async translator nodes (`ainvoke`) for the parallel pattern to actually deliver concurrent HTTP.

Built as part of the IBM RAG and Agentic AI Professional Certificate — Course 8, Module 1. The IBM lab specifies `langgraph==0.6.6`, `langchain-openai==0.3.27`, `pygraphviz==1.14`, and `ChatOpenAI(gpt-4o-mini)` with an insecure `httpx.Client(verify=False)` for the Cloud IDE cert-workaround. This implementation uses `langgraph==1.2.7`, `langchain-anthropic==1.4.8` (default) + `langchain-openai==1.3.3` (cross-provider comparison), `draw_mermaid_png()` via mermaid.ink instead of pygraphviz, Claude Haiku 4.5 as the default model, and drops the `httpx` workaround entirely (unnecessary on a local box, actively harmful because it disables TLS verification).

---

## What It Does

Four independent LangGraph applications, each demonstrating one pattern:

**Sequential (`sequential.py`)** — Prompt chain for a job application: `generate_resume_summary` → `generate_cover_letter`. State is a `ChainState` TypedDict; each node returns a partial dict; LangGraph merges into state. Fixed linear order, no branching. Same shape as any preprocessor → generator pipeline.

**Routing 2-way (`routing.py`)** — Intent classifier for summarise vs translate. `router_node` uses `llm.with_structured_output(RouterDecision)` to return a typed `Literal["summarize", "translate"]` intent; `route_by_intent` reads state and returns the routing key; explicit path_map maps intent to handler. Two branches, exhaustively narrowed.

**Routing 4-way (`routing_exercise.py`)** — Same pattern scaled to four service categories: ride_hailing, restaurant_order, groceries, default_handler. Proves the routing pattern extends from N=2 to N=N without structural change. Default handler is the graceful degradation surface.

**Parallel (`parallel.py`)** — Three concurrent translators (French, Spanish, Japanese) fan out from `START`, reduce into a single `translations` list via `Annotated[list[Translation], operator.add]`, aggregator formats the combined output. Translator nodes are `async def` using `llm.ainvoke(...)` — structural parallelism is necessary but not sufficient; the LLM calls must be async at the transport layer or three concurrent nodes still serialise on the network.

Each pattern has its own run script (`run_sequential.py`, `run_routing.py`, `run_routing_exercise.py`, `run_parallel.py`) that prints topology, renders a Mermaid PNG, invokes the graph, and prints the result. `run_parallel.py` additionally spawns a subprocess per provider (`LLM_PROVIDER=anthropic|openai`) for the cross-provider timing comparison.

---

## Stack

| Component | Implementation |
|---|---|
| LLM (default) | `claude-haiku-4-5` via `langchain-anthropic==1.4.8` |
| LLM (comparison) | `gpt-4o-mini` via `langchain-openai==1.3.3` |
| LLM provider selection | `LLM_PROVIDER` env var at import time; factory in `shared.py` |
| Graph construction | `langgraph==1.2.7` — `StateGraph` per pattern, `add_edge(START/END, ...)` over `set_entry_point/set_finish_point` |
| Structured output | `llm.with_structured_output(RouterDecision)` — Pydantic `BaseModel` with `Literal[...]` intent field |
| State shape | `TypedDict` per pattern (`ChainState`, `RouterState`, `ExerciseRouterState`, `ParallelState`) |
| Concurrent-write reducer | `Annotated[list[Translation], operator.add]` on `ParallelState.translations` |
| Async I/O (parallel pattern) | `async def` node functions with `await llm.ainvoke(...)`, `asyncio.run(app.ainvoke(...))` at entry |
| Checkpointer | `InMemorySaver` wired at compile time on every pattern |
| Node-name constants | Module-level constants for every node string, referenced in `add_node`, `add_edge`, and path_map |
| Visualisation | `get_graph().draw_mermaid_png()` via mermaid.ink — no `pygraphviz` |
| Architecture | Flat modules — no onion port at V1 |

---

## Setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

`.env` at the lab root — both keys needed even if only running Anthropic scripts, because `run_parallel.py`'s subprocess-per-provider mode reads both at compile time via the `LLM_PROVIDER` env var:

```
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
```

Each pattern runs from its own script:

```powershell
python run_sequential.py
python run_routing.py
python run_routing_exercise.py
python run_parallel.py           # spawns one subprocess per provider
```

---

## File Layout

```
course8-module1-lab1-v1/
├── shared.py                     # LLM factory (Anthropic default, OpenAI optional), graph helpers, content extractor
├── sequential.py                 # ChainState, generate_resume_summary, generate_cover_letter, sequential graph
├── routing.py                    # RouterState, RouterDecision (2-way), router_node, route_by_intent, handlers, routing graph
├── routing_exercise.py           # ExerciseRouterState, RouterDecision (4-way), same shape scaled
├── parallel.py                   # ParallelState (with reducer), async translators, aggregator, parallel graph
│
├── run_sequential.py             # entry: sequential pattern demo
├── run_routing.py                # entry: routing pattern demo (2-way)
├── run_routing_exercise.py       # entry: routing pattern demo (4-way)
├── run_parallel.py               # entry: parallel pattern demo, subprocess per provider
│
├── requirements.txt
├── .env                          # ANTHROPIC_API_KEY, OPENAI_API_KEY
│
├── graph_sequential.png          # rendered by describe_graph, per-pattern
├── graph_routing.png
├── graph_routing_exercise.png
├── graph_parallel_anthropic.png
└── graph_parallel_openai.png
```

---

## Key Concepts

### `with_structured_output` supersedes `bind_tools([Pydantic])` for schema extraction

The notebook binds a Pydantic `Router` model as a "tool," invokes the LLM, then unwraps `response.tool_calls[0]['args']['role']` with a defensive `if response.tool_calls: else fallback` branch. `bind_tools` is designed for tool-calling — where the LLM chooses which tool to invoke and supplies args. Using it for pure schema extraction means every consumer pays the tool-calling ceremony (existence check, index into first call, key lookup by field name).

`with_structured_output(RouterDecision)` returns a runnable whose `.invoke()` yields a typed Pydantic instance directly. Access via `.intent`, no `.tool_calls` archaeology, and pyright sees the return as `RouterDecision | dict[str, Any]` — the dict fallback path warrants `isinstance` narrowing at the seam, but that's honest documentation of the union rather than three levels of runtime unwrap. The field description on `Literal[...] = Field(..., description="...")` carries the routing prompt at schema-level; no separate prose scaffolding needed.

### `Annotated` reducer over distinct keys is the point of the parallel pattern

The notebook's parallel example gives each translator its own state key (`french`, `spanish`, `japanese`), avoiding the concurrent-write question by making it structurally impossible for two nodes to collide. Works, but sidesteps the pattern LangGraph reducers exist to solve — multiple parallel nodes writing to the *same* logical output. The modern shape uses one `translations` field with `Annotated[list[Translation], operator.add]`: each translator returns a single-element list, and LangGraph's reducer concatenates them into the combined list. Delete the reducer and concurrent writes silently drop depending on execution order.

### Structural parallelism is necessary but not sufficient

LangGraph's fan-out from `START` marks three translator nodes as ready to run at the same super-step. But if the node code calls sync `llm.invoke(...)`, the underlying `httpx.Client` blocks the thread until the response returns; other translators can't proceed until the sync call unblocks. Measured wall-clock stayed close to `3 × single_call_time` — pattern declared parallel, transport layer serialising.

Async translators (`async def` + `await llm.ainvoke(...)`) use `httpx.AsyncClient` under the hood, don't block the event loop, and LangGraph's async scheduler interleaves the three HTTP calls genuinely. Parallel wall-clock drops toward `max(single_call_time)` — the pattern delivers what its structure promises.

### Explicit path_map on `add_conditional_edges` is for introspection, not runtime

`add_conditional_edges(source, router_fn)` without a path_map runs correctly — LangGraph uses the router function's return value directly as the next node name. But `workflow.get_graph()` performs static analysis and can't resolve an arbitrary Python function's possible return values without the map. The rendered Mermaid diagram would show `router → END` only, with the handler nodes floating disconnected. Supplying `{intent: node_name, ...}` fixes the diagram at zero runtime cost. Shortcuts that work at execution can still break introspection.

### Node-name constants are rename-safety, not stylistic

Every node string (`"router_node"`, `"summarize"`, `"translate_french"`) appears 2-3 times per pattern module: `add_node`, `add_edge`, path_map entries. Duplicated string literals mean renaming a node requires finding and updating every reference — one miss produces a silent mis-dispatch at compile time (LangGraph raises on unknown node names, but only after graph construction). Module-level constants (`ROUTER_NODE = "router_node"`) collapse the references to a single source; renames land as one edit.

---

## Findings

**Cross-provider parallel timing shows the pattern's cost/latency tradeoff clearly, but variance is the story.** Across five runs of the parallel pattern (three languages, Claude Haiku 4.5 and gpt-4o-mini in subprocess per provider), single-call latency ranged from 1.38s to 3.67s and parallel wall-clock from 1.38s to 2.84s. Speedup (single ÷ parallel) ranged from 0.6× to 2.4×. Point measurements swing by ~40% run to run — network jitter, provider load, endpoint variance. Single measurements are enough to prove structural parallelism works (parallel wall-clock consistently below `3 × single`); they're not enough to compare providers. Median of 5-10 runs per configuration is what would survive interrogation. Real evidence for the "always benchmark, don't reason about" discipline any production LLM system needs.

**Async execution compresses per-provider latency differences.** On the same run where Anthropic's single-call was 44% faster than OpenAI's (2.55s vs 3.67s), the two parallel wall-clocks landed within 10% of each other (1.41s vs 1.54s). Parallel execution is bounded by the slowest branch, not the sum — so cross-provider differences that dominate serial workloads flatten under parallel structure. A production observation: if the workload is latency-sensitive and structurally parallel, provider choice matters less than in serial cases where every call sums.

**The default handler prompt is under-specified, and Haiku produces non-deterministic output at temperature 0.** Two of four `run_routing_exercise.py` runs produced the correct redirect message; two produced misreads where Haiku interpreted the prompt as *"here is a customer service response for you to critique"* and returned a meta-analysis of the redirect it was supposed to *be*. Same model, same prompt, same temperature (implicitly 0.0 via `langchain-anthropic` defaults), non-deterministic classification result. Prompt-engineering ceiling: `default_handler_node`'s prompt is ambiguous about *who is speaking*. Fix is prompt-level, not pattern-level — but worth naming that the pattern being wired correctly doesn't insulate against under-specified prompts.

**Static intent maps and LLM routing solve different problems.** The 4-way router correctly classified `"I need milk, bread, eggs, and vegetables for the week"` as `groceries` without any of those literal words appearing in the schema's field description — the model matched semantically, not lexically. A static keyword map would need every literal maintained explicitly. LLM routing wins on natural-language coverage; it loses on determinism, cost, and debuggability. The decision between them isn't stylistic — it's a golden-set eval question. Where the failure mode is *intent classification*, LLM routing is a legitimate lever; where the failure mode is *retrieval or generation*, swapping the router changes nothing. The observation matters wherever a production system uses static intent classification and considers a fuzzier upgrade.

**Schema field descriptions carry load-bearing prompt engineering invisibly.** The `Literal[...] = Field(..., description="Return 'ride_hailing' if the user wants a ride, taxi, or transport. Return 'restaurant_order' if...")` block on `RouterDecision.intent` is what teaches the LLM what each option *means*. Delete the description and the model has closed-set values with no semantic guidance — classification quality collapses. The Literal constrains; the description informs. Both are load-bearing, and it's not obvious which is doing which work until you delete one.

**LangGraph node signatures are `dict → dict` at the runtime API surface — every seam where a stronger contract meets that permissive substrate needs explicit narrowing.** `router_llm.invoke()` returns `RouterDecision | dict[str, Any]` under `with_structured_output`. `response.content` on `BaseMessage` is `str | list[str | dict]`. `state["messages"][-1]` types as `BaseMessage` where topology guarantees `AIMessage`. Each seam needs `isinstance` narrowing at runtime — one `TypeError` raised if the invariant breaks, one clean pyright-narrowed value otherwise, zero `cast()` or `# type: ignore` needed. The pattern is consistent across every LangGraph project touched to date, and it's not framework-specific: it's the general shape of "typed contract meets permissive substrate."

---

## What This Doesn't Cover

- **Onion port.** V1 is flat — one module per pattern, shared LLM factory, no interfaces / DI / composition root. The architectural work is banked as a follow-up if the pattern coverage ever needs to demonstrate orchestration-as-implementation-detail rather than orchestration-as-code.
- **Test coverage.** No pytest suite for V1. Nodes could be unit-tested via mocked `BaseChatModel`, graph topology could be pinned via node/edge inspection, reducer behaviour could be pinned via `get_type_hints(..., include_extras=True)` on the state class. Deferred to a separate pass; adds time without changing the demonstration.
- **Median-of-N timing.** Single point measurements per configuration. Timing evidence is directional (async ≪ sync at scale, parallel wall-clock < sum) but not comparison-of-record. A serious cross-provider comparison would run each configuration 5-10 times and report median with variance.
- **Prompt quality.** LLM outputs across all patterns are generic and heavy on scaffolding ("results-driven," "actionable insights," redundant table structures). Prompt-engineering is out of scope for a pattern-demonstration lab; production versions of any of these patterns would need substantially better prompts, and the parallel translator prompts specifically would benefit from constraint enforcement (single translation, no alternatives, no notes) rather than the current permissive shape.
- **`Send` API for dynamic fan-out.** The parallel pattern fans out to a static three languages via three fixed edges from `START`. If N were determined at runtime (variable-length list of languages from state), the modern shape would be `Send` from a dispatcher node. Not applicable here; noted as the next lever when the shape appears.
- **Reflection or cyclic patterns.** All three patterns are DAGs. LangGraph supports cycles (reflection loops, ReAct-style iteration) but none of these three exercise that capability. Module 2's Reflection/Reflexion/ReAct labs cover cyclic patterns; this lab is deliberately linear.

---

**Completed:** 6 July 2026