# Course 8 - Lab 33: Orchestration & Evaluation Design Patterns with LangGraph

> Code: [`course8-module1-lab2-v1/`](course8-module1-lab2-v1/)

Two agentic design patterns - Orchestrator-Worker (dynamic dispatch via `Send()` with async workers) and Reflection (generator-evaluator loop with two-persona generation and explicit termination reason) - implemented as separate LangGraph modules with a shared LLM factory. Modernised against the notebook's 0.6.6 shape: `with_structured_output(TargetGradeDecision)` over `response.content.lower()` for classifier extraction, async `chef_worker` nodes so `Send()` fan-out delivers concurrent HTTP, `MAX_REFLECTION_ITERATIONS` module constant with `>=` cap check (fixing the notebook's `>` off-by-one), a dedicated `finalize` node that populates `terminated_reason` so cap-hit doesn't silently return "Accepted", and `NotRequired` on `InvestmentState`'s graph-populated fields so reflection's invoke seam accepts partial dicts (`MealPlanState` uses empty defaults for the same purpose since its `str`/`list` types accept them honestly).

Built as part of the IBM RAG and Agentic AI Professional Certificate - Course 8, Module 1 (Lab 33, following Lab 32's workflow patterns). The IBM lab specifies `langgraph==0.6.6`, `langchain-openai==0.3.27`, `pygraphviz==1.14`, `ChatOpenAI(gpt-4o-mini)` with `litellm.ssl_verify=False` scaffolding that's dead code - `litellm` isn't wired to any downstream pipe. This implementation uses `langgraph==1.2.7`, `langchain-anthropic==1.4.8`, `python-dotenv==1.1.1`, `draw_mermaid_png()` via mermaid.ink, Claude Haiku 4.5 as the model, and adds Ruff as the format-on-save + lint layer scoped to the lab folder via `pyproject.toml` + `.vscode/settings.json`.

---

## What It Does

Two independent LangGraph applications, each demonstrating one pattern:

**Orchestrator-Worker (`orchestrator_worker.py`)** - Meal planner that decomposes a raw dish list via `planner_pipe` (`with_structured_output(Dishes)`), dispatches one async `chef_worker` per dish via `Send()`, merges via `Annotated[list[str], operator.add]` reducer, and joins into `final_meal_guide`. Dynamic fan-out where N is determined at runtime by the orchestrator's output.

**Reflection (`reflection.py`)** - Investment advisor with a target-grade classifier, a two-persona generator (Cathie Wood cold-start, Ray Dalio refinement, branched on `state.get("feedback")`), a Warren Buffett structured-output evaluator, a router (`ACCEPTED` on convergence or cap-hit, `REJECTED` on gap), and a `finalize` node that populates `terminated_reason` on the accepted path.

Each pattern's `__main__` invokes the compiled graph via `await app.ainvoke(...)` with a fresh `uuid` per `thread_id`, renders a Mermaid PNG, and prints the result. `reflection.py` also times the run via `time.perf_counter()`.

---

## Stack

| Component | Implementation |
|---|---|
| LLM | `claude-haiku-4-5` via `langchain-anthropic==1.4.8` |
| LLM factory | `get_llm(temperature=DEFAULT_TEMPERATURE)` in `shared.py` - override to `0.0` for classifiers, `0.7` (default) for generators |
| Graph construction | `langgraph==1.2.7` - `StateGraph` per pattern, `add_edge(START/END, ...)` |
| Structured output | `llm.with_structured_output(Schema)` - `Dishes`, `TargetGradeDecision`, `Feedback` |
| State shape | `TypedDict` per pattern; `NotRequired` on `InvestmentState`'s graph-populated fields (`Grade` Literal can't take empty defaults); `MealPlanState` uses empty `str`/`list` defaults |
| Dynamic dispatch | `Send()` API in `dispatch_to_chefs`; `add_conditional_edges` with list not dict for `Send`-shaped routers |
| Concurrent-write reducer | `Annotated[list[str], operator.add]` on `MealPlanState.completed_menu` and `WorkerState.completed_menu` |
| Async I/O | `async def chef_worker` with `await chef_pipe.ainvoke(...)`, `asyncio.run(_run())` at entry |
| Reflection loop cap | `MAX_REFLECTION_ITERATIONS = 5`, `>=` check in `route_investment` |
| Termination distinguisher | `terminated_reason: Literal["converged", "iteration_cap"]` populated by `finalize` node |
| Checkpointer | `InMemorySaver` wired at compile time on both patterns; `thread_id` UUID per invocation |
| Node-name constants | Module-level constants for every node string |
| Visualisation | `get_graph().draw_mermaid_png()` via mermaid.ink |
| Lint / format | `ruff==0.14.5` via `pyproject.toml` (`.vscode/settings.json` for editor integration, gitignored as personal preference) |
| Architecture | Flat modules - no onion port at V1, no Pydantic state at V1 |

---

## Setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

`.env` at the lab root:

```
ANTHROPIC_API_KEY=sk-ant-...
```

Each pattern runs from its module directly:

```powershell
python orchestrator_worker.py
python reflection.py
```

---

## File Layout

```
course8-module1-lab2-v1/
├── shared.py                     # LLM factory, describe_graph, render_graph_png
├── orchestrator_worker.py        # Dish/Dishes, MealPlanState, WorkerState, nodes, graph, __main__
├── reflection.py                 # Grade + TerminatedReason Literals, InvestmentState,
│                                 # TargetGradeDecision + Feedback schemas, nodes, graph, __main__
│
├── requirements.txt
├── pyproject.toml                # Ruff config (line-length 100, py313, E/F/I/W/UP)
├── .vscode/settings.json         # format-on-save + fix-on-save, Ruff as default formatter
├── .env                          # ANTHROPIC_API_KEY
│
├── graph_orchestrator_worker.png # rendered topology, dashed edge on the Send dispatch
└── graph_reflection.png          # rendered topology, dashed edges labelled accepted/rejected
```

---

## Key Concepts

### `Send()` fan-out changes the `add_conditional_edges` third-arg shape

Lab 32's routing pattern used `add_conditional_edges(source, router_fn, {intent: node_name})` - path dict because the router returns a string routing key. §4 orchestrator-worker uses `add_conditional_edges(source, dispatch_fn, [target_node])` - path list because the dispatcher returns `Send` objects, and the list is the allowed-target set for graph introspection. Different arg shape for different dispatcher shape.

### Async workers under `Send()` - the Lab 32 §2 finding, one layer up

`Send` fan-out declares N branches for LangGraph's async scheduler to run in the same super-step. Sync `llm.invoke(...)` inside those branches blocks the transport layer, so structural fan-out serialises on HTTP. `async def chef_worker` with `await chef_pipe.ainvoke(...)` switches transport to `httpx.AsyncClient` and lets the scheduler interleave. Same discipline transfers whether the fan-out is static (§2) or dynamic (§4).

### `terminated_reason` on state - behavioural §7 fail-loudly fix

The notebook's router returns `"Accepted"` on both convergence AND cap-hit. Caller can't distinguish success from silent bailout. A `finalize` node runs on the accepted path, checks whether grades actually match, populates `terminated_reason` with `"converged"` or `"iteration_cap"`. Caller reads the truth. Delivered as one node + one state field + one Literal type - makes the AI_PATTERNS.md §7 "fail loudly on max-iteration hit" observation visible in output.

### Two-persona generator role differentiation

`generate_plan` branches on `state.get("feedback")`: cold-start Cathie Wood, refinement Ray Dalio. Not a runtime router - an in-function branch on state field presence. First iteration has no feedback so Cathie fires; subsequent iterations take Ray. The role split isn't aesthetic; packing "propose something bold" and "respond to critique" into one prompt would degrade both because the instructions pull in different directions.

### Anthropic prompt contract - system+human, not system-only

Notebook uses system-only `ChatPromptTemplate.from_messages([("system", ...)])`. Anthropic's Messages API rejects requests with no `user`/`human` message: `messages: at least one message is required`. OpenAI's Chat Completions API is lax and accepts system-only prompts, so the IBM notebook works there. Every port from IBM's OpenAI-shaped notebook needs system-only prompts split into system (instructions) + human (the request with `{slot}` variables). Cross-provider gotcha, not a code bug.

---

## Findings

**Buffett's critique reads as prescription, not evaluation.** Across three runs of the reflection pattern, Buffett's structured `feedback` field consistently included recommendation-shaped language ("Recommendation for refinement: Anchor 50-60%..."). When Ray Dalio fired at iteration 2 in one run, he opened the refined plan by quoting Buffett's own recommendations back. The loop mechanically converges - grade stabilises at target, `terminated_reason` reports `converged` - but the convergence is echo, not correctness. Structural cousin of Course 7's Reflexion persona-locked responder failure. Direct evidence for AI_PATTERNS.md §7's "cheap evaluator prerequisite" - LLM-as-critic doesn't grade, it prescribes; the next generation pattern-matches to the prescription rather than the original target.

**Buffett's grade of the same Cathie plan swung across runs at temperature 0.0.** First direct-node run graded the initial plan `high risk`; the compiled-graph run graded a structurally similar plan `aggressive` and converged in one iteration. Same evaluator, same target, "deterministic" temperature. Structured output constrained the closed-set field to a valid `Grade` value, but *which* value swung a full level across otherwise identical setups. The "cheap evaluator prerequisite" isn't just about cost - it's about the evaluator being *stable*, and LLM critics fail on stability regardless of temperature setting.

**Latency variance dominates iteration count at N=2 measurements.** Reflection run 1 (single iteration): 107.59s wall-clock. Reflection run 2 (two iterations with Ray Dalio refinement): 47.5s. More iterations, dramatically less time - the opposite of what iteration-count-dominates-cost intuition predicts. Likely a mix of Anthropic cold-start, provider load variance, and prompt-caching on repeated Buffett system prompts. Which factor dominates is unknowable from N=2. Same discipline line as Lab 32: point measurements are directional not comparative; median-of-N is where "we timed a run" stops being defensible.

**Two-persona generator role differentiation is load-bearing when refinement fires.** In the one run where Ray Dalio fired, the refined plan was structurally different from Cathie's cold-start - position size limits, valuation-metric anchors, rebalancing rules, behavioural guardrails. Cathie's plans across three runs read as manifestos; Ray's read as strategies. The failure mode of skipping the split is silent quality degradation on whichever mode dominates the prompt - usually cold-start, because that's what gets prompt-engineered first.

**Mermaid static topology can't render dynamic fan-out.** `graph_orchestrator_worker.png` shows `chef_worker` as a single box even though N workers run at runtime per meal count. LangGraph knows the branch is dynamic (that's why the edge is dashed), but Mermaid has no vocabulary for "this node executes N times where N is runtime-determined." Same limitation would hit CrewAI, MAF's `BuildConcurrent`, ADK's `ParallelAgent` - property of "static diagram of dynamic dispatch," not framework-specific. The topology diagram is the *design*, not the *deployment* - same shape as microservices architecture diagrams drawing "the service" as one box while horizontal scaling means N instances at runtime.

---

## What This Doesn't Cover

- **Onion port.** V1 is flat - schemas, prompts, pipes, nodes, graph, main all in one module per pattern. Deferred to V2 alongside the Pydantic state migration.
- **Pydantic state modelling.** V1 uses `TypedDict` with `NotRequired` on graph-populated fields. Production shape is `BaseModel` with `default_factory` - cleaner invocation seam, no `.get()` at read sites, no ambiguity between "unset" and "empty string." Deferred to V2.
- **Test coverage.** No pytest suite for V1. Nodes could be unit-tested via mocked `BaseChatModel`, graph topology could be pinned via node/edge inspection. Same deferral as Lab 32.
- **Median-of-N timing.** Single wall-clock measurements per pattern. Serious cross-run comparison would run each configuration 5-10 times and report median with variance.
- **LLM-evaluator anti-pattern not fixed.** The reflection pattern demonstrates the §7 "cheap evaluator prerequisite" failure by construction. Production reflection with real stability would use deterministic evaluators (schema validation, unit tests, retrieval-grounded fact-check). Demonstrating the anti-pattern at working code is the finding.
- **Prompt quality.** Persona prompts are theatrical (Cathie Wood / Ray Dalio / Warren Buffett) and outputs are heavy on scaffolding language. Pattern demonstration is the point.

---

**Completed:** 7 July 2026