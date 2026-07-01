# Course 7 — Lab 30: ReAct Agent with LangGraph

> Code: [`course7-module2-lab3-no-refactor/`](course7-module2-lab3-no-refactor/)

ReAct agent that interleaves reasoning and tool calls until the model decides it has enough. The canonical build plus both exercises, each in its own folder — same StateGraph loop and dispatch node across all three; they differ in tool count and shape. No onion port — the architecture is triple-banked across Module 2 (Labs 28, 29, 30) and would repeat itself for a fourth time; the value here is what the loop exposes about tool-output shape and termination semantics, not folder structure.

Built as part of the IBM RAG and Agentic AI Professional Certificate — Course 7, Module 2, Lesson 2. The IBM lab specifies `langgraph==0.3.34`, `langchain-community.tools.tavily_search.TavilySearchResults`, `gpt-4o-mini`, `pygraphviz` for the render, and a `MessagesPlaceholder(variable_name="scratch_pad")` shape carried over from the older `AgentExecutor` era. This implementation uses `langgraph==1.2.6`, `langchain-tavily.TavilySearch` via the dedicated post-sunset package, `gpt-4.1-mini` via `langchain-openai`, and `draw_mermaid_png()` against mermaid.ink for the render. The two exercises canonically shipped with the notebook (calculator, news summariser) are implemented in full as separate modules under their own folders — each one surfaced findings the canonical demonstration didn't advertise.

---

## What It Does

Three ReAct agents, one graph topology, three tool surfaces.

**`canonical/react_canonical.py`** — the base build. Two tools (`search_tool` via Tavily, `recommend_clothing` via a keyword ladder), one system prompt instructing step-by-step reasoning, and the same StateGraph loop used in every variant: `agent → should_continue → tools → agent → …` until the model emits an `AIMessage` with no `tool_calls`. Test query: *"What's the weather like in Zurich, and what should I wear based on the temperature?"*

**`exercise1-calculator/react_calculator.py`** — same graph, three tools. Adds `calculator_tool`, which delegates to an AST-walking safe evaluator (`safe_eval.py`) with an explicit allowlist of operators, functions, and constants. Rejects everything outside the allowlist with a descriptive `ValueError` the LLM can read. Test query: *"Calculate 15% of 250 plus the square root of 144."*

**`exercise2-news/react_news.py`** — same graph, two tools. Replaces `recommend_clothing` with `news_summarizer_tool`, an LLM-as-tool built on a second `ChatOpenAI` instance with a focused summarisation system prompt. The outer agent decides when to route to it; the inner model does the summarisation. Same pattern as Lab 26's `sql_db_query_checker`. Test query: *"Find recent AI news and summarize the top 3 articles."*

Each `react_*.py` produces its own Mermaid artefact (`graph_*.mmd`/`graph_*.png`) via `render_graph_artefacts()`. All three artefacts are structurally identical — the graph topology is invariant across tool surfaces.

---

## Stack

| Component | Implementation |
|---|---|
| LLM (all) | `gpt-4.1-mini` via `ChatOpenAI`, `temperature=0.0` |
| LLM-as-tool (news) | Second `ChatOpenAI` instance, same model, no bound tools, `SystemMessage` + `HumanMessage` invocation |
| Graph construction | `StateGraph(AgentState)` — `MessageGraph` deprecated in 1.x |
| State reducer | `Annotated[Sequence[BaseMessage], add_messages]` |
| Conditional routing | `add_conditional_edges` with explicit path_map (`{"continue": "tools", "end": END}`) |
| Web search | `langchain-tavily.TavilySearch`, `max_results=5` (post-sunset from `langchain-community`) |
| Safe expression eval | AST walker with explicit operator/function/name allowlists (`safe_eval.py`) |
| Architecture | Canonical, no onion port |
| Visualisation | `get_graph().draw_mermaid()` / `draw_mermaid_png()` via mermaid.ink — no `pygraphviz` |
| Test surface | 40 tests (40 `safe_eval` unit tests, 3 canonical smoke, 4 news smoke) |

---

## Setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

`.env` at the lab root:

```
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
```

Each variant runs from its own folder:

```powershell
cd canonical; python react_canonical.py; cd ..
cd exercise1-calculator; python react_calculator.py; cd ..
cd exercise2-news; python react_news.py; cd ..
```

Tests from lab root:

```powershell
pytest
```

40 tests, no API key required (safe_eval is pure, smoke tests mock the LLM and Tavily).

---

## Key Concepts

### The ReAct Loop Terminates on Model Decision, Not Application Budget

`should_continue` inspects the last message and routes to `tools` if `tool_calls` is non-empty, else to `END`. The termination signal is the model choosing not to emit a next action — not a message-count ceiling (Lab 28), not a tool-count ceiling (Lab 29). The reasoning node owns termination; the tool node just executes and cycles back. In Reflection either node could theoretically end the loop and Lab 28 gave the decision to `len(messages) > 6` — a message-count ceiling, not by design. In Reflexion the decision was `MAX_ITERATIONS=4` measured in tool calls, which is Lab 28's anti-pattern relabelled. ReAct centralises termination in the reasoning node, where absence of a next-action is the signal by construction.

### `MessagesPlaceholder(variable_name="scratch_pad")` — Legacy AgentExecutor Shape

The canonical chains `chat_prompt | model.bind_tools(tools)` where `chat_prompt` contains `MessagesPlaceholder(variable_name="scratch_pad")`, and every `call_model` invocation passes `{"scratch_pad": state["messages"]}`. The placeholder injects the system message on every call by piping the accumulated message history into a variable literally named `scratch_pad`. The `scratch_pad` name is a leftover from the pre-StateGraph `AgentExecutor` era, where the agent maintained a separate `agent_scratchpad` variable distinct from user input. In modern `StateGraph` land, `state["messages"]` is the scratchpad — there's no separate channel. Kept for canonical fidelity; a cleaner modern shape would bind the system message once via `.bind()` or prepend a `SystemMessage` in `call_model`.

### Type Safety Lives at Graph Node Boundaries, Not in the Type System

Two runtime narrowing assertions (`tool_node`, `should_continue`) plus an `AgentState` annotation at the input construction site — three seams where the graph's typed API meets construction-site dict literals or attribute access on `BaseMessage`. LangGraph's node signature is `dict → dict`, structurally correct because nodes are polymorphic across state shapes, but it throws away every invariant the graph author established through routing. `state["messages"][-1]` types as `BaseMessage`; `.tool_calls` lives on `AIMessage`. The graph topology guarantees an `AIMessage` at this point — pyright cannot see topology. Adding `if not isinstance(last, AIMessage): raise TypeError(...)` documents the invariant in code, gives a runtime failure at the seam if the topology ever changes, and satisfies pyright without `cast()` or `# type: ignore`. Same architectural category as V3a/V3b's translator boundary — the place where a strong contract meets a permissive substrate needs an explicit check.

---

## Findings

**Module 2's termination-signal arc closes at ReAct with a caveat.** Reflection terminated on `len(messages) > 6` — a message-count ceiling that caught convergence by accident. Reflexion terminated on `MAX_ITERATIONS=4` measured in tool calls — a tool-count ceiling, structurally identical to Lab 28's pattern relabelled. ReAct terminates on `if not last_message.tool_calls` — the model itself signals termination by choosing not to emit a next action. Framework-provided budget → framework-provided budget → model-native decision. But model-native termination is not the same as model-native reasoning: the loop terminates *correctly* even when the reasoning underneath it is structurally hollow. Termination correctness and reasoning correctness are orthogonal.

**Termination correctness ≠ causal correctness.** The Zurich trace at `temperature=0.0` emitted parallel calls at turn 1 — `search_tool(query="current weather in Zurich")` and `recommend_clothing(weather="current weather in Zurich")` — with the clothing tool receiving the raw query string, not the search result. Both tools ran independently. The keyword ladder in `recommend_clothing` fell through to the fallback `"A light jacket should be fine."` because no keyword matched "current weather in zurich." The final `AIMessage` synthesised: *"The current weather in Zurich is about 18.3°C with a light rain shower. Based on this temperature, a light jacket should be fine to wear."* — an answer that reads as reasoned-from-evidence when both tools returned in isolation and the model concatenated their outputs while pretending they were causally linked. The ReAct loop *decorates* the final answer with tool outputs; it does not *reason from* them. Structural cousin of Lab 29's citations-as-laundering — the surface form performs "I searched, then I concluded" whether or not the chain is intact.

**Exercise 2 laundered "no data" into a confident summary of three articles.** The news query surfaced five search-result teasers (Reuters, WSJ, Indian Express, NBC, AI News), each ~15 words of headline-plus-fragment. The model recognised these were insufficient and fired three more parallel searches (~one wasted round). Those also returned teasers. The model then invoked `news_summarizer_tool` three times, passing the ~15-word teaser strings as `news_content`. The first summariser call was honest: *"The provided text does not contain detailed news articles or sufficient information to generate a factual summary."* The next two summarised fragments as if they were articles. The final `AIMessage` synthesised the three summariser outputs into a "summary of the top 3 AI news articles," presenting teaser fragments and one explicit denial as if they were article summaries. The retrieval layer *explicitly denied* having anything and the model routed around the denial. No layer in the pipeline flagged the contradiction. Third instance across Module 2 of "loop mechanics healthy, semantic content unchecked" — and the strongest, because the semantic content included an explicit denial the model overrode.

**Structured tool outputs enable multi-turn reasoning; unstructured outputs don't.** Exercise 1 (calculator) produced *actual* multi-turn ReAct: turn 1 emitted two parallel calls (`0.15 * 250 = 37.5`, `sqrt(144) = 12`), turn 2 emitted a third call (`37.5 + 12 = 49.5`) that consumed the first turn's tool results. The second turn's tool call is causally downstream of the first turn's outputs. Canonical and Exercise 2 both did parallel-and-hope with post-hoc synthesis. The difference is tool output shape. The calculator returns a numeric string the model can thread trivially into the next expression. `search_tool` returns 500 words of JSON the model has to *interpret* before feeding into another tool; instead of doing the interpretation, the model shortcuts by passing the raw query string. Tool return schema determines whether ReAct sequences correctly. Structured, machine-parseable outputs enable genuine multi-step reasoning. Prose or JSON blob outputs push the model toward one-shot parallel calls with post-hoc narration.

**Type-safety-via-topology is the graph's untyped seam.** Three narrowing points across the file, one per node that dots into a `BaseMessage` and one at the `AgentState` construction site. Every place the graph's typed boundary meets construction-site dict literals or attribute access on `BaseMessage` needs an explicit annotation or narrowing check. `AgentState` construction requires `inputs: AgentState = {...}` for pyright to accept a `list[HumanMessage]` value against the `Annotated[Sequence[BaseMessage], add_messages]` field — generic invariance across `dict` value types blocks the assignment without the annotation. The pattern is consistent: LangGraph node signatures are `dict → dict` at the runtime API surface, and pyright rejects every strong-contract-meets-permissive-substrate seam without explicit assertions.

**Production line:** ReAct's control flow is correct by design where Reflection's and Reflexion's were correct by accident — the termination signal comes from the model's own tool-calling contract, not a framework-provided budget. The loop guarantees termination; it does not guarantee that each step is causally downstream of the last, that tool outputs are actually consumed as inputs, or that "no data" from a tool is treated as different from "data." External evaluation against ground truth — Hooperman's golden-question shape — remains the only mechanism that would answer "was the reasoning real."

---

## What This Doesn't Cover

- **Causal-chain verification.** Nothing in the graph enforces that each tool call's input is causally downstream of prior tool outputs. The Zurich trace and Exercise 2's news synthesis both demonstrate this gap; neither is a solvable-inside-ReAct problem.
- **Retrieval quality vs relevance scoring.** The news exercise surfaced Tavily results with high relevance scores for teasers that contained no substantive article content. Same category as Lab 29's Facebook-post finding: retrieval ranking is not quality ranking.
- **Streaming beyond `stream_mode="values"`.** The lab uses full-state emission per node. `"updates"` mode (partial state deltas) is what a production trace would use to avoid re-serialising the transcript each step.
- **HITL middleware and structural guardrails.** No `interrupt_before` on tool nodes, no rate limits on Tavily calls, no PII filters at the tool boundary. The `safe_eval` walker is the only structural boundary in the file; a production ReAct build would have several.
- **Cross-turn convergence detection.** The graph terminates on tool-call absence but does not detect the model looping on the same tool with the same input across multiple turns. Exercise 2 fired four `search_tool` calls in one session, three of them variants on the same query. A production build would cap or dedupe.


---

**Completed:** 1 July 2026