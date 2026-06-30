# Course 7 — Lab 28: Reflection Agent with LangGraph

> Code: [`course7-module2-lab1-no-refactor/`](course7-module2-lab1-no-refactor/)

Canonical-first reflection agent — a generate/critique loop where one chain drafts a LinkedIn post and a second chain critiques it, with the critique role-swapped into a `HumanMessage` so the generator treats it as input rather than its own prior output. Run twice: once against `gpt-4.1-mini`, once against local `llama3.2:latest` (3B). No onion port — the architecture is already over-evidenced across five prior agent labs; the value in this lab is the eval/hallucination finding, not folder structure.

Built as part of the IBM RAG and Agentic AI Professional Certificate — Course 7, Module 2, Lesson 1. The IBM lab specifies `langgraph==0.3.31`, `langchain-ibm`, `MessageGraph`, and a LinkedIn-post generator over WatsonX/Granite. This implementation uses `langgraph==1.2.6`, `StateGraph` with the `add_messages` reducer, `gpt-4.1-mini` via `langchain-openai`, and `llama3.2:latest` via `langchain-ollama` for the local comparison run.

---

## What It Does

Two chains, one graph, one role-swap:

- `generate_chain` — drafts or refines a LinkedIn post against a system prompt instructing it to treat any `HumanMessage` in history as user feedback to refine against.
- `reflect_chain` — critiques the most recent draft against a six-point rubric (quality, structure, engagement potential, industry relevance, formatting, CTA effectiveness).
- `reflection_node` relabels the critique as a `HumanMessage` before appending it to state — the mechanism that makes the generator treat critique as input rather than continuing its own prior turn.
- `should_continue` routes `generate → reflect` while `len(messages) <= 6`, else `generate → END` — a message-count ceiling that caps the loop at 3 generate/reflect round trips, not a convergence check.

---

## Stack

| Component | Implementation |
|---|---|
| LLM (primary) | `gpt-4.1-mini` via `ChatOpenAI`, temperature=0.0 |
| LLM (local comparison) | `llama3.2:latest` (3B) via `ChatOllama`, temperature=0.0 |
| Graph construction | `StateGraph(AgentState)` — `MessageGraph` is deprecated in 1.x |
| State reducer | `Annotated[Sequence[BaseMessage], add_messages]` |
| Conditional routing | `add_conditional_edges` with explicit path_map (`{"reflect": "reflect", END: END}`) |
| Architecture | Canonical, single file — no onion port |
| Visualisation | `get_graph().draw_mermaid()` / `draw_mermaid_png()` — no `pygraphviz` |

---

## Setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -q langgraph==1.2.6 langchain==1.3.11 langchain-core==1.4.8 langchain-openai==1.3.3 langchain-ollama==1.1.0 python-dotenv
```

Two entry points: `app.py` (OpenAI) and `app_local.py` (Ollama). `.env` needs `OPENAI_API_KEY` for the former; the latter requires `ollama serve` running with `llama3.2:latest` pulled (`ollama list` to confirm).

---

## Key Concepts

### `StateGraph` + `add_messages` Replaces `MessageGraph`

`MessageGraph` is deprecated in LangGraph 1.x, scheduled for removal in 2.x — the framework's own guidance is `StateGraph` with a `messages` key. The state-level difference matters beyond the import swap: `MessageGraph`'s state *is* the bare message list — every node receives and returns `list[BaseMessage]` directly. `StateGraph(AgentState)` nodes receive the full state dict and return a partial dict; the `add_messages` reducer handles merging rather than the node itself managing list concatenation.

```python
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
```

`Annotated`'s second argument is the actual `add_messages` callable — not a string. A markdown sample in the IBM courseware itself got this wrong (`Annotated[List[...], "add_messages"]`, the reducer's name as a string literal rather than the function), which would silently fall back to last-write-wins overwrite semantics on every node transition rather than accumulating the transcript. Caught before writing the actual state class.

### The Role-Swap Is the Mechanism, Not a Formality

`reflection_node` returns `{"messages": [HumanMessage(content=res.content)]}` even though `res` (the critique) is itself an `AIMessage` from `reflect_chain.invoke(...)`. The relabel is what makes the generator treat the critique as something to react to rather than its own prior turn to continue — there's no separate "critique" channel in the message-stream abstraction; message *role* is the only signal carrying that distinction. The resulting transcript alternates `[Human, AI, Human, AI, ...]` where every other "Human" turn is actually machine-generated critique. This is a deliberate fiction the framework leans on, and it has a real cost: `state["messages"]` alone cannot later distinguish genuine user input from role-swapped critique. Any audit trail wanting to recover true provenance needs a parallel log outside the message stream — the same shape as Hooperman's eval log existing because the user-facing answer doesn't carry its own provenance.

### Explicit Path_Map Is Required for Graph Introspection, Not Just Runtime

`add_conditional_edges("generate", should_continue)` with no explicit path_map runs correctly — LangGraph doesn't need the map to execute the routing function and follow its return value. But `workflow.get_graph()` performs static analysis of the compiled graph, and without a path_map it cannot resolve an arbitrary Python function's possible return values. The rendered Mermaid diagram showed `generate → __end__` only, with `reflect` floating as a disconnected node — wrong relative to the actual topology, despite the graph executing correctly. Supplying the explicit map (`{"reflect": "reflect", END: END}`) fixed the diagram with zero runtime change. Shortcuts that work at execution time can still break the introspection layer — the same category as V3b's node-name-constants finding (rename-safety for tooling, not just correctness for execution).

---

## Findings

**Reflection is LLM-as-judge run in a loop — same mechanism as Lab 26's `sql_db_query_checker`, compounded by iteration count.** The checker cost ~1,778ms for one validation call against a 1.1ms query. Reflection runs that pattern N times with a growing transcript: prompt tokens climbed 90 → 369 → 786 → 998 across three `gpt-4.1-mini` generation calls in this lab — an ~11x growth to produce a 150-character post whose substance was decided after round one.

**The loop converged at round one, then idled for two more full round trips.** Round 2's draft incorporated round 1's critique (added a CTA, added an `@IBM` tag). Rounds 3 and 4 reissued near-identical critique points (image, tagging) against a post the generator had stopped meaningfully revising — the generator's final message was "your revised post is clear, engaging, and well-crafted," agreeing with its own prior output rather than refining further. `len(messages) > 6` caught this by accident, as a message-count ceiling, not by design. Production needs explicit convergence detection — diffing the draft against the previous round — not a fixed iteration budget.

**The critic's rubric is form-only by construction, confirmed in practice.** Six points — quality, structure, tone, engagement, formatting, CTA — and not one of them can ask whether a claim is true. Across every round of both runs, every critique addressed presentation; none touched factual grounding. This run didn't generate a fabrication to launder, but the mechanism is now demonstrated rather than predicted: reflection reliably improves *form* and has no path to improving *factual accuracy*, because the critic has no external signal, only its own judgment.

**`llama3.2:3b` produced silent empty-string critiques three rounds running — a plumbing failure, not a capability failure.** Every `reflect_chain.invoke()` call inside the graph returned an `AIMessage` with `content=""`. No exception, no warning — `should_continue` kept counting messages, the loop kept running, and the generator received an empty `HumanMessage` each round. Rather than fail or stall, the generator hallucinated plausible continuation language ("it seems like you're ready for another iteration!") and invented new draft directions unprompted, three rounds running, from literally nothing.

**Root-caused via isolation test to prompt structure, not model size.** Calling `reflect_chain.invoke()` standalone with a single instruction-laden `HumanMessage` ("Critique this LinkedIn post: ...") produced a full, substantive six-point critique — 604 output tokens, properly structured. The identical model, identical system prompt, identical rubric failed silently when delivered as system-prompt-over-accumulated-`[Human, AI]`-history and succeeded when the instruction was folded inline into a single human turn. `llama3.2:3b` cannot be assumed to extract "critique the AIMessage above" from conversational structure the way `gpt-4.1-mini` does — and the failure mode is invisible, which is the dangerous part for production: nothing in the trace distinguishes "the critic had nothing to add" from "the critic silently produced nothing."

**Fix confirmed end to end — and surfaced a second, sharper failure.** Restructuring `reflection_node` to pass `last_draft` as inline instruction content fixed the empty-completion problem completely; every round produced a real critique. But the corrected local loop actively drove the output away from the user's explicit 160-character constraint — round 1 already exceeded it, and by round 4 the post had ballooned into a multi-section narrative with headers and bullet points ("My Journey to Becoming an IBM Software Developer: A Story of Innovation and Collaboration"). The critic never once flagged the broken length constraint across four rounds; each round's feedback reinforced the drift by praising the added structure and narrative depth as improvements. Local reflection isn't just weaker than the frontier-model run — it can actively optimise toward a different, unstated goal than the one the user gave it, while costing roughly 3x the prompt tokens and 5–10x the wall-clock per call (4.5–6.9s per local call vs sub-second for OpenAI).

**Production line:** reflection improves form, not truth, and a capable critic can still drift from explicit user constraints if nothing in the rubric checks against them. Neither failure mode is fixable by looping harder — both need an external, ground-truthed signal the critic can check against. Hooperman's 33 golden questions are exactly that signal, sitting unused. Wiring a reflection loop's critic against retrieval ground truth from those golden questions — rather than the critic's own unaided judgment — is the real experiment, not a courseware exercise.

---

## What This Doesn't Cover

- **Convergence detection.** The loop terminates on message count, not on detecting that the draft has stopped changing. A real implementation would diff successive drafts and stop early.
- **External-signal reflection.** Both runs used a self-judging critic with no ground truth to check against — the core limitation this lab surfaces, not something it solves.
- **8B-class local model comparison.** The 3B failure was decisive and root-caused; an `llama3.1:8b` or `qwen2.5:7b` run was judged unnecessary against the cert-velocity strategy once the 3B exhibit was confirmed.
- **Constraint enforcement.** Nothing in either run's prompt or graph hard-enforces the 160-character limit; it's prompt-only guidance the local-model loop demonstrably ignored with no structural backstop.

---

## What's Next

Module 2 continues with concept-coverage labs at the stated terminal-per-lab velocity — no V1/V2/V3 ladders, no onion ports, findings banked and moving forward. The Hooperman external-eval-signal reflection experiment is logged as a post-cert closure trigger, not in-scope here.

---

**Completed:** 30 June 2026