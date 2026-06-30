# Course 7 — Lab 29: Reflexion Agent with External Knowledge Integration

> Code: [`course7-module2-lab2-no-refactor/`](course7-module2-lab2-no-refactor/)

Canonical-first Reflexion agent — a responder/executor/revisor loop where one chain produces a structured first draft with self-reflection and search queries, a Tavily-backed tool node retrieves external evidence against those queries, and a second chain revises the draft with citations. Schemas (`AnswerQuestion`, `ReviseAnswer`) carry the structural commitments; the graph wires them together with a message-count ceiling on iteration. No onion port — the architectural arc for Module 2 is deferred to V2/V3 work spanning the module, not per-lab. The value in this lab is the directionality finding, not folder structure.

Built as part of the IBM RAG and Agentic AI Professional Certificate — Course 7, Module 2, Lesson 2. The IBM lab specifies `langgraph` (unpinned, `--upgrade`), `langchain==0.3.21`, `langchain-openai==0.3.10`, `openai==1.68.2`, `langchain-community==0.3.24`, `MessageGraph`, `TavilySearchResults` from `langchain-community`, `gpt-4.1-nano`, and persona prompts modelled on two named real-world public figures (Dr. Paul Saladino and Dr. Peter Attia). This implementation uses `langgraph==1.2.6`, `langchain==1.3.11`, `langchain-core==1.4.8`, `langchain-openai==1.3.3`, `langchain-tavily` (the modern Tavily integration package), `StateGraph` with the `add_messages` reducer, `gpt-4.1-mini`, and fictional personas with the same archetypes as the IBM canonical.

---

## What It Does

Three nodes, one cycle, one structured-output contract:

- `respond` — drafts an `AnswerQuestion` tool call with three fields: `answer` (the ~250-word first draft), `reflection` (a nested `missing`/`superfluous` self-critique), and `search_queries` (1-3 follow-up queries to ground the next round).
- `execute_tools` — reads the last `AIMessage`'s `tool_calls`, runs Tavily against each `search_queries` entry, and emits a `ToolMessage` per tool call with JSON-serialised results.
- `revisor` — drafts a `ReviseAnswer` tool call (subclasses `AnswerQuestion`, adds `references: List[str]`) with citations against the Tavily results.
- `event_loop` — counts `ToolMessage`s in state; routes `revisor → execute_tools` while count < 4, else `revisor → END`. Four tool rounds = one initial draft + four revisions in practice.

---

## Stack

| Component | Implementation |
|---|---|
| LLM | `gpt-4.1-mini` via `ChatOpenAI`, default temperature |
| External knowledge | `TavilySearch(max_results=3)` from `langchain-tavily` |
| Structured output | `llm.bind_tools(tools=[AnswerQuestion])` and `[ReviseAnswer]` — Pydantic v2 schemas as JSON-schema tool specs |
| Graph construction | `StateGraph(AgentState)` with `messages: Annotated[Sequence[BaseMessage], add_messages]` |
| Conditional routing | `add_conditional_edges` with explicit path_map (`{"execute_tools": "execute_tools", END: END}`) |
| Architecture | Canonical, single file — no onion port |
| Visualisation | `get_graph().draw_mermaid()` (source to `.mmd`) and `draw_mermaid_png()` (PNG via mermaid.ink) |

---

## Setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -q langgraph==1.2.6 langchain==1.3.11 langchain-core==1.4.8 langchain-openai==1.3.3 langchain-tavily==0.2.18 python-dotenv
```

Single entry point: `app.py`. `.env` needs `OPENAI_API_KEY` and `TAVILY_API_KEY`. Tavily free tier is 1,000 credits/month — comfortable margin for repeated runs.

---

## Key Concepts

### Schemas Are the Reflexion Pattern's Load-Bearing Surface

`AnswerQuestion` carries three concerns in one tool call: the answer itself, a structured self-critique (`Reflection.missing` and `Reflection.superfluous`), and the search queries that will ground the next round. `ReviseAnswer` inherits all three and adds `references: List[str]`. The schema is doing double duty — Python type at the application boundary, JSON schema at the LLM boundary — and is *also* the eval rubric, encoded structurally: Lab 28's reflection was free-form text buried in a system prompt, here it's a typed two-field object the model cannot drift the shape of. The schema-and-prompt co-design matters: the responder prompt's step 5 (`list 1-3 search queries separately`) is what conditions the model to populate `search_queries`, and `search_queries: List[str]` is what makes the executor's iteration possible without regex parsing. Neither half works alone.

```python
class Reflection(BaseModel):
    missing: str = Field(description="What information is missing")
    superfluous: str = Field(description="What information is unnecessary")

class AnswerQuestion(BaseModel):
    answer: str
    reflection: Reflection
    search_queries: List[str]

class ReviseAnswer(AnswerQuestion):
    references: List[str]
```

`bind_tools` returns a tool-call AIMessage whose `args` is dict-shaped, not Pydantic-validated on the return path. Schema enforcement happens at the OpenAI API boundary (function-calling validates the JSON against the schema) but not re-validated into a Python object on the way back. Trade: the lab keeps simplicity at the cost of pyright autocomplete on the args dict; `with_structured_output` would return a Pydantic instance but loses the tool-call channel the executor reads.

### Schema Inheritance Lets One Executor Function Handle Two Agent Phases

`class ReviseAnswer(AnswerQuestion)` means the responder's tool call (3 fields) and the revisor's tool call (4 fields) both expose `search_queries`. The executor reads that field by name and runs Tavily — it never asks which phase it's serving. Schema hierarchy is doing the work that a polymorphism check would do in a stricter system. Cheap, and the only mechanism by which `execute_tools` stays single-purpose across draft and revision rounds.

### Closures Over Injected Dependencies as a Half-Step Toward DI

`build_graph(tavily_tool, initial_chain, revisor_chain)` takes the dependencies as parameters and defines `respond_node`, `revisor_node`, `execute_tools_node`, and `event_loop` as inner functions capturing them. Module scope holds zero runtime state at import time; nothing executes until `main()` is called. This is a form of dependency injection done via lexical scope rather than constructor parameters — pyright sees the captures, the test surface narrows to "test `build_graph()` end-to-end" rather than "test each node in isolation." A full onion port (`IAnswerCritic`, `IPersonaProvider`, `IConvergenceDetector` ports) is deferred to L29-1 in `cert-deferrals.md` and tied to the post-cert blog satellite rather than the lab.

### Library Migration: `langchain-community` → `langchain-tavily`

`langchain-community` was archived on 22 May 2026 ("No new releases are expected"). The IBM canonical cell imports `TavilySearchResults` and `TavilySearchAPIWrapper` from that sunset package; the modern equivalent collapses both into one class:

```python
from langchain_tavily import TavilySearch
tavily_tool = TavilySearch(max_results=3)
```

The contract changed alongside the import. Old `TavilySearchResults.invoke()` returned `List[Dict]` with `url`/`content` keys. New `TavilySearch.invoke()` returns the richer Tavily API response: `{query, answer, follow_up_questions, images, results, response_time, request_id}`, where `results` is the list. Downstream code that indexed `search_results[0]["content"]` needs to dereference `search_results["results"][0]["content"]`. Same class of migration as Lab 26's `langchain_community` sunset — the package is gone, dedicated integration packages (`langchain-tavily`, `langchain-ollama`, `langchain-anthropic`) are the replacement pattern.

---

## Findings

**Lab 28's convergence-at-round-one pattern holds, with retrieval present.** The initial draft and the final round-4 answer recommend the same foods, cite the same mechanisms, and reach the same conclusion ("carnivore elimination protocol at breakfast"). What changes across rounds is the addition of citation markers, the words "metabolic flexibility" and "individual variability", and minor sentence-level polish. No new evidence enters the answer. The agent reached its conclusion in round 1 and spent rounds 2-4 polishing it. Lab 28 found this with a self-judging critic; Lab 29 finds it with an external-retrieval critic. **The convergence problem isn't fixed by adding external signal — it's deeper than the signal source.**

**Citations-as-laundering, not citations-as-grounding.** The revisor prompt explicitly demanded peer-reviewed research, RCTs, or meta-analyses. It produced four citations across four rounds: a wellness-blog product page, a topic-aggregator portal entry, a Harvard nutrition source page whose actual position contradicts the claim it's anchoring, and a popular-science opinion piece. None of the requested source types — no peer-reviewed articles, no RCTs, no meta-analyses. The `references: List[str]` schema field requires URLs; it doesn't require the URLs to support the claim, to match the source-type the prompt requested, or to come from cardinal evidence. **The structural commitment was to *cite*, not to *be grounded*. The schema enforced one and not the other.** This is sharper than Lab 28's form-vs-truth finding because *form here includes the appearance of evidence*, and that appearance satisfies every automated check the system performs.

**Persona-prompted directionality overrode evidence-grounded revision.** The responder system prompt encoded a specific position (animal-based superior, plants-as-toxins, fiber-unnecessary). The revisor prompt encoded a rigorous evidence-based methodology that should, on paper, surface disagreement when the search results don't support the responder's framing. They didn't — the revisor produced four rounds of confirmatory revision, never once flagging that the search results included sources explicitly contradicting the claims. **The persona prompt sits above the revisor in the system-prompt hierarchy, and the revisor cannot meaningfully disagree with the responder's conclusions.** It can only re-dress them. *External knowledge integration without external authority cannot correct a directional prompt.* The lab's name promises external knowledge integration; the trace demonstrates that integration is not sufficient.

**The medical-content failure is structural, not incidental.** The test question encoded a specific clinical profile: pre-diabetic with heart disease. The agent recommended a breakfast pattern (high saturated fat, high cholesterol, zero fibre) that runs against mainstream cardiology consensus for this exact patient profile — and did so confidently, with citation scaffolding, with biomarker awareness baked into the language ("monitor HbA1c, lipid panel, inflammatory markers"). The clinical-jargon polish makes the output *more* dangerous to a non-expert reader, not less; the citation scaffolding signals authority that the underlying claims don't earn. **Production line:** a persona-anchored Reflexion agent on a contested clinical topic, with retrieval, produces medically dangerous advice that is harder to flag as wrong than a free-form persona answer would have been. No human in the loop, no expert review, no external-authority signal beyond Tavily's relevance ranking — and Tavily's relevance ranking has no stake in clinical correctness.

**Persona prompts naming real living public figures are out of scope for production.** The canonical responder persona was modelled on Dr. Paul Saladino; the revisor instructions named "Dr. David Attia" surrounded by Dr. Peter Attia's actual public positions. Implementation swapped both to a fictional researcher (Dr. Marcus Vale) with identical archetypes — same mechanics, same content bias, lab teaches the identical move. The failure mode of naming real figures in persona prompts is reputational and legal rather than factual; the fictional swap is cost-free and the lab's pedagogical content is unchanged.

**`MAX_ITERATIONS=4` as a `ToolMessage`-count ceiling is the same anti-pattern as Lab 28's `len(messages) > 6` — relabelled, more expensive to ignore.** Counting tool-message visits is no more a convergence signal than counting messages was. The revisor could be producing identical output across rounds 3 and 4 and the loop wouldn't notice; in practice, it was producing near-identical output, and the loop didn't notice. The honest convergence check is diffing successive answers and stopping below a threshold. Out of scope for the lab, but the cost (5 LLM calls per question vs Lab 28's 3) makes the absence sharper.

**Production line for the lab note.** Reflection improves form, not truth. Reflexion improves *apparent grounding*, not truth. The failure mode of agentic reflection is not absence of evidence — it's evidence subordinated to directional priors set elsewhere in the system. The architectural fix is not more retrieval; it's *authority-shaped retrieval*, where the signal can override persona. Hooperman's 33 golden questions are exactly that signal. Wiring a Reflexion loop's critic against ground-truthed retrieval — where the schema demands not "include a citation" but "match the answer against an authoritative source set" — is the experiment this lab points at and does not perform.

---

## What This Doesn't Cover

- **Convergence detection.** The loop terminates on `ToolMessage` count, not on detecting that the answer has stopped changing. A real implementation would diff successive `answer` fields and stop early.
- **External-authority critic.** Tavily provides relevance, not authority. A clinically grounded Reflexion agent would check claims against a curated source set (medical guidelines, peer-reviewed corpus) the persona cannot override. Logged as L29-1 in `cert-deferrals.md` for post-cert work.
- **Streaming output.** `app.invoke` blocks for 60-120s with no progress signal. `app.stream(stream_mode="updates")` would surface per-node deltas; not implemented because the goal was the trace, not the developer experience.
- **Tool-choice forcing.** `bind_tools(tools=[AnswerQuestion])` permits but doesn't force the tool call. `tool_choice="AnswerQuestion"` would close the trapdoor where the model could return free-form prose and `IndexError` the downstream code. One-line hardening, omitted to match IBM canonical.
- **Local-model run.** Lab 28's 3B failure under `bind_tools` made the local Reflexion run a confirmation rather than evidence; tool-calling reliability is the harder regime for small models and the lab's existing trace is sufficient for the directionality finding. The 7B+ floor flagged in the module plan would apply if ever revisited.
- **Persistent state / checkpointing.** Each `app.invoke` runs cold. `MemorySaver` or `SqliteSaver` is a one-line wire-up to make individual rounds resumable; not in scope.

---

**Completed:** 30 June 2026