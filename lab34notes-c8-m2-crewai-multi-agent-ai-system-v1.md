# Course 8 - Lab 34: CrewAI 101 - Building Multi-Agent AI Systems

> Code: [`course8-module2-lab1-v1/`](course8-module2-lab1-v1/)

Three-agent sequential content pipeline in CrewAI. `research_agent` (Senior Research Analyst, SerperDevTool) hands off to `writer_agent` (Tech Content Strategist), which hands off to `social_agent` (Social Media Strategist). One task per agent, wired with `Crew(process=Process.sequential)` - CrewAI threads prior task outputs into the next task's context implicitly (no `context=[...]` declaration in the code). Anthropic Claude Haiku 4.5 via CrewAI's native provider (installing `anthropic` transitively activates it - the `anthropic/` model-string prefix routes there in preference to LiteLLM). Framing lab for Module 2's CrewAI sub-section, following Module 1's LangGraph labs 32 and 33.

## Run it

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt  # anthropic, crewai, crewai-tools, python-dotenv
Copy-Item .env.example .env
# Fill in ANTHROPIC_API_KEY and SERPER_API_KEY in .env
python content_pipeline.py
```

Sign up at [serper.dev](https://serper.dev) for a free-tier API key.

## File layout

~~~
course8-module2-lab1-v1/
├── content_pipeline.py           # LLM + agents + tasks + crew, main() at bottom
├── smoke_serper.py               # Serper connectivity smoke test (optional)
├── requirements.txt
├── pyproject.toml                # Ruff config
└── .env.example                  # committed template (copy to .env locally)
~~~

---

## Key Concepts

### `Process.sequential` threads all prior task outputs implicitly

No `context=[research_task]` on `writer_task`, no `context=[research_task, writer_task]` on `social_task`. Under `Process.sequential`, every task's execution context includes every prior task's raw output - the framework threads them automatically and doesn't expose the wiring. This is the AI_PATTERNS.md §1 CrewAI-column vocabulary: `Crew(process=Process.sequential)` is the pattern's CrewAI expression, and implicit context threading is its behavioural characteristic. LangGraph's `add_edge` chain makes state passing explicit via state-key writes; CrewAI hides both the flow and the state, and gains conciseness in exchange for observability.

Downstream implication: for a three-stage pipeline (research → blog → social), the social task sees *both* the research report and the blog post, and the LLM decides at prompt-interpretation time which to treat as primary. There's no code-level lever to force "summarise the blog, not the research" - LangGraph exposes that (`context=[writer_task]`), CrewAI doesn't. The design tradeoff is visible in the code shape: CrewAI's `Task` list is a total order; LangGraph's `add_edge` graph is a topology.

### `AnthropicCompletion` native provider vs LiteLLM - same string, different plumbing

`LLM(model="anthropic/claude-haiku-4-5", ...)` is the notebook's mental model of a LiteLLM router directive. In CrewAI 1.15.2's `LLM.__new__`, that prefix triggers a check for a native SDK provider (`crewai.llms.providers.anthropic.completion.AnthropicCompletion`); if the `anthropic` package is installed, native takes precedence. LiteLLM only runs when native routing fails, or when `is_litellm=True` is passed explicitly. Net effect: install `anthropic`, get the native provider without changing the model string.

The behavioural difference matters where the LiteLLM shim strips detail. Prompt caching surfaces via `cached_prompt_tokens` and `cache_creation_tokens` on `UsageMetrics`, both of which are populated by the native provider from Anthropic's SDK response and both of which the LiteLLM aggregator collapses into the generic `prompt_tokens` counter. Same string, different observability.

### `CrewOutput | CrewStreamingOutput` return-type union - isinstance narrow at seam

`Crew.kickoff()` in 1.15.2 has a union return type. Streaming path returns `CrewStreamingOutput` (no `tasks_output` field); non-streaming path returns `CrewOutput`. Whether the LLM has `stream=True` set is what drives the branch. Pyright can't prove which branch runs at the call site, so accessing `result.tasks_output` without narrowing fails static analysis.

Same discipline as Lab 33's `isinstance(usage, UsageMetrics)` narrow on `result.token_usage`: assert-narrow at the framework boundary, no `cast`, no `type: ignore`, and the narrow doubles as a diagnostic if the framework's return shape drifts under a future upgrade rather than `AttributeError`ing at a downstream field access.

### `TaskOutput.messages` is the observation surface for framework activity

`TaskOutput.messages` (new in 1.15.x) is a `list[LLMMessage]` TypedDict - `role`, `content`, `tool_calls`, `tool_call_id`. No token counts per message, so token accounting stays CrewOutput-level. What it *does* expose is per-task LLM-turn count and, if you grep the tool_call names, delegation activity. `len(task.messages)` is the cheapest question you can ask about "how chatty was this agent" without parsing the verbose stdout dump. The finding below on delegation cost was surfaced entirely by comparing message counts across runs.

---

## Findings

**`expected_output` is the strongest single lever on token spend, stronger than `max_iter` or `max_tokens`.** First run with unbounded `expected_output="A detailed report on {topic}, including trends, emerging technologies, and their impact"` produced 3000+ tokens of research that truncated silently at `max_tokens=2000` mid-generation (report ended at `5. Goal-Oriented Execution:` with a bare colon). The writer downstream still produced a coherent blog because Anthropic's model handles cut-off input gracefully - but coherent doesn't mean intended. Constraining `expected_output` to `"400-word summary of {topic} covering 3-4 key trends"` on the next run produced clean structured output within cap *and* cut the writer's input context by ~4x, because the writer sees the shorter report. Compounding cost saving downstream. `max_tokens` truncates what the LLM did produce; `expected_output` constrains what it tries to produce in the first place. Production shape sets both, and treats prompt-shape as the primary lever.

**`allow_delegation=True` inflates the delegator's own turn count invisibly.** `writer_agent` with `allow_delegation=True` (notebook default): 5 messages, 146k total tokens, 18 requests. `writer_agent` with `allow_delegation=False`: 2 messages, 73k total tokens, 10 requests. Same prompt, same input, same output quality. The 60% drop in writer's message count and 50% drop in total tokens is entirely delegation-cost avoidance. Two things bank from this: the cost accumulates on the delegator's own ReAct loop (not, as I first hypothesised, on the target task's message log - the target task was already complete by the time writer fires under sequential), and the framework hides both the fact of the delegation call and its accounting from the delegator's raw output. Blog quality was indistinguishable between runs. Bank as an AI_PATTERNS.md §5 (Handoff) finding: CrewAI's delegation primitive is invisible at the DAG level and the delegator's task output - visible only in message-count deltas or a full trace.

**Silent Pydantic extra-field acceptance on Agent is a framework-boundary config-typo hazard.** `Agent(..., max_iterations=3)` (wrong field name - real one is `max_iter`) was accepted at construction, stored nowhere, and had zero runtime effect. Ran the pipeline against a 25-iteration cap that I thought was 3. Bug caught by comparing expected vs actual message counts across runs. Bank as a discipline finding for CONVENTIONS_TESTING.md: framework-boundary Pydantic models need either `model_config = ConfigDict(extra="forbid")` overridden on subclass instantiation (which CrewAI's `Agent` doesn't allow), or a schema-pinning test that instantiates `Agent(role="x", goal="y", backstory="z", max_iter_typo=3)` and asserts a `ValidationError`. Otherwise config typos become invisible cost bugs and the wrong caps ship to production. Same class as YAML config drift, applied to Pydantic construction.

**CrewAI framework and crewai-tools ship in lockstep with strict version equality.** `crewai-tools==1.15.2` declares `crewai==1.15.2` exact in its dependency metadata - not `>=1.15,<1.16` as ecosystem norms would suggest. Independent pinning fails: pinning `crewai==1.15.1` alongside `crewai-tools` (unpinned) either fails resolution with a conflict, or silently upgrades `crewai` to match. Bank as an ecosystem-behaviour finding: the framework/tools pin discipline that works in LangGraph (pin the framework, let langgraph-checkpoint / langchain-anthropic / langchain-community float within their own version tracks) doesn't work in CrewAI's ecosystem. You pin both together or you let both float; the middle path doesn't exist. Cross-references AI_PATTERNS.md's growth note on ecosystem maturity - CrewAI's version-locking model is younger, tighter-coupled, and more brittle than LangGraph's.

**Notebook drift: SerperDevTool renamed `query` → `search_query` between crewai-tools 0.38 and 1.15.2.** The notebook's smoke-test cell `search_tool.run(query="...")` raises `ValidationError` on 1.15.2 - the Pydantic schema field renamed. Not fatal to the pipeline itself, because the agent invokes the tool through the LLM's tool-calling contract which reads the schema at runtime, but bites any imperative use in walkthroughs or scripts. Bank as an ecosystem-drift finding at the CrewAI/tool-library boundary: the framework's tool wrappers churn their public schema between minor versions, and any code (or notebook) that calls tools imperatively rather than through an agent ages fast. Same class as the notebook's expectation that `LLM(model="anthropic/...")` routes through LiteLLM - the string works, the plumbing has changed underneath.

---

## What This Doesn't Cover

- **Prompt caching runtime verification.** Verified in source that CrewAI 1.15.2's experimental agent executor (`crewai/experimental/agent_executor.py`) calls `mark_cache_breakpoint()` on both the system prompt (per-agent stable prefix) and the user prompt (per-task stable prefix), and the native Anthropic provider translates the marker into `{"type": "ephemeral"}` cache_control on the message content blocks. Framework wiring is correct. `cached_prompt_tokens: 0` on three consecutive runs - `cache_creation_tokens` (the write metric, distinct from the read metric printed) was not measured, so can't distinguish "wired but not firing" from "wired, writing on run 1, TTL expired before run 2." Deferred; the one-line addition to print `cache_creation_tokens` next to `cached_prompt_tokens` would answer it directly on the next kickoff. Interview-story potential once verified.
- **Onion port.** V1 is flat - constants at module scope, factories omitted (module-scope construction over factory functions), `main()` at the bottom. No dependency-injection seam. Deferred to V2 alongside any Pydantic domain migration where CrewAI's state model warrants it.
- **Test coverage.** No pytest suite for V1. `CONVENTIONS_TESTING.md` applies; the highest-value test would be the schema-pin test called out in the finding above - instantiate `Agent(bad_field=...)` and assert `ValidationError` to prove the extra-field-acceptance discipline. Deferred until after the framework's behaviour is pinned in code.
- **Median-of-N timing.** Three wall-clock measurements per configuration change (delegation on vs off, constrained vs unconstrained `expected_output`). Enough to see the sign and rough magnitude of the effect; not enough to publish a comparison. Same discipline line as Lab 33 - median-of-N with variance is where "we ran it three times" stops being defensible.
- **Prompt quality.** Role/goal/backstory strings are the notebook's originals with the constrained `expected_output` additions the only prompt-engineering intervention. Blog outputs are readable but heavy on framing language ("Perfect! Based on my research analysis..."); social outputs (Exercise 3) were run once as a three-agent kickoff - 3 posts under 280 chars, no framing text, cited facts sourced through the research task. The constrained expected_output on the social task held cleanly, unlike the same run's research task which drifted 3x over its 400-word target. Prompt-shape adherence has visible variance run-to-run - bank as a nuance to the "expected_output is the strongest cost lever" finding. Pattern demonstration is the point.

---

**Completed:** 8 July 2026