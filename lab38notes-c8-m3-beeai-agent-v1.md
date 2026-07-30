# Course 8 - Lab 38: BeeAI Framework - Quickstart to Multi-Agent

> Code: [`course8-module3-lab1-v1/`](course8-module3-lab1-v1/)

Modernised port of IBM Skills Network's C8 M3 BeeAI lab: twelve sections from `ChatModel` basics through a four-agent travel planner with `HandoffTool`. `beeai-framework==0.1.81` (head; lab shipped `0.1.35`, 46 releases stale) on Python 3.12 + Anthropic Haiku 4.5 via LiteLLM. Kept CrewAI-cert model continuity for cross-lab cost comparability. Watsonx-swap deferred as post-lab experiment.

BeeAI is pre-1.0 (0.1.x) with active development — 46 releases across ~10 months of session-relevant history. The bulk of the session's findings clustered around framework-vs-model behaviour under scheduling constraints, the gap between typed API surfaces and actual measurable behaviour, and a load-bearing install saga on Windows driven by LiteLLM's Rust-core migration in June 2026.

## Run it

```powershell
python -m venv venv                     # explicit -3.12 required (see below)
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env             # populate ANTHROPIC_API_KEY
python smoke_chat.py                    # start here; each file runs standalone
```

Pinned:
```
--only-binary litellm

beeai-framework[wikipedia]==0.1.81
litellm==1.91.4
python-dotenv==1.2.2
```

The `--only-binary litellm` directive is load-bearing. Without it, pip resolves litellm to 1.93.0 (sdist-only on Windows) and hits the Rust build wall. See "Windows install requires substrate management" below.

## File layout

```
course8-module3-lab1-v1/
├── helpers/
│   ├── __init__.py
│   ├── metrics.py         # print_run_metrics(state): iterations, tokens, cache-rate, cost
│   └── hitl.py            # stdin_approval_handler for AskPermissionRequirement
├── smoke_chat.py          # § 2: ChatModel.run(messages)
├── prompt_template.py     # § 3: PromptTemplate + Pydantic schema
├── structured_output.py   # § 4: run(response_format=Schema) → .output_structured
├── minimal_agent.py       # § 5: RequirementAgent, tools=[]
├── wikipedia_agent.py     # § 6: WikipediaTool + max_invocations + trajectory
├── reasoning_agent.py     # § 7: adds ThinkTool (permitted)
├── controlled_agent.py    # § 8: force_at_step + only_after + min_invocations
├── react_agent.py         # § 9: composed-ReAct at scheduling layer
├── hitl_agent.py          # § 10: AskPermissionRequirement + custom handler
├── custom_tool_agent.py   # § 11: Tool subclass, AST-based safe arithmetic
├── travel_planner_multi_agent.py      # § 12: coordinator + 3 specialists via HandoffTool
├── requirements.txt
├── pyproject.toml
└── .env.example
```

---

## Key concepts

### Windows install requires substrate management

LiteLLM entered a Rust-core migration in June 2026 (litellm ≥ 1.81). Rust-cored versions ship sdist-only for Windows — no `cp3XX` wheels. Pip resolves version-first and won't backtrack from sdist to older wheels; hits `puccinialin` Rust auto-installer, needs full MSVC toolchain, fails. BeeAI 0.1.81's `litellm>=1.83.14` transitive floor puts head-of-line BeeAI past the wheel boundary on Windows. Working configuration: Python 3.12.10 + `--only-binary litellm` in `requirements.txt`. Pip's resolver then picks litellm 1.91.4 (last wheel-shipping 1.91.x maintenance line) and the install completes clean.

Three failed pip runs on 3.12 and 3.13 before the diagnosis landed. Two of my theories along the way were wrong (Python-version wheel gap, then pin-set-as-constraint-pressure); the actual mechanism was version-first resolution ignoring wheel availability. LiteLLM's own tracking issue #31261 confirms the Rust-migration install friction is known ongoing work through Dec 2026. BeeAI's `beeai-framework-py-starter` uses uv on macOS/Linux with Ollama — Windows + pip + cloud provider is off the tested path.

### API renames between 0.1.34 and 0.1.81

Five confirmed renames across the ~10-month window between the lab's shipped pin and current head: `ChatModel.create(messages=...)` → `ChatModel.run(messages)` (positional list, `.last_message.text`); `ChatModel.create_structure(schema=...)` → `ChatModel.run(messages, response_format=...)` (`.output_structured`); `beeai_framework.agents.experimental` → `beeai_framework.agents.requirement`; `RequirementAgentOutput.answer` (never existed at 0.1.81) → `RequirementAgentOutput.output[-1]`; `AskPermissionRequirement(["StringName", ...])` positional list of strings fails with a tool-name resolution error at 0.1.81 and needs the `[instance, ...]` shape. Rename set is coherent library hygiene consistent with pre-1.0 versioning. Third-party tutorials (Arize, Langfuse) and the framework's own on-main `openai_example.py` still use old API surfaces at time of session.

### Composed-ReAct at scheduling layer requires tool-floor constraints

BeeAI's `force_at_step=1` + `force_after=Tool` + `consecutive_allowed=False` on ThinkTool is described as reconstructing ReAct at the scheduling layer. On Haiku 4.5, the three-parameter composition alone doesn't fire the pattern — model reasons its way out of using the action tool, so `force_after=Tool` never has anything to fire against. Adding `min_invocations=1` on the action tool forces it to fire, then `force_after=Tool` triggers Think between calls, and the actual `Think → Wikipedia → Think → Wikipedia → Think → FinalAnswer` rhythm emerges.

Section 9 shipped without the tool-floor constraint (WikipediaTool commented out in the lab code). On strong instruction-followers, that config is degenerate. `react_agent.py` in the repo is the production shape — re-added WikipediaTool with `min_invocations=1` — and reflects the correction. Weaker models (e.g. Granite, which the framework demos use extensively) respond more readily to prompt suggestions to use tools, so the shipped config likely fires the pattern on those models.

### Requirement scheduler enforces constraints; model's belief about tool availability diverges from enforcement

The scheduling primitives (`force_at_step`, `only_after`, `min_invocations`, `consecutive_allowed`) empirically hold at enforcement. The model's Think content across five separate runs consistently hallucinated "Wikipedia is disabled" or "Allowed: False" for tools that were in fact available and about to be forced-fired. Mechanism appears to be: scheduler communicates per-step allowed-tools to the model via system-prompt injection; model reads "not this turn" as "not available at all" and reasons around the tool. Enforcement then either overrides (tool fires despite model's stated reasoning) or defers to model's choice (tool skipped).

`reasoning_agent.py` with `force_at_step=1` on Think and permitted Wikipedia: Wikipedia skipped, cost dropped 67% vs permitted baseline, output ungrounded. `controlled_agent.py` with `min_invocations=1` on Wikipedia: Wikipedia fired twice (both empty results), Think content said "Wikipedia disabled" while it was being forced to fire. `hitl_agent.py` with `min_invocations=0`: Wikipedia never invoked, HITL never fired, model reasoned Wikipedia unavailable. The "consistent behavior across LLMs" property that the primitives are designed to provide holds under hard-constrained configurations; flexible configurations preserve model-dependency.

### `state.cost.total_cost_usd` overstates true billed cost by 4-20× on multi-agent workflows

`travel_planner_multi_agent.py` reported `state.cost.total_cost_usd=0.9787` on one run. Anthropic Console showed $0.24 as the entire day's spend across ~15 runs of various shapes. Framework overstates by minimum 4×, plausibly 10-20× on this specific run. Mechanism unresolved — candidates are LiteLLM stale pricing rates, framework double-counting nested-agent calls, cache-tier pricing not applied. Multi-agent shape amplifies whatever the base error is. Cache-tier absence is confirmed at single-agent scale: cost held roughly stable across three consecutive runs where cache-hit rate flipped 0% → 87% → 87%, indicating cached-read pricing isn't applied.

For cost-critical use cases, provider dashboards remain source of truth. `state.cost` is directionally useful for identifying which runs are expensive relative to each other, but the absolute figures are not billing-grade.

### Cache hit rate is a function of Anthropic's TTL, not framework instrumentation

BeeAI's `ChatModelUsage` preserves `cached_prompt_tokens` through LiteLLM's aggregation layer (contra my initial reading of `types.py`). Three consecutive `wikipedia_agent.py` runs on identical input: 87% cache → 0% cache (TTL elapsed) → 87% cache (previous run warmed prefix within TTL). Fully reproducible. Multi-agent shape natively benefits: `travel_planner_multi_agent.py` cache hit was 90.6% because the coordinator's prefix (system prompt, instructions, tool descriptions) is reused across every iteration.

Resolves L34/L35/L36's `cached_prompt_tokens=0` finding on CrewAI as CrewAI-specific — `mark_cache_breakpoint()` calls exist in CrewAI 1.15.2's `AnthropicCompletion` source but the resulting `cache_control` markers aren't reaching the API. Not a LiteLLM aggregation defect.

### HandoffTool is subroutine-call semantics, not state-transition

Coordinator invokes `HandoffTool` wrapping a specialist agent; framework runs the specialist to completion in a fresh instance with its own memory, tools, and requirements; specialist's output text returns as tool result to coordinator; coordinator continues. No shared state, no session continuation. `GlobalTrajectoryMiddleware` propagates through the handoff boundary — nested tool calls appear inline with the coordinator's calls, distinguished by tool-namespace prefixes.

Contrast L34's finding on CrewAI `hierarchical` where delegated agent behaviour was opaque. BeeAI's observability at the handoff boundary is materially thicker at the code-level typed surface. Observability shows *what fired*, not *what the model believed about what fired* (see scheduler-vs-model belief above) — audit-trail-shaped rather than decision-audit-shaped.

### AskPermissionRequirement interrupt only fires if the model chooses to invoke the tool

The primitive prevents tool invocation without approval, but *also* signals unavailability to the model via the same scheduler-to-prompt channel. On strong models the "requires approval" signal reads as "unavailable" — model doesn't request the tool, approval handler never called. `hitl_agent.py` with default config never fired the prompt. `min_invocations=1` on Wikipedia forced invocation, HITL fired, custom handler surfaced query args and captured approval cleanly. Default handler shows tool name only; custom handler in `helpers/hitl.py` shows tool arguments alongside — the difference between rubber-stamp and informed approval.

For governance use cases, the primitive's presence changes model behaviour to skip the gated tool entirely — a reviewer seeing "no Wikipedia call, approval control in place" cannot distinguish "not needed" from "silently avoided due to gating signal" without also seeing the model's reasoning content.

### `PromptTemplate` HTML-escapes variables by default

`PromptTemplate` uses `chevron` (a Mustache implementation) with default HTML escaping. `<`, `>`, `&`, `"` in template variables render as HTML entities. Prompt in `prompt_template.py` had `<2% forecast error` in one field; rendered output showed `&lt;2% forecast error`. Modern LLMs often silently recover; smaller models won't. Standard Mustache-spec behaviour inherited from the underlying template library; per-variable triple-brace `{{{var}}}` bypasses the escape, or `html.unescape()` post-render handles it globally.

### `Tool` base class properties disagree with framework docs' subclass pattern

`Tool[Input, Options, Output]` declares `name`, `description`, and `input_schema` as `@property` on the base class. Framework's own canonical `RiddleTool` example at `framework.beeai.dev/modules/tools` shows subclasses overriding all three as class attributes. Works at runtime (Python duck-typing). Pyright rejects it as property-to-attribute type violation. Fix in `custom_tool_agent.py` is property overrides. Any BeeAI subclass shipped in a Pyright-strict codebase needs the property-override pattern rather than the class-attribute pattern the docs demonstrate.

### Multi-agent handoff amplifies cost 5-10× versus equivalent single-agent

`travel_planner_multi_agent.py` produced a competent travel plan (three-day breakdown, phrase lists, cultural etiquette, weather packing). Coordinator's Think #1 explicitly reasoned about delegating to all three specialists in sequence — genuine synthesis plan named upfront, not blind fan-out. Estimated true billed cost ~$0.05-0.10 (framework's $0.98 figure is unreliable per above). Whether the multi-agent shape earns its cost versus a single well-prompted call is genuinely uncertain and worth naming as the load-bearing question. Rough estimate: single-agent would produce ~85% of the quality at ~2% of the cost for this query shape.

## Findings

**Windows install is materially harder than the other frameworks in the cert.** Six failing pip runs, three wrong theories from me before the mechanism landed. Fix is one directive in `requirements.txt`. LangGraph, CrewAI, and LangChain all installed clean first try on the same substrate.

**BeeAI's declarative surface is most meaningful when tightly constrained; loosening constraints returns model-dependent uncertainty.** The three-parameter ReAct composition, `AskPermissionRequirement` without `min_invocations`, and `force_at_step` alone all produce different trajectories on Haiku than the framework demos on weaker models. The primitives-vs-pattern-of-primitives distinction is load-bearing. The "consistent behavior across LLMs" property holds under hard-constrained configurations.

**Framework observability at the code-level surface is genuinely thick.** Trajectory middleware propagates through handoff boundaries; tool calls surface with arguments and results; `state.usage` preserves Anthropic's cache metrics through LiteLLM. Materially better than CrewAI's typed observability at the same layer, provided `state.cost` isn't used for anything requiring billing accuracy.

**`state.cost` is directional not billing-grade.** 4-20× overstatement on multi-agent runs. Directional signal (which runs cost more than others) holds; absolute figures don't. Provider dashboards remain the source of truth for actual spend.

**Cache metrics resolve the L34/L35/L36 CrewAI caching-gap question.** `cached_prompt_tokens=0` across three CrewAI labs is CrewAI-specific. BeeAI-via-LiteLLM on the same model with the same key produces non-zero cache metrics under warm-TTL. F-L36-5 deferred question closed; sharpens to "why does CrewAI's native provider drop the markers between `mark_cache_breakpoint()` and API call."

**Strong models skip tools whose value they can substitute for, use tools whose value they can't.** `custom_tool_agent.py` invoked SimpleCalculator on all four queries including nested-parens `(10 + 5) * 3 - 7` — the case where model reasoning would be tempted to compute mentally. Same model on Wikipedia consistently reasoned itself out of tool use across five configurations. Tool utility from the model's perspective is the load-bearing variable, not force-vs-permit config.

**Course material and framework head are at ~10 months of drift.** Six concrete instances: stale framework pin, hand-rolled reimplementation of a first-class primitive, `.create()` API, `agents.experimental` imports, dead configuration in § 9, string-list `AskPermissionRequirement` on § 12. Plus pedagogical drift — § 11 teaches `eval`-with-character-set as "safe calculator" (security anti-pattern); § 9's title claims ReAct but the shipped config doesn't demonstrate it on strong instruction-followers. Consistent with the framework being pre-1.0 with active churn.

## Prior art carried across

- **L34: `allow_delegation=True` hid nested delegation cost.** Cross-refs BeeAI's HandoffTool — surfaces in trajectory (good), cost surface is directional-only (see above).
- **L34: `expected_output` as cost lever.** Same principle applied to BeeAI's per-agent instructions — trimmed multi-bullet role descriptions on travel planner specialists after noticing per-iteration inclusion cost.
- **L34: native Anthropic provider requires explicit pin.** Held on CrewAI; does not apply to BeeAI (no native provider path — everything routes through LiteLLM).
- **L34: `cached_prompt_tokens=0` on CrewAI native.** Resolved as CrewAI-specific; not present on BeeAI-via-LiteLLM.
- **L35: `.pydantic=None` silent failure on parse.** BeeAI's `.output_structured` narrow catches this at boundary; failure mode empirically unresolved (needs pathological schema test).
- **L36: agent tool-selection is waterfall not classification.** Held under BeeAI in a new shape — "model reasons about tool availability" is the same underlying dynamic. Model estimation of tool utility drives selection; estimation is unreliable when the model overrates its own capability.
- **L36: tool descriptions ride in every prompt.** Held. BeeAI's tool descriptions are class attributes on `Tool` subclasses (§ 11) or the `description` arg on `HandoffTool` (§ 12); both are prompt content that reaches the LLM.
- **L37: `@tool` docstring is framework contract at decoration.** Same principle applies at BeeAI's `Tool` subclass level — `description` field is not documentation, it's tool-selection signal.

New to L38:

- LiteLLM Rust-core migration boundary at litellm 1.80 on Windows, `--only-binary litellm` as the durable fix.
- Scheduler-vs-model belief divergence as a real observability gap (audit trails show tool calls but not model-belief about tool availability).
- Composed-ReAct at scheduling layer requires tool-floor constraints (not just reasoning-side constraints) to reconstruct the actual pattern on strong models.
- HandoffTool as subroutine semantics (fresh agent instance, no shared state), with nested trajectory observability.
- `state.cost.total_cost_usd` directional not billing-grade — provider dashboard remains authoritative for absolute figures.
- Mustache HTML-escape default in `PromptTemplate` (silent prompt corruption on `<`, `>`, `&`, `"`; standard Mustache-spec behaviour).
- `Tool` base class `@property` vs docs' class-attribute subclass pattern (property-override required under strict Pyright).

## What this doesn't cover

- **Memory strategies.** Every file used `UnconstrainedMemory` per lab pattern. `TokenMemory`, `SummarizeMemory`, `SlidingMemory` untested. Standing context flagged UnconstrainedMemory as demo-only; production shape unexplored.
- **OTEL/Phoenix instrumentation.** Trajectory middleware was sufficient for lab-level observability; real OTEL deferred. `BeeAIInstrumentor().instrument()` must run before any `beeai_framework` import per Arize docs.
- **Streaming.** `ChatModelParameters(stream=True)` visible in framework docs, not exercised.
- **Non-Wikipedia research tools.** `OpenMeteoTool` used once in travel planner; `DuckDuckGoSearchTool`, `ArxivTool`, `SearxngTool` untested.
- **`ReActAgent` vs `RequirementAgent+requirements` comparison.** Framework ships a first-class `ReActAgent` at `beeai_framework.agents.react`. Whether it produces the same trajectory as composed-primitives or is a distinct implementation is empirically open.
- **Cost overstatement mechanism.** Confirms direction but not root cause. Distinguishing between LiteLLM stale pricing, nested-agent double-counting, and cache-tier miss is the cheapest experiment (single-agent cache-warm run, compare `state.cost` against provider dashboard delta).
- **CrewAI cache_control marker path.** L36's F-L36-5 deferred half now clear (BeeAI works, so CrewAI-specific). Remaining half: source-read CrewAI 1.15.2's `AnthropicCompletion` from `mark_cache_breakpoint()` to actual API call.
- **`PromptTemplate` schema-template consistency check.** Cheap probe: template references a field the schema doesn't declare. Does BeeAI catch mismatch at construction, at render, or silently? Framework's schema-strictness measurable in one probe.
- **Watsonx-swap experiment.** Post-lab, single BeeAI script, provider only. Isolates provider-agnostic claim from framework-behaviour claim.
- **Custom `Requirement` subclasses.** `ConditionalRequirement` and `AskPermissionRequirement` used throughout; framework docs show `Requirement[RequirementAgentRunState]` subclass shape for custom scheduling logic. Not exercised.

---

**Completed:** 23 July 2026