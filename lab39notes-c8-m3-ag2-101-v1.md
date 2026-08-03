# Course 8 - Lab 39: AG2 101 (AutoGen) - Complete Tutorial

> Code: [`course8-module3-lab2-v1/`](course8-module3-lab2-v1/)
>
> **Distribution: `ag2==1.0.1` (`import ag2`).** Not AG2 Classic (`autogen`).
> Same lineage, different framework - see F-L39-1.

Modernised port of IBM Skills Network's C8 M3 AG2 lab. The notebook ships against AG2 Classic and does not run at all on a fresh install as of August 2026; this port targets the v1.0 protocol-driven framework that replaced it, on Python 3.14 + Anthropic Haiku 4.5 via AG2's native Anthropic client.

Seven scripts, every notebook cell covered. The lab's centre of gravity turned out not to be the API surface but the **failure mode**: four of the eighteen findings came from code that ran clean, produced fluent output, and was structurally wrong. Nothing raised. The only things that caught them were reading the channel WAL and reading the adapter source.

## Run it

```powershell
py -3.14 -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env             # populate ANTHROPIC_API_KEY
python smoke_chat.py                    # start here
```

Pinned:
```
ag2[anthropic]==1.0.1
python-dotenv==1.2.2
matplotlib==3.11.1
```

Python 3.14. The full dependency tree resolves to binary wheels on Windows — `pydantic-core`, `tiktoken`, `jiter`, `regex`, `pyyaml`, `cffi`, `matplotlib`, `numpy` all ship `cp314-win_amd64`, `cryptography` on `cp311-abi3`. No build toolchain, no resolver directives.

## File layout

```
course8-module3-lab2-v1/
├── helpers/
│   ├── __init__.py
│   ├── config.py             # AnthropicConfig factory, MODEL constant
│   └── metrics.py            # print_usage(UsageReport)
├── smoke_chat.py             # cell 10: two-agent exchange
├── specialised_agents.py     # cell 13: three roles, one question
├── plotting_agent.py         # cell 15: capability-as-tool
├── hitl_agent.py             # cell 18: hitl_hook bug triage
├── group_workflow.py         # cell 20: Hub + TransitionGraph channel
├── tools_agent.py            # cell 22: @tool prime check
├── structured_output.py      # cell 25: response_schema
├── requirements.txt
├── pyproject.toml
├── .env.example
└── .vscode/settings.json
```

---

## Key concepts

### The distribution forked; head-of-line is now ambiguous

`v1.0.0b0` (3 July 2026) promoted `autogen.beta` to top level and removed the classic framework from `ag2ai/ag2`. Classic moved to `ag2ai/ag2-classic` and publishes as `autogen` (0.14.1, maintenance mode, security and critical fixes only). `ag2` 1.0.1 shipped 29 July 2026.

`pip install ag2` → v1.0, `import ag2`. `pip install autogen` → Classic, `import autogen`. Neither is a drop-in for the other; the agent model, orchestration and imports all changed.

Modernisation-over-notebook-parity assumed a single head to port *toward*. Here the line split, so "head of line" needed a decision rather than a lookup. Chose v1.0 because Classic is frozen by design — porting onto a maintenance-mode line is notebook parity wearing a different hat.

### One agent class, composition over hierarchy

Classic's agent taxonomy is gone. Grepped the whole package: zero occurrences of `AssistantAgent` or `UserProxyAgent`. There is `Agent`, and behaviour composes through `config`, `tools`, `hitl_hook`, `response_schema`, `knowledge`, `assembly`.

`UserProxyAgent` split along the seam it was conflating. It was simultaneously "the human's representative" and "the thing that runs code on your machine" — two concerns sharing only the property of not being an LLM. v1.0 separates them into `hitl_hook` and code-execution tools. Cell 15 uses it purely as an executor with `human_input_mode="NEVER"`, which is the documented Classic idiom and reads oddly only in hindsight.

### Conversational code execution was scaffolding, not a pattern

Classic's assistant/proxy loop existed because the model could only emit code as prose, so something had to scrape fenced blocks from the transcript and run them. Modern models emit structured tool calls; the framework executes and feeds results back inside the same loop. The retry behaviour survives without the second participant.

AG2 v1.0 also refuses to run model-written code on a local backend by default — `SandboxCodeTool` requires an explicit Docker or Daytona environment, and the framework's own docstring says local subprocess execution "has no real isolation". `LocalCommandLineCodeExecutor` with `use_docker=False` was the standard Classic pattern; the framework changed its own stance on it.

`plotting_agent.py` goes one step further than swapping in a sandbox: because the task is enumerable, it becomes a typed function. The model chooses arguments, not code, so there is nothing arbitrary left to sandbox. `SandboxCodeTool` is the right reach only when the task genuinely isn't enumerable.

### HITL is a call the tool makes, not a gate the model sees

`ctx.input(message)` emits a `HumanInputRequest` onto the stream and awaits a `HumanMessage`. The `hitl_hook` answers it. `ctx: Context` is injected by fast_depends and excluded from the JSON schema (`CONTEXT_OPTION_NAME = "__ctx__"`), so the model's view of a gated tool has no trace of the approval step.

Default with no hook is `HumanInputNotProvidedError` — fail loud, not auto-approve.

### Routing as data

`TransitionGraph` is a JSON-serialisable object: `initial_speaker`, a list of `Transition(when=..., then=...)`, a `default_target`, a `max_turns`. It survives `to_dict()` / `loads()` and lives in channel metadata, so the topology can be printed, diffed and stored. Classic's `speaker_selection_method="auto"` asks a manager LLM per round, so the topology only exists at runtime and can differ across identical runs.

AG2 v1.0 still offers LLM-chosen ordering via the `discussion` adapter. The point is that declared routing is now available and is the workflow adapter's shape.

### The channel log is the system of record

`hub.read_wal(channel_id)` returns every envelope: invites, per-participant acks, opened, each substantive turn, closed-with-reason. `ChannelState` moves `PENDING → ACTIVE → CLOSED` and that state is *derived* from folding the log, not stored beside it — delete the metadata and it rebuilds from nine envelopes.

Channels also carry governance nobody configured: `turn_within` expectations at 120s (warn) and 600s (auto-close), a two-hour `expires_at`, tracked `pending_acks`, a `named_windowed_summary` view policy. `Envelope` has an `idempotency_key` reserved for cross-process transports and ignored in-process because the hub serialises under a per-channel lock. That is an author who has thought about exactly-once delivery, not someone who added a log for debugging.

Where it stops short of a unified event-sourced substrate: four stores, not one. Channel WAL, `KnowledgeStore` for agent memory, `Resume` for observed capability stats, a separate audit log. Plausibly deliberate — a channel is ephemeral and swept on TTL, cross-session memory is not, and unifying them means either keeping channel noise forever or losing memory when a conversation closes.

---

## Findings

**F-L39-1 — the notebook's install cell no longer resolves to a runnable environment.** Cell 3 is `!pip install ag2[openai]`, unpinned. When the notebook was written that resolved to Classic; since the 3 July 2026 fork it resolves to `ag2` 1.0.1, so cell 4's `from autogen import ConversableAgent, ...` raises `ImportError` and every subsequent cell depends on it. A different failure shape from the stale-pin-with-renames pattern in L34/L38 — the package name stayed valid while its contents were replaced. Unpinned installs are the underlying exposure; the fork is roughly a month old and post-dates the course materials.

**F-L39-2 — AG2's own v1.0 migration guide is written against the pre-promotion namespace.** The group-chat migration guide imports `from autogen.beta.network import ...` throughout; in 1.0.1 it is `ag2.network`. Symbols survived the move, so the fix is mechanical, but `MemoryKnowledgeStore` is in `ag2.knowledge` and needs its own correction. Documentation for a migration *to* v1.0, not updated at the promotion that defined v1.0.

**F-L39-3 — the module's reading and lab document different generations of Classic.** The reading teaches `AutoPattern` + `run_group_chat` + `ReplyResult` + `ContextVariables`. The notebook uses `GroupChat(speaker_selection_method="auto")` + `GroupChatManager` + `initiate_chat` — the pre-`AutoPattern` surface. Both were valid Classic at their respective times; the pairing is worth knowing before reconciling one against the other.

**F-L39-4 — AG2 v1.0's Anthropic client injects `cache_control` and enables prompt caching by default; not observed in this lab.** `AnthropicConfig.prompt_caching: bool = True`, and `anthropic_client.py::_inject_cache_control` writes `{"type": "ephemeral"}` onto message content blocks. Opposite of CrewAI 1.15.2's native provider, which reported `cached_prompt_tokens=0` across L34/L35/L36. **But every run in this lab reported `-` on both cache fields**, including a 4,464-token retry turn. Prompt sizes here (325–4,464) may simply sit under Anthropic's minimum cacheable prefix, or the markers may not be reaching the API. Unresolved — see deferrals.

**F-L39-5 — AG2 v1.0 reports tokens and makes no cost claim.** `Usage` carries `prompt_tokens`, `completion_tokens`, `total_tokens`, `cache_read_input_tokens`, `cache_creation_input_tokens`, `thinking_tokens`. No cost field anywhere on `UsageReport`. A framework that declines to estimate cost cannot mis-estimate it; whether that reads as discipline or gap depends on whether you would rather have a wrong number or no number.

**F-L39-6 — the caller/executor split is gone.** Classic's `register_function(fn, caller=agent_a, executor=agent_b)` separated the agent that *decides* to call a tool from the one that *runs* it — AG2's most distinctive decomposition against CrewAI (agent-owns-tools) and BeeAI (`Tool` subclass). v1.0 has no `register_function`; tools are `@tool` callables on `Agent(tools=[...])`, schemas generated from type hints, DI via `fast_depends`. **The one axis on which AG2 differed has been removed by AG2.**

**F-L39-7 — `@tool` takes its description from the docstring and does NOT fail when it is missing.** `tool()` resolves `description or f.__doc__ or ""`. L37 found CrewAI's decorator hard-fails at decoration on a missing docstring (loud, catchable). AG2 ships an empty description silently, degrading tool selection with no error. Same discipline rule, stronger justification: `@tool` docstrings are exempt from the cut-restatement rule.

**F-L39-8 — `MemoryStream` is one agent's conversation log, not a multi-agent channel.** Events are `ModelRequest` / `ModelResponse` with **no author attribution**. Two agents sharing a stream each inherit the other's output as their own prior turn — the student in `smoke_chat.py` claimed authorship of the tutor's explanation ("the explanation I just gave you"). Not confusion; a correct reading of a log that cannot distinguish speakers. Multi-agent conversation is not available on bare `Agent`; that is what `ag2.network` exists for.

**F-L39-9 — usage is per-stream, so multi-agent cost requires manual aggregation.** No process-level rollup. Worse inside a network channel: agents run inside the plugin glue, so no `AgentReply` surfaces to the caller and there is no per-turn usage to read at all. `group_workflow.py` ships without instrumentation for this reason.

**F-L39-10 — cost ordering: constraining output is cheap, granting capability is expensive.** All measured on Haiku 4.5, same framework, prompt tokens per call:

| Surface | Prompt tokens |
|---|---|
| Role prompt only (`specialised_agents`) | ~56 |
| Response schema, 4 fields (`structured_output`) | 325 |
| One tool, 1 param, 1-line docstring (`tools_agent`) | 1,273 |
| One tool, 4 params + Args block (`plotting_agent`) | 1,717 |

A single tool costs more prompt budget than an entire role description — 20-30× the role-prompt floor. Adding parameters and documentation scales at roughly 100 tokens each. Most editing effort goes into the role prompt, which is the cheapest thing on the list. Sharpens L34's `expected_output` finding: the lever is on the schema surface, not the persona.

**F-L39-11 [open] — persona shaped presentation, not conclusion.** Three roles (tech expert, creative writer, business analyst) given an identical question converged on the same judgement — "probably not, start with a modular monolith, extract later" — differing only in framing and structure. Completion tokens varied 3× (1,303 / 416 / 450), driven by one clause: "with code examples when appropriate". Confounded: the question has a strong consensus answer in training data. Untested on a genuinely contested question.

**F-L39-12 — Classic's agent-type hierarchy collapsed to one class plus composition; `UserProxyAgent`'s two conflated responsibilities separated.** See Key Concepts above.

**F-L39-13 — a constraint stated in a tool docstring is resolved silently; the same constraint enforced only in code is reported to the user.** With `filename: Output file name, a bare name ending in .png` in the Args block, a request for `.jpg` was satisfied as `.png` with no mention to the user — the model never perceived a conflict, it simply complied with the schema. With the hint removed, `_safe_output_path` raised, the error went back to the model, and the model recovered *and* narrated it: "the tool saves files as PNG format rather than JPG, so I saved it as cosine_wave.png instead." Guardrail-in-docstring is cheaper and invisible; guardrail-in-code costs a round trip and is auditable. **The auditable option costs 2.5× the tokens** (1,759 → 4,464 prompt tokens; `records` 2 → 3).

**F-L39-14 — `hitl_hook` closes L38's approval-gate failure mode structurally.** BeeAI's `AskPermissionRequirement` announced the gate through the same scheduler-to-prompt channel that enforced it, so strong models reasoned around the tool and the approval never fired — leaving a reviewer unable to distinguish "not needed" from "silently avoided". AG2 cannot fail that way: the human ask is a line of code in the tool body, invisible in the JSON schema. Confirmed on the same model class that skipped BeeAI's gate. The prompt is also written by the tool author, so it carries real decision context rather than a bare "approve tool call?".

**F-L39-15 — `hitl_hook` gates execution, not decision.** Two escalations, two human prompts, still `records: 2` — both tool calls were emitted in one assistant turn before either human answered. Parallel tool calls with serialised human gates. The reviewer's answer to the first prompt cannot influence what the second asks, because the model committed to its full plan before hearing anything. Batch approval, not iterative oversight. Ordering within the batch is not priority-ordered either.

**F-L39-16 — registering an `Agent` with a hub changes what `agent.ask()` does.** `hub.register(agent)` installs the network plugin (`attach_plugin=True`) which grants delegation tools. A subsequent plain `ask()` on that agent silently opened a *consulting* channel to a peer, obtained a full lesson plan, and returned a summary of it. Five model calls where three were expected, and the delegation lives in a different channel's WAL entirely — invisible from the workflow channel being inspected. No error, plausible output. The same call is a plain model turn before registration and potentially multi-agent after it.

**F-L39-17 — `TransitionGraph.sequence` runs N-1 models for N steps.** `initial_state` sets `expected_next_speaker = graph.initial_speaker`; the seed envelope is folded as that speaker's turn and `select_next` runs against it. So the opening seat is consumed by whatever the caller sends, and its occupant never invokes a model. Consequences observed in order: a three-role graph left `TEACHER_PROMPT` as dead configuration; dropping to two roles left only *one* thinking agent, and the reviewer — handed a bare topic — wrote the plan itself rather than reviewing one. The seat is real and necessary; it belongs to whoever supplies the seed. When that is a person, `register_human` says so honestly: no `Agent`, no plugin, therefore no delegation and no phantom channel (F-L39-16 cannot recur). Classic had no non-LLM participant except `UserProxyAgent`, which is why the notebook dressed a human up as a teacher.

**F-L39-18 — the framework's failure mode is fluent output under a wrong structure.** Four separate findings (F-L39-8, 13, 16, 17) presented as working code with plausible results and no exception. Conversation direction inverted, a phantom channel opened and billed, a user instruction silently overridden, an agent role collapsed. The only things that surfaced any of them were reading the channel WAL and reading the adapter source. Neither type checking nor tests would have caught any of the four.

---

## Prior art carried across

- **L34: native Anthropic provider requires explicit pin.** Held in modified form — `ag2[anthropic]` pulls `anthropic[vertex]>=0.116.0` transitively, so no separate pin, but the extras marker is load-bearing (bare `ag2` installs no provider).
- **L34: `expected_output` is the strongest lever on token spend.** Sharpened by F-L39-10 — the lever sits on the schema/tool surface, an order of magnitude above the role prompt.
- **L36/F-L36-4: `cached_prompt_tokens=0` on CrewAI native Anthropic.** Contrasted at the source level (AG2 injects the markers CrewAI drops) but not confirmed empirically here — see F-L39-4.
- **L36: agent tool-selection is waterfall, not classification.** Held in a new shape. `is_prime` fired even for the trivially-even 72, where L38's Haiku reasoned around Wikipedia across five configurations. Consistent with the underlying mechanism: a local deterministic function has no substitutable value; the model believed it already knew what Wikipedia would say.
- **L37: `@tool` docstring is framework contract at decoration.** Held with a worse failure mode — see F-L39-7.
- **L38: HandoffTool is subroutine semantics.** Sharpened. AG2 v1.0 ships both semantics as distinct primitives: `Agent.as_tool()` for delegation, and `EV_HANDOFF` / `ToolCalled` transitions for genuine state transition. BeeAI conflated them under one name.
- **L38: `AskPermissionRequirement` only fires if the model chooses the gated tool.** Closed structurally on AG2 — see F-L39-14.
- **L38: framework cost APIs are directional, not billing-grade.** Not applicable — AG2 v1.0 exposes no cost surface (F-L39-5).

New to L39:

- Head-of-line can *fork* rather than drift; "modernise against head" needs a rule for which head.
- First-party migration documentation can be stale against its own migration target.
- A framework can remove its own differentiator between minor versions.
- Guardrail placement (docstring vs code) determines whether a constraint is auditable or silent.
- Registration mutates agent behaviour — the same method call means something different afterwards.
- Declared routing as a serialisable artefact, with a derived-state channel log underneath it.

---

## What this doesn't cover

- **Structured-output failure mode.** Every run had a satisfiable schema. Whether `content()` returns `None`, raises `ValidationError`, or elides fields under a schema the model cannot satisfy is untested. Same question left open on CrewAI (L35 found silent `None`) and BeeAI (L38-5). Deferred.
- **Cache measurement.** F-L39-4 is source-confirmed and empirically absent. Needs a deliberately oversized system prompt to establish whether this is a threshold effect or a wiring gap.
- **Harness primitives.** `KnowledgeConfig`, assembly policies, compaction, aggregation. Opt-in and entirely unexercised — a bare `Agent` has zero harness middleware.
- **Observers and telemetry.** `observer` decorator and the Observer API exported and unused; `helpers/metrics.py` reads `UsageReport` post-hoc instead. `BaseHubListener` used for live tracing only.
- **Evaluation.** `ag2.eval` ships agent-as-judge and pairwise judging with threshold scorers. Revises the "AG2 has no native eval story" claim in `agentic-framework-landscape.md`; untested.
- **Connectivity.** MCP (client and server), A2A, ACP, NLIP all ship in `ag2.*` and none are exercised.
- **Conditional routing.** Only `TransitionGraph.sequence` used. `ToolCalled`, `ContextEquals`, custom `TransitionTarget` subclasses and `RoundRobinTarget` untouched — that is L40's territory.
- **Cross-process network.** `LocalLink` only. `WsLink`, federation via Passport + Visa, and `HumanClient` driven by a real UI untested.
- **Multi-agent cost.** `group_workflow.py` has no usage instrumentation because F-L39-9 makes per-agent usage unreachable inside the plugin glue. The multi-agent-earns-its-cost question from L38 stays open on AG2.
- **Streaming.** `AnthropicConfig.streaming` defaults False; `Agent.run()` returns an observable `AgentRun` handle. Not exercised.

---

**Completed:** 3 August 2026