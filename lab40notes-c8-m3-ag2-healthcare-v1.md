# Course 8 - Lab 40: AG2 (AutoGen) Multi-Agent Healthcare Chatbot

> Code: [`course8-module3-lab3-v1/`](course8-module3-lab3-v1/)
>
> **Distribution: `ag2==1.0.1` (`import ag2`).** Not AG2 Classic (`autogen`).
> Same target as L39 — see F-L39-1.

Modernised port of IBM Skills Network's C8 M3 AutoMed lab. The notebook builds a four-agent medical consultation on Classic's `GroupChat` / `GroupChatManager`; this port targets v1.0's declared routing over a network channel, on Python 3.14 + Anthropic Haiku 4.5.

The lab's centre of gravity turned out to be a single measured fact: **a declared `TransitionGraph` orders turns and does not carry content.** Every agent turn was caused by the patient's seed envelope, so `pharmacy` never received `diagnosis`'s output and `consultation` received neither. The notebook's stated premise — "the pharmacy agent follows up on the diagnosis" — does not hold, and the output reads like a pipeline anyway.

The exercise (mental-health chatbot) was deliberately not ported. See *What this doesn't cover*.

## Run it

```powershell
py -3.14 -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env             # populate ANTHROPIC_API_KEY
python usage_probe.py --offline         # wiring only, no spend
python consultation_workflow.py         # the lab
```

Pinned:
```
ag2[anthropic]==1.0.1
python-dotenv==1.2.2
```

Python 3.14, full tree resolves to binary wheels on Windows. No resolver directives.

## File layout

```
course8-module3-lab3-v1/
├── helpers/
│   ├── __init__.py
│   ├── config.py             # AnthropicConfig factory, MODEL constant (from L39)
│   ├── routing.py            # record_consultation_status → set_context
│   └── usage.py              # TokenLedger, TokenCounter, per-call input digest
├── usage_probe.py            # session 1: bare vs registered vs registered+tool
├── consultation_workflow.py  # cells 23/25/29 — the AutoMed system
├── repro_graph.py            # offline run of the real graph, scripted tool call
├── repro_handoff.py          # same shape with Handoff routing; causation chain
├── requirements.txt
├── pyproject.toml
├── .env.example
└── .vscode/settings.json
```

---

## Key concepts

### Routing is data, and the data is not the behaviour

`TransitionGraph` is JSON-serialisable: `initial_speaker`, `transitions`, `default_target`, `max_turns`. You can print it, diff it, store it in channel metadata. L39 recorded that as AG2's strongest answer to "what was the topology of that run".

It is still true, and it is narrower than it sounds. The graph governs *who speaks next in this channel*. It does not govern what they receive (F-L40-15), and it does not prevent an agent from leaving the channel entirely (F-L40-12).

### Termination is structural where the notebook made it a prompt instruction

The notebook's `consultation_agent` is told to end its response with `CONSULTATION_COMPLETE`. Nothing reads that string — there is no `is_termination_msg` handler in the notebook — so termination is purely `max_round=5`, which over three round-robin agents is not a whole number of cycles.

v1.0 has somewhere structural to put it: a tool writes `context_vars`, a `ContextEquals` transition reads it, `max_turns` becomes a backstop rather than the mechanism. `max_turns` still exists on `TransitionGraph`, so the notebook's parameter ports directly; what changes is that it stops being the only thing ending the run.

### The patient seat is not an agent

The notebook gives `patient` an `llm_config` it never uses: `initiate_chat` sends a literal string. F-L39-17 established that the opening seat is consumed by whatever the caller sends, so its occupant never invokes a model. `register_human(Passport(name=..., kind="human"))` is the honest occupant — no `Agent`, no plugin, therefore none of F-L40-12's failure mode.

IBM's comment on the `GroupChat` call — "Patient only initiates" — is the same discovery expressed as a workaround, because Classic had no non-LLM participant except `UserProxyAgent`.

### Guardrails as tool distribution rather than prompt text

`set_context`'s docstring: "Loose semantics — any participant may call this regardless of turn order." Whoever holds the tool can end the consultation. In this lab that would let `pharmacy` — an agent whose job is recommending medications — close the session before `consultation` has ruled on escalation.

The adapter enforces nothing. Tool distribution is the entire control, and F-L40-13 makes it nearly free: 88 prompt tokens on the one agent that needs it, zero on the two that don't.

### Instrumentation is not optional on channel work

Three separate diagnostic surfaces were needed and none is on by default: a `BaseHubListener` printing envelopes with elapsed time (without it a stall and a slow turn are indistinguishable — F-L40-11), the channel WAL read after close, and a middleware recording what each model call actually received (F-L40-15 is invisible without it).

---

## Findings

**F-L40-1 — the notebook contains four dead code cells and a narrative its code does not implement.** `client = OpenAI()` is constructed and never used; `code_execution_config = {"use_docker": False}` is a module-level dict never passed to any agent, with a markdown cell explaining at length what it does for code execution that no agent performs; `python-dotenv` is installed and never imported; the exercise solution's `if not response:` fallback is unreachable because `initiate_chat` returns a truthy `ChatResult`. Separately, the prose claims `consultation` "fetches real-time updates from trusted healthcare medical research papers" and that recommendations are "based on the user's health history" — there is no tool, no retrieval, no network call anywhere in the notebook, and `messages=[]` is documented as ensuring no history is retained. The markdown section "Why is GPT-4o Used?" sits above a config specifying `gpt-4`. In a general lab this is sloppiness; in a healthcare lab the narrative-vs-code gap is the same failure mode the module has been documenting, relocated into the courseware.

**F-L40-2 — `max_round` is a message cap, and the notebook's gloss says otherwise.** The markdown states `max_round=5` "limits the conversation to five full cycles through all agents" — that would be fifteen messages. Five messages over three round-robin agents also means the run wraps into a fourth turn, so the "final summary" the lab describes is not final. The exercise's `max_round=3` carries the comment "Ensures the conversation does not stop too early", which states the opposite of what a ceiling does; it happens to work by arithmetic accident.

**F-L40-3 — middleware reaches agents running inside the network plugin; factory-as-accumulator is the shape that works.** F-L39-9 recorded per-agent usage as unreachable inside a channel, inferred from "no `AgentReply` surfaces to the caller". The inference was scoped to the wrong object: `ag2/network/client/handlers.py` drives registered agents through the ordinary public `agent.ask(...)`, so the normal middleware chain runs and `on_llm_call` sits below the reply. The trap is that `Agent._execute` builds a **fresh middleware instance on every `ask()`** — a counter held on the `BaseMiddleware` subclass resets between turns and reports the last turn only, cleanly and without error. The correct shape is the long-lived `MiddlewareFactory` owning the accumulator. Both facts are documented only in a docstring on `_AggregationMiddleware`, an unrelated internal class, and in the source of `MetricsMiddleware` (`ag2/middleware/builtin/metrics.py`), a purpose-built per-agent token counter that ships in `builtin/` and that L39's gap list missed entirely. Source over docs, for the second lab running (cf. F-L39-2).

**F-L40-4 — `ag2.testing.TestConfig` answers wiring questions at zero provider spend, and L39's gap list mis-filed the category.** A scriptable `ModelConfig` returning canned `ModelResponse` / `ToolCallEvent` sequences. Every structural question in this lab — does middleware reach the plugin glue, does the graph terminate, does a tool call land, does the relay work — was answered offline before any provider call. L39's "harness primitives unexercised" line treated `KnowledgeConfig`, assembly and compaction as runtime concerns and therefore did not look for a testing surface. The capability is reusable across the rest of the cert; the miss is the finding.

**F-L40-5 — `Hub.open` registers the default adapters; the `Hub(store)` constructor does not.** `create_channel(manifest_type=WORKFLOW_TYPE)` raises `NotFoundError: no adapter registered for 'workflow'@v1` on a constructor-built hub until `hub.register_adapter(WorkflowAdapter())`. Both entry points are correct. What makes it a trap is that `Hub.open`'s docstring lists the built-ins it registers as "`consulting@v1`, `conversation@v1`, `discussion@v1`" — omitting `workflow`, the fourth one it actually registers. L39 banked that omission as a first-party doc defect; there are now two ways to trip on it.

**F-L40-6 — `ContextEquals` treats a missing key as `None`, and `_select` sorts by `priority` rather than list order.** `state.context_vars.get(self.key) == self.value`, with the docstring stating missing-compares-as-None explicitly, so `ContextEquals(key, None)` fires on turn one against an empty context — before any participant has spoken. Separately, `WorkflowAdapter._select` iterates `sorted(graph.transitions, key=lambda t: t.priority)`, so a transition list written in intended-evaluation order does not evaluate in that order. In this lab the exit conditions carry `priority=0` against pipeline steps at `1`; without that, `consultation`'s `FromSpeaker` transition wins and the graph falls through to `default_target` instead of terminating on the recorded status.

**F-L40-7 — two unrelated surfaces called "context" in the same channel.** `NetworkPlugin` installs a `context` tool that is read-only: `action: Literal["search", "quote"]` over past envelopes. `WorkflowState.context_vars` is routing state that `ContextEquals` reads. They share a name and nothing else. A model given the plugin's `context` tool and asked to record a status has a plausibly-named tool that runs clean and does nothing to the graph. The only write path is `workflow_helpers.set_context`, called from a tool body holding `ChannelInject`.

**F-L40-8 — structured output is unavailable to a channel-registered agent.** Three independent confirmations. `WorkflowAdapter.build_round_envelope` encodes `reply.body or ""` into the packet, so only text reaches the WAL and downstream participants; the validated instance lives on `AgentReply.content()`, which no caller sees inside the plugin glue. `HubClient.register` takes a bare `Agent`, which resolves to `Agent[str]` because `TResult = TypeVar("TResult", default=str)` under PEP 696. And `TResult` is invariant, so `Agent[ConsultationSummary]` is not assignable and no widening helps. The workaround — rendering the Pydantic model's field descriptions into the system prompt — is precisely CrewAI's `output_pydantic` mechanism (L35), minus the post-hoc validation. On this axis AG2's channel path is behind both CrewAI and BeeAI; the bare-`Agent` path L39 exercised is where AG2's real structured output lives. **This relocates L39-2**: the pathological-schema probe must run on a bare `Agent`, because the failure mode being tested does not exist on the channel path.

**F-L40-9 — a tool that mutates channel state is invisible to the model that called it, so the tool's return value is load-bearing prompt content.** `set_context` writes `context_vars` and `WorkflowAdapter.fold` applies it, but fold treats `EV_CONTEXT_SET` as auxiliary and does not re-run `_select` on it — nothing observable changes until the caller's own packet is folded. Returning a bare acknowledgement (`"recorded: escalate"`) left the model with no evidence its write had landed, and it called the tool three times in succession before the run was killed. A prompt asserting "the consultation does not close until you call this" made it worse by stating a causal relationship that is false. The fix is a directive return value: status recorded, do not call again, write your final message now. Same category as F-L39-7's docstring-as-contract, one level further out.

**F-L40-10 — `tool_choice` is absent from `AnthropicConfig` while present on four sibling providers.** `xai`, `mistral`, `zai` and `openai` configs all expose it; the Anthropic config does not, though the Anthropic API supports it. A tool that must fire cannot be required on Haiku through AG2 — enforcement falls back to the system prompt, which is exactly the surface F-L39-13 found gets complied with silently. Untested candidate workaround: `AnthropicConfig.extra_body`.

**F-L40-11 — a turn that raises inside the plugin is trapped and swallowed; the channel stalls rather than fails.** `on_turn_failed`'s own docstring: "The default notify handler traps `agent.ask` and `build_round_envelope` exceptions and emits this event. The channel stays alive; no reply envelope is posted." With no listener registered and no logging configured there is no signal at all, and the channel simply stops advancing until the 600s auto-close expectation fires. From a console that is indistinguishable from a slow provider turn. A `BaseHubListener` printing envelopes with elapsed time is the minimum instrumentation for any channel work. Note `HubListener` is a Protocol: every method must be `async` and parameter names are load-bearing (`reason`, not `error`), and `BaseHubListener`'s own overrides are unannotated so `inspect.signature` tells you nothing.

**F-L40-12 — registration grants capabilities that defeat the declared topology, and the bill is real.** `hub.register(agent)` installs `NetworkPlugin`'s identity-level tools on `agent.tools`: `delegate`, `peers`, `channels`, `tasks`, `context`. The `TransitionGraph` declared `consultation` terminal. Instead of calling the routing tool, `consultation` called `delegate`, opened a consulting channel to a peer mid-turn, and the workflow stalled waiting on an exchange nobody was driving; the orphaned channel's `invite.ack` was later rejected against a closed channel. Measured: 6 model calls and 19,439 prompt tokens for `consultation` against 1 call and ~2,488 for each peer — 80% of the run's spend. With `attach_plugin=False` the workflow runs identically and produces the same correct escalation at **2,141 total prompt tokens against 24,415, an 11.4× reduction**. This is F-L39-16 (registration mutates what `ask()` means) recurring against declared routing: the graph constrains who speaks next in *this* channel and does not constrain an agent from opening another one. `attach_plugin=False` should be the default for any participant that does not genuinely delegate.

**F-L40-13 — first-tool cost is not per-tool cost, and F-L39-10's figure was measured on a bare agent.** Channel participation itself costs ~37 prompt tokens over a bare agent (76 vs 39). The five plugin tools cost ~2,378. Adding a sixth, purpose-built routing tool on top of that baseline costs **88** — against F-L39-10's 1,273 for one one-param tool on a bare agent. Most of that 1,273 was the fixed cost of enabling tool-calling at all, a scope condition not stated when it was recorded. Consequence: marginal tool cost in an already-tooled context is an order of magnitude below the bare-agent figure, which makes tool-distribution guardrails (F-L40-14) nearly free. **This supersedes the session-1 reading** that registration costs ~2,415 — that number was real but was measuring the plugin, not registration.

**F-L40-14 — `set_context` has loose participant semantics, so tool distribution is the only exit control.** "Any participant may call this regardless of turn order." Whoever holds the routing tool can terminate the consultation. Restricting it to `consultation` is the whole guardrail, it is enforced structurally rather than by prompt, it holds regardless of model behaviour, and per F-L40-13 it costs 88 tokens. The healthcare framing is what gives it teeth: without the restriction, `pharmacy` could close the session before the escalation decision was made.

**F-L40-15 — workflow channels route turns and do not relay content.** Measured, not inferred. With distinctive scripted outputs, every agent's model call received only `[patient]: <seed>` and an empty history projection: `pharmacy` never saw `diagnosis`'s output, `consultation` saw neither. The causation chain is decisive — both agents' packets carry `causation_id` pointing at the patient's `ag2.msg.text`, not at the predecessor's packet. It is a fan-out, not a pipeline. **The same result holds for dynamic `Handoff` routing as for static `FromSpeaker`/`AgentTarget`**, so it is not a choice-of-primitive error: `routing=handoff` is correctly recorded on the packet and the next turn is still caused by the seed. What makes this worth recording rather than filing and forgetting is that every piece of the relay machinery verifies correct in isolation — the packet carries the body, `render_envelope` projects `EV_PACKET` via `_packet_text`, `extract_turn_input` handles `EV_PACKET`, and the default `NamedWindowedSummary(recent_n=6)` view projects prior bodies when run directly over a completed WAL. None of it is reached. Bounded to `ag2==1.0.1`, `LocalLink` in-process, `WorkflowAdapter`; reproduced on two machines, offline and live. Consequence for the lab: the notebook's collaboration premise does not hold, and the port pays 3× a single agent's model calls for three independent opinions arranged in a line.

**F-L40-16 — the framework's auditability surface accurately recorded a system that was not doing what its architecture claimed.** `causation_id` threading is the feature `agentic-framework-landscape.md` cites as AG2's answer to "what was the topology of that run". It worked perfectly, and it is what exposed F-L40-15 — the WAL recorded a fan-out faithfully while the rendered transcript read as a collaborating pipeline, complete with a final summary that appeared to weigh what came before. An accurate system of record describing an architecture that isn't delivering is a sharper cautionary shape than a missing log, and a better exhibit than "framework has a bug".

**F-L40-17 — type checking and log reading catch disjoint failure classes.** L39 concluded that reading the WAL and the adapter source were the only things that caught its four fluent-but-wrong findings. On this lab pyright caught four structural errors that ran clean and produced plausible output: a `str`-subclassing enum passed as a bare string (`provider="anthropic"`), an `Agent[ConsultationSummary]` against a registration API typed `Agent[str]`, a sync `HubListener` implementation against an async Protocol, and an unguarded `str | None` causation id. None would have raised; two would have silently mis-typed the run. The WAL caught the behavioural findings and the type checker caught the interface ones, and neither substitutes for the other. Operational corollary: a probe script that is never opened in the editor is never typechecked, and its result is being trusted on the strength of not crashing.

---

## Prior art carried across

- **L39/F-L39-16: registration mutates agent behaviour.** Held and sharpened — now demonstrated against a declared topology with a measured 11.4× cost (F-L40-12).
- **L39/F-L39-17: the seed seat runs no model.** Held exactly. `register_human` is the honest occupant, and IBM's "patient only initiates" is the same discovery as a Classic workaround.
- **L39/F-L39-13: guardrail placement decides auditability.** Applied rather than re-derived — `pharmacy`'s prescription-only constraint is prompt-only and therefore silent; the routing exit is structural (F-L40-14). The lab ships with the asymmetry deliberately visible.
- **L39/F-L39-10: cost ordering.** Corrected in scope (F-L40-13) — the 1,273 figure is first-tool, not per-tool.
- **L39/F-L39-9: per-agent usage unreachable in a channel.** Corrected (F-L40-3). Original claim and the inference that produced it preserved above.
- **L39/F-L39-7: `@tool` description comes from the docstring.** Extended (F-L40-9) — the tool's *return value* is prompt content too.
- **L39/F-L39-18: the failure mode is fluent output under a wrong structure.** Held, and refined by F-L40-17 — the type checker catches a class the WAL cannot see, and vice versa.
- **L35: `output_pydantic` as prompt-embedded schema.** Arrived at independently as the only available mechanism on a registered agent (F-L40-8).
- **L38: does the multi-agent shape earn its cost?** Answerable on AG2 for the first time, via F-L40-3's middleware. Answer for this shape: no — three agents, 3× the calls, and F-L40-15 means they cannot build on each other.

New to L40:

- A declared, serialisable topology can be accurate, inspectable, and not descriptive of what the system does.
- Capabilities granted at registration are not constrained by the topology declared at the same seam.
- Fixed-cost-vs-marginal-cost is a distinction token measurements need and rarely make.
- A framework can trap and swallow turn failures by design, making silence the default failure signal.

---

## What this doesn't cover

- **The mental-health exercise, deliberately.** Its distinctive instruction is that the therapy agent works "only based on the analysis from the Emotion Analysis Agent" — which F-L40-15 says cannot happen. Porting it would re-demonstrate an existing finding at the cost of a second script, and a mental-health chatbot with no crisis path, no escalation, and a broken relay is worse to ship than not to. IBM's own solution contains an unreachable `if not response:` fallback, added because something was not triggering and patched around rather than diagnosed — which is evidence about the courseware worth more than the port would have been.
- **Whether the notebook itself relayed.** Classic's `GroupChat` maintains a shared `messages` list, so on paper it would have. Untested, because the notebook does not run (F-L39-1). "The original worked and the port doesn't" is a plausible assumption, not a finding.
- **Upstream status of F-L40-15.** Not yet checked against `ag2ai/ag2` issues. If unknown, the reproduction is fifteen lines and worth filing.
- **Other adapters and transports.** `discussion` and `conversation` adapters untested; `WsLink` untested. F-L40-15 is bounded to `WorkflowAdapter` over `LocalLink`.
- **Multi-turn channels.** Every turn here descends from a single seed. A channel where the human sends a second `EV_TEXT` mid-run is the most likely case to behave differently and is the next probe worth building.
- **`response_schema` failure modes (L39-2).** Relocated to a bare `Agent` by F-L40-8; still open.
- **Prompt caching (L39-3).** `cache_read` and `cache_write` reported zero across every run including 2,542-token turns. Suggestive of a wiring gap rather than a threshold effect, but Haiku 4.5's current minimum cacheable prefix was not confirmed, and that is the number the inference rests on. Still open.
- **`extra_body` as a `tool_choice` workaround (F-L40-10).** Untested.
- **Eval.** `ag2.eval` still unexercised, two labs running.

---

**Completed:** 3 August 2026