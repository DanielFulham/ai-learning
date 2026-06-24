# Course 7 — Lab 27: LangGraph 101 (V3a — Event-Sourced QA Workflow)

V2's onion-architected QA workflow layered with an append-only event store, per-service event translator, and per-run streaming consumer. SQLite and in-memory event store concretes behind one Protocol. Run-summary projection over the event log. The hallucination from V2 is preserved as the eval-pipeline argument — the event log is what makes it observable as data, not just behaviour. 174 tests pass. Pyright clean on defaults, no `cast`, no `# type: ignore`.

Built as part of the IBM RAG and Agentic AI Professional Certificate — Course 7, Module 1, third commit sequence in the Lab 27 sequence. V1 (canonical-faithful) is in `course7-module1-lab1-v1-canonical/`; V2 (onion-architected) is in `course7-module1-lab1-v2-onion/`. V3b (Auth + sensitive-field policy + SQLite checkpointer) and V3c (Counter + cross-aggregate projection) follow in their own commit sequences.

---

## What It Does

V2's QA workflow with an event-sourced substrate around it:

- **Same two-node QA graph** — `ContextNode` looks up canned context by keyword match; `QANode` invokes the LLM. Linear topology. V2's behaviour preserved, two narrow lifts only.
- **Per-run event log** — every node update becomes a typed domain event written to an append-only store. `QuestionReceived` fires at run start; `ContextRetrieved` (with the matched context, or `None` on a keyword-match miss) fires after `ContextNode`; `AnswerGenerated` or `ModelInvocationFailed` (mutually exclusive) fires after `QANode`. Three events per successful run.
- **Two event store concretes** — `InMemoryEventStore` (list-backed, test default) and `SqliteEventStore` (single append-only table, indexed on `aggregate_id`, ORDER BY rowid for append-order replay) behind `AgentEventStoreInterface`.
- **Per-run streaming consumer** — `EventTranslatingStreamConsumer` is constructed inside `QAAgentService.run()` with a fresh `run_id`. Decorator over an inner `StreamConsumerInterface` (V2's `ConsoleStreamConsumer` for demos, `NullStreamConsumer` for tests). Inner fires first, then translation, then store append. Failures propagate.
- **Per-service translation** — `translate_qa_update` is a pure function from `(node_name, state_delta, run_id, clock)` to `list[QAEvent]`. The service binds its own translator. Auth and Counter translators land with V3b and V3c.
- **One projection** — `summarise_run` derives a `RunSummary` (per-aggregate) from one run's event sequence. `ThreadHistoryProjection` (cross-aggregate) lands in V3c.
- **CLI demo** — `python demo.py [--provider {ollama,openai}] [--persistence {memory,sqlite}] [--db-path PATH] [--quiet]` runs three canned QA questions, prints the live consumer output during each run, then prints the `RunSummary` block for each by replaying events from the store.

---

## Stack

| Component                | Implementation                                                              |
| ------------------------ | --------------------------------------------------------------------------- |
| LLM providers            | `langchain-ollama==1.1.0`, `langchain-openai==1.3.2` (same as V2)            |
| Default provider         | `OllamaChatModelProvider(model_name="llama3.2:latest", temperature=0.0)`     |
| Graph runtime            | `langgraph==1.2.5` (same as V2)                                              |
| LangChain core           | `langchain==1.3.10`, `langchain-core==1.4.8` (same as V2)                    |
| Event store              | `AgentEventStoreInterface` — `InMemoryEventStore` or `SqliteEventStore`     |
| Checkpointer             | `AgentCheckpointerInterface` — `InMemoryCheckpointer` (SQLite lands in V3b)  |
| Stream consumer          | `StreamConsumerInterface` — `EventTranslatingStreamConsumer` wraps inner    |
| Inner consumer concretes | `ConsoleStreamConsumer` (V2-inherited with `_redact`), `NullStreamConsumer`  |
| Translator               | `application/event_translation/qa_translator.py` — pure function            |
| Projection               | `application/projections/run_summary_projection.py` — pure function          |
| Persistence              | Stdlib `sqlite3`, no new runtime dependencies                                |
| Type checker             | `pyright==1.1.402`, defaults, fully green                                    |
| Test surface             | 174 tests            |

---

## Setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
ollama pull llama3.2:latest
ollama serve
```
Run the demo:

```powershell
python demo.py                                # in-memory default, Ollama
python demo.py --persistence sqlite           # writes to ./events.db
python demo.py --persistence sqlite --db-path runs.db --quiet
python demo.py --provider openai              # requires OPENAI_API_KEY in .env

# Flags compose freely:
python demo.py --provider openai --persistence sqlite
python demo.py --provider openai --persistence sqlite --db-path openai-runs.db --quiet
```

Tests and pyright run without external dependencies:

```powershell
pytest                                        # 174 tests
pyright                                       # 0 errors
```

Inspect events from a SQLite run:

```powershell
python -c "import sqlite3; c=sqlite3.connect('events.db'); print('\n'.join(f'{r[0][:8]} {r[1]:25} agg={r[2][:8]} at={r[3]}' for r in c.execute('SELECT event_id, event_type, aggregate_id, occurred_at FROM events ORDER BY rowid')))"
```

---

## File Layout

```
course7-module1-lab1-v3-event-sourcing/
├── demo.py                                       # entry point — argparse dispatch
├── requirements.txt
├── pytest.ini
├── conftest.py
│
├── domain/                                       # pure value objects, no imports out
│   ├── error_info.py                             # ErrorInfo — exception type + message
│   ├── qa_exchange.py                            # QAExchange + error_info field (V3a)
│   ├── state_schemas.py                          # QAState (Auth/Counter land later)
│   ├── agent_checkpoint.py                       # opaque DTO at the checkpointer boundary
│   ├── run_result.py                             # exchange + run_id pair
│   ├── run_summary.py                            # per-run projection result
│   └── events/
│       ├── base.py                               # BaseAgentEvent, kw_only frozen dataclass
│       └── qa_events.py                          # QuestionReceived, ContextRetrieved,
│                                                 # AnswerGenerated, ModelInvocationFailed
│                                                 # + closed QAEvent union
│
├── interfaces/                                   # Protocols infra/application satisfy
│   ├── chat_model_provider_interface.py
│   ├── stream_consumer_interface.py
│   ├── agent_event_store_interface.py            # append + events_for_run
│   └── agent_checkpointer_interface.py           # save + load returning Optional
│
├── infra/                                        # I/O-touching concretes
│   ├── ollama_chat_model_provider.py
│   ├── openai_chat_model_provider.py
│   ├── console_stream_consumer.py                # V2 lift with _redact preserved
│   ├── null_stream_consumer.py                   # one-liner no-op
│   ├── in_memory_event_store.py                  # list-backed, defensive copy
│   ├── in_memory_checkpointer.py                 # dict-backed, last-write-wins
│   └── sqlite_event_store.py                     # append-only single table
│
├── application/                                  # orchestration via interfaces
│   ├── llm_text.py                               # invoke_text — V2 narrowing helper
│   ├── qa_nodes.py                               # V3a behaviour changes
│   ├── graph_builders.py                         # build_qa_graph only (V3a)
│   ├── qa_agent_service.py                       # V3a constructor surface
│   ├── event_translating_stream_consumer.py      # decorator + per-run lifecycle
│   ├── container.py                              # composition root, V3a wiring
│   ├── lab_app.py                                # qa + event_store
│   ├── event_translation/
│   │   └── qa_translator.py                      # pure function
│   ├── projections/
│   │   └── run_summary_projection.py             # pure function
│   └── interfaces/
│       └── qa_agent_service_interface.py
│
└── tests/                                        # mirrors source layout
    ├── domain/...
    ├── infra/...
    ├── application/...
    └── test_demo.py
```

---

## Key Concepts

### Per-Run Streaming Consumer Lifecycle

V2's `ConsoleStreamConsumer` was per-process — one instance constructed in the container, shared across all services. V3a inverts this: `EventTranslatingStreamConsumer` is constructed **inside** `QAAgentService.run()`, on every call, with a fresh `run_id`. The lifecycle shift is deliberate.

Why: V2's consumer printed lines. Lines don't need run identifiers. V3a's consumer writes events. Events do. The `aggregate_id` on every event is the run_id; the translator's clock and event-id generation happen per call. Holding all that state on a process-lifetime singleton conflates concerns; constructing per-run makes the lifecycle visible at the call site.

The container still constructs the **inner** consumer process-lifetime (one `ConsoleStreamConsumer` for the demo). The translating consumer wraps it per run. Decorator pattern with mixed lifetimes — the decorator owns run state; the inner owns process state.

### Translator is the Canonical Source of Event IDs

`event_id` (`uuid4()`) and `occurred_at` (`clock()`) are generated by the translator at the moment of dispatch. The base `BaseAgentEvent` declares them with no defaults — defaulting them to `field(default_factory=uuid4)` would let any code construct an event, which violates the contract that events come from one place. Pyright catches the missing argument at every call site; tests pin the no-default property explicitly.

`aggregate_id` is the run UUID, generated by the service in `.run()`, threaded down to the translating consumer via constructor injection. The pattern: service owns run lifecycle, translator owns event identity, neither owns the other's concern.

### Decorator, Not Fan-Out

`EventTranslatingStreamConsumer` satisfies `StreamConsumerInterface` and wraps an inner `StreamConsumerInterface`. Each `on_update` calls inner first (dev visibility survives translation crashes), then translates, then appends every returned event to the store.

The alternative considered and rejected was service-side fan-out — `service._consumers: list[StreamConsumerInterface]` with explicit dispatch. The decorator wins on three counts: (1) the dispatch loop in `QAAgentService.run()` doesn't change at all, just the variable's type; (2) one cross-cutting concern lives in one concrete; (3) wrapping composes — V3b's `RedactingStreamConsumer` could wrap the translating consumer the same way.

`NullStreamConsumer` is the inner the decorator wraps when console output isn't wanted. One-liner. Keeps `inner: StreamConsumerInterface | None` out of the API.

### Observability Consistency Lift on `context_provider_node`

V2's `context_provider_node` returned `state` unchanged on a keyword-match miss. That depended on LangGraph emitting `updates` chunks for no-op nodes — which it does today, but the contract is implicit. V3a returns an explicit `{"exchange": replace(current, context=None)}` delta on every path. The translator now emits `ContextRetrieved(context=None)` reliably; ThreadHistoryProjection's "context didn't match the question's subject" query in V3c has a stable basis.

The hallucination behaviour itself — keyword-matching "guided project" returns LangGraph context for any subject — is preserved. V3a doesn't try to detect or fix the hallucination. It records it.

### Failure-Precedence Dispatch in the Translator

`QAExchange` carries both `answer` and `error_info`. On a recoverable LLM failure, `qa_node` populates **both** — a user-safe message in `answer` ("I couldn't reach the model right now. Please try again."), and the full diagnostic in `error_info` (`ErrorInfo(exception_type=..., exception_message=...)`).

The translator branches: `error_info is not None` → `ModelInvocationFailed`; else → `AnswerGenerated`. Diagnostic takes precedence over user-safe message at the event log layer. The UI gets the clean text; the event log gets the truth.

V2's `qa_node` did `except Exception: return f"An error occurred: {e}"` — exception text leaked into the user-facing answer, no diagnostic surface anywhere. V3a's narrow catch (`httpx.HTTPError`, `ConnectionError`, `TimeoutError`) separates user-safe-message from log-diagnostic and lets logic bugs propagate.

### Singleton Event Store at the LabApp Surface

The container constructs one `event_store`. The `QAAgentService` holds it. `LabApp` exposes it. The demo reads from it for the projection. One reference end-to-end. When V3b adds the Auth service and V3c adds Counter, both will receive this same reference; that's what makes `ThreadHistoryProjection`'s cross-aggregate query meaningful (it joins across runs of different workflows on a shared store).

Tests pin this explicitly: `app.event_store is qa_service._event_store`.

### Schema Versioning from Day One

Every event carries `schema_version: int = 1`. Stored as a column in SQLite (not a JSON field) so migration logic can read it without parsing payloads. Adding `correlation_id` for V3c's cross-aggregate causation becomes a default bump and a backward-compat read path, not a migration. New event types are new dataclasses under the per-service union — additive, not breaking.

### Open Across Services, Closed Within

`QAEvent = QuestionReceived | ContextRetrieved | AnswerGenerated | ModelInvocationFailed` — closed within the QA service. The translator returns `list[QAEvent]`; pyright + `assert_never` enforces exhaustiveness at the translator boundary via `match`-case.

Across services, the event taxonomy is open: V3b adds `AuthEvent = LoginAttempted | LoginFailed | LoginSucceeded`, V3c adds `CounterEvent = TickAdded | CounterTerminated`. Each service's translator returns its own union; the store accepts `BaseAgentEvent` covariantly via `Sequence[BaseAgentEvent]`.

The shape is "open across services, closed within" — extensible globally, exhaustive locally.

### Two-Stage Narrowing Still the Pattern

V2's two-stage narrowing pattern (one `isinstance(chunk, dict)` for access, one `isinstance(value, DomainType)` for the type) transfers verbatim to V3a. The QA service uses it on the LangGraph stream; the translator uses it on `state_delta` to verify the exchange. No `cast`, no `# type: ignore`. The translator's failure mode is loud — `raise ValueError("missing or malformed exchange")` rather than silent `None`.

---

## Production Insights

**The substrate is what carries the production-readiness story, not the agent.** The output of `python demo.py` without Ollama running showed three QA runs degrading gracefully — `ModelInvocationFailed` events recorded, user-safe messages returned, the projection summarised "Status: failed" cleanly. The agent didn't have to do anything special; the substrate did the work. Production observability is what makes a system operable; V3a demonstrates that the substrate IS the production layer.

**Latency is in the event log for free.** The ContextRetrieved → AnswerGenerated gap is the LLM call duration. RunSummary could derive `duration_ms`; ThreadHistoryProjection in V3c can answer "p95 latency over the last 100 runs" with one SQL query against the event store. The microsecond precision in `occurred_at` was a default of `datetime.now(timezone.utc)` — no engineering needed to land observability data.

**The eval-pipeline argument is now in the artefact.** The third demo question ("What is the best guided project?") produced a fluent, confident, plausible, internally-consistent answer recommending a non-existent LangGraph guided project about bank transfers. Unit tests pin contracts; eval pipelines catch the gap between contract and intent. V3a's contribution: the event log records the gap as data — context retrieved was unrelated to the question's subject. V3c's `ThreadHistoryProjection` can quantify the hallucination rate over time without changing the substrate.

**Per-event JSON payloads stay narrow.** Storing `event_id`, `aggregate_id`, `occurred_at`, `schema_version` as columns (not JSON fields) means the payload column carries only the event's semantic content. `QuestionReceived: {"question": "..."}`. Nine bytes of overhead beyond the question. The column layout is what makes the schema honest: indexing, migration, schema evolution all work without payload parsing.

**Per-run consumer lifecycle is the right inversion.** The V2-to-V3a shift from per-process to per-run consumer is small in code (the service constructs the consumer inside `.run()` instead of receiving it). The conceptual shift is large — process-lifetime singletons can't hold per-call state; constructing per call makes the lifecycle visible. Same pattern would apply to any cross-cutting concern that's per-run, not per-process: tracing spans, request IDs, audit contexts.

---

## What V3a Doesn't Cover (V3a → V3b Findings)

V3b's substrate sits next to V3a's, not on top of it. The Auth workflow comes over from V2 with the sensitive-field policy applied at the translator boundary.

- **Auth translator.** `translate_auth_update(node_name, state_delta, run_id, clock)` — same shape as `translate_qa_update`. `ValidateCredential` → `LoginAttempted` (password field discarded at translation time, not stored). `Success` → `LoginSucceeded`. `Failure` → `LoginFailed`. Loop semantics: each pass through `InputNode` emits its own `LoginAttempted`, preserving the user's full retry journey in the event log.
- **Sensitive-field policy as first-class translator concern.** `LoginAttempted` payload schema omits `password` entirely — the verdict is the diagnostic, not the credential used. V3a's `_redact` helper in `ConsoleStreamConsumer` stays as the dev-console policy. The translator drops at write time; the storage layer never sees cleartext passwords.
- **`prompt_secret` on `InputProviderInterface`.** V2's `ConsoleInputProvider` uses `input()` so password input echoes to the terminal. V3b adds `prompt_secret` (using `getpass.getpass`); the V2 input-echo gap closes alongside the storage-side gap. Both halves of the sensitive-field policy ship in V3b together — splitting them makes V3b's narrative incoherent ("we fixed storage but the input still echoes").
- **SQLite checkpointer concrete.** `SqliteCheckpointer` lands alongside `SqliteEventStore` in the same DB. Auth runs are short enough to make resume-from-checkpoint a meaningful demo. The DI shape is already in place from V3a (`AgentCheckpointerInterface` + `InMemoryCheckpointer`); the SQLite concrete is the second concrete.
- **`LabApp` adds `auth: AuthAgentServiceInterface`.** Container wires the Auth service with the same `event_store` and `inner_consumer` references the QA service holds. The singleton contract becomes load-bearing — V3a tested it for one service; V3b tests it for two.
- **`v2-style demo.py` subcommands.** V3a's `demo.py` runs the three QA questions only. V3b extends to argparse subcommands matching V2 (`qa`, `auth`, `all`). The integrated `all` flow shows Auth + QA event interleaving in one event log.
- **Event log composition assertions land as the V3a/V3b testing pattern.** V3a has unit tests over each unit; V3b adds composition tests that assert against the full event log of a run. Pattern: "after `demo.py all`, the store contains exactly one `LoginSucceeded` followed by three `QuestionReceived`/`AnswerGenerated` pairs." The V2 lab note's two composition bugs (wrong-username-no-recovery, scripted-response-size-mismatch) would have been caught by event log assertions; V3b is the first commit where that test surface exists.

---

## V3a → V3b Findings (Pinned for the Next Commit)

1. **Per-service translator is the boundary that earned its keep.** V3a's QA translator is small (one function, four branches, ~60 lines). V3b's Auth translator will be slightly bigger (three nodes plus loop semantics, sensitive-field handling). The duplication-vs-abstraction call tipped in favour of duplication; that holds. Auth's translator gets its own file, its own union, its own tests.
2. **Sensitive-field policy is a translator concern, not a separate concrete.** V3b applies the policy inline in `translate_auth_update` — password gets discarded at write time. No `RedactingStreamConsumer` shipping in V3a; if a future surface needs separate redaction policy (multi-tenant differing rules), it lands as a decorator then.
3. **`prompt_secret` lands with the translator change, not separately.** Storage-side and input-side gaps close together. Coherent V3b narrative; small enough not to justify two commits.
4. **`SqliteCheckpointer` shares the SQLite database with `SqliteEventStore`.** One DB file, two tables (`events` and `checkpoints`). Container picks the SQLite path once via `use_sqlite_persistence=True` + `db_path`; both concretes use the same file. Demo for resume-from-checkpoint: kill the auth REPL mid-loop, restart with the same `run_id`, resume from saved state.
5. **`LabApp` field additions are additive only.** V3a → V3b: `LabApp(qa, event_store)` → `LabApp(qa, auth, event_store)`. No field removals. V3c then adds `counter`.
6. **`CounterTerminated` should fire explicitly in V3c, not derive from `TickAdded.n == 13`.** The principle: when choosing between "event fired" and "absence of subsequent event signals termination", firing is the more useful shape for projections. The cost is one event per run; the benefit is `ThreadHistoryProjection`'s "longest counter cycle" query becomes a filter, not a groupby with threshold reasoning. Decided in V3a planning; binding for V3c.
7. **The hallucination demo stays preserved across all commits.** V3a recorded it as data; V3b and V3c don't change the hallucination behaviour. V3c's `ThreadHistoryProjection` is the surface that quantifies it — "show me all AnswerGenerated events where the retrieved context didn't match the question's subject" — but the LLM-side detection is a separate eval problem and out of scope until a future lab.

---

## Connection to Subsequent Labs

**Lab 28 (Module 2, Reflection).** Lesson-pool projection is structurally the same shape as `RunSummaryProjection` applied to a different question. The substrate from V3a (event store, projection-as-pure-function) is the firing surface. The Reflexion critic LLM's output becomes a new event type emitted by a Reflexion-specific translator; the lesson pool projects over those events. No new substrate needed.

**Lab 29 (Tweet Reflection).** Same shape as Lab 28. Domain extension only.

**Lab 30 (ReAct).** `ToolCalled` and `ToolReturned` event types are reserved in V3a's schema but have no firing site in Lab 27. Lab 30's tool-calling agent is the natural home. Schema-versioned from day one so adding `mcp_server_name` later is a version bump, not a rewrite.

**Module 3 (Multi-agent).** `correlation_id` field added to `BaseAgentEvent` here — schema version bump from 1 to 2, backward-compat read path for V1 events. Cross-agent traces become joins over event streams keyed by `correlation_id`. The five-layer architecture's row 5 (cross-cutting observability) becomes concrete code via projections over the unified event log.

---

**Started:** 23 June 2026  
**Completed (V3a):** 24 June 2026