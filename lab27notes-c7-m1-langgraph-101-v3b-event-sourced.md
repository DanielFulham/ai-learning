# Course 7 — Lab 27: LangGraph 101 (V3b — Auth + Sensitive-Field Policy + SQLite Checkpointer)

V3a's event-sourced QA substrate extended with a second service — Auth — sharing the same event store, the same translator pattern, and the same per-run streaming consumer lifecycle. Sensitive-field policy applied at the translator boundary (password discarded at write time, never persisted). `prompt_secret` on the input-provider interface closes the input-side credential gap. `SqliteCheckpointer` lands as the second checkpointer concrete, sharing the SQLite database file with `SqliteEventStore` when persistence is enabled. Composition assertions over the integrated event log pin the cross-service singleton contract that earlier C-prompts only tested per-service. 306 tests pass. Pyright clean on defaults, no `cast`, no `# type: ignore`.

Built as part of the IBM RAG and Agentic AI Professional Certificate — Course 7, Module 1. V3b is the terminal version of this lab; the Counter workflow and cross-aggregate projection were scoped out in favour of moving the certificate forward and landing a clean terminal substrate. The seven-commit V3b sequence is documented in `lab27notes-c7-m1-langgraph-101-v3b-checklist.md` (the binding checklist) and the per-slice methodology audit trail in this lab's gitignored `docs/v3b-c8-deferrals.md`.

---

## What It Does

V3a's QA substrate with a second service on the same substrate:

- **Two services, one store.** `LabApp` exposes `qa`, `auth`, `event_store`, and `checkpointer`. The container constructs the event store once and passes the same reference to both services. Composition assertions pin the singleton contract: `app.event_store is app.qa._event_store is app.auth._event_store`.
- **Auth workflow with loop-on-failure topology.** Three-node graph: `InputNode` reads credentials via the input-provider interface, `ValidateCredentialsNode` checks them, conditional router branches to `SuccessNode` (terminal) or `FailureNode` (loops back to `InputNode`). The failure loop preserves the user's full retry journey in the event log — each pass through `InputNode` emits its own `LoginAttempted`.
- **Per-service translator, no premature abstraction.** `translate_auth_update` mirrors `translate_qa_update`'s shape — pure function from `(node_name, state_delta, run_id, clock)` to `list[AuthEvent]`. The two translators share no code; duplication is the right shape until pattern volume justifies abstraction.
- **Sensitive-field policy at the translator boundary.** `LoginAttempted` payload omits `password` — the verdict is the diagnostic, not the credential used. The `_encode_auth_login_attempted_payload` codec helper makes the field-discard decision explicit and named. Storage layer never sees cleartext passwords.
- **`prompt_secret` on `InputProviderInterface`.** `ConsoleInputProvider.prompt_secret` uses `getpass.getpass`; `ScriptedInputProvider.prompt_secret` matches the scripted behaviour for tests. The V2 input-echo gap closes alongside the storage-side gap — both halves of the sensitive-field policy land together.
- **`SqliteCheckpointer` as the second checkpointer concrete.** Mirrors `SqliteEventStore`'s `db_path` + stdlib `sqlite3` shape. `save` is INSERT OR REPLACE (last-write-wins); `load` returns `None` for unknown run_ids. Shares the SQLite database file with the event store when persistence is enabled: one DB, two tables (`events` + `checkpoints`).
- **CLI demo (QA only).** `python demo.py [--provider {ollama,openai}] [--persistence {memory,sqlite}] [--db-path PATH] [--quiet]` runs three canned QA questions, prints the live consumer output during each run, then prints the `RunSummary` block for each by replaying events from the store. The auth flow is exercised at the test surface (composition assertions) and was confirmed at C8 pre-draft via an off-tree debug script; the polished `qa`/`auth`/`all` subcommand demo predicted by V3a's Finding 6 was not built (see Findings → Terminal-state acceptability).

---

## Stack

| Component                | Implementation                                                              |
| ------------------------ | --------------------------------------------------------------------------- |
| LLM providers            | `langchain-ollama==1.1.0`, `langchain-openai==1.3.2`                         |
| Default provider         | `OllamaChatModelProvider(model_name="llama3.2:latest", temperature=0.0)`     |
| Graph runtime            | `langgraph==1.2.5`                                                           |
| LangChain core           | `langchain==1.3.10`, `langchain-core==1.4.8`                                 |
| Event store              | `AgentEventStoreInterface` — `InMemoryEventStore` or `SqliteEventStore`     |
| Checkpointer             | `AgentCheckpointerInterface` — `InMemoryCheckpointer` or `SqliteCheckpointer` |
| Stream consumer          | `StreamConsumerInterface` — `EventTranslatingStreamConsumer` wraps inner    |
| Inner consumer concretes | `ConsoleStreamConsumer`, `NullStreamConsumer`                                |
| Input provider           | `InputProviderInterface` — `ConsoleInputProvider` or `ScriptedInputProvider` (tests) |
| Translators              | `application/event_translation/qa_translator.py`, `application/event_translation/auth_translator.py` — pure functions |
| Projection               | `application/projections/run_summary_projection.py` — pure function          |
| Persistence              | Stdlib `sqlite3`, no new runtime dependencies                                |
| Type checker             | `pyright==1.1.402`, defaults, fully green                                    |
| Test surface             | 306 tests                                                                    |

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
pytest                                        # 306 tests
pyright                                       # 0 errors
```

Targeted single-file pytest invocations require `PYTHONPATH=tests` so the `_helpers` package resolves:

```powershell
$env:PYTHONPATH = "tests"
pytest tests/application/test_container_event_log_composition.py -q
```

Inspect events from a SQLite run:

```powershell
python -c "import sqlite3; c=sqlite3.connect('events.db'); print('\n'.join(f'{r[0][:8]} {r[1]:25} agg={r[2][:8]} at={r[3]}' for r in c.execute('SELECT event_id, event_type, aggregate_id, occurred_at FROM events ORDER BY rowid')))"
```

---

## File Layout

```
course7-module1-lab1-v3-event-sourcing/
├── demo.py                                       # entry point — QA-only demo (Finding 6)
├── how-to-run.md
├── requirements.txt
├── pytest.ini
├── conftest.py
│
├── domain/                                       # pure value objects, no imports out
│   ├── error_info.py                             # ErrorInfo — exception type + message
│   ├── qa_exchange.py                            # QAExchange + error_info field
│   ├── auth_credentials.py                       # AuthCredentials — username + password + verdict (V3b)
│   ├── auth_result.py                            # AuthResult — credentials + run_id (V3b)
│   ├── state_schemas.py                          # QAState, AuthState
│   ├── agent_checkpoint.py                       # opaque DTO at the checkpointer boundary
│   ├── run_result.py                             # QA exchange + run_id pair
│   ├── run_summary.py                            # per-run projection result
│   └── events/
│       ├── base.py                               # BaseAgentEvent, kw_only frozen dataclass
│       ├── qa_events.py                          # QuestionReceived, ContextRetrieved,
│       │                                         # AnswerGenerated, ModelInvocationFailed
│       └── auth_events.py                        # LoginAttempted, LoginFailed, LoginSucceeded (V3b)
│
├── interfaces/                                   # Protocols infra/application satisfy
│   ├── chat_model_provider_interface.py
│   ├── stream_consumer_interface.py
│   ├── input_provider_interface.py               # prompt + prompt_secret (V3b)
│   ├── agent_event_store_interface.py            # append + events_for_run
│   └── agent_checkpointer_interface.py           # save + load returning Optional
│
├── infra/                                        # I/O-touching concretes
│   ├── ollama_chat_model_provider.py
│   ├── openai_chat_model_provider.py
│   ├── console_stream_consumer.py
│   ├── null_stream_consumer.py
│   ├── console_input_provider.py                 # input() + getpass.getpass (V3b)
│   ├── in_memory_event_store.py
│   ├── in_memory_checkpointer.py
│   ├── sqlite_event_store.py
│   └── sqlite_checkpointer.py                    # shared db_path with event store (V3b)
│
├── application/                                  # orchestration via interfaces
│   ├── llm_text.py                               # invoke_text — narrowing helper
│   ├── qa_nodes.py
│   ├── auth_nodes.py                             # InputNode, ValidateCredentialsNode, etc. (V3b)
│   ├── graph_builders.py                         # build_qa_graph + build_auth_graph (V3b)
│   ├── qa_agent_service.py
│   ├── auth_agent_service.py                     # mirrors QAAgentService shape (V3b)
│   ├── event_translating_stream_consumer.py
│   ├── container.py                              # composition root, V3b wiring
│   ├── lab_app.py                                # qa + auth + event_store + checkpointer
│   ├── event_translation/
│   │   ├── qa_translator.py                      # pure function
│   │   └── auth_translator.py                    # pure function + codec helper (V3b)
│   ├── projections/
│   │   └── run_summary_projection.py             # pure function
│   └── interfaces/
│       ├── qa_agent_service_interface.py
│       └── auth_agent_service_interface.py       # (V3b)
│
└── tests/                                        # mirrors source layout
    ├── _helpers/
    │   └── scripted_input_provider.py            # test-only input provider for auth flow
    ├── domain/...
    ├── infra/...
    ├── application/
    │   ├── ...
    │   ├── test_container_auth_wiring.py         # C6 wiring assertions
    │   ├── test_container_shared_db_path.py      # C5 checkpointer-share-store integration
    │   └── test_container_event_log_composition.py  # C7 composition assertions (F15)
    └── test_demo.py
```

---

## Key Concepts

### Two Services, One Store

The substrate's load-bearing claim is that two services with disjoint event taxonomies can share a single append-only event store, and reads partition cleanly by `aggregate_id`. V3a tested the singleton contract for one service (QA). V3b puts a second service (Auth) on the same store and adds composition assertions that pin the cross-service identity: `app.event_store is app.qa._event_store is app.auth._event_store`, and the transitive `app.qa._event_store is app.auth._event_store`.

Why composition assertions, not just per-service tests: the failure mode of "two services accidentally constructed with two different stores" is invisible at the per-service test surface. Each service's tests pass; the cross-service projection silently sees only one half of the log. The composition test makes the failure mode loud at the test surface.

### Auth Translator and the Codec Helper

`translate_auth_update` is a pure function with the same shape as `translate_qa_update`. Three branches: `ValidateCredentialsNode` emits `LoginAttempted` then either `LoginSucceeded` or `LoginFailed` depending on the verdict; `InputNode` and the success/failure nodes do not emit (the verdict is the event, not the prompt-for-input action).

The sensitive-field policy lands as an explicit codec helper called inside the translator: `_encode_auth_login_attempted_payload(credentials)` returns a payload dict with `username` and `message`, never `password`. The helper is named, not inline — the field-discard decision is visible at the translation boundary, not buried in a dict-construction expression. Same pattern generalises to any future translator with field-policy concerns; for V3b's one site, the helper earns its keep by making the policy auditable.

### `prompt_secret` and the Input-Side Gap

V2's `ConsoleInputProvider` used `input()` for all prompts, including passwords — the credential echoed to the terminal. V3b adds `prompt_secret` as a method on `InputProviderInterface`. `ConsoleInputProvider.prompt_secret` uses `getpass.getpass`; `ScriptedInputProvider.prompt_secret` reads from the same scripted sequence as `prompt` (so tests get parity with production behaviour without a side channel).

The storage-side fix (translator discards `password`) and the input-side fix (`prompt_secret` doesn't echo) ship together. Splitting them would have made the V3b narrative incoherent — "we fixed storage but the input still echoes" is the wrong shape for a sensitive-field-policy lab.

### SQLite Checkpointer, Shared Database File

`SqliteCheckpointer` lands as the second `AgentCheckpointerInterface` concrete. Single `checkpoints` table keyed by `run_id`, `state` as a JSON blob. `save` is INSERT OR REPLACE; `load` returns `None` for unknown run_ids. The DI seam was already in place from V3a (`AgentCheckpointerInterface` + `InMemoryCheckpointer`); the SQLite concrete is the second concrete behind the same interface.

The container picks one `db_path` once via `use_sqlite_persistence=True` + `db_path`; both `SqliteEventStore` and `SqliteCheckpointer` use the same file. Two tables in one database — `events` and `checkpoints` — keyed independently. The shared-file test (`test_container_shared_db_path.py`) pins this: same container instance, same `db_path`, both concretes report the same `_db_path` attribute, the resume round-trip works across container instances.

### Composition-Root Symmetry — Three Injection Seams

V3a's `initialise()` had two injection seams: `event_store` and `chat_model_provider`, each defaulting to `None` and constructed in the function body when not injected. V3b adds three more: `checkpointer`, `inner_consumer`, `input_provider`. Every dependency in the composition root now has the same shape — optional injection parameter, default `None`, production concrete constructed when not injected.

The symmetry matters for testability. A dependency without an injection seam is unmockable from the test surface without patching internals; that's a structural inconsistency at the composition root. Both the checkpointer seam (D6) and the input-provider seam (D12) were caught at C5/C6 pre-draft because the C-prompts initially scoped them as constructor-only wiring. The methodology-retrospective section below treats both halts as positive signals — the discipline working.

### Per-Service Translator Pattern, Still Not Abstracted

V3a's QA translator has two dispatch branches; V3b's Auth translator has three. Five branches across two services is not enough variance to justify a `TranslationSpec` abstraction or a global dispatch table. Duplication wins; the cost of the abstraction would exceed the savings, and the test surface for each translator stays focused on its service's domain.

### Loop-on-Failure Topology

The Auth graph's failure branch loops back: `FailureNode → InputNode`, no edge from `FailureNode` to `END`. Any terminating run that includes a failure has the minimal four-event shape `[LoginAttempted, LoginFailed, LoginAttempted, LoginSucceeded]` — the second `LoginAttempted` carries the retry credentials (the `FailureNode` clears the previous credentials so the new prompt-for-input reads fresh). This was caught at C7 v1's pre-execution review (D15): the v1 prompt's two-event `[LoginAttempted, LoginFailed]` assertion would have required mocking the graph, deviating from C6's real-graph pattern and silently losing the loop-on-failure property. The v2 prompt's four-event assertion captures the architectural truth.

### Singleton Event Store, Now Load-Bearing

V3a tested singleton-store sharing for one service; V3b makes it load-bearing for the composition assertions. The test surface pins three identities: `app.event_store is qa_service._event_store`, `app.event_store is auth_service._event_store`, and `qa_service._event_store is auth_service._event_store`. The third is the transitive — without it, two services holding the "same" reference via two paths through `LabApp` could still drift (a hypothetical bug where the LabApp constructor copies the store into one field and aliases it into another). The transitive test makes the no-drift property explicit.

### Open Across Services, Closed Within (Two Services Now)

The pattern V3a introduced — closed unions within a service (`QAEvent = QuestionReceived | ContextRetrieved | AnswerGenerated | ModelInvocationFailed`), open across services — now carries two services. `AuthEvent = LoginAttempted | LoginFailed | LoginSucceeded` is closed at the Auth boundary; the translator returns `list[AuthEvent]`; `match` + `typing.assert_never` enforces exhaustiveness. The store accepts `BaseAgentEvent` covariantly via `Sequence[BaseAgentEvent]`.

Adding a service (the Counter case the V3 series scoped out, or any future service in a downstream lab) is additive — a new union, a new translator, a new entry in the container's combined event-type registry. The pattern scales without rewriting earlier services.

---

## Production Insights

**The substrate carries production-readiness, not the services.** The same observation V3a made about the QA service holds for both services in V3b. Auth runs degrade gracefully under failure (the loop-on-failure topology preserves the user's retry journey in the event log); QA runs degrade gracefully under model unavailability (the failure-precedence dispatch from V3a still routes diagnostic to `ModelInvocationFailed` and user-safe text to `answer`). Neither service does anything special; the substrate does the work. Production observability is what makes a system operable across multiple services on a shared substrate.

**Sensitive-field policy at the translator boundary, not at the consumer.** V3a's `ConsoleStreamConsumer._redact` was the dev-console policy — it kept passwords out of the streamed line output for demo purposes. V3b's translator discards `password` from `LoginAttempted` payloads — the storage layer never sees cleartext. The two policies coexist because they protect different surfaces (console-stream readers vs the event log); they don't substitute for each other. The translator's drop is the load-bearing fix because the event log is the persistent artefact.

**Three injection seams, one composition pattern.** `checkpointer`, `inner_consumer`, `input_provider` all gain optional injection seams in V3b. The pattern is the same: parameter defaults to `None`, function body constructs the production concrete when nothing was injected, tests pass spec'd mocks or test-only concretes (`ScriptedInputProvider`). Composition-root symmetry is a testability property — the failure mode of a missing seam is "this dependency is unmockable without patching," which the C5 and C6 pre-draft confirmations both caught (D6, D12).

**Composition assertions are the cross-service integrity check.** Per-service unit tests cannot catch "two services accidentally use two stores"; only a test that holds both services and asserts on the shared reference can. C7's `test_container_event_log_composition.py` pins this for V3b's two services; the same pattern scales to N services — one assertion per pairwise identity, one transitive across all pairs. Cheaper than every cross-service projection re-deriving the shared-store assumption at runtime.

**The C8 debug-script exercise was the substrate's smoke test.** During C8 pre-draft, a throwaway script (`debug_auth_qa.py`, created off-tree and deleted after one run) wired the container with a mocked QA graph and a `ScriptedInputProvider` carrying a failure-then-retry credential sequence. The run output: auth events `[LoginAttempted(wrong_user), LoginFailed(wrong_user), LoginAttempted(test_user), LoginSucceeded(test_user)]`, then QA events `[QuestionReceived, ContextRetrieved, AnswerGenerated]`, all seven in one event store, `events_for_run` partitioning cleanly by run_id. The composition assertions in C7 pin this at the test surface; the debug script confirmed it at the runtime surface. The substrate is correct on both surfaces.

**Targeted pytest needs `PYTHONPATH=tests`.** The root `conftest.py` inserts the repo root into `sys.path` but not the `tests/` directory; the `_helpers` package only resolves during full-suite runs because pytest's collection happens to import a helper test file. Targeted single-file runs (`pytest tests/application/<file>.py`) fail without `PYTHONPATH=tests` prepended. Logged as D18 and accepted-as-pinned for V3b; a one-line `conftest.py` fix would close it in a standalone slice. The methodology lesson: tooling micro-issues that surface during execution belong in the deferrals doc, not in the implementation slice that hit them.

---

## V3b Methodology Retrospective

V3b shipped as seven C-slices using the binding-prompt sequence methodology documented upstream in `DELIVERY_METHODOLOGY.md`. The retrospective covers the seven slices, the two pre-execution halts, the pyright-gate catch, the preview-layer-unreliability finding, and the operating principles that surfaced across the sequence. The methodology lessons referenced below are upstream — the canonical doc lives in a separate repo, and the V3b lab does not edit it. This section is the receipts.

### Seven C-slices in six commits

The V3b checklist recommended eight C-slices (C1 through C8). The actual commit log compresses C1 and C2 into one commit (`449d1c18 C1+C2: Auth event types, registry union, shared QA node-name constants`), closing F01 and F02 together. The remaining slices each ship as their own commit:

- `449d1c18` — C1+C2: Auth event types, registry union, shared QA node-name constants (F01, F02)
- `80855f92` — C3: Auth workflow lifted onto V3b substrate (F03, F04, F05)
- `ab7133cb` — add `prompt_secret()` to `InputProviderInterface` for password input (closes F06; subject line did not follow the `Cn:` prefix convention)
- `b00fee57` — C5: SqliteCheckpointer + shared db_path + injection seam (F07)
- `2e93cc7e` — C6: AuthAgentService + input-provider seam + LabApp surface (F08, F17)
- `56b9d20f` — C7: composition assertions for integrated event log (F15)
- (this commit) — C8: V3b terminal documentation pass

C1+C2 combined because the registry-union fix (F01) and the node-name constants (F02) shared the same file surface and the same review pass; splitting them would have meant two commits with no independent verification gain. C4's subject line drift is a one-off; subsequent commits restored the `Cn: <description> (F-numbers)` shape.

### Two pre-execution halts (C5 and C7)

**C5 halt:** the original C5 prompt scoped F07 as wiring-only — "wire `SqliteCheckpointer` to the same `db_path`." The pre-draft confirmations surfaced that no `SqliteCheckpointer` concrete existed yet. F07 was actually a build + wire + expose slice (D5), and the composition root needed a `checkpointer` injection seam for symmetry with the other dependencies (D6). The prompt was revised twice before any code was written. The methodology cost: roughly 60 minutes of planning agent time across the revisions. The methodology saving: no halt-and-recover during execution, no half-built slice merged.

**C6 pre-execution review catch:** the original C6 prompt framed `AuthAgentService` as "mirrors `QAAgentService` exactly." The pre-execution review caught two halt-worthy findings before any code was written. First, the prompt's "mirror" framing broke for the `run()` method — QA's `RunResult` is typed with the QA-specific `QAExchange`, which Auth cannot return; the prompt punted the decision to execution time. Second, seven auth-flow files were missing from the required-reading list despite the executing agent needing all of them to write `run()` correctly. The revisions added an `AuthResult` domain type, a binding `run()` contract decision, and the seven required-reading entries. The re-review caught a regression where a new test was instructed to "mirror `test_run_result.py`" without adding that file to required reading — the same failure class as the original finding, reintroduced by the fix. A one-line addition closed it.

The pre-execution review pattern was banked from this slice. Without it, C6 would have halted twice during execution at the same points the review caught beforehand. The overhead (~20 minutes of agent time) saved roughly 60–90 minutes of execution-time halt-and-recover work.

**C6 also surfaced D14 — the planning agent's instructed-to-do-the-wrong-thing failure mode.** The C6 prompt included an instruction to update `LabApp`'s docstring narrative as part of the field-additive slice. Updating a docstring's narrative prose is a documentation-pass concern; the methodology assigns it to the terminal commit (C8), not to an implementation slice. The human caught it at execution time and reverted the docstring change to import-and-field only. The lesson: future C-prompt drafts must explicitly exclude doc-prose tidies from non-terminal slices. The mirror image of the receipts already banked in the upstream methodology doc — there the prompt was *missing* a needed instruction; here the prompt *contained* an instruction it should not have.

**C7 v1 halt:** the original C7 prompt specified the auth-failure composition test as asserting a two-event sequence `[LoginAttempted, LoginFailed]`. The auth graph's failure branch loops via `FailureNode → InputNode` with no edge to `END` from the failure side; any terminating auth run that includes a failure has the minimal four-event shape `[LoginAttempted, LoginFailed, LoginAttempted, LoginSucceeded]`. A two-event assertion would have required mocking the graph, deviating from C6's real-graph pattern, and silently losing the loop-on-failure property. The planning agent had not pre-read `graph_builders.py` for topology — only for node-name constants. The executing agent caught it at the pre-execution boundary and halted; v2 reframed the test as four-event and captured the architectural truth.

Lesson banked upstream: sibling-method tests that diverge from their happy-path counterpart on the *control-flow path* need topology pre-confirmation, not just signature pre-confirmation. Pre-draft confirmations check existence and shape; pre-execution reviews check assumption-against-reality. Different checkpoints catch different failure modes.

### Pyright-gate catch at C7 (D17)

C7's cross-aggregate test included `app.event_store is app.qa._event_store` — a direct attribute access through a Protocol-typed field. `LabApp.qa` is typed `QAAgentServiceInterface`; the Protocol does not declare `_event_store` (a private implementation detail of the concrete `QAAgentService`). Pyright fired `reportAttributeAccessIssue`. The sibling test in `test_container_auth_wiring.py` (`TestContainerF17Singleton`) resolved this by `isinstance`-narrowing to the concrete classes before the identity assertions; the C7 v2 prompt instructed the executing agent to read that sibling test during required reading but omitted both the narrowing step and the transitive identity assertion.

Root cause: the planning agent wrote the assertion code blocks for shape rather than type-checking them mentally or via tooling against the actual interface surface they reference. The pre-execution review read the assertions for shape correctness without running them through the type-checker the codebase enforces.

Lesson banked upstream: the planning agent must type-check (mentally or via tooling) the prompt's example code blocks against the actual interface surface they reference, not just write them for shape. Sibling-test patterns must be read with the assertion-shape implication in mind during pre-draft confirmation. The catch came at the pyright gate (step 11 of C7 v2's order of operations) — late enough to be unsatisfying, early enough that no incorrect test shipped.

### Preview-layer unreliability finding (D19)

Across C7's execution session, the Claude Code preview layer rendered duplicated content in at least six distinct previews — three `Create file` / `Edit file` diff renders and three bash heredoc body displays. Each occurrence showed a different paragraph or two-line block duplicated; sometimes the line-number column restarted mid-block; sometimes blank lines between paragraphs were also hidden in the same render. The duplication was non-deterministic across previews of similar content — different paragraphs duplicated in each render, ruling out content-construction corruption as the cause.

On-disk content was clean in every case. The verification was direct: `grep -c '^(def |class )'` on both test files returned all 1s (no duplicate definitions); pyright clean (`reportRedeclaration` would have fired on real duplicates); `cat -A` on the commit-message scratchpad showed preserved blank lines; sentinel `grep -c` on seven unique phrases in the committed message returned all 1s; the committed git object read clean prose end to end.

The retroactive question — "were the test-file preview duplications real but silently shadowed by Python's last-definition-wins semantics?" — was conclusively answered by pyright plus the grep counts: no duplicates landed.

Cost during C7: roughly 30 minutes across six halt-and-verify cycles, each a per-edit approval round trip. Cost would have been higher if the test files had silently shipped duplicates that pytest tolerated (Python shadows rather than errors on duplicate definitions at module or class scope; pytest would have collected the second definition and the assertions would have passed on the surviving one).

Operating principle for C8+ (and banked upstream): trust on-disk verification (`grep`, `cat -A`, import-check, pyright, git-object reads) as the load-bearing check. Preview-based per-edit approval is unreliable for any content where prose correctness or structural duplication matters. C8's own verification gate enforces this — sentinel `grep -c` counts on unique phrases in every prose artefact this commit touches.

### Accepted-as-pinned at V3b terminal

Four items ship V3b terminal without closure, documented as accepted-as-pinned in D20 of the deferrals doc:

- **D7** — DRY refactor of `initialise()`'s `db_path` guards. Two duplicated guards, working behaviour, code-aesthetic drift. Single-author lab, failure mode is over-validation not under-validation.
- **D10** — `test_checkpointer_field_accessible` symmetry test missing from `test_lab_app.py`. Field coverage is load-bearing in `test_container_shared_db_path.py`; the missing test is readability symmetry, not coverage.
- **D18** — `conftest.py` does not insert `tests/` into `sys.path`. Full-suite runs work; targeted runs need `PYTHONPATH=tests`. Documented in the Setup section of this note.
- **demo.py subcommand gap (V3a→V3b Finding 6)** — V3a predicted V3b would extend `demo.py` to `qa`/`auth`/`all` subcommands. The seven-commit sequence did not scope this. The composition assertions in C7 and the C8 pre-draft debug-script exercise both confirmed substrate correctness; the polished demo surface is future production work, not a V3b-completeness requirement.

The methodology lesson under all four: "accepted-as-pinned" is a legitimate terminal-state outcome. V3b does not need to be flawless to ship; it needs to be honestly documented. Future-me reading the deferrals doc and this lab note sees the deliberate non-closure decisions and the reasoning, not a hidden gap.

### Operating principles surfaced across the sequence

Three operating principles became explicit across V3b's seven commits, all banked upstream in the methodology doc but worth restating in the V3b receipts:

1. **Pre-draft confirmation needs topology reading, not just signature reading.** D15's root cause was a planning agent that pre-read node-name constants without pre-reading the graph's control-flow shape. Signature confirmation catches existence and shape drift; topology confirmation catches assumption drift on the control-flow path. Both checkpoints, not one.

2. **Planning agents must type-check example code blocks against the actual interface surface they reference.** D17's root cause was prompt code blocks written for shape rather than for type correctness. Pyright is part of the verification gate; the prompt's example assertions are part of the prompt; failing to run the latter through the former before drafting is a known failure mode.

3. **Prose deliverables need on-disk verification, not preview-based review.** D19's root cause was the preview layer rendering duplicated content non-deterministically. On-disk verification by sentinel grep is the load-bearing check for any prose artefact; per-edit preview approval is unreliable for prose. C8's verification gate is built around this principle from the start.

---

## Connection to Production Work

V3b's substrate is courseware-built but production-shaped. The patterns transfer directly to two production workstreams already in flight and one architectural pattern carried across:

**Hooperman (production RAG chatbot, Shamrock Rovers FC, live on Lambda).** Hooperman currently writes evaluation results to S3 as flat JSON files — a passable shape for golden-question evaluation but not for runtime observability. V3b's substrate would land as a per-request event log over the same questions Hooperman already answers in production: `QuestionReceived` (the user's question), `ContextRetrieved` (the documents the retriever returned, with their similarity scores), `AnswerGenerated` (the LLM's response). The composition pattern — one event store, multiple service runs appending to it, `events_for_run` partitioning by request — gives Hooperman the latency-observability story V3a demonstrated (microsecond-precision `occurred_at`, p95/p99 latency as a SQL query against the store) without instrumentation surface beyond what V3b already ships.

**Agent observability infrastructure (side project, in flight).** A side project sits at the same architectural seam V3b's substrate occupies: an event log around an agent's runtime, not part of the agent itself. The work explores reproducing V3b's translator + per-run streaming consumer shape at one language layer and the event store + projection pattern at another, applied to production-grade workloads. V3b is the courseware-scale rehearsal of the patterns. The `EventTranslatingStreamConsumer` decorator-over-inner pattern in particular maps directly to the "instrument any agent without modifying it" claim the side project is built around.

**Supporter 360 (fan CDP with text-to-SQL agent).** Supporter 360 will run a SQL agent against fan data with an exec-team-facing interface. The defence-in-depth pattern V3a/V3b carries — read-only DB URI + system-prompt restrictions + LLM-disposition layer — applies directly. Adding an event log around the SQL agent (which V3b's substrate would supply) gives the exec team an audit trail: every question, every generated query, every result row count, every error. The same projection pattern (RunSummary per question, an aggregate projection over all questions in a time window) is what makes the audit trail readable. Not built yet; V3b is the reference shape.

The architectural pattern carried across all three is the same: event sourcing as observability, not as system-of-record. The agent owns its state; the substrate listens and writes a parallel log. The substrate is what makes the system operable in production; the agent is what answers questions. Two concerns, two surfaces, one shared store.

---

**Started:** 23 June 2026
**Completed (V3a):** 24 June 2026
**Completed (V3b):** 29 June 2026
