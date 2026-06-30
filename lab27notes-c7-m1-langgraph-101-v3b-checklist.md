# Course 7 - Lab 27: V3b Implementation Checklist (Binding)

> Code: [`course7-module1-lab1-v3-event-sourcing/`](course7-module1-lab1-v3-event-sourcing/)

Binding execution document for V3b. This checklist maps 1:1 to the pinned findings in the V3a note so V3b can be delivered as an auditable commit sequence, independent of chat history.

Source of truth for finding text: lab27notes-c7-m1-langgraph-101-v3a-event-sourced.md, section "V3a -> V3b Findings (Pinned for the Next Commit)".

---

## How to use this checklist

- Keep commit scope narrow: one checklist item (or one tightly coupled pair) per commit.
- Every item must have objective evidence (tests, pyright, note update, or file diff proof).
- If an item is explicitly deferred or out-of-scope, close it by documentation pinning, not by partial code.
- Do not weaken V3a constraints: no cast(), no type: ignore, no behavior-only-through-tests assumptions.

---

## 1:1 Checklist (Pinned Finding -> V3b Action)

## A. Architectural fixes V3b must land

- [ ] F01 - Union-of-services event registry in container
  - Pinned finding: QA-only registry causes write-accept/read-fail once Auth events share the SQLite store.
  - V3b action: replace QA-only wiring with combined registry (QA + Auth) for SqliteEventStore construction.
  - Evidence of done:
    - Container defines combined registry and passes it to SqliteEventStore.
    - Integration test proves Auth event can be appended and read back from same store instance.
    - Existing QA event replay still passes.

- [ ] F02 - Node-name constants owned by graph builder
  - Pinned finding: duplicated node-name literals create hidden runtime coupling.
  - V3b action: export node-name constants from graph builder module(s), import into translator(s).
  - Evidence of done:
    - Translator has no duplicate node-name string literals for owned graph nodes.
    - Rename-safe behavior demonstrated by tests anchored to shared constants.

- [ ] F03 - Sensitive-field policy as explicit codec helper in Auth translator
  - Pinned finding: field-discard policy should be visible and local at translation boundary.
  - V3b action: implement explicit helper (for example, auth payload encoder) invoked by Auth translator path that emits LoginAttempted.
  - Evidence of done:
    - Password is absent from persisted LoginAttempted payload.
    - Helper function is used in translator code path (not ad-hoc inline dict building only).
    - Tests assert policy behavior at event payload boundary.

## B. V3b core deliverables

- [ ] F04 - Keep per-service translator duplication (no premature abstraction)
  - Pinned finding: current branch count does not justify dispatch framework/spec abstraction.
  - V3b action: implement Auth translator in same style as QA translator (pure function, service-local union return), without generic translation framework.
  - Evidence of done:
    - Auth translator exists as service-local module.
    - No TranslationSpec/global dispatcher introduced.

- [ ] F05 - Sensitive-field policy remains translator concern (no new redaction concrete)
  - Pinned finding: policy belongs at translation boundary for V3b.
  - V3b action: do not introduce RedactingStreamConsumer or equivalent policy concrete in V3b.
  - Evidence of done:
    - Policy implemented in translator/codec helper only.
    - No new consumer decorator introduced for this policy.

- [ ] F06 - prompt_secret ships in same commit sequence as Auth translator
  - Pinned finding: input-side and storage-side credential handling must land together.
  - V3b action: add `prompt_secret` as a method on `InputProviderInterface`; `ConsoleInputProvider` uses `getpass.getpass`; `ScriptedInputProvider` adds matching behavior for tests.
  - Evidence of done:
    - `InputProviderInterface` defines `prompt_secret`.
    - `ConsoleInputProvider.prompt_secret` uses `getpass.getpass`.
    - `ScriptedInputProvider` implements `prompt_secret` for test parity.
    - Auth path uses `prompt_secret` for password entry.

- [ ] F07 - SQLite checkpointer shares DB file with SQLite event store
  - Pinned finding: one DB, two tables (events + checkpoints).
  - V3b action: wire SqliteCheckpointer to same db_path selected by container persistence flags.
  - Evidence of done:
    - Container uses one db_path for both SQLite concretes.
    - Test pins checkpoint resume path: save checkpoint -> reconstruct service/container from same db_path -> load returns saved checkpoint.

- [ ] F08 - LabApp grows additively: add auth, keep qa and event_store
  - Pinned finding: additive app surface from V3a to V3b.
  - V3b action: LabApp includes auth service while preserving existing fields/contracts.
  - Evidence of done:
    - Existing V3a consumers compile unchanged where intended.
    - New app surface includes auth and shares singleton event_store reference.

## C. Deferred (must be pinned, not half-implemented)

- [ ] F09 - Read/write event-store interface split deferred to V3c
  - Pinned finding: authority leak acknowledged; fix deferred intentionally.
  - V3b action: do not split interfaces yet; explicitly document defer decision in V3b note.
  - Evidence of done:
    - No partial CQRS split merged in V3b.
    - Defer rationale recorded in V3b notes.

- [ ] F10 - CounterTerminated explicit event deferred to V3c
  - Pinned finding: explicit termination event preferred over inference.
  - V3b action: document as V3c responsibility only.
  - Evidence of done:
    - No counter termination inference logic added in V3b code.
    - V3b note references V3c ownership.

- [ ] F11 - Serializer shape expansion deferred to V3c payload diversity
  - Pinned finding: list/dict/complex union support expansion is a later concern.
  - V3b action: keep V3a serializer scope; add explicit defer note rather than speculative refactor.
  - Evidence of done:
    - No broad serializer rewrite in V3b.
    - Defer note included with trigger condition (payload diversity demand).

## D. Out of scope across V3 series (document boundaries)

- [ ] F12 - Multi-tenancy remains out of scope
  - Pinned finding: no tenant_id, no tenant-scoped read in V3 series.
  - V3b action: keep single-tenant assumption explicit in notes.
  - Evidence of done:
    - V3b docs keep boundary visible; no partial tenant field additions.

- [ ] F13 - Schema migration dispatch remains out of scope until real version bump
  - Pinned finding: schema_version exists; migration dispatch not yet needed.
  - V3b action: avoid speculative migration framework; document trigger for future landing.
  - Evidence of done:
    - No migration dispatcher added in V3b.
    - Notes state expected landing trigger.

- [ ] F14 - SQLite concurrency tuning is production migration path, not V3 work
  - Pinned finding: WAL + busy timeout + retry-on-locked belongs to production hardening path.
  - V3b action: preserve boundary and reference it in V3b docs when discussing persistence.
  - Evidence of done:
    - No behavior change forced into V3b solely for lock tuning.
    - Boundary remains explicitly documented.

## E. Testing and behavior continuity pins

- [ ] F15 - Add event-log composition assertions for integrated flows
  - Pinned finding: composition bugs are caught by full log assertions.
  - V3b action: add integrated tests asserting event sequences for auth+qa runs.
  - Evidence of done:
    - At least one integration test asserts full event sequence semantics, not only unit behavior.
    - Test fails meaningfully if flow order/content regresses.

- [ ] F16 - Preserve hallucination behavior; only improve observability surface
  - Pinned finding: behavior remains; quantification improves in later projections.
  - V3b action: do not alter hallucination behavior in QA path while adding Auth and substrate work.
  - Evidence of done:
    - Existing QA behavior tests remain valid.
    - Notes explicitly state preservation.

- [ ] F17 - Singleton event store contract enforced for two services
  - Pinned finding: V3a tested singleton store sharing for one service; V3b's second service makes it load-bearing for V3c cross-service projections.
  - V3b action: container test asserts QA service, Auth service, and LabApp all hold the same event_store instance.
  - Evidence of done:
    - Test: app.event_store is qa_service._event_store.
    - Test: app.event_store is auth_service._event_store.
    - Test: qa_service._event_store is auth_service._event_store.

---

## Recommended commit slicing for V3b

- [ ] C1: Registry union fix (F01) + tests
- [ ] C2: Shared node-name constants (F02) + translator imports + tests
- [ ] C3: Auth translator + sensitive codec helper (F03), bound by F04 (no premature abstraction) and F05 (no new redaction concrete) + tests
- [ ] C4: prompt_secret and auth input flow (F06) + tests
- [ ] C5: SqliteCheckpointer wiring and shared db_path (F07) + tests/demo note
- [ ] C6: LabApp additive surface (F08) + container wiring + tests
- [ ] C7: Composition assertions for integrated event log (F15)
- [ ] C8: V3b lab note draft with all deferred/out-of-scope/continuity items pinned (F09-F14, F16)

---

## V3b release gate

V3b is done only when all of the following are true:

- [ ] All applicable checklist items above are checked with evidence.
- [ ] Deferred and out-of-scope items are explicitly pinned in V3b notes.
- [ ] pytest passes for the V3b tree.
- [ ] pyright is clean on defaults.
- [ ] Auth-event-on-shared-store round-trip verified: integration test appends a `LoginSucceeded` via Auth service, replays the shared event store, and reads it back as a typed `LoginSucceeded` instance.
- [ ] No V3a behavior drift outside explicitly declared V3b scope.

---

Created: 2026-06-24
Purpose: binding pre-implementation checklist for Lab 27 V3b commit sequence.
