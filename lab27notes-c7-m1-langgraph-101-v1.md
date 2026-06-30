# Course 7 — Lab 27: LangGraph 101: Building Stateful AI Workflows (V1 — Canonical)

> Code: [`course7-module1-lab1-v1-canonical/`](course7-module1-lab1-v1-canonical/)

Canonical IBM lab translated faithfully to Python script form, run locally on Ollama. Three worked examples — Auth (conditional + loop), QA (linear + LLM), Counter (cyclic + termination) — exercising every primitive in `langgraph==1.2.5`'s `StateGraph`. No tests, no interfaces, no architectural lift. V1 stays close to the IBM shape so the canonical-vs-current API gap is visible. V2 introduces the onion structure; V3 introduces the event-sourced layer.

Built as part of the IBM RAG and Agentic AI Professional Certificate — Course 7, Module 1, the first lab in the LangGraph track.

---

## What It Does

Three workflows in two files:

- **Auth (`app.py`)** — five nodes (`input_node`, `validate_credentials_node`, `success_node`, `failure_node`, `router`), conditional edges via a router function, plain edge loop-back from failure to entry. Demonstrates conditional routing and cyclic flow.
- **QA (`app.py`)** — three nodes (`input_validation_node`, `context_provider_node`, `llm_qa_node`), linear flow, single LLM call inside a `try/except`. Demonstrates linear flow with an LLM node.
- **Counter (`counter.py`)** — two nodes (`add_node`, `print_out_node`) plus a `stop_condition` router, cyclic with `add` → `print_out` → conditional edge back to `add` or to `END`. Demonstrates termination conditions and `dict[bool, str]` path_map.

Three different graph topologies behind the same `StateGraph` primitives. State is `TypedDict(total=False)` in all three.

---

## Stack

| Component             | Implementation                                                     |
| --------------------- | ------------------------------------------------------------------ |
| LLM                   | `llama3.2:latest` (3B, q4) via `langchain-ollama==1.1.0`, `temperature=0` |
| Graph runtime         | `langgraph==1.2.5` (lab pinned to `0.2.57` — pre-1.0)              |
| LangChain core        | `langchain==1.3.10`, `langchain-core==1.4.8`                       |
| State schema          | `TypedDict(total=False)` — partial-init friendly                   |
| Node contract         | `def node(state) -> dict` returning partial updates                |
| Routing               | `add_conditional_edges(node, router_fn, path_map)`                 |
| Entry point           | `set_entry_point("Name")` (deprecated, still works in 1.x)         |
| Architecture          | None — V1 is the canonical script before architectural lift        |
| Test surface          | None — V1 ships without tests                                      |

---

## Setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
ollama pull llama3.2:latest   # if not already local
ollama serve                  # or run the Ollama app
```

Run each workflow:

```powershell
python app.py        # runs Auth + QA (Auth prompts interactively)
python counter.py    # runs Counter, 13-iteration cycle
```

No `.env` required (no API keys). Lab runs fully local.

---

## File Layout

```
course7-module1-lab1-v1-canonical/
├── app.py                                        # Auth workflow + QA workflow
├── counter.py                                    # Counter exercise (cyclic + termination)
├── requirements.txt
```

Two-file split is intentional. Auth and QA share the same script because they're both walked-through worked examples in the IBM lab. Counter is presented as a stub-and-fill exercise in the lab — separate file matches the pedagogical break. V2 will refactor all three into one onion-structured package regardless.

---

## Key Concepts

### StateGraph and the Node Contract

LangGraph's primitive is `StateGraph(SchemaType)` — a state machine where state is a typed dict and nodes are pure-ish functions that take the full state in and return a *partial* dict of what they updated. The framework merges the partial dict into the bag on the way out.

```python
class AuthState(TypedDict, total=False):
    username: Optional[str]
    password: Optional[str]
    is_authenticated: Optional[bool]
    output: Optional[str]


def validate_credentials_node(state):
    username = state.get("username", "")
    password = state.get("password", "")
    if username == "test_user" and password == "secure_password":
        is_authenticated = True
    else:
        is_authenticated = False
    return {"is_authenticated": is_authenticated}
```

Two contracts encoded above. **State**: a flat `TypedDict` where every field is optional (`total=False`) because the bag fills progressively. **Node**: takes the whole bag, returns only the keys it changed. Default merge is last-write-wins per field; no reducer required.

### Three Topologies, One Primitive Set

| Workflow | Topology              | Primitives used                                            |
| -------- | --------------------- | ---------------------------------------------------------- |
| Auth     | Conditional + loop    | `add_edge`, `add_conditional_edges`, loop-back `add_edge` to entry |
| QA       | Linear + LLM          | `add_edge` only                                            |
| Counter  | Cyclic + termination  | `add_edge`, `add_conditional_edges` with `bool` path_map   |

All three compile to the same Pregel-shaped execution model. The graph topology is what differs — primitives are identical.

### Conditional Routing via Path Map

`add_conditional_edges(source, router_fn, path_map)` runs `router_fn(state)` after `source` completes, looks up the return value in `path_map`, and dispatches to that destination. The Auth lab uses this with an indirection layer:

```python
def router(state):
    if state['is_authenticated']:
        return "success_node"
    else:
        return "failure_node"

workflow.add_conditional_edges(
    "ValidateCredential", router,
    {"success_node": "Success", "failure_node": "Failure"},
)
```

The router returns lowercase keys; the path_map translates to capitalised node names. Identity mapping would also work — the indirection is a lab artefact, not a meaningful pattern.

### Loop-back as a Plain Edge

```python
workflow.add_edge("Failure", "InputNode")
```

Cycles in LangGraph are not special. A loop-back is just a normal edge whose target happens to be an earlier node. The state bag carries forward across the cycle — verified by the Auth trace, where the second `InputNode` invocation skips the username prompt because the bag already contains `username`.

### Termination Conditions

The Counter exercise terminates on `n >= 13` via a `bool`-returning router with a `dict[bool, str]` path_map:

```python
def stop_condition(state: ChainState) -> bool:
    return state.get("n", 0) >= 13

workflow.add_conditional_edges(
    "PrintOutNode", stop_condition,
    {True: END, False: "AddNode"},
)
```

`END` is the framework's terminal sentinel. Returning `True` from the router routes to `END`; returning `False` routes back to `AddNode` for another iteration.

---

## Canonical vs Current — API Drift

The IBM lab pins `langgraph==0.2.57` (pre-1.0). V1 ran on `1.2.5` (released after LangGraph 1.0 GA in October 2025). Drift observed in this lab:

**`set_entry_point` is deprecated but still works.** The 1.x StateGraph docstring describes it as "equivalent to calling `add_edge(START, key)`." V1 keeps the lab's `set_entry_point` call. V2 switches to `add_edge(START, ...)`.

**`add_conditional_edges` path_map shape unchanged.** Both `{"success_node": "Success", ...}` (Auth) and `{True: END, False: "AddNode"}` (Counter) work as written. The `path_map=` keyword is now formally named but positional calls still work.

**`END` import path unchanged.** `from langgraph.graph import StateGraph, END` works identically. V2 adds `START` to the same import.

**`langchain-ibm==0.3.10` dropped entirely.** The IBM lab's ChatWatsonx call uses an unauthenticated Skills Network endpoint that only resolves inside their environment. Local `langchain-ollama==1.1.0` is the equivalent. The swap is one line in `app.py`.

**Pyright tension with `total=True`.** The IBM lab declares state TypedDicts without `total=False`. This fails pyright on `app.invoke({"username": "test_user"})` because the dict literal is missing three required fields. `total=False` is correct for LangGraph state and matches current 1.x convention.

**`dict[bool, str]` path_map trips pyright.** Known issue in the type stubs (`add_conditional_edges`'s TypeVar mis-narrows on non-string keys). Works at runtime. V2 changes the Counter's router to return string literals and uses a `dict[str, str]` path_map.

**`BaseMessage.content` is typed `str | list[str | dict]`.** Multimodal responses can be lists; text-only responses are always `str` but pyright can't narrow without a runtime check. V1 uses an `isinstance` guard at the call site:

```python
response = llm.invoke(prompt)
if not isinstance(response.content, str):
    raise TypeError(f"Expected str, got {type(response.content).__name__}")
return {"answer": response.content.strip()}
```

V2 owns the narrowing inside the `ChatModelProviderInterface` adapter so callers see `str`.

---

## Architecture Walkthrough — Auth End-to-End

When `python app.py` runs the Auth workflow with `inputs = {"username": "test_user"}`:

```
app.py
  app.invoke({"username": "test_user"})
    → InputNode                                                 [no layer split in V1]
      → state.get('username') == "test_user", skips username prompt
      → input("Enter your password: ")  → "test"
      → returns {"password": "test"}
    → ValidateCredential
      → state["username"]="test_user", state["password"]="test"
      → returns {"is_authenticated": False}
    → router(state) → "failure_node"
    → path_map["failure_node"] → "Failure"
    → Failure → returns {"output": "Not Successfull, please try again!"}
    → add_edge("Failure", "InputNode") → loop back
    → InputNode
      → state.get('username') != "" (still "test_user"), skips username prompt
      → input("Enter your password: ") → "secure_password"
      → returns {"password": "secure_password"}
    → ValidateCredential
      → returns {"is_authenticated": True}
    → router(state) → "success_node"
    → path_map["success_node"] → "Success"
    → Success → returns {"output": "Authentication successful! Welcome."}
    → add_edge("Success", END) → END
  → final state: {username, password, is_authenticated=True, output=success}
```

Every transition is a node invocation; every node returns a partial dict; LangGraph merges and passes the merged bag forward. No layer crossings to mark — V1 is all in one file. V2's walkthrough will tag application, infra, domain at each step.

---

## Traces

### Auth — failure then success

```
{'username': 'test_user'}
Enter your password: test
Username : test_user Password : test
{'username': 'test_user', 'password': 'test', 'is_authenticated': False, 'output': 'Not Successfull, please try again!'}
Enter your password: secure_password
Username : test_user Password : secure_password
{'username': 'test_user', 'password': 'secure_password', 'is_authenticated': True, 'output': 'Authentication successful! Welcome.'}
Authentication successful! Welcome.
```

State bag accumulates across the cycle. Second `InputNode` skips the username prompt — the bag already contains it.

### QA — clean answer, then hallucination

**Q:** "What is LangGraph?"
**A:** "Based on the provided context, LangGraph is a Python library used for designing state-based workflows. It allows users to build complex applications by connecting modular nodes with conditional edges, simplifying the process of creating and managing these workflows."

**Q:** "What is the best guided project?"
**A:** "Based on the provided context, I would recommend creating a 'Simple Bank Transfer Workflow' as the best guided project... [five fabricated justifications about bank transfers]"

Same graph, same model, same temperature. First call grounded. Second call hallucinated. Retrieval failure upstream, hallucination downstream. The context provider keyword-matched on "guided project" and returned the LangGraph context for a question about project recommendations; the LLM filled the gap.

### Counter — 13 iterations, exits on boundary

```
Adding 1 to n and selecting a random letter: n=1, letter='v'
Current n: 1 Letter: v
...
Adding 1 to n and selecting a random letter: n=13, letter='l'
Current n: 13 Letter: l
Final state: {'n': 13, 'letter': 'l'}
```

Cyclic flow with termination working as designed.

---

## Production Insights

**Graph correctness ≠ system correctness.** The Auth and QA workflows both ran perfectly. The QA workflow returned a structurally clean, confidently-worded, completely fabricated answer about "Simple Bank Transfer Workflow" for a question about project recommendations. The state machine executed every transition correctly; the answer was hallucination. Production agents need golden questions and eval pipelines, not just working topology. Rovers Chatbot has 33 golden questions and a passing threshold for exactly this reason; the IBM lab has neither and presents this output as success.

**`TypedDict(total=False)` is the correct shape for LangGraph state, not `total=True`.** State starts empty and fills progressively; not every field is populated at entry. The IBM lab's `total=True` defaults force partial-init via `cast` or `# type: ignore`. `total=False` matches LangGraph 1.x convention and removes the friction.

**LangGraph is a state machine framework with LLM-friendly ergonomics, not "agent orchestration."** The Pregel/BSP model (2010 Google paper) is the actual lineage. The marketing layer pitches cycles as a LangGraph differentiator versus LangChain — but `create_agent` has been cyclic since it shipped, and `create_agent` IS a LangGraph state machine under the hood. The accurate framing is "LangGraph exposes the state machine that `create_agent` hides." This matters because once you see the framework as a state machine, the production patterns (validated domain state, observability via event streams, defence-in-depth via interfaces) are the same patterns you'd apply to AWS Step Functions, Wolverine sagas, Temporal workflows, or C#'s `async`/`await` compiler-lowered state machines.

**`set_entry_point` is courseware drift.** It still works in 1.x but emits a DeprecationWarning. The lab predates the LangGraph 1.0 cutover by ~6 months. Anyone learning LangGraph from this courseware will write code that compiles today and breaks at the 2.0 release. V1 surfaces this; V2 fixes it.

**`response.content`'s `str | list` union is real but always-str at runtime for text-only prompts.** The cleanest production move is to own the narrowing inside an adapter that exposes `str`. `cast(str, response.content)` hides the assumption; an `isinstance` guard with a `TypeError` makes the assumption explicit. V2's `ChatModelProviderInterface` adapter does this once at the boundary so callers don't repeat the check.

**Side-effects inside nodes (`print(...)`, `input(...)`) are notebook-shape, not production-shape.** The IBM lab's `input_node` blocks on stdin and the QA nodes print state internally. V1 keeps both faithful to the lab. V2 moves observation to a streaming consumer (`graph.stream(stream_mode="updates")`) and input behind an `InputProviderInterface`. V3 makes the streaming consumer the event store's writer.

---

## What the Lab Doesn't Cover

- **Checkpointer / persistence.** `MemorySaver`, `SqliteSaver`, thread IDs, resumable runs. Introduced in subsequent modules.
- **Streaming consumption.** `graph.stream(stream_mode="values" | "updates")`. Not used; lab uses `.invoke()` only.
- **Tool-calling.** No `@tool`, no `bind_tools`, no `ToolMessage`, no agentic dispatch loop. Module 1 is foundations only.
- **`messages` reducer.** No `Annotated[list, add_messages]`. State fields use default last-write-wins. Module 2's Reflexion lab introduces the reducer pattern.
- **Eval discipline.** Two QA invocations; no expected-output assertions; no golden questions. The hallucination is only visible because a human read the output.
- **Validation at state boundaries.** Raw `state["is_authenticated"]` access; no domain invariants. V2's domain layer encodes "authenticated implies username exists" as a `__post_init__` check.

---

## V1 → V2 Findings (Pinned for the Next Commit)

1. `total=False` on all state TypedDicts (lab uses `total=True`).
2. `set_entry_point` → `add_edge(START, ...)` (deprecation captured).
3. Counter router returns string literals → `dict[str, str]` path_map (fixes pyright).
4. `response.content` narrowing moves into `ChatModelProviderInterface` adapter.
5. `input()` moves behind `InputProviderInterface`.
6. `print()` in nodes moves to streaming consumer.
7. State TypedDicts hold a single domain object field (`tick: CounterTick`, `credentials: AuthCredentials`) instead of loose primitive fields. Invariants encoded in `__post_init__`.
8. Tool factories arrive for V2's fourth worked example (minimal tool-calling agent) — none needed for Auth/QA/Counter.

---

## Connection to Subsequent Labs

**Lab 28 (Reflexion).** Self-critique loop — agent generates output, critiques itself, regenerates. Module 2's first multi-step agent. State will need the `messages` reducer; nodes will need to read prior LLM output. V3's event substrate (RunStarted, NodeEntered, LlmInvoked, RunCompleted) fires for the first time against a workload that actually has meaningful events to record.

**Lab 29+ (multi-agent).** Multiple agents coordinating via shared state. The state-as-validated-domain-object discipline becomes load-bearing — multiple writers mean reducers and invariants are no longer optional.

**Blog feeders surfaced from V1:**

1. *LangGraph is a state machine — and you've probably built one before* (satellite). Parallels to C# async/await compilation, AWS Step Functions, Wolverine sagas, Temporal workflows. Ships off V2.
2. *Event-sourced agent observability* (flagship). V3 + Lab 28. The architectural argument that `stream_mode="updates"` IS the event stream; snapshot-first checkpointing is a usage pattern, not a framework limitation.

---

**Completed:** 23 June 2026