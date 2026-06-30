# Course 6 — Lab 26: Natural Language SQL Agent

> Code: [`course6-module3-lab2/`](course6-module3-lab2/)

Onion-architected SQL agent that answers natural-language questions about a Chinook database. Hand-rolled tools matching the canonical `SQLDatabaseToolkit` descriptions and output formats. SQLAlchemy access behind `SqlDatabaseInterface`; OpenAI behind `ChatModelProviderInterface`. Logging middleware via `wrap_tool_call` records every tool invocation to a domain-layer `AgentTrace`. Read-only enforced at the SQLite URI as defence-in-depth against DML.

Built as part of the IBM RAG and Agentic AI Professional Certificate — Course 6, Module 3, Lesson 2. The IBM lab specifies `langchain==0.2.1`, `create_sql_agent`, `langchain-experimental`, watsonx + Granite, and MySQL. This implementation follows the official LangChain 1.x SQL agent tutorial (https://docs.langchain.com/oss/python/langchain/sql-agent) instead, using `create_agent`, `gpt-4.1-mini`, and SQLite. The architectural decisions are mine; the IBM lab's example queries are reused as the smoke test set.

---

## What It Does

Four hand-rolled tools wired into one agent. The LLM decides which to call:

- `sql_db_list_tables()` — comma-separated list of all tables
- `sql_db_schema(table_names)` — DDL + 3 sample rows for the named tables, with allow-list validation
- `sql_db_query(query)` — executes SQL, catches exceptions so the LLM can read errors and retry
- `sql_db_query_checker(query)` — LLM-as-tool: validates a SQL query via a separate model call

DML attempts are refused at the engine level (read-only SQLite URI). The system prompt also instructs the LLM not to emit DML — two layers, one probabilistic (prompt), one deterministic (engine).

---

## Stack

| Component             | Implementation                                                     |
| --------------------- | ------------------------------------------------------------------ |
| LLM                   | `gpt-4.1-mini` via `init_chat_model`, temperature=0.0              |
| Agent construction    | `langchain.agents.create_agent` (LangChain 1.x)                    |
| Tool decoration       | `@tool` from `langchain.tools`, four hand-rolled factories         |
| Middleware            | `@wrap_tool_call` for tool-invocation logging                      |
| Database access       | `SqlAlchemyDatabase` wrapping SQLAlchemy 2.x                       |
| Read-only enforcement | SQLite URI mode: `sqlite:///file:Chinook.db?mode=ro&uri=true`      |
| LLM provider coupling | `OpenAIChatModelProvider` behind `ChatModelProviderInterface`      |
| Architecture          | Strict onion — domain, interfaces, application, infra              |
| Test surface          | 101 tests, no API key, no network, no real DB file required        |

---

## Setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create `.env`:

```
OPENAI_API_KEY=sk-...
```

Run the demo:

```powershell
python sql_agent_demo.py
```

Run the tests:

```powershell
pytest tests/ -v
```

---

## File Layout

```
course6-module3-lab2/
├── sql_agent_demo.py                              # entry point: REPL
├── conftest.py
├── pytest.ini
├── requirements.txt
├── Chinook.db
│
├── domain/
│   └── models.py                                  # ToolCallRecord, AgentTrace
│
├── interfaces/
│   ├── sql_database_interface.py
│   └── chat_model_provider_interface.py
│
├── application/
│   ├── sql_agent.py                               # SqlAgent service
│   ├── container.py                               # composition root
│   ├── middleware.py                              # log_tool_call factory
│   ├── interfaces/
│   │   └── sql_agent_interface.py
│   └── tools/
│       ├── list_tables.py
│       ├── get_schema.py
│       ├── run_query.py
│       └── check_query.py
│
├── infra/
│   ├── sqlalchemy_database.py
│   └── openai_chat_model.py
│
└── tests/
    ├── domain/                                    # 8 tests
    ├── infra/                                     # 22 tests
    └── application/                               # 71 tests
```

---

## Key Concepts

### Hand-Rolled Tools

`langchain_community` was sunset, effective immediately, with the maintainers' guidance pointing toward implementing tools directly in application code rather than waiting for replacement packages. There is no standalone successor for the SQL utilities.

Four `@tool`-decorated factories living in `application/tools/`, each closing over `SqlDatabaseInterface` (or the LLM, for the checker). The tool names and descriptions are preserved verbatim from the canonical toolkit — the LLM is conditioned on those exact strings; drifting them would shift behaviour. Zero `langchain_community` imports remain in the codebase.

### Tools Compose Atomic Interface Methods

`SqlDatabaseInterface` exposes five atomic methods: `dialect`, `get_table_names`, `get_table_ddl`, `get_sample_rows`, `run`. The `sql_db_schema` tool composes `get_table_ddl` and `get_sample_rows`. The split isolates the sample-row path from the DDL path — the seam where future PII redaction would intercept.

The interface returns primitives (strings, lists). SQLAlchemy types never cross the boundary; the LLM consumes text.

### Middleware Captures Tool Calls to a Domain-Layer Trace

`make_log_tool_call_middleware(trace)` returns a `wrap_tool_call` middleware that appends a `ToolCallRecord` for every tool invocation. Closure-captures the trace. `AgentTrace` exposes records as an immutable tuple — the same pattern from Lab 25 that emerged from a Copilot review pointing out that returning the underlying list let callers mutate the trace through the property.

Observational only — records and returns the handler's response unchanged. Transformational middleware (PII redaction, row limit, HITL) would be separate instances composed alongside it.

### Defence in Depth at the Engine Boundary

The system prompt instructs the LLM not to emit DML. The model honours this — verified by asking the agent to add a new record, which it refuses without attempting execution.

That's the probabilistic layer. The deterministic layer is the read-only URI:

```
sqlite:///file:Chinook.db?mode=ro&uri=true
```

Verified independently: instantiating `SqlAlchemyDatabase` directly with the read-only URI and bypassing the LLM, an INSERT raises `(sqlite3.OperationalError) attempt to write a readonly database`. The credential boundary is the safety mechanism, not the prompt.

### LLM-as-Tool: `sql_db_query_checker`

The checker is a mini-agent inside the agent. It receives a SQL string, invokes the LLM with a SQL-expert system prompt, returns either the validated query or a rewritten version.

On a complex LEFT JOIN query, the checker added ~1,778ms — measured via the logging middleware. The underlying query executed in 1.1ms. The checker was three orders of magnitude slower than the query it was checking. That cost is the price of self-validation.

The checker fires conditionally. Empirically, it skips trivial queries (`SELECT COUNT(*) FROM Album`) regardless of the system prompt's `"You MUST double check"` instruction. For complex queries, it fires. The LLM weighs cost against benefit at runtime.

---

## Architecture Walkthrough — One Call

When `python sql_agent_demo.py` runs `agent.ask("How many albums are in the database?")`:

```
sql_agent_demo.py
  initialise(database_uri="sqlite:///file:Chinook.db?mode=ro&uri=true")
    → OpenAIChatModelProvider().create() → BaseChatModel        [infra]
    → SqlAlchemyDatabase(uri)                                    [infra]
    → make_list_tables(db), make_get_schema(db),
      make_run_query(db), make_check_query(llm) → [BaseTool]    [application/tools]
    → AgentTrace()                                               [domain]
    → make_log_tool_call_middleware(trace) → AgentMiddleware    [application]
    → SqlAgent(llm, tools, dialect, trace)                      [application]
  agent.ask("How many albums?")
    → agent.invoke({"messages": [user message]})
      → AIMessage(tool_calls=[sql_db_list_tables])
      → middleware.wrap_tool_call → handler → trace.append(...)
      → AIMessage(tool_calls=[sql_db_schema(Album)])
      → middleware.wrap_tool_call → handler → trace.append(...)
      → AIMessage(tool_calls=[sql_db_query("SELECT COUNT(*)...")])
      → middleware.wrap_tool_call → handler → trace.append(...)
      → AIMessage(content="There are 347 albums.")
    → return final.content
```

Every transition is explicit. The agent never imports infra; tool factories use interface types; the container is the only place infra and application meet.

---

## Hardening

**Read-only URI at the SQLAlchemy connection.** Single URI change, six tests proving INSERT/UPDATE/DELETE/DROP refused at the engine. Defence-in-depth against prompt-injection-driven DML.

---

## Potential Additions

Three further hardening surfaces were considered but not implemented. The seams exist; the policies do not.

**Row-limit middleware via `wrap_tool_call`.** Truncating oversized query results before they reach the model's context. Character-based truncation loses structured semantics; row-based truncation is misleading on wide rows. Timeout enforcement is dialect-specific — SQLite cannot cancel running queries. The work does not justify itself at the SQLite-demo scale; on a production database with millions of rows, it would.

**PII redaction at the schema-reconnaissance layer.** Observed empirically: a user asking *"how many employees are there?"* causes the agent to invoke `sql_db_schema` on `Employee`, which returns 3 sample rows. The first employee's phone number, email, and address land in the model's context for the rest of the conversation. The user never sees this; the trace logs do. The seam exists at `get_sample_rows`; the redaction policy would live in the composition root. The Chinook dataset is fictional — the architectural concern is real, the leak is not.

**HITL middleware on `sql_db_query`.** The DML attack vector is closed by the read-only URI. The PII exfiltration vector via direct query is partly defended by the read-only URI plus system prompt. HITL would defend against prompt-injected exfiltration queries — plausible in production, contrived here. The implementation (checkpointer, thread_id, state persistence) is significant work for limited demo value.

---

## Production Insights

**Hand-rolling tools is the correct response to a sunset framework package** when the migration target doesn't exist and the maintainers' guidance points to application-level implementation. The work is modest (~80 lines across four files); the result is fully owned.

**Defence in depth means more than one layer.** Read-only URI + system prompt is two layers. One is probabilistic; one is deterministic. Either alone is insufficient.

**The `sql_db_query_checker` costs ~2 seconds per invocation.** The checker is the LLM validating itself. Measured via the logging middleware. For trivial queries, the LLM skips it. For complex queries, it fires. The system prompt's `"MUST"` instruction is a tiebreaker, not a rule.

**The `sql_db_schema` sample-row feature is a PII leak vector.** Aggregate questions trigger schema reconnaissance; schema reconnaissance returns sample rows; sample rows contain PII. The user's answer is PII-clean; the model's context is not. The trace logs surface this even when the user does not.

**Middleware closes over the trace from the composition root.** The container owns the trace; the agent receives a reference. Tests inspect the trace independently; the demo prints it on exit without needing the agent to expose serialisation.

---

## What This Doesn't Cover

- **Conversation memory.** Each `agent.ask(question)` starts a fresh conversation.
- **Streaming.** `agent.invoke()` returns the final message; `agent.stream(stream_mode="values")` would expose intermediate steps.
- **Multi-database routing.** One `database_uri` per `initialise()` call.
- **PostgreSQL or MySQL.** Only SQLite is exercised. Read-only enforcement is SQLite-specific; other dialects need different patterns.
- **Eval suite.** The 101 tests verify behaviour at every layer; they do not validate semantic correctness end-to-end. Ambiguous questions ("Which country spent the most by invoice?" — total or average?) produce confident answers in one interpretation without flagging the ambiguity.

---

**Completed:** 15 June 2026