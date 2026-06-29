# How to Run — Lab 27 V3a (Event-Sourced QA Workflow)

V3a of the LangGraph 101 lab. V2's onion-architected QA workflow with an
event-sourced substrate layered around it — every node update becomes a typed
domain event written to an append-only store, with in-memory and SQLite
concretes behind one `AgentEventStoreInterface`. The QA agent service stays
unchanged in shape; the observability layer is what's new.

For the architectural narrative and pinned decisions see
`../lab27notes-c7-m1-langgraph-101-v3a-event-sourced.md`. V1
(canonical-faithful) is in `course7-module1-lab1-v1-canonical/`; V2
(onion-architected) is in `course7-module1-lab1-v2-onion/`.

V3b ships QA + Auth. The V3 series is terminal at V3b; Counter was scoped
out.

## Setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
ollama pull llama3.2:latest   # if not already local
ollama serve                  # or run the Ollama app
```

## Run

```powershell
python demo.py                                # in-memory default, Ollama
python demo.py --persistence sqlite           # writes to ./events.db
python demo.py --persistence sqlite --db-path runs.db --quiet
python demo.py --provider openai              # requires OPENAI_API_KEY in .env

# Flags compose freely:
python demo.py --provider openai --persistence sqlite
python demo.py --provider openai --persistence sqlite --db-path openai-runs.db --quiet
```

Each invocation runs the three canned QA questions, prints structured
streaming output during each run (`[NodeName] {state_delta}` lines from
`ConsoleStreamConsumer`), then prints a `RunSummary` block per run by
replaying the event log via `summarise_run`. The third question
(`"What is the best guided project?"`) preserves V1/V2's keyword-match
hallucination — context_provider_node returns LangGraph context for any
"guided project" question regardless of subject. V3a records the
hallucination as data; the eval-pipeline argument is what catches it.

`--quiet` suppresses the streaming trace lines — only the summaries print.
Useful for clean demos where the event log replay is the observation
surface.

## Test

```powershell
pytest                            # 174 tests, no API key required
pyright                           # 0 errors, no cast, no # type: ignore
```

The test suite runs without an Ollama server, without an API key, in under
2 seconds. SQLite infra tests use `tmp_path` fixtures so no real DB files
are created.

## Provider selection

Defaults to Ollama (`llama3.2:latest`, no API key needed). To use OpenAI,
copy the example env file and fill in your key:

```powershell
Copy-Item .env.example .env
notepad .env   # or any editor — paste your key after OPENAI_API_KEY=
```

Then run with the OpenAI provider:

```powershell
python demo.py --provider openai
```

`demo.py` calls `load_dotenv()` at startup, so the key in `.env` becomes
available via `os.environ` for the lab process. The `.env` file is
gitignored — it never enters version control. Production deployments inject
env vars via the runtime and skip `.env` entirely; `load_dotenv()` is a
no-op when no file is present.

Get a key from https://platform.openai.com/api-keys.

## Persistence

In-memory is the default. Events live in a Python list inside the process;
exiting `demo.py` discards them. Fine for iteration and demos.

`--persistence sqlite` writes to `./events.db` (or `--db-path PATH`). The
schema is one append-only `events` table indexed on `aggregate_id`. The
store is durable across runs — re-running the demo appends to the existing
log rather than truncating.

Inspect a SQLite run:

```powershell
python -c "import sqlite3; c=sqlite3.connect('events.db'); print('\n'.join(f'{r[0][:8]} {r[1]:25} agg={r[2][:8]} at={r[3]}' for r in c.execute('SELECT event_id, event_type, aggregate_id, occurred_at FROM events ORDER BY rowid')))"
```

Or inspect the JSON payload of one event:

```powershell
python -c "import sqlite3, json; c=sqlite3.connect('events.db'); r=c.execute('SELECT event_type, payload FROM events ORDER BY rowid LIMIT 1').fetchone(); print(r[0]); print(json.dumps(json.loads(r[1]), indent=2))"
```

`.db` files are gitignored at the repo root.

## Adding a new provider, persistence backend, or event type

A third LLM provider is a new file in `infra/`, a new branch in
`application/container.py`, and (optionally) a new option on the
`--provider` flag in `demo.py`. No changes to interfaces, domain,
application services, or tests for any unaffected layer.

A third persistence backend (Postgres, Redis, etc.) is a new file in
`infra/` satisfying `AgentEventStoreInterface`, a new branch in the
container, and a new option on `--persistence`. The translator, projection,
service, and tests stay unchanged — the interface is the seam.

A new event type for QA is a new frozen dataclass in
`domain/events/qa_events.py`, a one-line addition to the `QAEvent` union,
and a new dispatch branch in `translate_qa_update`. Pyright's
`assert_never` catches the union extension at the translator boundary if
it's added without the dispatch. Schema versioning is automatic — the new
event ships with `schema_version=1`; bumping happens later when a field
changes on an existing type.