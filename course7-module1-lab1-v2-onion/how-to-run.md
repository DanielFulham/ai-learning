# How to Run — Lab 27 V2 (Onion-Architected)

V2 of the LangGraph 101 lab. Three workflows (Auth, QA, Counter) restructured
into strict onion architecture with two LLM provider concretes behind a
`ChatModelProviderInterface` seam.

For the architectural narrative and pinned decisions see
`../lab27notes-c7-m1-langgraph-101-v2-onion.md`. V1 (canonical-faithful) is
in the sibling `course7-module1-lab1-v1-canonical/`.

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
python demo.py auth      # interactive — uses ConsoleInputProvider
python demo.py qa        # runs three canned questions; third triggers V1's hallucination
python demo.py counter   # runs the 13-iteration cycle to termination
python demo.py all       # all three back-to-back; Auth uses ScriptedInputProvider
```

Each run produces structured streaming output — `[NodeName] {state_delta}`
lines per node execution — followed by the workflow's final result. The
trace lines come from `ConsoleStreamConsumer` subscribed to LangGraph's
update stream; they're the production observation surface, not log noise.

## Test

```powershell
pytest                            # 116 tests, no API key required
pyright                           # 0 errors, no cast, no # type: ignore
python scripts/draw_graphs.py     # emits docs/graphs/{auth,qa,counter}.mmd
```

The test suite runs without an Ollama server, without an API key, in under
3 seconds. The Mermaid script needs no external dependencies either — it
mocks the dependencies the graph topology doesn't depend on.

## Provider selection

Defaults to Ollama (`llama3.2:latest`, no API key needed). To use OpenAI, copy
the example env file and fill in your key:

```powershell
Copy-Item .env.example .env
notepad .env   # or any editor — paste your key after OPENAI_API_KEY=
```

Then run with the OpenAI provider:

```powershell
python demo.py qa --provider openai
```

`demo.py` calls `load_dotenv()` at module load, so the key in `.env` becomes
available via `os.environ` for the lab process. The `.env` file is gitignored
— it never enters version control. Production deployments inject env vars via
the runtime and skip `.env` entirely; `load_dotenv()` is a no-op when no file
is present.

Get a key from https://platform.openai.com/api-keys.

Adding a third provider is a new file in `infra/`, a new `bool` parameter on
`initialise()` (or promotion to an enum if the count grows beyond two), and
one branch in `application/container.py`. No changes to interfaces, domain,
application services, or tests for any unaffected layer.