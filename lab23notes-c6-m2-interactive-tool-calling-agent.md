# Course 6 — Lab 23: Interactive Tool-Calling Agent

Onion-architected single-turn tool-calling agent that exposes the dispatch loop
hidden inside `langchain.agents.create_agent`. Two demos (arithmetic and tip
calculator) share one agent, one container, and one LLM provider through a
test-injectable seam. Demonstrates parallel-tool-call handling, provider-agnostic
application code, and a test suite that runs without an API key or network.

Built as part of the IBM RAG and Agentic AI Professional Certificate — Course 6,
Module 2.

---

## What It Does

The lab exercises the four manual tool-calling primitives — `@tool`,
`bind_tools`, `AIMessage.tool_calls`, `ToolMessage` — without the
`create_agent` wrapper. Two demonstrators run on top:

- **Arithmetic demo** — three tools (`add`, `subtract`, `multiply`) over
  natural-language input ("one plus 2", "three times two")
- **Tip demo** — one tool (`calculate_tip`) over realistic bills with float
  amounts and percentage handling

Both demos share the same `ToolCallingAgent`, the same `initialise()` container,
and the same `OpenAIChatModelProvider`. The only difference between them is
which tool list gets passed at composition time.

---

## Stack

| Component             | Implementation                                                     |
| --------------------- | ------------------------------------------------------------------ |
| LLM                   | `gpt-4.1-nano` via `init_chat_model`                               |
| Agent construction    | Manual `@tool` + `bind_tools` + dispatch loop, parallel-call-safe  |
| Tool decoration       | `@tool` from `langchain.tools` (LangChain 1.x canonical)           |
| Message types         | `langchain.messages` (LangChain 1.x canonical)                     |
| LLM provider coupling | `OpenAIChatModelProvider` behind `ChatModelProviderInterface`      |
| Architecture          | Strict onion — application, interfaces, infra, tools-in-app        |
| Test surface          | 20 tests, all passing without an API key                           |
| Entry point           | `if __name__ == "__main__"` with `initialise()` from container     |

---

## Setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -U langchain langchain-openai python-dotenv pytest
```

Create `.env` next to the demo scripts:

```
OPENAI_API_KEY=sk-...
```

Run the demos:

```powershell
python arithmetic_demo.py
python tip_demo.py
```

Run the tests (no API key required):

```powershell
pytest tests/ -v
```

---

## File Layout

```
course6-module2-lab2/
├── arithmetic_demo.py                            # entry point: arithmetic agent
├── tip_demo.py                                   # entry point: tip agent
├── conftest.py                                   # pytest sys.path setup
├── requirements.txt
├── .env
│
├── application/
│   ├── __init__.py
│   ├── agent.py                                  # ToolCallingAgent (the loop)
│   ├── container.py                              # composition root, optional LLM
│   └── tools/
│       ├── __init__.py
│       ├── arithmetic.py                         # add, subtract, multiply
│       └── tip.py                                # calculate_tip
│
├── interfaces/
│   ├── __init__.py
│   └── chat_model_provider_interface.py          # Protocol infra satisfies
│
├── infra/
│   ├── __init__.py
│   └── openai_chat_model.py                      # OpenAIChatModelProvider
│
└── tests/
    ├── __init__.py
    ├── application/
    │   ├── __init__.py
    │   ├── test_agent.py                         # 4 tests, LLM mocked
    │   ├── test_container.py                     # 3 tests, provider mocked
    │   └── tools/
    │       ├── __init__.py
    │       ├── test_arithmetic.py                # 5 direct tool tests
    │       └── test_tip.py                       # 5 direct tool tests
    └── infra/
        ├── __init__.py
        └── test_openai_chat_model.py             # 4 tests, init_chat_model patched
```

---

## Key Concepts

### The Four-Step Manual Loop

The dispatch loop reduces to four primitives:

1. **`bind_tools`** — attaches JSON Schemas of the tools to the LLM client.
   The model sees the schemas; the runtime sees nothing yet.
2. **First `invoke`** — sends the user query plus the schemas. The model
   decides which tool(s) to call and emits an `AIMessage` with empty `.content`
   and a populated `.tool_calls` list.
3. **Dispatch** — for each call in `tool_calls`, look up the tool by name in a
   dispatch dict, invoke it with the model-supplied args, wrap the result in a
   `ToolMessage` echoing the `tool_call_id`.
4. **Second `invoke`** — re-send the full history (HumanMessage + AIMessage +
   ToolMessages). The model reads its own tool request and the result, then
   writes the natural-language answer.

This is what `langchain.agents.create_agent` runs internally for a single
turn. Building it manually surfaces the message-stream semantics that the
wrapper hides — useful when debugging `create_agent` traces, instrumenting
spans, or making informed decisions about which abstraction level a given
deployment needs.

### Parallel Tool Calls

A model can emit multiple tool calls in a single `AIMessage`. For queries
like `"add 3 plus 2 and multiply 4 by 5"` this is obvious; for ambiguous
phrasings like `"three times two"`, gpt-4.1-nano sometimes emits a second
redundant call as well. Either way, the dispatch loop must produce one
`ToolMessage` per `tool_call_id` — the OpenAI API rejects the next request
with HTTP 400 if any IDs are unanswered:

```
openai.BadRequestError: 400 — An assistant message with 'tool_calls' must be
followed by tool messages responding to each 'tool_call_id'.
```

The agent iterates every call and builds one `ToolMessage` per ID:

```python
tool_messages = [
    ToolMessage(
        content=str(self._tool_map[call["name"]].invoke(call["args"])),
        tool_call_id=call["id"],
    )
    for call in response.tool_calls
]

chat_history.extend([response, *tool_messages])
```

Single-call paths still work because the loop runs once. The agent test
suite pins both cases — `test_single_tool_call_dispatches_and_summarises`
and `test_parallel_tool_calls_all_dispatched`.

### Provider DI Seam

The container picks the concrete LLM provider. The agent receives a fully-built
`BaseChatModel` and never knows the brand. The interface
(`ChatModelProviderInterface`) exists to make the seam testable.

```python
def initialise(
    tools: list[BaseTool],
    provider: ChatModelProviderInterface | None = None,
) -> ToolCallingAgent:
    if provider is None:
        provider = OpenAIChatModelProvider()
    llm = provider.create()
    return ToolCallingAgent(llm, tools)
```

Production calls `initialise(tools=[add])` with no provider; the default
`OpenAIChatModelProvider` is constructed. Tests pass
`provider=MagicMock(spec=ChatModelProviderInterface)` and the OpenAI client
is never built — no `OPENAI_API_KEY` required.

### Application Doesn't Know Its Provider

The entry points have zero imports from `infra/`:

```python
# arithmetic_demo.py
from application.container import initialise
from application.tools.arithmetic import add, subtract, multiply

agent = initialise(tools=[add, subtract, multiply])
```

The string `"openai"` and the model name `"gpt-4.1-nano"` appear in exactly one
file: `infra/openai_chat_model.py`. Swapping providers is a one-line change
in `container.py`; demos and tests are untouched.

Main doesn't know more than application does. Application doesn't know more
than the interface defines. Infra is the only layer with provider-specific
knowledge.

### Pattern Continuity Across Labs

The previous lab (DataWizard, Lab 22) applied this onion shape to a five-tool
data-science agent. This lab applies the same shape to a three-tool arithmetic
agent and a one-tool tip calculator. Folder structure, testing strategy, DI
seam — identical.

Strict onion isn't justified by system size; it's justified by *pattern
repeatability* across deployments. The same shape extends naturally to the
next tool surface, the next provider, the next deployment target.

---

## Tests as Architectural Specification

The 20 tests pin both behaviour and design decisions:

**Behaviour:**
- Tool functions return correct values for representative inputs
- Tip calculator handles round percentages, realistic bills, fractional
  percents, and boundary cases (zero bill, zero percent)
- Agent dispatches single tool calls correctly
- Agent dispatches parallel tool calls correctly
- Agent passes content through when the model doesn't call a tool
- `initialise()` returns a `ToolCallingAgent`
- `initialise()` calls `provider.create()` exactly once
- Provider stores model name; passes correct arguments to `init_chat_model`

**Design decisions:**
- `pytest.approx` on `calculate_tip` floating-point case — currency is float
- `tip_percent: float` — fractional percents (12.5%) must work
- Container returns fresh agent per call — pins the stateless contract
- `bind_tools` called once on construction — pins the constructor contract
- Tool dispatch via `tool.name` lookup — pins the derived-from-tools invariant
  on the tool_map

Each design-decision test, if changed, forces a conversation about whether
the test or the new code is correct. Architecture without test coverage is
suggestion; architecture with test coverage is contract.

---

## Architecture Walkthrough — One Call

When `python arithmetic_demo.py` runs `agent.run("three times two")`:

```
arithmetic_demo.py
  agent.run("three times two")
    → ToolCallingAgent.run()                                    [application]
      → llm.bind_tools([add, subtract, multiply])               [once in __init__]
      → first invoke: send query + tool schemas
        → AIMessage(content='', tool_calls=[multiply(3,2), ...])
      → for each tool_call in response.tool_calls:
        → self._tool_map["multiply"].invoke({"a": 3, "b": 2})
          → multiply(3, 2)                                      [application/tools]
          → returns 6
        → ToolMessage(content="6", tool_call_id="call_xyz...")
      → chat_history.extend([response, *tool_messages])
      → second invoke: send full history
        → AIMessage(content='Three times two equals 6.')
      → return final.content
```

Every transition is explicit. The reader can trace the exact path from
natural-language input to natural-language output, with model decisions
surfaced at every boundary.

---

## Production Insights

**The model decides parallelism; the dispatch loop must follow.** Iterate
`response.tool_calls`, build one `ToolMessage` per call, echo the
`tool_call_id` unchanged. Single-call shortcuts work in toy examples and
break on real traffic the moment the model emits a second call.

**`ToolMessage.content` is always stringified for the wire.** The LLM reads
text in the message stream, not Python objects. `add(2, 3)` returns
`int(5)`; the framework stringifies to `"5"` before transit. Structured
returns (dicts, Pydantic models, dataframes) need explicit JSON
serialisation at the tool boundary — `str(dict)` produces Python's repr,
not JSON. Same principle as numpy serialisation: cast at the infra
boundary.

**`init_chat_model` validates credentials on construction, not on invoke.**
Container tests fail without credentials unless the LLM is injectable. The
optional `provider` parameter to `initialise()` is the cleanest fix —
production behaviour unchanged, tests inject a mock provider.

**Two LLM calls per tool-using turn is the floor.** First call selects;
second call summarises. Multi-step tool chains multiply this — three tool
calls means four invocations, each carrying the full growing transcript.
The round-trip multiplier is the single biggest cost driver in agent
design, not the per-token price.

**The application layer should not reveal the concrete infra name.** Entry
points import `from application.container import initialise` and never
`from infra.openai_chat_model import ...`. Provider swaps become one-line
changes in `container.py`; everything else is untouched.

**Anti-corruption layers over framework APIs are ceremony when there's
nothing to corrupt against.** Wrapping LangChain's `bind_tools` + `invoke`
in a project-local Protocol restates the same surface in different words.
Type the application against `BaseChatModel` directly; the framework
coupling is the contract, not a leak. Reserve anti-corruption layers for
APIs there's genuine reason to swap.

**The manual loop is the natural span boundary for agent observability.**
Each `invoke` is a span; each tool dispatch is a span; the `tool_call_id`
is the correlation primitive. `create_agent` hides these inside LangGraph;
the manual version surfaces them at obvious instrumentation points.

---

## What This Doesn't Cover

- **Multi-turn iteration** — this is a single-turn agent. A tool result
  never triggers another tool call. For chains like *"What's (3+2) × 4?"* —
  where the result of `add(3,2)` should feed `multiply(5,4)` — use
  `langchain.agents.create_agent`, which loops until the model stops
  emitting tool calls.
- **Streaming** — `.invoke()` returns the full response; `.stream()` would
  yield each token. The demos use `.invoke()` for clarity.
- **Conversation memory** — each `agent.run(query)` starts a fresh
  `chat_history` with just the HumanMessage. The model has no memory of
  prior queries within the same demo run.
- **Middleware** — `create_agent` supports LangGraph middleware for
  retries, PII redaction, human-in-the-loop interrupts. None of that
  exists here. The manual loop is the minimum viable agent step.
- **Eval harness** — the demos verify the loop works by inspection. A real
  eval setup would generate questions programmatically, compare against
  expected semantic content, and report pass/fail rates.
- **Cost / latency observability** — instrumentation hooks are absent but
  the structure exposes them clearly. Each `invoke` and each tool dispatch
  is a natural span boundary; `tool_call_id` is the correlation primitive.

---

## What's Next

The same onion shape extends naturally to the next tool surface, the next
provider, the next deployment target. The folder structure tells the story;
the test suite proves the seams; the composition root makes the provider
choice the single swap surface.

---

**Completed:** 7 June 2026