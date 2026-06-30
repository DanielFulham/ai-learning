# Course 6 — Lab 21: AI Math Assistant (LangChain Tool Calling)

> Code: [`course6-module1-lab1/`](course6-module1-lab1/)

LangChain agent with a math toolkit and a Wikipedia lookup. User asks a math or factual question, the agent decides which tool to call, the tool runs, the agent synthesises an answer. Pure tool-calling agent pattern — no retrieval, no vector store, no embeddings. Built as part of the IBM RAG and Agentic AI Professional Certificate — Course 6, Module 1.

**Not a faithful port.** The IBM notebook targets `initialize_agent` and `langgraph.prebuilt.create_react_agent`, both deprecated. This lab runs against `langchain.agents.create_agent`, the canonical 1.x API. The diff between what the notebook teaches and what the modern API produces is the most valuable artefact here.

---

## What It Does

- **Define tools** — math operators (add, subtract, multiply, divide), a power tool, and a Wikipedia lookup, all via `@tool` decorator with type-hinted schemas
- **Construct agents** — single-tool, multi-tool, and multi-tool-plus-retrieval agents using `create_agent`
- **Run invocations** — eleven distinct calls across two models (GPT-4.1-nano via OpenAI, Mistral 7B via Ollama)
- **Walk message lists** — custom `print_trace()` helper inspects the `HumanMessage / AIMessage / ToolMessage` sequence each agent returns
- **Test harness** — four-case loop comparing tool output to expected values

No file scaffolding, no Flask, no UI. The output is the terminal trace of each agent invocation.

---

## Stack

| Component | IBM Notebook | This port |
|---|---|---|
| Agent framework | `initialize_agent` / `langgraph.prebuilt.create_react_agent` | `langchain.agents.create_agent` |
| Primary LLM | `ibm/granite-4-h-small` via watsonx | `gpt-4.1-nano` via OpenAI |
| Local LLM | Not in notebook | `mistral:latest` via Ollama |
| Tool definition | Mix of `Tool()` constructor and `@tool` decorator | `@tool` decorator throughout |
| Wikipedia | `WikipediaAPIWrapper` from `langchain_community.utilities` | Direct `wikipedia.summary()` with User-Agent |
| Credentials | None (IBM Cloud IDE auto-injects) | `python-dotenv` + `.env` with `OPENAI_API_KEY` |
| Message format | `("human", query)` tuples | `{"role": "user", "content": query}` dicts |

---

## File Layout

```
course6-module1-lab1/
├── app.py                     — entry point with --section dispatcher
├── app_local.py               — Ollama variant, subset of sections
├── tools.py                   — all @tool definitions (shared)
├── tracing.py                 — print_trace() helper (shared)
├── requirements.txt
├── .env                       — OPENAI_API_KEY
├── .gitignore
└── sections/
    ├── s1_direct_tool_tests.py
    ├── s2_single_tool_agent.py
    ├── s3_tool_introspection.py
    ├── s4_four_tool_agent.py
    ├── s5_harness.py
    ├── s6_wikipedia.py
    └── s7_power_tool.py
```

One file per lab section. Each section module exports a `run()` function constructing its own LLM and agent, fully self-contained. `app.py` dispatches via `--section <name>` or runs all in sequence. Single-responsibility files mean `python app.py --section harness` runs just the harness without paying for six sections of OpenAI calls first.

---

## Three Generations of API Churn

The single most striking thing about running this lab is how much LangChain has shifted in 18 months. Three distinct generations of agent API have been deprecated:

| Generation | What it was | Status |
|---|---|---|
| Legacy | `initialize_agent` with `agent="zero-shot-react-description"` | Rehomed into `langchain-classic` |
| Transitional | `from langgraph.prebuilt import create_react_agent` | Deprecated in LangGraph v1.0 (Oct 2025), redirects to current |
| Current | `from langchain.agents import create_agent` | Canonical 1.x; `langchain-community` integrations sunsetting |

The framework has been progressively de-magicked. Each rename signals an architectural assumption hardening into a constraint:

- "Models support native tool calling" → no `agent_type` parameter needed
- "Tool calls are structured JSON, not text" → no `handle_parsing_errors` needed
- "Conversations are message lists, not strings" → no `max_iterations`, just a `recursion_limit` in config
- "Agent loops live on LangGraph" → `langgraph.prebuilt` rehomed into `langchain.agents`

All three generations are physically present in the freeze output as separate packages — `langchain-classic`, `langgraph-prebuilt`, `langchain` itself. The migration paths aren't yet clean. The production signal: prefer thin `@tool` wrappers you control over batteries-included integrations you don't. `@tool`-decorated functions are stable Python. `WikipediaAPIWrapper` is not.

---

## Tool Definition Patterns

### Legacy `Tool()` constructor

```python
add_tool = Tool(
    name="AddTool",
    func=add_numbers,
    description="Adds a list of numbers and returns the result.",
)
```

Name and description as constructor arguments. The function's docstring is inert — the LLM never sees it. The `description` string is the contract.

### Modern `@tool` decorator

```python
@tool
def add_numbers(inputs: str) -> dict:
    """Adds a list of numbers provided in the input string.

    Example Input:  "Add the numbers 10, 20, and 30."
    Example Output: {"result": 60}
    """
    numbers = [int(num) for num in re.findall(r'\d+', inputs)]
    return {"result": sum(numbers)}
```

Docstring becomes the LLM-facing description. Type hints become the JSON schema for arguments. The agent sees a *typed* contract rather than a one-line prose description.

For `add_numbers_with_options(numbers: List[float], absolute: bool = False)` the schema produces `numbers` as an array and `absolute` as an optional boolean — exactly the OpenAI / Anthropic function-calling JSON schema the model receives natively.

---

## The Docstring is a Load-Bearing Artefact

The single most important insight: **with `@tool`-decorated functions, the docstring is a runtime artefact, not developer comfort.** The LLM reads the docstring to decide whether to call the tool, what arguments to construct, and how to format the inputs. The example block in the docstring is the *prior* the LLM uses for argument shape.

Evidence from the actual run:

**Multiply "four"** (section 4c): The LLM hedged with two parallel tool calls — one with `'2, 3, 4'` (returned 24), one with `'2, 3, four'` (returned 6). The docstring example was `"2, 3, 4"` (clean integers). Both variants matched some interpretation of the prompt. The synthesis sentence in the final answer is itself revealing: "considering 'four' as a string that can be interpreted as the number 4" — the LLM rationalised the hedge after the fact rather than committing to one interpretation.

**Add "two"** (section 2b): The LLM passed `'10, 20, two, and 30'` as a single tool call. The docstring example was `"Add the numbers 10, 20, and 30."` (verbose natural-language form). The LLM matched the docstring's shape and kept the word "two" intact.

Same model, two tools with similar regex parsing, different word-number handling. The diff is the docstring example shape.

**Docstrings don't just describe — they prescribe.** Sloppy docstrings produce sloppy tool arguments. Stronger models trust the docstring *more*, not less, so a sloppy contract becomes more dangerous as the model improves.

---

## Silent-Wrong-Answer Patterns

The richest output of running this lab is a catalogue of failure modes the IBM notebook's lesson plan cannot teach. The notebook was built around the legacy `initialize_agent` parser-loop hazard ("the agent keeps trying the same approach repeatedly... runs out of time"). That hazard is architecturally gone in `create_agent` — native tool calling has no text-parsing layer to fail on. The new failure mode is the inverse: **agents accept wrong answers silently.**

### Verified traces from this run

`temperature=0` and `gpt-4.1-nano` throughout, except where noted. The exact wording shifts run-to-run (hosted API non-determinism — see the temperature experiment in Lab 20 notes), but the failure patterns are stable across runs.

| Call | Query | Tool behaviour | Agent answer | Verdict |
|---|---|---|---|---|
| 2a | GDP totals across US/Canada/Mexico | Two parallel calls, both return 195 (regex stripped decimals) | "$195 trillion" | **Wrong** — silent acceptance |
| 2b | "Add 10, 20, two and 30" | Single call with `'10, 20, two, and 30'`, returns 60 | "60" | **Wrong** — correct answer is 62 |
| 2c | "Add the numbers -10, -20, -30" | Single call, returns 60 (regex strips minus signs) | "60" | **Wrong** — three negatives summing positive is arithmetically impossible, model didn't notice |
| 2c-local | Same prompt, Mistral 7B via Ollama | Two parallel calls: one valid returning 60, one with fabricated args `{'result': -60}` (output shape passed as input), framework rejects with Pydantic error | "-60" | **Right** — model read its own error in the message stream and corrected via internal arithmetic |
| 4a | "What is 25 divided by 4?" | Single call, returns 6.25 | "6.25" | **Right** — happy path |
| 4b | "Subtract 100, 20, and 10" | Buggy tool returns -130 | "-130" | **Wrong** — LLM trusted obviously-wrong tool |
| 4c | "Multiply 2, 3, and four" | Two parallel calls: 24 and 6 | Synthesised both: "24... and 6 considering 'four' as a string" | **Right answer present** but with confused justification |
| 4d | "Divide 100 by 5 and then by 2" | Two identical parallel calls, both return 20.0 | "10" (LLM did second division itself) | **Right** — through tool bypass |
| 6b | "Population of Canada × 0.75" | Wikipedia → multiply called with single operand `'41500000'`, returns 41500000 unchanged | "31.125 million" | **Right** — LLM did the math itself |
| 7b | "Calculate 5 to the power of 2" | Single call, typed args `{base: 5, exponent: 2}`, returns 25.0 | "25" | **Right** — typed-args path |

### Four failure modes

**1. Trust-the-tool (silent-wrong-answer).** GDP, "two and 30", subtract-bug. Tool returns a plausible-looking but wrong number; LLM accepts it; final answer is fluent and wrong. Most dangerous mode because nothing in the trace looks broken.

**2. Trust-the-tool on arithmetically-impossible output.** Negative numbers (2c). Three negative integers cannot sum to a positive integer — the wrongness is detectable from the prompt alone without doing any arithmetic. The model accepted 60 anyway. This is a sharper version of mode 1: the model is not doing sanity-checking against the prompt at all, just trusting the tool's output as ground truth.

**3. Override-the-tool (silent-right-answer-through-LLM-arithmetic).** Sequential divide (4d), population × 0.75 (6b). Tool returns something inadequate; LLM bypasses and produces the correct answer through internal arithmetic. *Worse* than visible failure because it builds confidence in a system whose underlying components are broken.

**4. Hedged execution.** GDP (2a), multiply-four (4c), sequential divide (4d). The agent issues *parallel* tool calls when uncertain about argument shape — sometimes with different arguments (GDP, multiply-four), sometimes identical (sequential divide). At minimum this is wasted latency and tokens; at worst the final synthesis confuses the user with conflicting tool outputs presented as if both were meaningful.

The lab's framing — "the model accepts plausible output but corrects implausible output" — does not survive contact with the data. The model accepts arithmetically-impossible output too. Surface-level pattern recognition does not extend to "do the signs of the inputs match the sign of the output."

---

## The Tool Contract is Name + Docstring + Behaviour

The lab's deliberate-bug subtraction tool is the cleanest demonstration of contract drift in agent systems:

```python
@tool
def subtract_numbers(inputs: str) -> dict:
    """... Example Input: "100, 20, 10". Example Output: {"result": -130}."""
    result = -1 * numbers[0]   # negate the first number
    for num in numbers[1:]:
        result -= num
    return {"result": result}
```

Three sources of truth:
- The **name** (`subtract_numbers`) implies `100 - 20 - 10 = 70`
- The **docstring example** explicitly states `-130`
- The **implementation** produces `-130`

Two of three agree. The name lies. The agent trusts the docstring example (which matches the implementation), the user expects the name's promise. **A tool's contract is its name + its docstring + its example + its behaviour. Drift between any of those is a silent-wrong-answer source.**

The fix (`new_subtract_numbers`) is one character: remove `-1 *` from the first-number initialisation. Behaviour now matches name and example.

---

## Direct Tool Tests Catch What LLM-Mediated Tests Hide

Section 3 of the lab tests the math tools directly, with no LLM. Three results worth flagging:

```
subtract_numbers.invoke('10 20 30 and four a b') = {'result': -60}
multiply_numbers.invoke('2, 3, and four ')       = {'result': 6}
divide_numbers.invoke('100, 5, two')             = {'result': 20.0}
```

The word "four" is silently dropped. "two" is silently dropped. Non-numeric noise ("a b") is ignored. None of this is visible in LLM-mediated tests because the LLM normalises word-numbers before the tool sees them.

**Lesson:** direct invocation tests expose the tool's parsing brittleness; LLM-mediated tests hide it behind the model's natural-language compensation. Production tool suites need both. The agent-test alone is insufficient because it tests model+tool jointly — failures in either are attributed to whichever layer the engineer happens to suspect.

---

## Test Harness — Contract Testing Dressed Up as Eval

The lab includes a four-case test harness. Cases are data, the loop invokes the agent, walks the message list to find the first `ToolMessage`, parses its content as JSON, compares the `result` field to an expected value.

**This is contract testing, not eval testing.** Concrete demonstration from this run:

```
Test Case 1: Subtract 100, 20, 10 (expected 70)         → PASS
Test Case 2: Multiply 2, 3, 4 (expected 24)             → PASS
Test Case 3: Divide 100 by 5 and then by 2 (expected 10.0)
   Tool returned: 20.0 (only first division)
   Agent answered: "First, dividing 100 by 5 gives 20. Then, dividing 20 by 2 gives 10."
   Harness verdict: FAIL
Test Case 4: Subtract 50 from 20 (expected -30)
   Tool returned: 30 (passed 50, 20; subtracted in literal order)
   Agent answered: "The result of subtracting 50 from 20 is -30."
   Harness verdict: FAIL

Final: 2 PASSes, 2 FAILs.
```

**Both FAILs are false negatives.** The agent answered correctly in both cases. The harness reports 50% failure rate; the user-visible accuracy is 100%.

Three categories of false signal the harness produces:

1. **Silent-wrong-answer pass.** If a tool returned the expected-but-wrong value, the test would PASS while the agent's answer was wrong. (Not triggered in this run; structurally possible.)
2. **Silent-right-answer FAIL.** Tool output diverges from expected, but LLM compensates in the final message. Reported as FAIL despite correct user-visible outcome. Both test 3 and test 4 hit this.
3. **JSON parse failure.** If a tool ever returns non-JSON content (e.g. the error-string branch of `sum_numbers_with_complex_output`), `json.loads(...)["result"]` throws. The harness handles this explicitly now; without the try/except, it would crash and be indistinguishable from a wrong-answer.

A real eval pipeline needs semantic match on the agent's *final answer* against user intent, not tool output against expected value. Tool-call inspection should be a *separate* signal. Multi-run scoring is required to handle non-determinism even at `temperature=0` (see Lab 20).

---

## Wikipedia Tool — Direct API vs Framework Wrapper

The lab uses `WikipediaAPIWrapper` from the deprecated `langchain-community` package. Migrating to direct `wikipedia.summary()` calls hit `JSONDecodeError` — Wikipedia's API returned an empty body, the parser failed.

Cause: the `wikipedia` package (1.4.0, last released 2014) doesn't set a User-Agent header. Wikipedia's API throttles or refuses anonymous requests.

Fix: `wikipedia.set_user_agent("app-name/version (contact) python-requests")`. Wikipedia's etiquette policy requests this format and rewards honest identification with better rate limits than spoofed browsers.

**Production lesson.** For any tool whose underlying library is small (`wikipedia`, `duckduckgo-search`, `requests`-based APIs), skip the LangChain wrapper entirely and write a thin `@tool` over the raw library. Six lines, zero framework versioning risk, full control of HTTP behaviour. Reserve LangChain wrappers for integrations where the wrapper does real work — auth flows, complex pagination, multi-step orchestration.

---

## Typed Args vs String Parsing — Power Tool

The IBM lab's final exercise wraps `calculate_power` with three nested regex attempts (`"5^2"`, `"2 to the power of 3"`, `"2 3"`), the `Tool()` constructor, and `initialize_agent`. Three deprecated patterns in one cell.

Modern equivalent, five lines:

```python
@tool
def calculate_power(base: float, exponent: float) -> dict:
    """Calculate base raised to the exponent (base ** exponent).

    Example: base=5, exponent=2 returns {"result": 25.0}
    """
    return {"result": base ** exponent}
```

Agent test: "Calculate 5 to the power of 2." Single tool call, args `{'base': 5, 'exponent': 2}`, result 25.0, fluent final answer. No hedging, no regex fallbacks.

**The notebook's regex-parser solution is the wrong layer.** A 20-line tool body defending against language ambiguity is doing work the LLM should do. With typed args, the LLM constructs the right floats and the framework validates them; the tool body is the actual computation, nothing else.

Whether parameter *names* matter independently of the docstring is a hypothesis this lab does not test (only the happy-path 5²=25 trace was run). Worth verifying separately.

---

## OpenAI vs Mistral 7B Local

Two backends, identical code structure (`app.py` vs `app_local.py`), identical tool definitions imported from the shared `tools.py`. Three prompts, three different failure-or-recovery patterns from Mistral on the same agent:

| Prompt | Mistral 7B local | GPT-4.1-nano |
|---|---|---|
| GDP sum | Right ($31.65T) — pre-converted decimals to integers (`27720000000000`) before calling, single tool call | Wrong ($195T) — hedged with two parallel calls, both returning 195 from decimal stripping |
| "Add 10, 20, two and 30" | 60 — two identical parallel calls, hedged final answer ("the numbers you provided", not naming the inputs) | 60 — single call, confident final answer ("The sum of 10, 20, two, and 30 is 60") |
| "Add -10, -20, -30" | Right (-60) — one valid call returning 60, plus one fabricated call passing output shape `{'result': -60}` as input; framework rejected with Pydantic error; model read its own error, recovered via internal arithmetic | Wrong (60) — single call, accepted arithmetically-impossible result |
| Speed | ~60s per call on CPU (model spilled VRAM) | ~3s per call |
| Cost | Electricity | ~$0.0001 per call |

**Mistral is not "more reliable" — it is differently unreliable.** Three patterns on the same agent: paranoid pre-processing (GDP), soft-language hedging (two-and-30), fabricate-and-recover (negatives). The right answer on the negative-numbers case came from a self-correction loop, not from tool correctness — the model passed the *output* shape `{'result': -60}` as if it were input arguments, the framework caught it, the model saw "Field required" in the next message and produced the right answer through internal arithmetic.

**The structured-tool-schema infrastructure is doing real protective work in that trace.** Without Pydantic validation, the malformed call would have either crashed or run with garbage. Instead the framework surfaced a structured error into the message stream, the LLM read it, the LLM recovered. The lab's framework critique is mostly negative — this is the one positive case.

The notebook's "Granite vs GPT switch to fix structured-tool failures" lesson is dead — `create_agent` handles any tool-calling-capable model uniformly. The new lesson the lab accidentally teaches: **same tool contract, two models, four different reliability profiles across three prompts.** "Capability" is not a scalar.

---

## VRAM Spill (Ollama Local)

Running `mistral:latest` via Ollama on a 4070 8GB initially spilled to CPU (`ollama ps` showed `100% CPU` instead of `100% GPU`). 7B model in fp16 + 16K context KV cache exceeded available VRAM after Windows + browser tabs took their share.

Fix: reduce context window via `ChatOllama(model=MODEL, temperature=0, num_ctx=4096)`. KV cache pre-allocates the full context length — 16K context = ~4GB pre-allocated before any token is sent.

**Production parallel.** Same shape as Lambda cold-start sizing — pre-allocation costs are the real budget item, not steady-state usage. "Model fits in VRAM" is misleading; "model + KV cache + system prompt + tool schemas all fit in VRAM" is the real check. Reducing context window is the cheap fix; quantisation (q4 instead of fp16) is cheaper still, but degrades structured-output fidelity faster than free-text generation — exactly the axis that matters for tool calling.

---

## Patterns Worth Retaining

**1. The agent loop is a state machine, not a Python while-loop.** `create_agent` returns a `CompiledStateGraph` from LangGraph. The "loop" is a conditional edge between an Agent Node and a Tools Node — the LLM controls iteration by emitting (or not emitting) `tool_calls`. Agent behaviour is a model-capability problem, not a framework problem.

**2. Tools execute; LLMs reason.** The whole agent pattern exists because these two roles are different. Tool calling is the seam between them. Drift at the seam (sloppy docstring, mismatched name, hidden side-effect) is where silent failures live.

**3. Native tool calling moves parsing into the LLM/framework layer.** With typed `@tool` schemas, the LLM constructs structured arguments and the framework validates them before the tool function runs. String-input tools push language understanding into the tool body, which is the wrong layer. Production tools take typed arguments.

**4. Docstring examples are runtime artefacts.** The LLM constructs tool arguments in the shape of the docstring's example. Canonical example → canonical args. Verbose example → verbose args. Invisible until you compare two tool traces with different docstring shapes side by side.

**5. The hazard moved from loud to silent.** Legacy `initialize_agent`: agents looped on bad tool outputs and timed out. Current `create_agent`: agents accept and move on. Failure changed from "fails loudly" to "fails silently." The legacy lesson is gone; the new one isn't yet in any textbook.

**6. Pydantic schema validation is a load-bearing safety net.** The Mistral negative-numbers trace produced a malformed tool call (output shape passed as input arguments). The framework rejected it with a structured error message that became part of the message stream, the LLM read its own error, and recovered via internal arithmetic. Without typed schemas, the malformed call would either crash or run with garbage. The structured-tool-schema infrastructure is doing real protective work here — worth flagging because most of the lab's framework critique is negative, and this is one of the few positive cases.

**7. Tool-result-was-load-bearing is an open eval question.** The traces show what happened. They don't show whether the tool's output drove the final answer or whether the LLM compensated and the tool call was theatre. The lab doesn't validate this hypothesis at scale — it provides one concrete example (population × 0.75) where the tool output was bypassed.

**8. API churn is the ecosystem maturing, not breaking.** Three generations of agent API in 18 months. Each rename signals an architectural assumption hardening into a constraint. Thin `@tool` decorators over your own functions survive the churn. Imports from `langchain-community` don't.

**9. Visibility moved from constructor flag to method choice.** Legacy `initialize_agent(verbose=True)` printed intermediate state synchronously during text parsing. `create_agent` runs on LangGraph; state transitions are events. `.invoke()` returns the final state; `.stream()` yields each tick. Production observability inherits from this — the same primitive that drives debug output drives LangSmith traces and OpenTelemetry spans.

---

## Rovers Connection

**The agent-tier failures here are the same shape as RAG-tier failures Hooperman has already navigated.**

The population × 0.75 trace (section 6b) is structurally identical to the canonical RAG fallback failure: deterministic component (tool / retriever) is called, returns something inadequate, LLM bridges the gap from internal knowledge, final answer is fluent, trace shows "tool was used / retrieval happened" — nothing flags that the answer wasn't actually grounded in the deterministic output. The countermeasure in RAG is grounding evaluation; the analogous countermeasure in agents is "was the tool's output load-bearing in the final answer." Neither check ships in any platform today.

**The harness limitation maps directly to Hooperman's eval framework.** The IBM lab's harness compares *tool output* to expected; both Test 3 and Test 4 in this run FAILed despite correct agent answers. Hooperman's golden questions compare *agent answer* via semantic match — that's the right shape, this lab demonstrates why exact-match-on-tool-output is the wrong shape.

**The temperature experiment from Lab 20 cashes out here too.** Test cases that PASS on one run might FAIL on the next, even at `temperature=0`. Multi-run scoring (N=3 or N=5) is required for any production eval on hosted APIs. The IBM harness as written would have flipped behaviour between runs — single boolean PASS/FAIL flattens both non-determinism and the silent-right-answer pattern into the same signal.

---

## What This Doesn't Cover

- **Streaming responses.** `create_agent` supports `.stream()` instead of `.invoke()`. The lab doesn't exercise this; production UX needs it.
- **Structured output (JSON mode).** Math tools return `dict`; Wikipedia returns `str`. No use of OpenAI's structured-output mode or Pydantic response schemas. Production agents needing typed final outputs should request them at the model level.
- **Middleware.** `create_agent` supports before/after model hooks and custom tool execution. Useful for rate-limiting hedged duplicate tool calls (see section 4d's two identical divides), logging, redaction. Not exercised.
- **Multi-agent coordination.** Single agent throughout. Multi-agent patterns (Agent A delegates to Agent B) are a separate course module.
- **Eval pipeline.** The four-case harness is contract testing; a real eval needs semantic match on agent answers, multi-run scoring, grounding checks. Hooperman's framework is closer to the production shape.
- **Cost tracking.** No token counting, no per-call cost logging. Parallel hedged tool calls (sections 2a, 4c, 4d) are a noticeable cost driver at scale and would be flagged here.
- **Observability.** No tracing (LangSmith / OpenTelemetry), no structured logs. The `print_trace()` helper is the only observability and runs at debugging time, not in production.