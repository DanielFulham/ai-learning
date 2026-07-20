# Lab 36 - CrewAI Agents-with-Tools vs Tasks-with-Tools

> Code: [`course8-module2-lab3-v1/`](course8-module2-lab3-v1/)

Course 8 Module 2. Modernises the Daily Dish notebook (watsonx Granite → Anthropic Haiku 4.5 native) and closes an open question the notebook raises: **when both an Agent and a Task declare tools, does the task-level list override the agent-level list, or augment it?**

Three approach scripts against identical fixture queries, plus a custom `@tool` decorator appendix. 14 kickoffs total. Task-level tools override - confirmed by running it.

## What the lab teaches vs what the code proves

The notebook shows two shapes:
1. **Agent-centric** - agent has all tools, LLM picks per query
2. **Task-centric** - agent has no tools, each task binds its own

The case in between is untested: agent has both tools, task binds one. The CrewAI docs and the notebook narrative both assert override behaviour without exercising it. This lab adds a third script (`override_test.py`) that closes the loop.

## Structure

```
course8-module2-lab3-v1/
├── data/
│   └── daily_dish_faq.pdf          # cached on first run
├── src/
│   ├── daily_dish/
│   │   ├── __init__.py
│   │   ├── config.py               # LLM factory + env + sandbox flag
│   │   ├── tools.py                # PDFSearchTool + SerperDevTool factories
│   │   ├── fixtures.py             # shared query set - no cross-approach imports
│   │   ├── agent_centric.py        # Approach 1
│   │   ├── task_centric.py         # Approach 2
│   │   └── override_test.py        # Approach 3 - the override test
│   └── calculator/
│       ├── __init__.py
│       ├── config.py               # LLM factory + env, self-contained
│       └── calculator.py           # @tool decorator demo (appendix)
├── .env                            # ANTHROPIC_API_KEY, SERPER_API_KEY
├── .env.example
├── requirements.txt
└── pyproject.toml                  # ruff config
```

Two peer subpackages. Each stands alone with its own `config.py`. No shared code between them - the calculator appendix uses `@tool` decorators and doesn't touch PDFSearchTool, SerperDevTool, or the fixture set. Duplicating ~10 lines of env loading and `build_llm()` was cheaper than extracting a shared surface for a lab that ships and stops.

## Setup

```
cd course8-module2-lab3-v1
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env  # then populate ANTHROPIC_API_KEY and SERPER_API_KEY
```

Requirements pinned:
```
anthropic==0.116.0
crewai==1.15.2
crewai-tools[rag]==1.15.2
python-dotenv==1.2.2
sentence-transformers==5.1.0
```

Two changes from L35's requirements: `[rag]` extras marker on crewai-tools (needed for PDFSearchTool's embedding + vector-db substrate), and explicit `sentence-transformers` (needed for local embeddings - see F-M2).

## How to run

```
python src\daily_dish\agent_centric.py       # Approach 1
python src\daily_dish\task_centric.py        # Approach 2
python src\daily_dish\override_test.py       # Approach 3
python src\calculator\calculator.py          # Appendix
```

Each script kicks off its crew against the same fixture list (or the calculator's own two-query set for the appendix) and prints per-query token deltas.

First run of any daily_dish script: PDF downloads to `data/`, sentence-transformers pulls `all-MiniLM-L6-v2` (~90MB) to `~/.cache/huggingface/`. Cached thereafter.

## Fixture queries

Three FAQ-answerable + one deliberately not-in-FAQ:

```
What are the timings?
What is the phone number?
What is the location?
What are some nearby parking options?
```

The parking query is load-bearing: it's the only fixture that reaches for Serper on Approach 1, so it's the only one that reveals override behaviour on Approach 3.

## Modernisation findings

**F-M1 - `crewai-tools[rag]` extras replace three langchain deps.** The original notebook installs `langchain-community + langchain-huggingface + sentence-transformers` separately for PDFSearchTool's embedding backend. crewai-tools 1.15.2's `[rag]` extras pulls `chromadb + lancedb + pdfminer + pdfplumber + pypdfium2 + openai + tiktoken` in one marker, dropping the langchain family entirely. Only sentence-transformers stays as an explicit pin, since `[rag]` defaults to OpenAI embeddings (see F-M2).

**F-M2 - PDFSearchTool's `provider="huggingface"` now means the HuggingFace Inference API (remote, requires api_key), not local embeddings.** This is a semantic shift, not a rename. Local sentence-transformers embeddings are now `provider="sentence-transformer"` (hyphen, no 's'). Straight-porting the original config (`provider="huggingface"` with a model path expecting local execution) would silently switch from local-free to remote-authenticated calls. Type inspection of `RagToolConfig` on crewai-tools 1.15.2 confirmed both shape and semantics.

The correct-current config shape:
```python
rag_config: RagToolConfig = {
    "embedding_model": {
        "provider": "sentence-transformer",
        "config": {"model_name": "all-MiniLM-L6-v2"},
    },
}
```

`vectordb` omitted deliberately - `RagToolConfig` is `total=False`, defaults to chromadb which is what `[rag]` installs. Only earn `vectordb` config if you need qdrant or non-default chromadb settings.

**F-M3 - Two more tool-availability mechanisms exist outside this lab's scope.** `PDFKnowledgeSource` attaches retrieval as Agent/Crew-level Knowledge rather than a Tool. Anthropic-specific dynamic tool injection (March 2026) selects tools per-turn instead of per-agent or per-task. Neither exercised here - banked as prior art for future labs.

**F-M4 - CrewAI 1.15.2 ships a mandatory file-path sandbox as CVE response.** CVE-2026-2285 (arbitrary local file read) and CVE-2026-2286 (SSRF), disclosed March 2026 via CERT/CC VU#221883. Remediation: `validate_file_path()` and `validate_url()` enforced in RagTool and 20+ tools. Any pre-CVE notebook that passes absolute paths outside the process cwd fails with `Blocked unsafe file path`. Escape hatch is `CREWAI_TOOLS_ALLOW_UNSAFE_PATHS=true`, documented for "tests and trusted pipelines."

This lab sets the flag in `config.py`:
```python
os.environ.setdefault("CREWAI_TOOLS_ALLOW_UNSAFE_PATHS", "true")
```

The correct-by-construction fix is packaging the code to run with cwd = lab root (matching the sandbox's boundary to the data location). Deferred to V2 - see deferrals below.

## Findings from running it

Full three-way token comparison across identical fixture queries:

| Query | A1 prompt | A1 comp | A2 prompt | A2 comp | A3 prompt | A3 comp |
|---|---|---|---|---|---|---|
| 1: timings | 5745 | 371 | 3133 | 248 | 4849 | 325 |
| 2: phone | 3029 | 216 | 3151 | 260 | 3860 | 256 |
| 3: location | 3026 | 264 | 3178 | 325 | 3842 | 240 |
| 4: parking | 5246 | 364 | 3134 | 265 | 3843 | 283 |
| **Total** | **17046** | **1215** | **12596** | **1098** | **16394** | **1104** |

**F-L36-1 - Agent tool-selection is waterfall, not classification.** Every query hits PDFSearchTool first, escalating to Serper only when the PDF result is insufficient - Query 4 shows this: the FAQ's minimal parking answer, then Serper for more. Instructing an agent to "pick the best tool" produces try-in-order behaviour, not classification.

**F-L36-2 - Task-centric costs 26% fewer prompt tokens despite needing two LLM calls per query instead of one.** 12596 vs 17046 across 4 queries. The savings come from Task 2 (drafting) carrying no tool descriptions - just the retrieved context from Task 1. Confirms L35's finding that tool descriptions dominate the prompt-token floor; more tasks doesn't mean more cost when the extra task is cheap.

**F-L36-3 - Task-centric answers are strictly grounded; agent-centric can hallucinate from Serper.** On Query 4, Approach 1's Serper result described a real namesake restaurant in Silver Spring, MD - fabricated detail for the fictional Daily Dish. Approach 2 stayed inside the FAQ (valet + street parking, nothing more). Task-centric can't reach beyond the corpus, so it can't confabulate.

**F-L36-4 - `cached_prompt_tokens=0` across all 14 kickoffs.** Third consecutive lab (L34, L35, L36) confirming CrewAI 1.15.2's native Anthropic provider doesn't wire through prompt caching.

**F-L36-5 - Task-level tools override agent-level tools during task execution, but only when the task's list is non-empty.** The lab's core finding. Agent declared `[pdf_tool, serper_tool]`, Task 1 bound only `[pdf_tool]`. On Query 4, Serper never fired - Approach 3's answer matched Approach 2's exactly (FAQ-grounded, no fabricated web data). Override is deterministic replacement at the invocation layer, confirming the CrewAI docs' claim - for Task 1. Task 2 sets no tools of its own, and CrewAI's `Task.check_tools` validator backfills an empty task-level list from the agent's tools whenever the agent has any (`crewai/task.py`). So Task 2 actually has both tools available, same as the agent - it just never invokes either, since Task 1's retrieved context is already enough to draft a response. `tools=[]` on a task does not mean "no tools" when its agent has tools; it means "inherit the agent's tools."

**F-L36-6 - Override isn't cost-free.** Approach 3 costs 16394 prompt tokens total, between A1 (17046) and A2 (12596) - the Serper tool description still rides in the prompt even though it's unreachable at task-execution time (~700-800 tokens/query more than Approach 2). Use `tools=[]` on the agent if nothing else needs the excluded tool; override at task level only if something else does.

## Approach comparison - synthesis

| | Approach 1 | Approach 2 | Approach 3 |
|---|---|---|---|
| Agent tools | `[pdf, serper]` | `[]` | `[pdf, serper]` |
| Task 1 tools | (single-task shape) | `[pdf]` | `[pdf]` |
| Task 2 tools | - | `[]` | `[pdf, serper]` (agent fallback, unused) |
| Tasks per query | 1 | 2 | 2 |
| Prompt token floor | ~3k (steady state) + tool desc | ~3.1k per task | ~3.8k per task |
| Query 4 behaviour | PDF → Serper escalation | PDF only | PDF only (override) |
| Query 4 grounding | Corpus + web (can hallucinate) | Corpus only | Corpus only |
| Best for | Broad queries where escalation matters | Strict-grounded pipelines | Reusable agent, per-task constraint |

## Prior art carried across from L34 and L35

Confirmed on L36:

- **L34 finding: `max_iter` as primary prompt-token lever.** Held. Kept default of 25; would tighten if this were a longer-running crew.
- **L34 finding: `allow_delegation=True` inflates cost invisibly.** Held. `allow_delegation=False` on every agent across all four scripts.
- **L35 finding: capability removal is the reliable constraint mechanism, not prompt instructions.** Sharpened by F-L36-5 - task-level override is the finest-grained capability-removal mechanism CrewAI offers, and it works at the invocation layer even when the agent-level declaration is broader.
- **L35 finding: tool descriptions ride in every prompt regardless of use.** Confirmed by F-L36-2 and F-L36-6.
- **L35 finding: prompt caching definitively zero on CrewAI 1.15.2 native Anthropic.** Confirmed a third time via F-L36-4.
- **L35 finding: `output_pydantic` as prompt-embedded schema + post-hoc extraction.** Not exercised in L36 - no structured outputs in this lab.

New to L36:

- Framework-level tool-selection is *waterfall*, not classification (F-L36-1). This changes how you'd write prompt instructions for tool routing - instructing "pick the best tool" doesn't produce classification behaviour.
- Approach shape has a trustworthiness dimension, not just a cost dimension (F-L36-3). Task-centric is safer in RAG contexts because the corpus is the ceiling.
- Override is real and cheap-ish but not free (F-L36-6). The agent-level declaration still costs prompt tokens even when overridden.

## Deferrals

Tracked in cert-deferrals.md.

**L36-1 - Onion architecture port.** Same shape as L34-1, L35-3. The lab as it stands is script-per-approach in the src-layout package shape. An onion port would extract domain (agent/task/crew concepts as pure types), application (the three approach shapes as use-case orchestrations), infrastructure (CrewAI/Anthropic/Serper adapters). Cert velocity constraint - defer to when the onion pattern is genuinely load-bearing rather than academic.

**L36-2 - Correct-by-construction sandbox fix.** Currently using `CREWAI_TOOLS_ALLOW_UNSAFE_PATHS=true` in config.py because data lives outside the process cwd under the current run shape. The correct fix is packaging the lab so cwd matches the data location - either editable install with `pip install -e .` and running as `python -m daily_dish.agent_centric` from lab root, or shifting data into the package tree. Both work; both have friction. Deferred pending a broader review of Python packaging shape across all labs.

**L36-3 - Calculator without duplication.** Calculator's `config.py` duplicates `daily_dish/config.py`'s env-loading and `build_llm()`. Fine at two-subpackage scale, breaks down at 5+ subpackages. Extract to `src/shared/config.py` when there are enough consumers to justify the abstraction cost. Not yet.

**L36-4 - Approach 2 with Serper.** As modelled here (following the notebook), Approach 2 has no path to Serper - Query 4 returns a narrower answer than Approach 1. A production task-centric shape would add a third task (`web_search_task`) with Serper bound, either always-on or conditional on Task 1's confidence. Explicit branching is the mechanism CrewAI Flows provides. Deferred to a future lab that specifically explores task branching.

**L36-5 - Investigate CrewAI native provider's prompt caching gap.** L34/L35/L36 all confirm `cached_prompt_tokens=0`. Root cause is a CrewAI-layer choice (likely not sending `cache_control` blocks to Anthropic). Anthropic SDK 0.116.0 supports caching without beta headers on any TTL. Would take source-diving crewai/litellm/anthropic paths. Deferred - cost of investigation not yet justified by savings.

## Test protocol

Each script self-tests via the fixture runner:

```
python src\daily_dish\agent_centric.py
python src\daily_dish\task_centric.py
python src\daily_dish\override_test.py
python src\calculator\calculator.py
```

Expected outputs per script are documented above (token totals, tool paths per query). Deviation from expected - particularly Serper invocation appearing in `override_test.py`'s Query 4 stdout - would invalidate F-L36-5 and warrant re-investigation.

Reproducibility notes:

- `temperature=0.0` on the LLM factory - closest thing to deterministic Anthropic responses; still not fully deterministic run-to-run.
- Fixture queries are literal strings, no normalization, no case changes.
- MiniLM embeddings are deterministic given the same corpus and query strings - retrieval order won't vary across runs.
- Token counts may drift ±5% run-to-run due to LLM non-determinism at temperature 0; the 26% Approach 1 → Approach 2 delta is well outside that noise floor.