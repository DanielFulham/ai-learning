# Course 6 — Lab 24: YouTube Tool-Calling Agent — Three Orchestration Strategies

Onion-architected YouTube agent that implements the same `VideoAgentInterface`
three ways: an imperative `while`-loop, a fixed two-step LCEL chain, and a
recursive LCEL chain that runs until the model stops emitting tool calls. The
choice of orchestration is a one-line change in the composition root; the
tools, infra clients, LLM provider, and tests are shared across all three.
Demonstrates that agent orchestration is an *implementation detail* behind a
stable interface, not a project-wide architectural commitment.

Built as part of the IBM RAG and Agentic AI Professional Certificate — Course
6, Module 2.

---

## What It Does

Six tools wired into one agent surface. The LLM decides which to call:

- `extract_video_id(url)` — regex over URL forms (watch, embed, shorts, live, youtu.be)
- `fetch_transcript(video_id, language="en")` — youtube-transcript-api
- `search_youtube(query, max_results=5)` — pytubefix Search, clamped at the boundary
- `get_full_metadata(url)` — title, views, duration, channel, likes, comments, chapters
- `get_thumbnails(url)` — list of thumbnails with resolutions
- `get_trending_videos(region_code, max_results=25)` — broken externally, documented as such

Three agents run on top, all implementing the same interface:

- **`ManualLoopAgent`** — imperative `while messages[-1].tool_calls:` loop.
  The substrate. No LCEL, no recursion. Easiest to debug and instrument.
- **`TwoStepChainAgent`** — LCEL chain hardcoded for exactly two tool calls
  then a summary. Works for "summarise this video", fails for anything
  needing one or three steps. Included as the pedagogical waypoint that
  shows what hardcoded orchestration costs.
- **`RecursiveAgent`** — LCEL chain that loops until the model emits an
  `AIMessage` with no `tool_calls`. The shape `create_agent` uses
  internally. Handles arbitrary tool sequences.

The container picks which agent to instantiate via an `AgentStrategy` enum.
Everything else — tools, LLM, infra clients — is identical across the three.

---

## Stack

| Component             | Implementation                                                          |
| --------------------- | ----------------------------------------------------------------------- |
| LLM                   | `gpt-4.1-nano` via `init_chat_model`                                    |
| Agent contracts       | `VideoAgentInterface.run(query: str) -> str` — three implementations    |
| Orchestration         | Manual `while`-loop, hardcoded LCEL, recursive LCEL                     |
| Tool decoration       | `@tool` from `langchain.tools` (LangChain 1.x canonical)                |
| Message types         | `langchain.messages` (LangChain 1.x canonical)                          |
| LLM provider coupling | `OpenAIChatModelProvider` behind `ChatModelProviderInterface`           |
| External clients      | `YouTubeTranscriptApi`, `pytubefix.Search`, `yt_dlp.YoutubeDL`          |
| Architecture          | Strict onion — domain (empty by design), interfaces, application, infra |
| Test surface          | 63 tests, all passing without an API key or network access              |
| Strategy selection    | `AgentStrategy` enum, one-line swap in `container.py`                   |

---

## Setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -U langchain langchain-openai python-dotenv youtube-transcript-api pytubefix yt-dlp pytest
```

Create `.env` next to the demo scripts:

```
OPENAI_API_KEY=sk-...
```

Run any demo:

```powershell
python manual_loop_demo.py
python two_step_chain_demo.py
python recursive_agent_demo.py
```

Run the tests (no API key, no network):

```powershell
pytest tests/ -v
```

---

## File Layout

```
course6-module2-lab3/
├── manual_loop_demo.py                           # entry: strategy=MANUAL_LOOP
├── two_step_chain_demo.py                        # entry: strategy=TWO_STEP_CHAIN
├── recursive_agent_demo.py                       # entry: strategy=RECURSIVE
├── conftest.py                                   # pytest sys.path setup
├── .env
│
├── application/
│   ├── __init__.py
│   ├── manual_loop_agent.py                      # imperative while-loop
│   ├── two_step_chain_agent.py                   # hardcoded LCEL chain
│   ├── recursive_agent.py                        # recursive LCEL chain
│   ├── container.py                              # composition root, AgentStrategy enum
│   ├── interfaces/
│   │   ├── __init__.py
│   │   └── video_agent_interface.py              # run(query: str) -> str
│   └── tools/
│       ├── __init__.py
│       ├── extract_video_id.py                   # pure regex — no infra
│       ├── fetch_transcript.py                   # wraps TranscriptClient
│       ├── search_youtube.py                     # wraps YouTubeSearchClient
│       ├── get_full_metadata.py                  # wraps YouTubeMetadataClient
│       ├── get_thumbnails.py                     # wraps YouTubeMetadataClient
│       └── get_trending_videos.py                # wraps YouTubeMetadataClient
│
├── interfaces/
│   ├── __init__.py
│   ├── chat_model_provider_interface.py          # Protocol — LLM provider
│   ├── transcript_client_interface.py            # Protocol — transcripts
│   ├── youtube_search_client_interface.py        # Protocol — search
│   └── youtube_metadata_client_interface.py      # Protocol — yt-dlp ops
│
├── infra/
│   ├── __init__.py
│   ├── openai_chat_model.py                      # OpenAIChatModelProvider
│   ├── youtube_transcript_api_client.py          # youtube-transcript-api
│   ├── pytubefix_search_client.py                # pytubefix.Search
│   └── yt_dlp_metadata_client.py                 # yt-dlp + private _YtDlpLoggerAdapter
│
└── tests/
    ├── __init__.py
    ├── application/
    │   ├── __init__.py
    │   ├── test_manual_loop_agent.py             # 5 tests, LLM mocked
    │   ├── test_two_step_chain_agent.py          # 1 test, LLM mocked
    │   ├── test_recursive_agent.py               # 4 tests, LLM mocked
    │   ├── test_container.py                     # 7 tests, provider mocked
    │   └── tools/
    │       ├── __init__.py
    │       ├── test_extract_video_id.py          # 13 tests, pure
    │       ├── test_fetch_transcript.py          # 4 tests, client mocked
    │       ├── test_search_youtube.py            # 4 tests, client mocked
    │       └── test_metadata_tools.py            # 6 tests, client mocked
    └── infra/
        ├── __init__.py
        ├── test_openai_chat_model.py             # 3 tests, init_chat_model patched
        ├── test_youtube_transcript_api_client.py # 3 tests, library patched
        ├── test_pytubefix_search_client.py       # 4 tests, library patched
        └── test_yt_dlp_metadata_client.py        # 9 tests, library patched
```

---

## Key Concepts

### Three Orchestrations Behind One Interface

`VideoAgentInterface.run(query: str) -> str`. Every implementation honours
this. The container instantiates the chosen one based on an `AgentStrategy`
enum value:

```python
def initialise(
    strategy: AgentStrategy = AgentStrategy.RECURSIVE,
    chat_model_provider: ChatModelProviderInterface | None = None,
) -> VideoAgentInterface:
    ...
    if strategy is AgentStrategy.MANUAL_LOOP:
        return ManualLoopAgent(llm, tools)
    if strategy is AgentStrategy.TWO_STEP_CHAIN:
        return TwoStepChainAgent(llm, tools)
    if strategy is AgentStrategy.RECURSIVE:
        return RecursiveAgent(llm, tools)
```

Demos differ in one parameter:

```python
# manual_loop_demo.py
agent = initialise(strategy=AgentStrategy.MANUAL_LOOP)

# recursive_agent_demo.py
agent = initialise(strategy=AgentStrategy.RECURSIVE)
```

### Why Three?

Each implementation answers a different production question:

- **Manual loop** — what `create_agent` does for one turn, made visible.
  When debugging an opaque agent failure, this is the version you can step
  through. Lab 23 used this shape.
- **Two-step chain** — what happens when orchestration is hardcoded. Works
  cleanly for "summarise this YouTube video" (two tools needed). Fails for
  trending queries (one tool) and multi-tool flows (three+ tools). Kept
  as evidence that bad orchestration can hide behind a clean interface
  until the workload changes.
- **Recursive chain** — what `create_agent` does for arbitrary tool
  sequences. The model decides when to stop. The shape every production
  agent framework converges on.

Same interface; testably different orchestration shapes; same tools and
infra underneath. This is the strategy pattern applied at the agent level.

---

## Architecture Walkthrough — One Call

When `python recursive_agent_demo.py` runs `agent.run("Summarize https://...")`:

```
recursive_agent_demo.py
  initialise(strategy=AgentStrategy.RECURSIVE)
    → OpenAIChatModelProvider().create() → BaseChatModel       [infra]
    → YouTubeTranscriptApiClient()                              [infra]
    → PytubefixSearchClient()                                   [infra]
    → YtDlpMetadataClient()                                     [infra]
    → make_*(client) × 6 → list[BaseTool]                       [application/tools]
    → RecursiveAgent(llm, tools)                                [application]
  agent.run(query)
    → first invoke: send query + tool schemas
      → AIMessage(content='', tool_calls=[extract_video_id(url)])
    → _recurse(messages):
      → should_continue? yes
      → execute extract_video_id via tool_map
        → tool's closure calls into pure regex                  [application/tools]
        → returns "T-D1OfcDW1M"
      → ToolMessage(content="T-D1OfcDW1M", tool_call_id=...)
      → invoke with new history
        → AIMessage(content='', tool_calls=[fetch_transcript(id)])
      → should_continue? yes
      → execute fetch_transcript via tool_map
        → tool's closure calls client.fetch(id, "en")
          → YouTubeTranscriptApiClient.fetch()                  [infra]
          → returns transcript text
      → ToolMessage(content=transcript, tool_call_id=...)
      → invoke with new history
        → AIMessage(content="Summary: ...")
      → should_continue? no — return messages
    → isinstance(messages[-1].content, str) — passes
    → return messages[-1].content
```

Every transition is explicit. Layer crossings are visible: the agent never
imports infra; the tool factories use interface types; the container is
the only place infra and application meet.

---

## Production Insights

**Agent orchestration is an implementation detail, not architecture.**
The same `VideoAgentInterface` admits three implementations with wildly
different control flow (imperative loop, fixed chain, recursive chain).
The choice belongs in config, behind the interface, where it can be
A/B tested or swapped per query class. Coupling business logic to a
specific orchestration shape (LCEL chain, LangGraph, manual loop) is the
same category mistake as coupling business logic to a specific database
client.

**Hardcoded orchestration shapes leak into the workload.** The two-step
chain works fine on "summarise this video" and fails on "show me trending
videos" because the latter needs one tool call, not two. The chain
silently does nothing useful — it runs the second invocation against a
state that doesn't justify a second call. Tests pass; production users
get garbage. Recursive orchestration costs more code internally and saves
more failures externally. Pay the cost early.

---

## What This Doesn't Cover

- **Max-iteration cap.** `RecursiveAgent` will loop until the model
  stops emitting tool calls. A misbehaving model can drive the context
  length to exhaustion. Production code should accept `max_iterations`
  in the constructor and raise with a clear error after exhaustion.
- **Streaming.** All agents use `.invoke()`. For token-level streaming,
  the recursive shape would need restructuring as a generator — LCEL's
  streaming model doesn't propagate through arbitrary Python recursion.
- **Conversation memory.** Each `agent.run(query)` starts a fresh
  conversation. No state survives between calls. The container returns
  a fresh agent each time anyway, so memory would need to live in a
  conversation manager above the agent layer.
- **Observability.** No spans, no metrics, no LangSmith integration.
  The recursive shape exposes natural span boundaries — each `invoke`,
  each tool dispatch, each `tool_call_id` — but nothing is wired.
  Hooperman has the cost-and-latency story; this lab does not.
- **`get_trending_videos`.** YouTube removed the global trending feed
  in 2024+. The tool is preserved with a `NOTE:` in its docstring so the
  refactor stays faithful to the lab while documenting the breakage.
  Production version would use the YouTube Data API v3.

---

## What's Next

The same shape extends naturally. New agent strategy (e.g. `LangGraphAgent`
using `create_agent` directly) — new class in `application/`, new enum
value, one line in `container.py`. New external service (e.g. captions
translation) — new interface, new infra class, new tool factory, one
line in `container.py`. New consumer of video data (e.g. a typed JSON
API endpoint) — domain layer earns its keep, with `VideoMetadata`,
`Thumbnail`, and `SearchHit` modelled in `domain/` and mapped to/from
infra returns in the application layer.

The folder structure tells the story; the test suite proves the seams;
the composition root makes orchestration choice the single swap surface.

---

**Completed:** 9 June 2026