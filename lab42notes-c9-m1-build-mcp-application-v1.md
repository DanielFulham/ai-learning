# Course 9 - Lab 42: Build an MCP Application (V1)

> Code: [`course9-module1-lab2/`](course9-module1-lab2/)

A command-line agent that uses two prebuilt MCP servers through one `MultiServerMCPClient` from `langchain-mcp-adapters`: Context7 (library documentation) over Streamable HTTP and a Metropolitan Museum of Art server over stdio. Their tools go to a `create_agent` agent with `InMemorySaver` memory, and a menu loop keeps one conversation going. No MCP server is built.

This is a port of IBM Skills Network's "Build an MCP Application" lab (the lab text is not included here) on Python 3.14 and Windows, using Anthropic Haiku 4.5 in place of `gpt-5-nano`. The lab's pinned install fails, and its Met server cannot search (see Findings).

## Run it

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py        # needs Node.js (npx), network access and a .env with ANTHROPIC_API_KEY
```

| | |
|---|---|
| Python | [python --version] |
| Node.js | [node --version] |
| `langchain-mcp-adapters` | 0.3.2 |
| `langchain` / `langgraph` | 1.4.3 / 1.2.12 |
| `langchain-anthropic` / `langchain-core` | 1.7.5 / 1.6.6 |
| `mcp` (capped below 2 by the adapters) | [pip show mcp] |
| Met server | `@cyanheads/met-museum-mcp-server@0.7.0` |

## File layout

```
course9-module1-lab2/
├── main.py                    # the agent and CLI loop
├── probe_met_tools.py         # lab's Met server: tool schemas (no model)
├── probe_met_search.py        # lab's Met server: one search call (no model)
├── probe_met_working.py       # lab's Met server: the tools that don't use search (no model)
├── probe_cyanheads_tools.py   # replacement Met server: tool schemas (no model)
├── pyproject.toml
└── requirements.txt
```

## Differences from the lab

- The lab's five pins are replaced by the set above; `langchain-openai` and `langchain-ibm` are dropped.
- `ChatAnthropic` (`claude-haiku-4-5`) replaces `ChatOpenAI`; the watsonx fallback is dropped (its `project_id` works only inside IBM's environment).
- `create_react_agent` becomes `create_agent`, with the same arguments.
- `config` is typed `RunnableConfig`, messages are sent as role/content lists, and replies are read with `.text`. The lab's forms fail type-checking (pyright 1.1.414, Python 3.12) or can return a list of content blocks.
- The Met server is `@cyanheads/met-museum-mcp-server@0.7.0`, pinned, in place of the unpinned `metmuseum-mcp`.
- `trace_turn` is added (not in the lab): per turn it prints the tools called, each result's `status` and the tokens used.

## Findings

**The lab's pinned install fails.** On Linux with Python 3.12 the pins install, resolve `mcp` 2.3.0, and the first import fails (`streamablehttp_client` was renamed in `mcp` 2.x). On Windows with Python 3.14, `langchain-ibm==0.3.18` cannot install because its releases from 0.3.10 declare `Requires-Python <3.14`. `langchain-mcp-adapters` caps `mcp<2` only from 0.3.2 (6 Aug 2026); `langchain` 1.4.0 adds `langchain.mcp` as a replacement, which needs a separate environment.

**The lab's Met server cannot search.** The Met retired `/public/collection/v1/search` on 2026-10-01: it now returns HTTP 410 and names `/v1.1/search`, which returned 200. `metmuseum-mcp` 1.0.0 (the latest npm release, February 2026) still failed with `status='error'` and "rejected the request (HTTP 410)". Its `list-departments` and `get-museum-object` tools still worked. `@cyanheads/met-museum-mcp-server@0.7.0` searches on `/v1.1`; asked for a van Gogh painting, the agent answered *The Potato Peeler*, 1885, object 438722, and the Met's own record matched every field checked except a "CC0 image" claim the record does not show.

**The failure was reported as temporary.** With the lab's server the agent said it hit "technical difficulties" and offered to retry; a 410 from a retired endpoint is permanent. What the model actually received was not printed, so this is the agent's summary.

**The lab's "persistent connection" for stdio does not hold for the default client.** The adapters' README says each tool invocation starts a new session. On a local stand-in server, `get_tools()` started the process once and each call started it again. Not measured on the real Met server.

**Six tool definitions cost about 4,000 input tokens, and every model call resends the thread.** Measured on the introduction turn (one call, a two-line prompt): 4,028 input tokens. A follow-up that used no tools still cost 7,809, and a turn with three model calls cost 16,500 to 26,000. Two sessions came to 46,605 and 28,545 input tokens, about $0.054 and $0.036 at Anthropic's listed Haiku 4.5 prices, assuming no prompt caching (not checked).

**Tool use followed the descriptions in the turns observed.** The model called `resolve-library-id` before `query-docs` every time, issued independent calls from both servers in the same step, and answered a follow-up from history. These are single runs and say nothing about rates.

## What this doesn't cover

- Stdio process starts and `open-met-explorer` (an MCP Apps tool the lab's server provides) were not tested on the real server, and the lab's failing search was not traced.
- The `MCPAdapter` path (V2); the claim that the adapters package will be archived rests on one maintainer comment.
- Tool-output safety and supply-chain review of the packages run through `npx` (both Met servers are pinned).
- One machine; Context7 and the model were observed in a few sessions.

---

**Completed:** 4 October 2026