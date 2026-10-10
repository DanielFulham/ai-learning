# Course 9 - Lab 43: Hello World of MCP Servers (V1)

> Code: [`course9-module2-lab1/`](course9-module2-lab1/)

A small MCP server built with FastMCP (two tools, two resources, one prompt) and every way the lab reaches it: an in-memory client, Streamable HTTP served from inside the same process, and stdio through a subprocess. A `create_agent` agent on Haiku 4.5 then uses the same server's tools over HTTP, over stdio, and over both at once through `MultiServerMCPClient`. One server definition feeds every transport; `raw_client_demo.py` drives it with the raw client and `agent_demo.py` drives it with the agent.

This is a port of IBM Skills Network's "Hello World of MCP Servers" notebook (the notebook is not included here) on Python 3.14 and Windows. The notebook was ported cell by cell, run and debugged, then restructured into onion layers. 58 tests pass without an API key.

The most useful result is about the server's own surface: two servers exposing the same tool names reach a model without any signal of which server answered.

## Run it

From the repo root:

```powershell
cd course9-module2-lab1
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Developed on Python 3.14 and Windows; commands are PowerShell (on Linux or macOS use `source venv/bin/activate`; the tests were also run on Linux, the demos only on Windows). Node.js is not needed. pip needs network access, and the first `in-memory` run downloads two small IBM sample files into `path/`. The HTTP demos need port 8000 free. Only the agent demos need an Anthropic API key: put `ANTHROPIC_API_KEY=your-key` in a `.env` file in the lab folder (or a parent folder).

Raw client, no model and no key (start here):

```powershell
python raw_client_demo.py in-memory    # tools, both resources and the prompt
python raw_client_demo.py http         # the same server over Streamable HTTP, served from this process
python raw_client_demo.py stdio        # the same server launched as a subprocess
python raw_client_demo.py local-tool   # a plain LangChain tool for contrast, no MCP
```

Agent on Haiku 4.5 (needs the key):

```powershell
python agent_demo.py stdio             # [HUMAN] ... [AI] tool call: add ... [TOOL] 3 ... [AI] answer
python agent_demo.py http
python agent_demo.py multi             # stdio and HTTP servers together; the tool is prefixed, e.g. stdio-client_add
```

Tests (58, no key and no network): `pytest`

The `http` and `multi` runs print uvicorn's startup and shutdown lines around the results; the in-process server stops cleanly (F-L43-8).

Resolved on 10 October 2026:

| | |
|---|---|
| Python | 3.14 (patch version not recorded) |
| `fastmcp` | 3.4.8 |
| `mcp` | 1.30.0 |
| `langchain-mcp-adapters` | 0.3.2 |
| `langchain` / `langgraph` | 1.4.4 / 1.2.14 |
| `langchain-anthropic` | 1.7.5 |
| `python-dotenv` / `pytest` | 1.2.4 / 9.1.1 |

Static checks: `ruff check` and `ruff format --check` are clean on the lab machine (rules E, F, I, W, UP, ANN; line length 100), and Pylance reports no problems on the lab's entry points (`raw_client_demo.py`, `agent_demo.py`). pyright was clean on a Python 3.13 run and was not re-run on 3.14.

## File layout

```
course9-module2-lab1/
├── domain/
│   └── conversation.py                       # ConversationTurn, AgentReply (frozen dataclasses)
├── interfaces/                               # Protocols that infra satisfies
│   ├── chat_model_provider_interface.py
│   ├── document_store_interface.py
│   ├── http_server_host_interface.py
│   └── tool_source_interface.py
├── infra/
│   ├── anthropic_chat_model_provider.py
│   ├── file_document_store.py                # the containment check (F-L43-1)
│   ├── http_tool_source.py
│   ├── stdio_tool_source.py
│   ├── multi_server_tool_source.py           # tool_name_prefix=True (F-L43-2)
│   └── in_process_http_server_host.py
├── application/
│   ├── interfaces/agent_service_interface.py # contract the entry point depends on
│   ├── calculator_server.py                  # build_server(document_store) -> FastMCP
│   ├── agent_service.py
│   ├── mappers.py                            # LangChain messages -> domain turns
│   └── container.py                          # the only file that imports infra and application
├── scripts/download_samples.py               # python -m scripts.download_samples
├── config.py                                 # port, URL, documents dir, sample URLs, stdio command
├── run_server.py                             # launched by the stdio clients
├── agent_demo.py                             # agent entry point: http | stdio | multi
├── raw_client_demo.py                        # raw-client entry point: in-memory | http | stdio | local-tool
└── tests/                                    # mirrors the source layout
```

`raw_client_demo.py` imports `fastmcp.Client` directly. That is deliberate: it is the lab's view of the protocol, so it sits outside the layers, while `agent_demo.py` reaches the agent only through `AgentServiceInterface`.

## Differences from the notebook

- Cells become functions in two entry points. The server is built by `build_server(document_store)`, so the in-memory, HTTP and stdio paths all serve one definition.
- The document resource reads through `FileDocumentStore`, which resolves the requested name and checks it is still inside the root (F-L43-1).
- The documents directory, sample URLs, port and stdio command live in `config.py`. The download script has `timeout=10`, skips a file that already has content, and prints no check-mark glyphs (F-L43-8).
- `streamablehttp_client` is replaced by `streamable_http_client` (F-L43-6). Results are narrowed with `isinstance(..., TextContent)` and `TextResourceContents` before `.text` is read.
- The in-process HTTP server runs uvicorn directly (`server.http_app()` under `uvicorn.Server`) as a task, waits for `started`, and sets `should_exit` in `finally` so the lifespan shuts down cleanly. A busy port raises `RuntimeError` instead of printing a message and returning.
- `MultiServerMCPClient` is built with `tool_name_prefix=True` (F-L43-2). The stdio server is started with `sys.executable`, so it runs in the venv's interpreter.
- The agent cells use `ChatAnthropic` (`claude-haiku-4-5`). The agent service returns an `AgentReply` of typed turns, and `agent_demo.py` prints them.

## Findings

**F-L43-1 - the document resource takes a client-supplied name, so the store enforces containment.** The resource `file://endpoint2/{name}` passes `name` from the client to the filesystem. `FileDocumentStore` resolves the joined path and returns nothing unless `is_relative_to(root)` and `is_file()` hold. The tests cover `../`, `..\`, an absolute path, and `sub/../name` (allowed because it stays inside the root); with the check deliberately broken the traversal tests fail. This is a hardening, not a reproduced vulnerability: the notebook's original resource was not run against a traversal name for these notes. Windows-specific forms (`C:\...`, UNC paths, reserved device names) were not tested.

**F-L43-2 - two servers with the same tool names reach the model unlabelled, and the API accepted them.** With both the stdio and HTTP servers loaded unprefixed, the agent was given `['add', 'subtract', 'add', 'subtract']` and Anthropic's API accepted the request through `langchain-anthropic` with Haiku 4.5. The transcript cannot say which server ran the call. With `tool_name_prefix=True` the tools are `stdio-client_add` and `http-client_add`, and `agent_demo.py multi` shows `[AI] tool call: stdio-client_add` for "whats 8 + 7?", so the choice is visible in the turn. This is one run on one model and provider; other providers may reject duplicates, and nothing here measures how the model picks between them.

**F-L43-3 - the published tool description is the docstring's summary line.** Each tool has a long docstring (summary, `Args`, `Returns`, `Example`). `list_tools()` returned `description='Add two integers together.'`, with the `Args` entries moved into per-parameter descriptions in `inputSchema` (`'The first integer.'`) and the `Returns` and `Example` text absent from the listing. The docstring is the whole source of what a model sees, and only part of it arrives. Not tested: how other docstring layouts are split.

**F-L43-4 - the server's `instructions` name a tool that does not exist.** The server is created with instructions that say "Call get_average() to analyze numerical data", and it defines only `add` and `subtract`. I kept the text as the notebook has it. Whether a model acts on it was not tested. Same shape as Lab 41's declared-versus-enforced finding: the text a server publishes is not checked against what it serves.

**F-L43-5 - a missing document is a normal result, not an error; the first resource reads nothing.** `file://endpoint2/random.txt` returns `uri` and `mimeType='text/plain'`, `meta=None`, and the text `Document 'random.txt' not found in path directory`. No exception is raised, so a caller checking only for errors presents the message as the document. The other resource, `file:///endpoint/{name}`, never touches the filesystem: it formats a string (`Document contents of README.txt`), which the test pins by asserting the store is not called.

**F-L43-6 - SDK drift between the two Course 9 stacks.** On `mcp` 1.30.0, `streamablehttp_client` is deprecated in favour of `streamable_http_client`, which yields `(read, write, get_session_id)`; the third value is a callable and returned `None` here. `Tool.inputSchema` is the field name on this stack (the listing prints `inputSchema=...`), whereas Lab 41, on `mcp` 2.3.0, found `input_schema` with `inputSchema` as a deprecated alias. Code written for one stack's field names will not carry across to the other.

**F-L43-7 - `fastmcp` is held at 3.4.8 while 4.1.0 is available.** Startup prints an update banner for 4.1.0. `fastmcp` 4.x needs `mcp>=2`, and `langchain-mcp-adapters` 0.3.2 caps `mcp<2` (Lab 42), so an agent over `MultiServerMCPClient` and a 4.x `fastmcp` cannot share one environment today. Lab 41 ran `fastmcp` 4.0.10 on `mcp` 2.3.0 in its own. Nothing in this lab was run on `fastmcp` 4.x.

**F-L43-8 - Windows output and shutdown noise.** A Python script whose output is redirected or piped on Windows writes with cp1252 and crashed with `UnicodeEncodeError` on a check-mark glyph; `$env:PYTHONUTF8="1"` fixes it for a redirected run, and the glyphs were removed from the download script instead. With the in-process HTTP task started through `run_http_async` and cancelled at the end of a run, uvicorn logged a `CancelledError` traceback from its lifespan handler; the results had always printed first. Running uvicorn directly and setting `should_exit` instead of cancelling removed the traceback on Windows (`agent_demo.py http` and `raw_client_demo.py http`). I did not trace why cancelling produced it.

## Prior art carried across

- **Declared versus enforced (Course 8; Lab 41 F-L41-5).** F-L43-3 and F-L43-4: the description a server publishes is partial, and its instructions are not checked against its tools.
- **Errors delivered as ordinary text (Lab 41 F-L41-7).** F-L43-5 is the same shape on a server written in this lab.
- **Adapters and prefixes (Lab 42).** F-L43-2 and F-L43-7 build on the adapters' behaviour and `mcp<2` cap recorded there.

## What this doesn't cover

- The notebook's original `read_document` against a traversal name; the containment check is a hardening, not a reproduced vulnerability.
- Windows path forms in the containment check (`C:\...`, UNC, drive-relative names, reserved device names).
- Whether a provider other than Anthropic accepts duplicate tool names, and how any model chooses between them (F-L43-2 is one run).
- `fastmcp` 4.x, and whether `http_app()` and the uvicorn shutdown behave the same there.
- Whether `MultiServerMCPClient` starts a new stdio process per tool call on this server (Lab 42 saw it on a stand-in server).
- Security of the HTTP server beyond the containment fix: it serves on 127.0.0.1 with no authentication, and origin checks and transport hardening were not reviewed.
- One machine, one Python 3.14, one set of pinned versions.

---

**Completed:** 10 October 2026
