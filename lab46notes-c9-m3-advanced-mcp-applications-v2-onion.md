# Course 9 - Lab 46: Advanced MCP Applications with Streamable HTTP, Roots, and Sampling (V2 - Onion-Architected)

> Code: [`course9-module3-lab2-v2-onion/`](course9-module3-lab2-v2-onion/)

[V1](lab46notes-c9-m3-advanced-mcp-applications-v1.md) is the lab's four files as written. This is the same system as two separate onion-layered apps: an HTTP/stdio MCP file server and a client with two entry points, a Gradio client page and a Gradio chat host for `gpt-4o-mini`. 210 tests pass without an API key or an external network: 62 on the server (63 collected; one is skipped) and 148 on the client. Counted on Windows with Python 3.14.0: one server test runs only on Windows (backslash paths) and the symbolic-link test is skipped where links cannot be created, so another platform reports different numbers.

The lab's stubs are kept as stubs: "roots" is the server's folder, and `analyze_code` is the canned string (F-L46-2, F-L46-3 in V1). The fixes are listed under Differences, each added test first.

The most useful result is about the server's HTTP mode: with FastMCP 3.4.8's defaults it accepted a foreign `Origin` and a foreign `Host`. Turning the checks on made the same requests return 403 and 421.

## Run it

Python 3.14 and PowerShell, from the repo root. Each app has its own virtual environment:

```powershell
cd course9-module3-lab2-v2-onion\server
py -3.14 -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

```powershell
cd ..\client
py -3.14 -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Put `OPENAI_API_KEY=...` in `client/.env` (gitignored); only the host app needs it. Then, in three terminals, each from its own folder with its venv active:

```powershell
python run_server.py http         # server/, port 8000 (default; `stdio` also works)
python gui_app.py                 # client/, page on http://127.0.0.1:7861
python host_app.py                # client/, page on http://127.0.0.1:7862
```

Both client entry points take an optional server and an optional roots text for the page header: `python gui_app.py http://127.0.0.1:8000 workspace`. A server script path instead of a URL makes the client launch that server over stdio with the server's own interpreter, so the server's venv must exist first.

Tests need no key and no network: run `pytest` inside `server/` and inside `client/`. `server/workspace/` is created on first run and is not committed.

Resolved on 18 October 2026:

| | |
|---|---|
| Python | 3.14.0 |
| `mcp` | 1.30.0 |
| `fastmcp` (server only) | 3.4.8 |
| `pydantic` | 2.14.1 |
| `gradio` / `openai` / `python-dotenv` (client only) | 6.30.0 / 2.54.0 / 1.2.4 |
| `pytest` / `ruff` | 9.1.1 / 0.17.1 |

Static checks: `ruff check` and `ruff format --check` are clean on both apps (rules E, F, I, W, UP, ANN; line length 100; target py314), and Pyright 1.1.414 reports 0 errors on both (basic mode; Pylance was not run on these apps).

## File layout

Each app has its own layers, `config.py`, `conftest.py`, `pytest.ini`, `pyproject.toml` and `requirements.txt`, and is run from inside its own folder.

```
course9-module3-lab2-v2-onion/
├── server/
│   ├── interfaces/file_store_interface.py   # FileStoreInterface, PathOutsideRootError
│   ├── infra/local_file_store.py            # the root check
│   ├── application/
│   │   ├── lab_server.py                    # build_server(file_store, name): tools, template, prompts
│   │   ├── http_security.py                 # Host and Origin options for HTTP mode
│   │   └── container.py                     # the only file that imports infra and application
│   ├── config.py / run_server.py            # [http|stdio]
│   └── tests/
└── client/
    ├── domain/                              # tool, resource template, prompt, conversation, chat reply
    ├── interfaces/                          # mcp_session, mcp_transport, llm_chat_model
    ├── infra/
    │   ├── stdio_transport.py / http_transport.py
    │   ├── mcp_sdk_session.py               # ClientSession; returns primitives
    │   └── openai_chat_model.py             # AsyncOpenAI, created on first use
    ├── application/
    │   ├── interfaces/                      # gui_service, host_service
    │   ├── gui_service.py / host_service.py
    │   ├── mappers.py / formatters.py
    │   └── container.py                     # initialise_gui, initialise_host
    ├── gui_app.py / host_app.py             # the only files that import Gradio
    ├── cli_arguments.py / config.py
    └── tests/
```

`build_server` takes the file store as a parameter and imports nothing from `infra`; its tests use an in-memory `fastmcp` `Client` with `MagicMock(spec=FileStoreInterface)`. The host service takes a session and a chat model as interfaces, so its tests run the full chat loop with mocks. It keeps no conversation: `chat(message, history)` returns a `ChatReply` carrying the new history, and the page holds it in `gr.State`. The OpenAI client is built on the first `complete()`, not in the constructor, because `OpenAI()` raises at construction without a key.

## Differences from the lab

- **Layers.** `LocalFileStore` holds the root check (resolve, then `is_relative_to`), UTF-8 reads and writes, and creates the root. The tools, the resource template and the prompts keep the lab's names, arguments and text; tools return `Error: ...` text and the resource raises, as the lab does. `analyze_code` is the lab's stub, unchanged.
- **Both transports**, chosen by configuration: the server runs `[http|stdio]`, and the client's container picks `HttpTransport` for an http(s) address (adding `/mcp` if missing) and `StdioTransport` otherwise. `HttpTransport` uses `streamable_http_client`; `streamablehttp_client` is deprecated in `mcp` 1.30.0 (Seen: the `@deprecated` marker in the SDK).
- **One client app with two entry points** sharing the domain, interfaces, infra and mappers. A separate protocol service was dropped: it would only have wrapped two session calls.
- **Dropped or replaced:** the lab's `tools_cache` and `prompts_cache` (never read); `getattr` fallbacks (fields are read directly); the unused `history` argument on the chat function; `type="messages"` on `gr.Chatbot` (F-L46-15); the synchronous OpenAI client (the async one is used). Both command-line arguments are optional.
- **Host results unpacked.** The model gets a resource's text and a prompt's messages as `[role]: text`, not Python reprs (F-L46-4 in V1). In one real run the role showed as `user`, where the lab's version printed three `[unknown]` lines.
- **Fixed, each test first (F-L46-13, F-L46-17 to F-L46-19).** Tool-argument JSON that is invalid or not an object is reported to the model as the tool's result; `mcp_get_prompt` accepts both argument shapes; `connect()` is guarded by a lock and cleans up after a failed handshake; a result the server flags as an error is shown as an error; the listing helper tools report failures as text; the conversation belongs to the page session, not the service.
- **Added, not in the lab:** the Host and Origin checks (F-L46-11), the tool-call display and the no-text message on the chat page (F-L46-16).
- **Kept as the lab has it:** one round of tool calls per message; the page header's "roots" text being display only; a session that is never closed (F-L46-12).

## Findings

**F-L46-11 - FastMCP's HTTP defaults accept a foreign Origin and Host, so the port turns the checks on.** Seen live and in-process on `fastmcp` 3.4.8: with the defaults, `initialize` over HTTP returned 200 for `Origin: http://evil.example` and for a foreign `Host`. `host_origin_protection` defaults to `False`; `True` validates every request and `"auto"` validates only loopback-bound servers. The options are `host_origin_protection`, `allowed_hosts` and `allowed_origins` on `run` and `http_app`. `run_server.py` now passes them from `application/http_security.py` (a `TypedDict`, so Pyright accepts the unpacking), with the hosts and origins listed in `config.py`. Seen on the running server afterwards: no `Origin` 200, the server's own `Origin` 200, a foreign `Origin` 403 (`Forbidden Origin`), a foreign `Host` 421 (`Misdirected Request`), `Host: localhost:8000` 200; the real client over `http://127.0.0.1:8000` still listed its tools. Eight server tests cover this in-process; switching the checks off failed three. Labs 44 and 45 use the same FastMCP defaults for their HTTP modes; I did not probe them. Not done: authentication or HTTPS, which a local lab server does without. The spec makes `Origin` validation a requirement for servers (from a chat summary of the 2026-07-28 revision; not rechecked).

**F-L46-12 - the session is never closed at shutdown, and cannot be with this design.** A script that connects over stdio and never calls `close()` ends with `RuntimeError: Attempted to exit cancel scope in a different task than it was entered in` (Seen, twice, including closing from a different task than the one that connected). The SDK requires a connection's cancel scopes to be exited by the task that entered them; here the connection is entered inside whichever Gradio event connects first, and a shutdown hook is another task. A fix needs a dedicated long-lived session task, which is a design change. I left it, as in the lab, where `cleanup()` is never called by Gradio. Whether the running pages print this on Ctrl+C was not checked.

**F-L46-13 - two overlapping events could both connect.** Gradio can run two events at once. `McpSdkSession.connect()` is now guarded by an `asyncio.Lock` and ignores a second call. The test uses a stand-in `ClientSession` and a transport that yields control; without the lock it failed, with it it passes. A first version of the test failed for another reason: `asyncio.gather` runs each `connect()` in its own task, which trips the same rule as F-L46-12.

**F-L46-14 - the root check and the encoded-slash cases.** The store's tests cover `../`, a nested `../`, an absolute path, listing the parent, `..\` on Windows, `sub/../name` (allowed) and a refused write leaving nothing outside; every store method refuses an outside path. Container tests send `file://workspace/..%2Fsecret.txt`, `%2e%2e%2f` and `..%5C` through the real server and store, and none returns the outside file's text. `list_files` with `..` and `write_file` to `../escaped.txt` are refused and write nothing. With the check disabled on a scratch copy, 11 tests failed. Seen on the real server: `read_file` of `../run_server.py` and `C:/Windows/win.ini` gave `Error: Access denied - path outside workspace roots`, `file://workspace/..%2Frun_server.py` gave `Error reading resource ...: Access denied - path outside workspace roots`, and a missing file gave `File not found`. In Lab 45 the same encoded forms got past a template that had no check; here I did not run the encoded forms against an unchecked store, so this is a pinned guard, not a reproduced leak. Not tested: Windows drive and UNC paths; the symbolic-link test is skipped because my account cannot create links.

**F-L46-15 - the lab's `gr.Chatbot(type="messages")` fails on Gradio 6.** Seen on 6.30.0: the `type` parameter is gone (messages are the only format). `gr.update`, `Dropdown`, `Textbox`, `Button`, `queue()` and `launch()` work as the lab uses them. The host page here builds `gr.Chatbot` without `type`, and its history is a list of `{"role", "content"}` dicts.

**F-L46-16 - the chat page now shows which tools ran, and never shows `None`.** Two things made a demo hard to read. A model answer with no text came back as `None`, and Gradio had nothing to show. And when the model refused a path it was not visible whether a tool had been called at all. `HostService.chat()` now returns a `ChatReply` (the text, plus each tool call's name, the arguments the model wrote and the result); no text becomes "The model returned no text."; and the page prints a "Tool calls" list, one line each with the result cut at 200 characters, before the answer. Seen in the service's history on the real server: the model's `read_file` calls for `../run_server.py` and `/etc/passwd` did reach the tool and got the exact `Error: Access denied - path outside workspace roots`; the chat replies alone were weak evidence, since the model sometimes rewrote the path (dropped the `../`, decoded `%2F`) and sometimes refused without calling anything. Unit tests cover the formatting and the no-text cases; two scratch-copy mutations (no-text message removed, truncation removed) each failed one test. The new display was **not re-run in the browser**.

**F-L46-17 - the model's own tool-call arguments can be wrong, and the port reports that back instead of crashing.** The lab's `json.loads` on the model's arguments is unguarded. The port returns `Error: the tool arguments were not valid JSON: ...` or `Error: the tool arguments must be a JSON object` as the tool's result, so the model can correct itself. Removing the object check failed 4 tests. `mcp_get_prompt` takes a filled nested `arguments` first and otherwise the keys beside `name` (removing the second path failed 2 tests). A result flagged `is_error` is shown as `Error: <text>` in the GUI and `Error executing tool: <text>` to the model (ignoring the flag failed 4 tests). One real run after these fixes, with the model passing the filename this time, does not show how often the original argument gap happens.

**F-L46-18 - one shared history for every browser session (Copilot review, High).** The lab keeps one conversation on the host object, and the first version of the port copied that: `initialise_host()` built one `HostService` that every Gradio session shared, so a second tab got the first tab's prompts and tool results sent to the model, and Clear in one tab reset both. `HostService` now keeps no conversation. `chat(message, history)` builds the model's messages from the history passed in and returns the new history in the `ChatReply`; the page keeps it in a per-session `gr.State`, and Clear resets the state. Two service tests pin this (two chats on one service stay apart; a chat continues from the history given); ignoring the passed history on a scratch copy failed one. The MCP session itself is still one per process, shared by all page sessions; its calls carry no per-user state, and the lock from F-L46-13 covers the connect. Not run: two real browser tabs.

**F-L46-19 - failed connects and failed listings (Copilot review, Medium, two comments).** (1) `McpSdkSession.connect()` pushed the transport and the session onto one long-lived exit stack before the handshake finished, so a failed `initialize()` left them registered and each retry added more until shutdown. Each attempt now has its own stack, and only a finished handshake is moved onto the long-lived one. The test checks, inside the running loop, that two failed connects opened two connections and closed two (before the fix it read 2 opened, 0 closed; `asyncio.run` closes leftover generators at exit, so a check after the run would have missed it). (2) `mcp_list_resources` and `mcp_list_prompts` called the session without the error handling the other tools have, so a dropped connection would end the chat event instead of reaching the model. They now return `Error listing resources: ...` and `Error listing prompts: ...`; removing the prompts guard failed 2 tests.

## Real runs

Seen from here, on this machine, with `gpt-4o-mini` and the real server; the model's replies differ between runs:

- The GUI app, driven with `gradio_client` over HTTP: list tools, resources and prompts; `write_file`, `read_file`, `list_files`; the root check; the resource read; the `analyze_code` stub text; the `review_code` prompt; bad JSON gave `Error: Invalid JSON format`; no tool selected gave the dropdown message. Over stdio through the same service: four tools, a read, and the outside path refused.
- The host app, five prompts in one conversation: the model created `hello.txt` (Seen on disk: exactly `Hello from HTTP MCP!`); said it could not access `/etc/passwd`; listed the resource template and asked for a filename; listed both prompts and returned the rendered `review_code` text; and wrote its own analysis for `analyze_code`, opening with "an analysis is ready to be sent for approval", which fits the stub's text having reached it.
- Python run from my shell could not verify `api.openai.com` (a TLS-inspection setting on this machine, the same one that breaks `uv` without `--system-certs`); I ran those checks through a wrapper that calls `truststore.inject_into_ssl()`. This is about my environment, not the app.

## Prior art carried across

- **Containment on a client-supplied path (Lab 43 F-L43-1; Lab 44 F-L44-3; Lab 45 F-L45-1).** F-L46-14, with the encoded forms pinned from the start.
- **One class per transport (Lab 43; Lab 45).** Stdio and HTTP behind one interface, chosen in the container.
- **Errors delivered as ordinary text (Lab 41 F-L41-7; Lab 43 F-L43-5; Lab 44 F-L44-4; Lab 45 F-L45-3).** The tools' `Error: ...` text and the missing-file text.
- **SDK drift between versions (Lab 43 F-L43-6).** `streamable_http_client` here; the Gradio 6 change in F-L46-15.
- **HTTP mode without origin checks (Lab 44 and Lab 45 "what this doesn't cover").** Now probed on the port, F-L46-11.

## What this doesn't cover

- Real roots and sampling (`ctx.sample`, `ctx.list_roots`, the client's `sampling_handler` and `roots` exist on `fastmcp` 3.4.8; behaviour not run).
- `fastmcp` 4.x, `mcp` 2.x and the 2026-07-28 revision (no `initialize` handshake, roots and sampling deprecated, new HTTP headers). A modernised port is not planned yet.
- Authentication, HTTPS, and Host and Origin checks on Labs 44 and 45.
- Closing the session at shutdown (F-L46-12).
- The tool-call display and the per-session history in a browser (the service tests show two histories stay apart; I did not open two tabs).
- Windows drive and UNC paths, and symbolic links, in the root check.
- Pylance on the ported apps; `gpt-4o-mini` availability beyond these runs.
- One machine, Python 3.14.0, one set of pinned versions.

---

**Completed:** 18 October 2026
