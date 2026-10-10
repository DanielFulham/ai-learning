# Course 9 - Lab 44: Build an Enhanced MCP Server (V1)

> Code: [`course9-module2-lab2/`](course9-module2-lab2/)

This is a port of IBM Skills Network's "Build an Enhanced MCP Server" lab (the lab's template repository is not included here) on Python 3.14 and Windows. I did the lab first from its own files, then rebuilt it as two separate onion-layered apps. 94 tests pass without an API key or an external network: 39 on the server and 55 on the client, with one more server test skipped on my machine.

## What the lab builds

- **A server** (`server/`) built with FastMCP. It has two tools (`write_file`, which reports progress as it writes, and `delete_file`), two resources (`file:///{file_name}` reads a file, `dir://.` lists the folder) and two prompts (`code_review`, and `documentation_generator`, which asks you for the file name and the documentation name through elicitation).
- **A client** (`client/`) with a text menu. It runs the prompts, reads the resources, and lets Claude Haiku 4.5 call the server's tools in a loop.

The client either starts the server itself as a subprocess over stdio, or connects to a server that is already running over Streamable HTTP. The two apps share no code.

## Run it

From the repo root, set up each app with its own virtual environment:

```powershell
cd course9-module2-lab2\server
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

```powershell
cd ..\client
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Commands are PowerShell. Node.js is not needed. The client's model calls need an Anthropic API key: put `ANTHROPIC_API_KEY=your-key` in a `.env` file in `client/`. The server only touches `server/workspace/`, which it creates on first use and which is not committed.

Start the client from `client/`. It launches `../server/run_server.py` itself over stdio:

```powershell
python cli_client.py
```

Or run the two separately, in two terminals:

```powershell
# terminal 1, in server/ with its venv active
python run_server.py http

# terminal 2, in client/ with its venv active
python cli_client.py http://127.0.0.1:8000/mcp
```

The menu is: 1 generate documentation, 2 review code, 3 read a file, 4 list the server's folder, 5 converse with Claude, q quit. The workspace starts empty, so options 1 to 3 need a file first. A first run that covers everything:

1. Choose 5 and type `create a file called hello.txt containing Hello, World!`, then `q` to return to the menu. Progress lines appear while the file is written.
2. Choose 4 to list the folder, then 3 and `hello.txt` to read the file back.
3. Choose 1, enter `hello.txt` for the file and `hello-docs` for the documentation name. The server asks for both through elicitation, and Claude writes `hello-docs.md`.
4. Choose 2 and enter `hello.txt` to get a review.

Tests need no key and no external network. Run `pytest` inside `server/` and inside `client/`; nothing uses the HTTP port.

Resolved on 11 October 2026:

|                               |                                   |
| ----------------------------- | --------------------------------- |
| Python                        | 3.14 (patch version not recorded) |
| `fastmcp`                     | 3.4.8                             |
| `mcp`                         | 1.30.0                            |
| `pydantic`                    | 2.14.0                            |
| `anthropic` (client only)     | 1.13.0                            |
| `python-dotenv` (client only) | 1.2.4                             |
| `pytest` / `ruff`             | 9.1.1 / 0.16.8                    |

Static checks: `ruff check` and `ruff format --check` are clean on my machine for both apps (rules E, F, I, W, UP, ANN; line length 100). pyright was clean on a Python 3.13 run of both apps. It was not re-run on 3.14, and Pylance in the editor was not checked.

## File layout

Each app has its own layers, `config.py`, `conftest.py`, `pytest.ini`, `pyproject.toml` and `requirements.txt`. Both use the top-level package names `domain`, `interfaces`, `infra` and `application`, so each app has to be run from inside its own folder.

```
course9-module2-lab2/
├── server/
│   ├── domain/file_entry.py                       # FileEntry (frozen dataclass)
│   ├── interfaces/file_store_interface.py         # FileStoreInterface, RawEntry, PathOutsideRootError
│   ├── infra/local_file_store.py                  # the root check (F-L44-3)
│   ├── application/
│   │   ├── file_server.py                         # build_server(file_store, name) -> FastMCP
│   │   ├── mappers.py                             # RawEntry -> FileEntry -> listing row
│   │   └── container.py                           # the only file that imports infra and application
│   ├── config.py                                  # server name, workspace root, host and port
│   ├── run_server.py                              # entry point: stdio, or `http`
│   └── tests/                                     # mirrors the source layout
└── client/
    ├── domain/                                    # conversation, prompt_spec, directory_item
    ├── interfaces/                                # chat_model, mcp_session, user_input
    ├── infra/
    │   ├── anthropic_chat_model.py
    │   ├── fastmcp_session.py                     # stdio or HTTP; handlers passed in as callables
    │   └── console_user_input.py
    ├── application/
    │   ├── interfaces/client_service_interface.py # contract the entry point depends on
    │   ├── chat_service.py                        # the tool loop (F-L44-6)
    │   ├── client_service.py                      # the menu
    │   ├── handlers.py                            # elicitation, progress and notification handlers
    │   ├── mappers.py / formatters.py
    │   └── container.py                           # the only file that imports infra and application
    ├── config.py                                  # model id, max tokens, default server script
    ├── cli_client.py                              # entry point: [server script or http URL]
    └── tests/                                     # mirrors the source layout
```

`build_server` takes the file store as a parameter and imports nothing from `infra`. The server tests call it through an in-memory `fastmcp` `Client` with a `MagicMock(spec=FileStoreInterface)`. The client's session tests use a small test server defined inside the test file, never the real one.

## Differences from the lab

- One file per side becomes two apps in layers. All file access goes through `FileStoreInterface`; `LocalFileStore` is the only code that touches the filesystem and the only place the root check lives (F-L44-3). The tools, resources and prompts keep the lab's names, arguments and prompt text.
- The root is `server/workspace/`, set in `config.py`, instead of the current directory at import. The tools can no longer reach the server's own code.
- The `dir://.` listing is built through mappers instead of the lab's `get_path` (F-L44-2). The client shows a server `{"error": ...}` reply as the server's message instead of failing on `'items'`.
- `write_file` makes an empty write first, then appends ten chunks, so empty content still creates a file as the lab's version does. The pause between chunks is `await asyncio.sleep` instead of the lab's blocking `time.sleep`. I did not measure the effect of that change.
- `documentation_generator` checks `isinstance(result, AcceptedElicitation)` instead of reading `result.data` unconditionally. A decline or cancel raises an error that names the action (F-L44-5).
- The error behaviour is kept as the lab has it: `write_file` re-raises and `delete_file` returns text (F-L44-4). Both are tested.
- The client uses `AsyncAnthropic` instead of the synchronous client inside async code, and `max_tokens` is 8192 (F-L44-6). It prints a warning when a reply ends at `max_tokens`, and a tool call in such a reply is not run.
- The elicitation, progress and notification handlers are plain callables passed into the session. The session starts the server through an explicit `StdioTransport` using the venv's interpreter, so launching `.js` and `.ts` servers, which the lab's check allows, is dropped.
- Removed: the server's `print("Starting File Operations Server...")` (on stdio, stdout carries the protocol; in the lab run I did not see the line in the client and did not find out why), the client's `print(first.text)` and `print(prompt_obj)` leftovers, and unused imports. The lab client's Pylance errors and Ruff F401 hits (untyped `messages` and `tools`, `.text` on a union of content blocks) are fixed with annotations and `isinstance` narrowing, without `cast` or `# type: ignore`.
- Added, not in the lab: an `http` mode for the server and a URL argument for the client.
- `anthropic` is pinned at 1.13.0, the version I tested, not the lab's 0.69.0. The client uses only the Messages API with tools.

## Findings

**F-L44-1 - the lab's pinned install does not import.** The lab pins `mcp==1.15.0`, `fastmcp==2.12.4`, `anthropic==0.69.0` and `python-dotenv==1.1.1`, and leaves `pydantic` unpinned. With `pydantic` 2.14.0, `import fastmcp` failed with an `eval_type_backport` ImportError on Linux (Python 3.13) and on Windows (Python 3.14). On `fastmcp` 3.4.8 with `mcp` 1.30.0 the import works. A two-run comparison on one machine points at the old `mcp` rather than `pydantic` alone; it is not a bisect. The port pins every direct dependency, `pydantic` included.

**F-L44-2 - the lab's directory listing fails every time, and the client hides why.** The lab's `get_path` returns `Path(relative_path).resolve().relative_to(BASE_DIR)`, a relative path, although its docstring says absolute. The `dir://.` code then calls `item.relative_to(BASE_DIR)` on absolute items and raises. On Windows the server returned `{"error": "Error listing files: '.DS_Store' is not in the subpath of ...\\start"}` (`.DS_Store` is a macOS file inside the template folder, incidental), and the client's `["items"]` lookup showed only `Error reading directory: 'items'`. Returning the resolved path fixed it. In the port the listing works over stdio and over HTTP on Windows.

**F-L44-3 - the lab's root check holds; the port keeps it in one class.** With `get_path` corrected, the lab's server refused `file:///..%2Fx` (`{"error": "Error reading file: Path is outside BASE_DIR"}`), a `write_file` outside the folder raised, and a `delete_file` outside it returned the same text. I ran this on Linux (Python 3.13) against the lab's server file, not on Windows. The lab's root is the whole project folder, so its tools could in principle reach the server's own source; I did not demonstrate that. In the port, `LocalFileStore` resolves each path under `workspace/` and raises `PathOutsideRootError` unless `is_relative_to(root)`. Its tests cover `../`, a nested `../`, an absolute path, and a refused write that leaves nothing outside; `sub/../name` is allowed because it stays inside. With the check replaced by `if False:`, five tests failed on Windows (the three refusal cases, the refused-write test and the backslash test) and ten still passed; restored, all pass. The backslash-traversal test runs only on Windows. The symlink test was skipped because my Windows account cannot create links. This is a result, not a reproduced vulnerability.

**F-L44-4 - the two tools report failure differently.** `write_file` logs the error and re-raises, so the client gets an error result. `delete_file` catches everything and returns `Error deleting file: ...` as ordinary text with `is_error` unset. A path outside the folder does this (Linux run above), and a directory or a missing file returns similar text after a warning (a directory was seen returning text in a Windows run). The lab's text says delete "raises an error". A caller that checks only for errors treats the message as a success. The port keeps both behaviours and tests each.

**F-L44-5 - elicitation works on `fastmcp` 3.4.8; the lab does not handle decline.** The lab's handler, `ctx.elicit(response_type=...)` and the three-way result ran unchanged. On the accept path the file name and documentation name reached the server. A decline (an empty file path) ended as `McpError: ... Documentation request was not accepted: decline`, and the client kept running. The lab reads `result.data` without checking the action, which Pylance flags (`DeclinedElicitation` has no `data`); I did not run it without the guard. In the port, accept ran with Haiku 4.5 over stdio and over HTTP, and the model then wrote `hello-docs.md`. Decline and cancel are tested in memory only.

**F-L44-6 - a cut-off tool call ended the lab's loop with no output.** With `max_tokens=4096`, a documentation request ended with `stop_reason` `max_tokens` while the model was writing the `content` argument of `write_file`. The lab's loop continues only on `tool_use`, so it exited and printed nothing. With 8192 the same request ran: a 14,892-character file, eleven progress notifications. That the content exceeded 4096 tokens is an inference, since I did not count tokens. The port warns on `max_tokens` and does not run a tool call in that reply. That is tested against a fake model and was not forced against the real API.

**F-L44-7 - a review prompt was once followed by an unrequested file write.** In the lab run, the review prompt, which asks only for an evaluation, was followed by the model calling `write_file` to create `CODE_REVIEW.md`. In the port run, the same menu option on a one-line text file returned a review as text and wrote nothing; the workspace held only the two files from earlier steps. One run each, one model, different files. It shows the write tool being within reach of a review, not how often it happens.

**F-L44-8 - the lab's logging claim is wrong as written.** The lab says its message handler prints the server's logs. It prints only the notification method names (`Received: notifications/message`, `Received: notifications/progress`). The log text itself appears through FastMCP's default handler (`INFO Received INFO from server: ...`). The port keeps this handler behaviour, including one `Received: notifications/progress` line after every progress update.

**F-L44-9 - the lab against the current protocol revision.** The lab and its video present `ctx.elicit` as real-time two-way interaction and `ctx.info` and `ctx.error` as logging. The [2026-07-28 changelog](https://modelcontextprotocol.io/specification/2026-07-28/changelog) says Multi Round-Trip Requests replace server-initiated requests such as `elicitation/create`, and that Roots, Sampling and Logging are deprecated but stay functional for at least twelve months. It also says `notifications/progress` keeps flowing on the originating request's response stream. All of the lab's behaviour ran on `mcp` 1.30.0, which speaks an earlier protocol revision. Nothing here was run against a 2026-07-28 server or client.

## Prior art carried across

- **Errors delivered as ordinary text (Lab 41 F-L41-7; Lab 43 F-L43-5).** F-L44-4 is the same shape: `delete_file` reports failure as a normal result.
- **Containment on a client-supplied path (Lab 43 F-L43-1).** F-L44-3 repeats the resolve-and-`is_relative_to` check, here inside `LocalFileStore`.
- **SDK drift between stacks (Lab 43 F-L43-6, F-L43-7).** F-L44-1 and F-L44-9: the lab's pins and feature set belong to an older stack than the one it now installs on.

## What this doesn't cover

- Whether `../outside.txt` typed into menu option 5 is refused end to end. The root check is tested on `LocalFileStore` directly.
- Windows path forms other than backslash traversal (`C:\...`, UNC, reserved device names), and symlinks on Windows.
- Subfolder file names (`docs/m.md`) through the client's `quote(file_name, safe="")`: they worked on Linux and were not run on Windows.
- Decline and cancel against the real client, and the `max_tokens` warning against the real API.
- `fastmcp` 4.x, `mcp` 2.x and the 2026-07-28 protocol (F-L44-9).
- The HTTP mode listens on 127.0.0.1 with no authentication. I did not review origin checks.

---

**Completed:** 11 October 2026
