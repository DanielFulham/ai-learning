# Course 9 - Lab 45: Build a Custom MCP Client with Python (V1)

> Code: [`course9-module3-lab1/`](course9-module3-lab1/)

A command-line MCP client and a small MCP server, as two separate onion-layered apps. The server is built with FastMCP (two tools, one resource template, one prompt). The client uses the official `mcp` SDK's `ClientSession` and reaches the server over stdio or Streamable HTTP, through separate transport classes behind one interface. 107 tests pass without an API key or an external network: 38 on the server and 69 on the client, with one more server test skipped on my machine.

This is a port of IBM Skills Network's "Build a Custom MCP Client with Python" lab (the lab text is not included here) on Python 3.13 and Windows. I did the lab first from its own files, then rebuilt it as two apps.

The most useful result is about the server's resource template: before the root check, it returned a file from outside its folder when the slash was URL-encoded (`..%2F`), although a plain `../` did not reach it.

## What the lab builds

- **A server** (`server/`) called `lab-server`. Two tools: `echo` (returns `Echo: <text>`) and `write_file` (writes a file under the server's folder). One resource template, `file://resources/{filename}`, which reads a file from the resources folder. One prompt, `review_file(filename)`.
- **A client** (`client/`) with a command loop: `tools`, `call`, `resources`, `read`, `prompts`, `prompt`, `history`, `help` and `quit`. `call` and `prompt` take their arguments as typed JSON. `history`, `help` and the JSON-error hints are the lab's extensions.

## Run it

From the repo root, set up each app with its own virtual environment:

```powershell
cd course9-module3-lab1\server
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

Commands are PowerShell. No API key and no Node.js are needed. Start the client from `client/`. It launches `../server/run_server.py` over stdio, using the server's virtual environment, so create that one first:

```powershell
python mcp_client.py
```

Or run the two separately, in two terminals:

```powershell
# terminal 1, in server/ with its venv active
python run_server.py http

# terminal 2, in client/ with its venv active
python mcp_client.py http://127.0.0.1:8000/mcp
```

Type each command at the `>` prompt. `call`, `read` and `prompt` then ask for more on the lines after them, and what you type there is the tool name, the URI, or the JSON arguments, not a command. A first run that covers every command; what you type is whatever follows a `>`, `Tool name:`, `URI:` or `Prompt name:` prompt, and the JSON on the line under each `Arguments` line. The rest is output:

```
> tools
  • echo: Echo back the input text.
  • write_file: Write content to a file.
> call
  Tool name: echo
  Arguments (as JSON, for example, {"text": "hello"}):
  {"text": "Hello MCP!"}
  Result: Echo: Hello MCP!
> call
  Tool name: write_file
  Arguments (as JSON, for example, {"text": "hello"}):
  {"path": "test.txt", "content": "Hello from MCP client!"}
  Result: Successfully wrote to test.txt
> resources
  • read_resource_file
    URI template: file://resources/{filename}
> read
  URI: file://resources/project_info.txt
(the sample file's text)
> prompts
  • review_file: Generate a prompt to review a file's contents. (args: filename)
> prompt
  Prompt name: review_file
  Arguments (as JSON):
  {"filename": "test.txt"}
(the rendered prompt)
> history
> help
> quit
```

The `write_file` call creates `server/workspace/test.txt`. The folder is created on first write and is not committed.

To add a resource, put a file in `server/resources/` and read it with `file://resources/<its name>`; the server needs no change.

Tests need no key and no external network. Run `pytest` inside `server/` and inside `client/`. One client test launches a small stdio server as a subprocess; nothing uses port 8000.

Resolved on 12 October 2026:

| | |
|---|---|
| Python | 3.13.13 |
| `fastmcp` (server only) | 3.4.8 |
| `mcp` | 1.30.0 |
| `pydantic` | 2.14.0 |
| `pytest` / `ruff` | 9.1.1 / 0.16.8 |

Static checks: `ruff check` and `ruff format --check` are clean on both apps (rules E, F, I, W, UP, ANN; line length 100; target py313). Pylance was clean on the lab's client file after the fixes in F-L45-6. It was not run on the ported apps, and pyright was not run.

## File layout

Each app has its own layers, `config.py`, `conftest.py`, `pytest.ini`, `pyproject.toml` and `requirements.txt`. Both use the top-level package names `interfaces`, `infra` and `application`, so each app has to be run from inside its own folder.

```
course9-module3-lab1/
├── server/
│   ├── interfaces/file_store_interface.py   # FileStoreInterface, PathOutsideRootError
│   ├── infra/local_file_store.py            # the root check (F-L45-1)
│   ├── application/
│   │   ├── lab_server.py                    # build_server(file_store, resource_store, name)
│   │   └── container.py                     # the only file that imports infra and application
│   ├── resources/                           # sample files the template serves
│   ├── config.py                            # name, workspace and resources folders, host, port
│   ├── run_server.py                        # entry point: stdio, or `http`
│   └── tests/                               # mirrors the source layout
└── client/
    ├── domain/                              # tool, resource template, prompt, history entry
    ├── interfaces/                          # mcp_session, mcp_transport, user_input
    ├── infra/
    │   ├── stdio_transport.py
    │   ├── http_transport.py
    │   ├── official_mcp_session.py          # ClientSession; returns primitives
    │   └── console_user_input.py
    ├── application/
    │   ├── interfaces/client_service_interface.py
    │   ├── client_service.py                # the command loop
    │   ├── mappers.py / formatters.py
    │   └── container.py                     # picks the transport; the only file that imports infra and application
    ├── config.py                            # how the default server is launched
    ├── mcp_client.py                        # entry point: [server script or http URL]
    └── tests/                               # mirrors the source layout
```

`build_server` takes the two file stores as parameters and imports nothing from `infra`. The server tests call it through an in-memory `fastmcp` `Client` with `MagicMock(spec=FileStoreInterface)`. The client's session tests run the SDK's own low-level server on the other end of in-memory streams, through a test transport, so they need no process and no `fastmcp`.

## Differences from the lab

- One file per side becomes two apps in layers. `LocalFileStore` is the only code that touches the filesystem and the only place the root check lives (F-L45-1). The tools, resource template and prompt keep the lab's names, arguments and text.
- The server's folders come from `config.py`. The tools write under `server/workspace/`, and the template reads from `server/resources/`. The lab used the current directory for both, so `write_file` could write into the resources folder; it cannot now.
- The client's two transports, `StdioTransport` and `HttpTransport`, implement one interface, and the container picks one from the command-line argument: a URL gives HTTP, anything else is a script launched over stdio. `OfficialMcpSession` takes a transport and returns primitives, mappers turn them into domain objects, and the command loop works on those.
- The server is launched with the server's own interpreter (`server/venv`), not `python` from the path. The apps have separate environments, and the client's has no `fastmcp`. The argument is optional; the lab requires a script path.
- The `resources` command reads each template's `name` and `uriTemplate` directly. The lab's `getattr` fallbacks would print `N/A` or `Unnamed resource` if a field were renamed. Non-text content parts are skipped explicitly.
- Added, not in the lab: an `http` mode for the server and a URL argument for the client.
- Return types, `isinstance` narrowing and a session helper (F-L45-6). The history is a typed record, and `help` also lists `history`, which the lab's help text does not because its extensions are separate.
- Kept as the lab has it: the check-mark glyph (F-L45-4), a tool result flagged as an error printing like any result (pinned by a test), a failed connection raising out of the client, and `input()` blocking inside the async loop.
- The lab's sample files are not included. `server/resources/` holds two files of my own text, and the lab's `notes.txt` exercise is not reproduced.
- The lab pins `mcp==1.16.0` and `fastmcp==2.12.5` and leaves `pydantic` unpinned. I did not try that install; Lab 44's older pins failed at `import fastmcp` with `pydantic` 2.14.0 (F-L44-1). The port pins every direct dependency at the Lab 43 and 44 versions, and the client does not install `fastmcp`.

## Findings

**F-L45-1 - the resource template read a file outside its folder through URL-encoded slashes, so the root check lives in one store class.** The lab's template builds `BASE_DIR / "resources" / filename` with no check. I ran the port's template through an in-memory client against a store that did not yet have the check, with `secret.txt` beside the resources folder. `file://resources/ok.txt` returned its text, and `file://resources/../secret.txt` returned `File not found: secret.txt`. But `file://resources/..%2Fsecret.txt`, `file://resources/%2e%2e%2fsecret.txt` and `file://resources/..%5Csecret.txt` all returned the content of `secret.txt`: the server decodes the slash before it fills `{filename}`. `LocalFileStore` now resolves each path under its root and raises `PathOutsideRootError` unless `is_relative_to(root)` holds. Its tests cover `../`, a nested `../`, an absolute path and, on Windows, `..\`; `sub/../a.txt` is allowed because it stays inside. Container tests send the three encoded URIs through the real stores. With the check replaced by `if False:` on a scratch copy, 10 tests failed; restored, all pass. The probe ran against the port's store before the check, not against the lab's own `mcp_server.py`, which I did not probe. Not tested: Windows drive and UNC paths, and a symbolic link (that test is skipped because my Windows account cannot create links).

**F-L45-2 - the lab's `write_file` joins the path onto the folder with no check (from reading, not run).** `BASE_DIR / path` accepts `..` segments, and under pathlib's rules an absolute path replaces `BASE_DIR`. The lab's own server was not probed. In the port the tool writes through the same store: a path outside the root returns an error result and writes nothing, tested with `../escaped.txt`. Inside the root, parent folders are still created as the lab does.

**F-L45-3 - a missing resource file is ordinary text, not an error.** `file://resources/missing.txt` returned `File not found: missing.txt` as the resource content, over stdio and over HTTP. No error is raised, so a caller that checks only for errors shows the message as the file. The port keeps the lab's behaviour and pins it with a test; any other read failure raises.

**F-L45-4 - the check-mark glyph crashes on redirected output on Windows.** With stdout piped and no `PYTHONIOENCODING`, `print("✓ Connected to MCP server")` raised `UnicodeEncodeError: 'charmap' codec can't encode character '✓'` and the client exited with code 1. With `PYTHONIOENCODING=utf-8` the same session ran. In a terminal window the line printed. Lab 43's F-L43-8 is the same problem; that lab removed the glyph, and this port keeps the lab's.

**F-L45-5 - the server banner prints despite the lab's logging line.** The lab's server sets the `fastmcp` logger to WARNING to keep output quiet. Running the server alone, the FastMCP banner and an "Update available: 4.1.0" notice still printed. In the client's merged output they also appear; the SDK relays the child's stderr, so they come from the server's stderr, which I did not check separately. In a piped run the banner's box characters printed as `▄`-style escapes, and I did not find out why. The port keeps the lab's logging line and prints nothing of its own to the server's stdout.

**F-L45-6 - the lab's client has Pylance and Ruff problems.** It has no annotations, so Ruff's `ANN` rules flag every method. Pylance flagged `self.session = None` followed by method calls (six errors), and `read_resource(uri)`, because the SDK takes an `AnyUrl` and not a `str`. Once the methods had real return types, it also flagged `.text` on a union of content types in `call`, `read` and `prompt`, and `.get(...)` on the same union; the unannotated code showed none of these. The fixes: return types from `mcp.types`, `isinstance` narrowing against `TextContent` and `TextResourceContents`, a `_require_session()` helper, `AnyUrl(uri)`, and a local variable in `connect` so the session is stored only after `initialize()` succeeds. Assigning `self.session` directly made Pylance report "Class definition for `ClientSession` depends on itself", and I did not trace why. No `cast` and no `type: ignore`. One line in the `prompt` command was over 100 characters.

**F-L45-7 - the lab's Extension 1 "full code" listing is incomplete (from reading, not run).** It leaves out the `self.history` initialisation, the `history.append` in `call_tool`, and the `cleanup` method that `main()` calls in `finally`. I built the extension from the lab's step-by-step blocks and did not run the listing.

**F-L45-8 - the lab's Best Practices ask for two things the client does not do.** Point 5 says to check tool schemas and point 6 says to validate URIs against the templates. The `tools` command prints only name and description, and `read` sends the URI as typed, so the server does the matching. Point 2 calls `session.initialize()` an MCP handshake that negotiates protocol version and capabilities. That fits this stack: the SDK reports 2025-11-25 as its latest protocol version. I did not record which version a run negotiated, and I did not run anything against a 2026-07-28 server or client.

**F-L45-9 - the same session gives the same output over stdio and over HTTP.** I ran a scripted session (tools, both calls, resources, two reads, prompts, the prompt, history) once over each. The server's log for the HTTP run showed `POST /mcp` 200, `GET /mcp` 200, a `POST` 202 and a closing `DELETE /mcp` 200. The HTTP transport uses `streamable_http_client`, which on `mcp` 1.30.0 replaces the deprecated `streamablehttp_client` (Lab 43 F-L43-6). It also yields a third value, a session-id getter, which the transport drops.

## Prior art carried across

- **Containment on a client-supplied path (Lab 43 F-L43-1; Lab 44 F-L44-3).** F-L45-1 and F-L45-2 repeat the resolve-and-`is_relative_to` check. The new part is the way in: a URL-encoded slash in a resource URI.
- **Errors delivered as ordinary text (Lab 41 F-L41-7; Lab 43 F-L43-5; Lab 44 F-L44-4).** F-L45-3 is the same shape.
- **Windows output encoding (Lab 43 F-L43-8).** F-L45-4.
- **SDK drift between stacks (Lab 43 F-L43-6).** The HTTP client in F-L45-9.
- **One class per transport (Lab 43).** Stdio and HTTP are separate classes behind one interface here too, chosen in the container.

## What this doesn't cover

- The lab's own `mcp_server.py` against a traversal path; F-L45-1 and F-L45-2 are about the port's store and, for `write_file`, from reading.
- Windows drive and UNC paths, and symbolic links, in the root check.
- The lab's pinned install (`mcp` 1.16.0, `fastmcp` 2.12.5).
- `fastmcp` 4.x, `mcp` 2.x and the 2026-07-28 protocol revision.
- Which protocol version a run negotiated (F-L45-8).
- Pylance and pyright on the ported apps.
- Tool schema checks and URI validation in the client (F-L45-8).
- The HTTP server listens on 127.0.0.1 with no authentication, and I did not review origin checks. The HTTP path was run by hand once; the client's automated tests cover the transport with a stand-in, not a real HTTP server.
- One machine, one Python 3.13.13, one set of pinned versions.

---

**Completed:** 17 October 2026
