# Course 9 - Lab 46: Advanced MCP Applications with Streamable HTTP, Roots, and Sampling (V1 - Canonical)

> Code: [`course9-module3-lab2-v1-canonical/`](course9-module3-lab2-v1-canonical/)

The lab's code as written, in four files: an HTTP MCP file server, a base client, a Gradio client app and a Gradio AI host app that lets `gpt-4o-mini` use the server's tools. Only what Ruff and Pylance forced was changed, and each change is listed below. The onion-architecture port is in [V2](lab46notes-c9-m3-advanced-mcp-applications-v2-onion.md).

This is IBM Skills Network's "Advanced MCP Applications with Streamable HTTP, Roots, and Sampling" lab (the lab text is not included here) on Windows. The lab is built for Python 3.11; this folder runs on 3.13.13.

The most useful result: the lab's "roots" and "sampling" are not MCP features as built. The client declares neither, the "roots" are a folder hard-coded in the server, and `analyze_code` returns a canned string without sending a sampling request.

## What the lab builds

- **A server** (`mcp_http_server.py`), "HTTP File Server", on `127.0.0.1:8000` at `/mcp`. Four tools (`read_file`, `write_file`, `list_files`, `analyze_code`), one resource template (`file://workspace/{filename}`) and two prompts (`review_code`, `analyze_security`). File access goes through `is_within_roots` against a `workspace/` folder.
- **A base client** (`mcp_http_client_base.py`) on the official `ClientSession` over Streamable HTTP.
- **A GUI client** (`mcp_http_client_app.py`, port 7861): a Gradio page to list and call tools, resources and prompts.
- **An AI host** (`mcp_http_host_app.py`, port 7862): a Gradio chat where `gpt-4o-mini` gets the server's tools plus four helper tools for resources and prompts, and one round of tool calls per message.

## Run it

From the repo root, in PowerShell:

```powershell
cd course9-module3-lab2-v1-canonical
py -3.13 -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Put `OPENAI_API_KEY=...` in a `.env` file in this folder (gitignored). Only the host app needs it. Then, in three terminals with the venv active:

```powershell
python mcp_http_server.py                              # terminal 1, port 8000
python mcp_http_client_app.py http://127.0.0.1:8000    # terminal 2, page on http://127.0.0.1:7861
python mcp_http_host_app.py http://127.0.0.1:8000      # terminal 3, page on http://127.0.0.1:7862
```

`workspace/` holds `test.txt` and `README.md` for the tests. Opening `http://127.0.0.1:8000/` in a browser shows "Not Found"; the server answers only at `/mcp`.

Resolved on 18 October 2026 (the lab's pins, Python 3.13.13):

| | |
|---|---|
| `mcp` / `fastmcp` | 1.16.0 / 2.12.5 |
| `pydantic` | 2.11.10 (not pinned by the lab) |
| `httpx` / `uvicorn` | 0.28.1 / 0.38.0 |
| `gradio` / `openai` | 5.49.1 / 2.6.1 |

`ruff check` is clean (rules E, F, I, W, UP, ANN; line length 100; target py313) and Pyright 1.1.414 (basic mode) reports 0 errors. The editor's Pylance was clean on the GUI file; the others were checked with Pyright only.

## Deviations from the lab

1. A `py -3.13` virtual environment with `Activate.ps1`, not `python3.11`; the venv sits beside the code.
2. A `requirements.txt` with the lab's six pins, instead of the lab's one-line `pip install`. See F-L46-1.
3. A `pyproject.toml` for Ruff.
4. Server: the `f` prefix removed from four f-strings with no placeholders (`F541`); imports sorted.
5. Base client: imports sorted, unused `asyncio` removed; `self.session` typed `ClientSession | None` behind a `_require_session()` helper; `connect()` keeps the session in a local variable until `initialize()` succeeds; `read_resource` passes `AnyUrl(uri)`; return types from `mcp.types` on every method.
6. GUI app: annotations throughout; `isinstance` narrowing in `gui_call_tool`, `gui_read_resource` and `gui_get_prompt` (non-text content is skipped); three long lines wrapped. The lab's `getattr` fallbacks are kept.
7. Host app: `python-dotenv` and `load_dotenv()` at the start of `main()` (F-L46-6); annotations including `list[ChatCompletionMessageParam]`; an `isinstance` filter on the model's tool calls; `getattr` after `hasattr` in the lab's dead branches. The behaviour the findings below describe is untouched.
8. The workspace test files are written as plain UTF-8, not with `echo >`, which writes a byte-order mark in this PowerShell. What a BOM does to `read_text()` was not run.

## Findings

**F-L46-1 - the lab's pins do not install on Python 3.14.** `pip install` failed building `pydantic-core`. From the packages' metadata: `gradio` 5.49.1 requires `pydantic<2.12`, so only `pydantic` 2.11.x fits, every 2.11.x needs `pydantic-core` 2.33.2, and that version has no 3.14 wheel. Why the source build failed (probably no Rust toolchain) is an inference; only the summary lines were pasted. On 3.13.13 the pins installed and `pip check` is clean. `import fastmcp` works (2.12.5), unlike Lab 44's older pins with `pydantic` 2.14.0 (F-L44-1), and prints an `AuthlibDeprecationWarning` from a dependency. The lab's server filters `DeprecationWarning`; whether that hid it at start-up was not reported.

**F-L46-2 - the client declares neither roots nor sampling.** Both bases call `ClientSession(read, write)` with no callbacks. In `mcp` 1.30.0, `ClientSession.initialize` declares the `roots` capability only when a `list_roots_callback` is passed and `sampling` only with a `sampling_callback` (Seen in the SDK source; the lab pins 1.16.0, which I did not read). The server never sends `roots/list`. The client's `roots_dir` only changes display text. The lab's text says the client "maintains full control over filesystem access" and that you learned to "allow clients to specify allowed directories". The check that protects the files is the server's own hard-coded folder and `is_within_roots`, which is where such a check belongs.

**F-L46-3 - `analyze_code` is a canned string.** Asked for the tool's raw output, the model returned the stub exactly (`[SAMPLING TRIGGER]` ... "Note: Full bidirectional sampling requires low-level MCP SDK."). No sampling request is sent. On `fastmcp` 3.4.8, `Context` has `sample`, `sample_step` and `list_roots`, and `fastmcp.Client` takes `roots`, `sampling_handler` and `sampling_capabilities` (Seen as attributes). A real round trip was **not run**.

**F-L46-4 - the host app hands the model Python reprs and drops prompt messages.** `execute_tool` falls through to `str(result)`, so a resource read reaches the model as `meta=None contents=[TextResourceContents(uri=AnyUrl(...), ..., text='...')]`. For prompts, iterating a `GetPromptResult` yields `(field, value)` tuples, so a retrieved prompt came back as three `[unknown]:` lines with empty content (Reported: the pasted reply; the three-tuple count follows from the types).

**F-L46-5 - the `mcp_get_prompt` argument shape contradicts the lab's text.** The lab says the model passes `name` and the prompt's arguments at the top level. The tool schema nests them under an `arguments` object with no declared properties, and `execute_tool` reads only `arguments.get("arguments", {})`. Reproduced on the server side: `get_prompt("review_code", {})` raises `Missing required arguments: {'filename'}`; with `{"filename": "test.txt"}` it works. In Daniel's run the model's first retrieval failed with a complaint about the missing filename. The exact arguments the model sent were not seen.

**F-L46-6 - the lab's `OpenAI()` never reads a `.env` file.** With no `OPENAI_API_KEY` in the environment it raises `OpenAIError` at construction (Seen). The host app here calls `load_dotenv()` (deviation 7); the lab does not.

**F-L46-7 - a resource template lists nothing the model can read.** "What resources are available? Then read one of them" listed `file://workspace/{filename}` and then asked for a filename instead of reading one; with a concrete URI it returned the file. A template is a pattern, not a list. Why the model asks rather than picks `test.txt` was not established.

**F-L46-8 - the root check held; I did not probe the encoded-slash and `list_files` cases here.** Reported and Seen: `read_file` of `../mcp_http_server.py` and `/etc/passwd` gave `Error: Access denied - path outside workspace roots`; `sub/../test.txt` returned the file (allowed, because `resolve()` collapses the `..`). **Not run against the canonical server:** `file://workspace/..%2Fmcp_http_server.py` (the form that got through in Lab 45, F-L45-1) and `list_files` with `{"directory": ".."}`. Both are tested in the V2 port.

**F-L46-9 - the lab's text overclaims.** "Enterprise-grade security" (line 7), "multi-tenant safe" (Conclusion) and that you implemented sampling approval. The server has no authentication, no HTTPS and no tenant concept.

**F-L46-10 - the lab's code has the Lab 45 type problems again, plus one.** As written, the base client had 7 Pylance errors (six on `self.session` being optional, one `str` where the SDK wants `AnyUrl`) and 11 Ruff findings; the GUI app had 19 Ruff findings and 14 Pylance errors (`.text` and `.get` on the content unions); the host app had 11 Ruff findings and, from Pyright, 7 errors, including `.function` on the custom-tool-call member of OpenAI's tool-call union. The fixes are deviations 5 to 7, with no `cast` and no `type: ignore`.

## Prior art carried across

- **The same type problems in the lab client (Lab 45 F-L45-6).** F-L46-10.
- **Pins that stop installing (Lab 44 F-L44-1).** F-L46-1.
- **The lab text overclaiming (Lab 41 and later).** F-L46-9.

## What this doesn't cover

- Real roots and sampling on `fastmcp` 3.4.8 or the SDK (F-L46-3).
- The encoded-slash and `list_files` probes against the canonical server (F-L46-8).
- Whether the server, which is the lab's, validates `Host` and `Origin` (see F-L46-11 in V2: it does not by default).
- `fastmcp` 4.x, `mcp` 2.x and the 2026-07-28 protocol revision, which deprecates roots and sampling and removes the `initialize` handshake. I took those from a chat summary of the spec and did not check them.
- The Lesson 2 video's claims.
- One machine, one Python 3.13.13, the lab's pins. The model's replies are from `gpt-4o-mini` and differ between runs.

---

**Completed:** 18 October 2026
