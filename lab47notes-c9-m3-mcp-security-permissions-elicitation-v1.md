# Course 9 - Lab 47: MCP Security with Permissions and Elicitation (V1 - Canonical)

> Code: [`course9-module3-lab3-v1-canonical/`](course9-module3-lab3-v1-canonical/)

The lab's code as written, in four files: a FastMCP server with four risk-tiered tools, a base client that applies allow, deny and ask policies and writes an audit log, a Gradio client app and a Gradio AI host app that lets `gpt-4o-mini` call the tools. Only what Ruff and Pyright forced was changed, and each change is listed below. The onion-architecture rebuild, with these findings fixed, is in [V2](lab47notes-c9-m3-mcp-security-permissions-elicitation-v2-onion.md).

This is IBM Skills Network's "MCP Security with Permissions and Elicitation" lab (the lab text is not included here) on Windows. The lab is built for Python 3.11; this folder runs on 3.13.13.

The most useful result: the lab's "elicitation" is not MCP elicitation, and its permission layer is a client-side check that can be bypassed in several ways, which I ran. `request_elicitation` prints and returns `{}`, and nothing calls it. "Approval" is a boolean the caller passes, or the word "yes" typed in the host's chat.

## What the lab builds

- **A server** (`mcp_permission_server.py`) over stdio only. Four tools: `read_file` (LOW), `write_file` (MEDIUM), `delete_file` (HIGH) and `execute_command` (CRITICAL, a simulation that only logs). Two resources (`file://audit/log`, `file://config/permissions`) and one prompt (`security_review`). Files live in `data/` beside the script.
- **A base client** (`mcp_permission_client_base.py`): `check_permission`, `log_audit`, `call_tool_with_permission(tool, args, approved=False)` and the protocol methods. Policies are in `data/permissions.json`; defaults are read allow, write ask, delete deny, execute deny.
- **A GUI client** (`mcp_permission_client_app.py`, port 7863): tabs for tools, resources, prompts and permissions, with an **Approve & Execute** button.
- **An AI host** (`mcp_permission_host_app.py`, port 7864): a chat where the model gets the server's tools plus four helper tools for resources and prompts. A reply of "yes" runs the stored pending call.

## Run it

From the repo root, in PowerShell:

```powershell
cd course9-module3-lab3-v1-canonical
py -3.13 -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Put `OPENAI_API_KEY=...` in a `.env` file in this folder (gitignored). Only the host app needs it. Then:

```powershell
python mcp_permission_client_app.py mcp_permission_server.py   # http://127.0.0.1:7863
python mcp_permission_host_app.py mcp_permission_server.py     # http://127.0.0.1:7864
```

Run both from this folder: the server and the client each use a `data/` folder, and they are one folder only then (see F-L47-7). Create `data/test.txt` with an editor or `Set-Content -Encoding utf8`, not `echo >` (F-L47-12). `data/` is gitignored.

Resolved on 23 October 2026 (the lab's pins, Python 3.13.13):

| | |
|---|---|
| `mcp` / `fastmcp` | 1.16.0 / 2.12.5 |
| `pydantic` | 2.11.10 |
| `httpx` / `uvicorn` | 0.28.1 / 0.38.0 |
| `gradio` / `openai` | 5.49.1 / 2.6.1 |
| `python-dotenv` | 1.2.4 (not pinned) |
| `ruff` | 0.17.2 (installed by `gradio`, which requires `ruff>=0.9.3`; not pinned) |

`pip check` is clean. `ruff check` and `ruff format --check` are clean (rules E, F, I, W, UP, ANN; line length 100; target py313) and Pyright (basic mode) reports 0 errors on the four files. The linters ran through `uvx` and I did not record their versions.

## Deviations from the lab

1. A `py -3.13` virtual environment named `venv`, not `python3.11` and `mcp_security_env`. The code sits at the folder root, with no `mcp_security_lab` subfolder, and `data/` is created beside it.
2. A `requirements.txt` instead of the lab's one-line `pip install`: the lab's four pins, plus `httpx` and `uvicorn` (the Lab 46 pins; this stdio lab imports neither) and `pydantic==2.11.10`, which the base client imports for `AnyUrl`. `gradio` 5.49.1 caps `pydantic` below 2.12, so 2.11.10 is the newest that installs.
3. A `pyproject.toml` for Ruff, and `ruff format` applied to all four files.
4. Imports sorted in all four files; the unused `asyncio` removed from the base client.
5. Base client: `self.session` typed `ClientSession | None` behind a `_require_session()` helper; `connect()` keeps the session in a local variable until `initialize()` succeeds; return types from `mcp.types` on every method; `read_resource` passes `AnyUrl(uri)`; arguments typed `dict[str, object] | None` and `dict[str, str] | None`; bare `dict` annotations replaced.
6. Base client: the deny and ask results are `TextContent`, not the lab's `type('obj', (), {'text': ...})` objects. Pyright accepted the old objects while the return type was unannotated and rejected them once it was `list[ContentBlock]`. Callers still read `.text`.
7. GUI app: annotations throughout; `isinstance` narrowing in place of `hasattr` and `getattr`; caches typed `list[str]`; long lines wrapped.
8. Host app: `python-dotenv` and `load_dotenv()` in `main()` (F-L47-11); `TypedDict`s for the pending approval and the risk assessment; `conversation_history` typed as `list[ChatCompletionMessageParam]`; an `isinstance` filter on the model's tool calls; `uri`, `name` and prompt arguments checked as strings before use (a non-string `uri` now returns "URI is required"); the dead `hasattr(result, 'text')` branch removed; the unused `assess_risk` result and `perm_info` variable no longer assigned.
9. `data/test.txt` is the file the lab's `echo` wrote in this PowerShell, moved into the server's `data/`; the tests used it unchanged. The seed `pytest.ini` and `conftest.py` were deleted (no tests in this folder).
10. The apps were run from PowerShell, not Git Bash (F-L47-13).

No `Any`, `cast`, `type: ignore`, `getattr` or `hasattr` in the four files (checked by search).

## Test results

Marks: **Seen** = checked in a file or output here; **Reported** = described to me, not pasted; **Not run**.

- **Test 1, allow `read_file`:** audit line Seen (`TOOL: read_file - Decision: ALLOWED - Reason: Policy: allow`). The Result box text was Reported as fine and not pasted.
- **Test 2, deny `delete_file`:** Result `Permission denied for tool: delete_file` Reported; audit line Seen (`DENIED ... Policy: deny`).
- **Test 3, ask `write_file`:** after **Approve & Execute**, Result `Successfully wrote to new_file.txt` Reported; audit lines Seen (`ASK ... Awaiting approval`, `ALLOWED ... Policy: ask`, then the server's `WRITE: new_file.txt`). The approval-request text was not pasted.
- **Test 4, host:** the three replies were pasted (Reported). Behind them, Seen in the audit log: `read_file` allowed, `delete_file` denied, `write_file` asked. Then "yes" wrote `greeting.txt` (Seen: 13 bytes, `Hello, world!`). The replies differ from the lab's expected text; the lab says the model's behaviour varies.

## Findings

**F-L47-1 - there is no MCP elicitation.** The server never calls `ctx.elicit`. `request_elicitation` is defined once in the base client, prints "automatic approval for demo" and returns `{}`, and no file calls it (search of the four files; the only other mentions are two server docstrings). `ClientSession(read, write)` takes no callbacks; whether that omits the elicitation capability I did not check in the SDK. The lab's Conclusion admits the elicitation is "conceptual", then lists a "Three-action model (Accept, Decline, Cancel)" as something learned; the code has none. Lab 44 ran real elicitation on `fastmcp` 3.4.8; it was not run here.

**F-L47-2 - the checks live in the client; the server enforces nothing.** The "(Risk: ...)" text is in the tool docstrings and nothing on the server reads it (from reading). Seen on a copy: a plain `mcp` `ClientSession`, with no permission layer, called `delete_file` and the file was deleted. Any other client runs every tool.

**F-L47-3 - approval is not tied to what the user saw.** In the host test the model said it could not create files and gave shell instructions; the approval request had gone to the model as the tool result and was not shown. "yes" then ran the stored write (Seen: `greeting.txt` on disk). The host stores one `pending_approval`, set when a tool result contains both "Permission required for tool:" and "Please approve this operation" (from reading). The GUI button sends `approved=True` for whatever is in the dropdown and text box at the click (from reading). The audit line for the approved run is the same as an automatic allow.

**F-L47-4 - deny cannot be approved, so HIGH and CRITICAL tools never reach any confirmation.** Seen with `approved=True` against a `deny`: `Permission denied for tool: delete_file`, file kept. With the default policies `delete_file` and `execute_command` are blocked outright; the "detailed" and "extensive" confirmation the docstrings promise is never asked for.

**F-L47-5 - a policy value that is not `deny` or `ask` runs the tool, and an argument-level rule beats a tool-level deny.** Seen: `read_file` set to `"alow"` ran. Seen: with `delete_file` at `deny` and an argument-level `allow` for one path, the file was deleted. The GUI only offers allow, deny and ask, and has no way to set an argument-level rule, so both routes need a hand-edited `permissions.json`. The Conclusion presents argument-level control as a feature.

**F-L47-6 - no path containment.** Seen with `read_file` at allow: `../mcp_permission_server.py` returned the server's source and `C:/Windows/win.ini` returned that file. The permission layer judges the tool, not the argument.

**F-L47-7 - the audit log does not record what the lab says.** Seen: lines have no arguments (`[ts] TOOL: read_file - Decision: ALLOWED - Reason: Policy: allow`); the lab's test pages expect `ALLOWED: read_file with args {...}`. An approved run is `ALLOWED ... Policy: ask`, so approvals are not recorded; no outcome is logged; saving a policy writes no line. The Conclusion lists "User approval events", "Tool execution outcomes" and "Full argument capture". The server and the client both append to `data/audit.log` in different formats (Seen: a `TOOL:` line and a `WRITE:` line 4 ms apart), because both ran from this folder. The client's `data/` is relative to the working directory and the server's is beside its script. Seen on a copy, with the client run from another directory: the client's `TOOL:` line went to that directory's `data/audit.log`, the server's `WRITE:` line to the script's own `data/`, and the server's `file://config/permissions` resource showed `read_file: allow` although the client had saved `deny`.

**F-L47-8 - the policy file and the audit log are reachable by the file tools.** Seen with `write_file` approved: `permissions.json` was replaced with `{"delete_file": "allow"}`, and an empty write to `audit.log` succeeded and left the file at 47 bytes (content not read). Seen: an `execute_command` argument containing a newline and a fake entry put the fake line in the log as its own line. The "audit trail for compliance" is a text file the audited tools can rewrite.

**F-L47-9 - resources and prompts are not checked.** Seen through the client: `file://config/permissions` and `file://audit/log` read with no policy. The host's `mcp_read_resource` helper has no check either (from reading), and its description suggests `file://audit/log`.

**F-L47-10 - the host's risk assessment is computed and thrown away.** Ruff flagged `risk_assessment` as assigned and never used (Seen). The level is a fixed name-to-level dict that ignores the arguments, and `requires_approval` is true for `deny`. The lab's text claims "dynamic risk assessment based on context". The Key Patterns snippet reads `risk['level']`; the key is `risk_level`, so it raises `KeyError` as written (from reading).

**F-L47-11 - the lab's `OpenAI()` never reads a `.env` file.** Same as Lab 46 F-L46-6. The host app here calls `load_dotenv()` (deviation 8).

**F-L47-12 - the lab's `echo` writes UTF-16, and the file tools read the platform default.** In this PowerShell, `echo "..." > data/test.txt` wrote 58 bytes starting `FF FE` (a UTF-16 little-endian byte-order mark, with CRLF). Seen: `Path.read_text()` under cp1252, which is what `read_file` does, returns `'ÿþS\x00a\x00m\x00...'`. Reported: through the host, the model said the file had unusual characters at the start. The Result box in the GUI was not pasted.

**F-L47-13 - the client could not connect when run from Git Bash.** `McpError: Connection closed` on **Load Tools**, with nothing printed by the server. A probe that connects the same way printed `FAILED: McpError Connection closed` in that window and `OK tools: [...]` with stderr redirected to a file (Seen from the pasted output). The apps ran in PowerShell. The cause is not established; my guess is the stderr handle Git Bash passes to the server, which I did not test. In my own shell, with the lab venv first on the PATH, the same client connected. The client starts the server with `command="python"`, so it depends on the PATH at launch.

**F-L47-14 - the lab's pins install on 3.13.13.** `pip check` is clean and `import fastmcp` works (2.12.5), with an `AuthlibDeprecationWarning` from a dependency. The pins do not install on Python 3.14 (Lab 46 F-L46-1); not retried here.

**F-L47-15 - the lab's code has the Lab 45 and 46 type problems again.** As written: base client 13 Ruff findings and 9 Pyright errors; GUI app 20 Ruff and 6 Pyright; host app 14 Ruff and 15 Pyright (including `.function` on the custom-tool-call member of OpenAI's union); server 0 Ruff and 0 Pyright. The fixes are deviations 4 to 8, with no `cast` and no `type: ignore`.

**F-L47-16 - the lab's text overclaims.** "Enterprise-grade" (introduction), "production-ready" (Deployment Summary), "production-grade" and "enterprise-level security" (Conclusion), "Fail-safe defaults protect against mistakes" and "Complete audit trail for compliance". The server has no authentication or HTTPS and runs as a stdio subprocess for one user; see F-L47-4 to F-L47-8.

**F-L47-17 - each app keeps its own copy of the policy, and a save overwrites the file.** Both apps load `permissions.json` once and write their whole dict on every save. Seen at the class level, with two client objects: one saved `read_file: deny`, the other then saved `write_file: allow`, and the file ended with `read_file` back at `allow`. A running app never sees the other's change. The GUI and the host running at once was not tried.

**F-L47-18 - the host's pending approval is state on one object.** The host keeps `pending_approval` and the conversation history on a single object. Seen at the class level: with a pending `write_file` set, a "yes" passed to that object's `chat()` ran the write (`p4.txt` written, the pending slot cleared); the call does not depend on who sent the message. Gradio calls the same bound method for every browser session, so a "yes" from a second tab would reach the same slot (from reading). A second real tab was not tried. This is the shared-state problem Copilot rated High in Lab 46 (F-L46-18), with an approval as the shared state.

## Prior art carried across

- **The same type problems in the lab client (Lab 45 F-L45-6, Lab 46 F-L46-10).** F-L47-15.
- **No path check on the lab's server (Lab 45 F-L45-1).** F-L47-6.
- **The `OpenAI()` constructor ignoring `.env` (Lab 46 F-L46-6).** F-L47-11.
- **Elicitation done for real (Lab 44).** F-L47-1.
- **The lab text overclaiming (Lab 41 and later).** F-L47-16.

## What this doesn't cover

- Real MCP elicitation, on any SDK version, and the 2026-07-28 protocol revision (Lab 44 F-L44-9), which I did not recheck.
- A "yes" typed in a second real browser tab, and the GUI and the host running at the same time (F-L47-17 and F-L47-18 were run at the class level).
- The GUI's Result box for Tests 1 to 3, and the approval-request text in Test 3: Reported, not pasted.
- The probes in F-L47-2, F-L47-4 to F-L47-9, F-L47-17 and F-L47-18 ran on a copy, driving the client or host classes directly with a plain UTF-8 `test.txt`, not through the GUI or the host's chat page.
- One machine, one Python 3.13.13, the lab's pins. The model's replies are from `gpt-4o-mini` and differ between runs.

---

**Completed:** 23 October 2026
