# Course 9 - Lab 47: MCP Security with Permissions and Elicitation (V2 - Onion-Architected)

> Code: [`course9-module3-lab3-v2-onion/`](course9-module3-lab3-v2-onion/)

[V1](lab47notes-c9-m3-mcp-security-permissions-elicitation-v1.md) is the lab as written, with 18 findings. This is the same system as two separate onion-layered apps on the latest versions, with those findings fixed from the start: a FastMCP server that enforces the policy itself and asks the user through MCP elicitation, and a client with a Gradio page and a Gradio chat host for `gpt-4o-mini`. 441 tests pass without a key or a network: 214 on the server (215 collected, one skipped) and 227 on the client.

The most useful result: `ctx.elicit` does not work on the 2026-07-28 protocol, which the latest SDKs negotiate by default. A tool now returns an "input required" result, the client answers and retries, and the approval can be bound to the exact arguments the user was shown (F-L47-19, F-L47-21).

Stack: Python 3.14.7, `mcp` 2.3.0, `fastmcp` 4.1.3 (server only), `gradio` 6.30.0, `openai` 3.37.0, `pydantic` 2.14.1, all pinned.

## Run it

PowerShell from the repo root, one virtual environment per app. On this machine `uv` needed `--system-certs` (TLS inspection); you may not.

```powershell
cd course9-module3-lab3-v2-onion\server
uv venv --python 3.14.7 venv
uv pip install --system-certs --python venv\Scripts\python.exe -r requirements.txt
cd ..\client
uv venv --python 3.14.7 venv
uv pip install --system-certs --python venv\Scripts\python.exe -r requirements.txt
```

Put `OPENAI_API_KEY=...` in `client/.env` (gitignored; only the host needs it). From `client/` with its venv active:

```powershell
python gui_app.py      # http://127.0.0.1:7863
python host_app.py     # http://127.0.0.1:7864
```

The client launches the server itself over stdio, using `server/venv`. Use PowerShell: from Git Bash the same launch failed (V1, F-L47-13). The policy is `server/state/permissions.json`, for example `{"write_file": "allow"}`; it is read on every call. `pytest` in each folder needs no key and no network.

## File layout

```
server/  domain/  interfaces/  infra/  application/  config.py  run_server.py  tests/
client/  domain/  interfaces/  infra/  application/  gui_app.py  host_app.py  tests/
```

Each app has its own layers, config and pins, and shares no code with the other. The contract between them is the protocol: tool names and schemas come from `list_tools`, and the server defines the resource addresses and the shape of the approval question.

- **Server:** `PermissionGate` (`decide` and `execute`, stateless, no "approved" flag), `PolicyBook`, `LocalFileStore`, `JsonPolicyStore`, `FileAuditLog`, `build_server`, `container.py`.
- **Client:** `McpSdkSession` on the official `mcp` client, `OpenAiChatModel`, `GuiService`, `HostService`, `ApprovalBroker` (holds a question until a person answers), `ApprovalDesks` (one broker per Gradio session). Only `gui_app.py` and `host_app.py` import Gradio.

## What V2 does about the V1 findings

| V1 | V2 |
|---|---|
| 1 no real elicitation | Real elicitation on the modern path (F-L47-19). |
| 2 server enforces nothing | Every tool goes through one gate on the server. |
| 3 approval not tied to what was shown | The question names the tool and arguments; an accept needs a token bound to them (F-L47-21). Seen in the real host run. |
| 4 a deny cannot be approved | Kept on purpose, now pinned by tests. Set a tool to `ask` to get a confirmation. |
| 5 bad policy values, argument-level rules | A bad value, a bad file or an argument-level key invalidates the file; every call is refused. |
| 6 no path containment | `LocalFileStore` resolves and checks every path. Seen: `../state/audit.log` and `C:/Windows/win.ini` refused. |
| 7 audit log gaps | One writer, JSON lines, the arguments and an event per step. Seen in the real trail. |
| 8 policy and log reachable, forged lines | They live in `state/`, outside the tools' `workspace/`; JSON escapes newlines. |
| 9 resources unchecked | Both resources go through the gate, but default to `allow`, so they are open out of the box. |
| 10 risk assessment discarded | Dropped. Risk is each tool's MCP annotations plus a "(Risk: ...)" label. |
| 11 `OpenAI()` ignores `.env` | The host entry point calls `load_dotenv()`. |
| 12 UTF-16 from `echo >` | UTF-8 throughout; a UTF-16 policy file is refused. `read_file` on one was not run. |
| 17, 18 two policy copies, shared pending approval | The clients cannot write the policy; each Gradio session has its own approval desk. |

Other differences: the GUI's "Approve & Execute" is a panel showing the waiting question with Approve and Decline; the Permissions tab is read-only; the host reads a bare yes, y, approve, ok or confirm as accept, no, n, deny or decline as decline, and cancel as cancel, so a sentence containing "yes" approves nothing; refused calls come back as MCP error results with the lab's wording. Kept from the lab: the four tools and defaults, the helper tools in the host, `execute_command` only simulating. No `Any`, `cast` or `type: ignore` in either app; Pyright reports 0 errors.

## Findings

**F-L47-19 - `ctx.elicit` is unavailable on the 2026-07-28 protocol.** `fastmcp` 4.1.3's client negotiates it by default, and `ctx.elicit()` then fails with "elicitation via server-initiated requests is unavailable on 2026-07-28 connections" (the string is in the library). The modern pattern is a tool returning an `InputRequiredResult`; the client answers and retries with the answer and an opaque `request_state`. I read this from the installed source and ran it, not from the protocol text. A client on the old handshake cannot be asked at all: the call fails and nothing runs (Seen, pinned by a test). This confirms Lab 44's note (F-L44-9).

**F-L47-20 - the official `mcp` 2.3.0 `Client` runs the retry loop; the low-level `ClientSession` negotiated the old handshake.** The `Client` negotiated 2026-07-28 and answered the question over stdio (Seen). My first spike used `ClientSession`, and `ctx.elicit` worked there, which implies the old handshake (inferred; I did not print that version).

**F-L47-21 - an approval can be bound to the exact arguments.** The gate returns a SHA-256 of the tool and sorted arguments as `request_state`, which the framework seals. An accept counts only with the matching token. Seen at the gate in tests; a replay with changed arguments over the wire was not run.

**F-L47-22 - the SDK client must be closed by the task that opened it.** Lab 46 left this open. `McpSdkSession` now runs the client in one dedicated task, so `connect()` and `close()` work from any task (Seen in tests).

**F-L47-23 - tool errors leaked the server's absolute path.** Found in the real GUI run: a missing file returned `[Errno 2] ... 'C:\\...\\server\\workspace\\missing.txt'`. Errors now name the relative path only.

**F-L47-24 - in Gradio, a second message waits behind the turn that is waiting for it.** Seen on 6.30.0: with the default concurrency limit it blocked until it timed out; with `concurrency_limit=None` it ran. The host's chat event sets it, so a "yes" can arrive while the turn is still waiting.

**F-L47-25 - an unanswered question is cancelled after 300 seconds, never approved.** Unit-tested. It also happened by accident in a real run: the audit log shows `approval_requested` then `cancelled`.

**F-L47-26 - three of my tests passed when they should have failed.** Mutation checks found the connect-lock test, the abandoned-question test and the path-leak tests (Windows doubles backslashes inside an `OSError` message, so a full-path comparison could not see the leak). All three were strengthened and re-checked. A mutation helper of mine also silently did nothing twice; it now stops if the text is not found.

## Tests

Names read `test_given_..._when_..._then_...`, written test first, with a docstring on each design pin. The server tests drive the real server with an in-memory client and a real elicitation handler. The client tests run the official client against an in-process server, plus one real stdio subprocess. Mutation checks on scratchpad copies covered the policy, the gate, the audit log, the file store, the container, the session, the broker and the host page; all were caught after F-L47-26.

## Real runs

Seen here, Windows and PowerShell; the model's replies vary.

- **Server over stdio:** `write_file` asked once with the arguments in the question, then wrote; `delete_file` denied; outside paths refused; the audit trail was complete.
- **GUI app** (a real process, two `gradio_client` sessions): session B could not see or answer A's question; A's approve wrote the file, A's decline wrote nothing (file absent on disk).
- **Host app with `gpt-4o-mini`:** A's chat showed the exact question; a "yes" from session B was an ordinary message and A's turn kept waiting; A's own "yes" wrote the file. A declined write wrote nothing. "Delete the file" came back as `Permission denied for tool: delete_file`.

Python from my shell could not verify `api.openai.com` (the same TLS inspection), so the host ran through a wrapper outside the repo that uses the system certificate store. That is my environment, not the app.

## What this doesn't cover

- A replay with changed arguments over the wire, `read_file` on a UTF-16 file, a second real browser tab, and running from Git Bash.
- The resources default to `allow`: the audit log, which lists arguments, is readable out of the box. Set `read_audit_log` to `ask` or `deny` to change that.
- Stopping a standalone server: Ctrl+C prints a traceback, and a stop signal sent while the client's pipe is still open did not end the process on Windows in my probe. The client stops the server by closing the pipe, which exits cleanly. I tried handling stop signals and dropped it; it is not resolved.
- Authentication and HTTPS: the server is a local stdio subprocess for one user.
- The 2026-07-28 protocol text; the notes rely on the installed libraries' source and my runs. Both major versions are new, and the pins are exact.
- Other platforms or Python versions; one machine. The symlink test is skipped here.

---

**Completed:** 24 October 2026
