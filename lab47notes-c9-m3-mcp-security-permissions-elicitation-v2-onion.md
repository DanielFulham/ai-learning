# Course 9 - Lab 47: MCP Security with Permissions and Elicitation (V2 - Onion-Architected)

> Code: [`course9-module3-lab3-v2-onion/`](course9-module3-lab3-v2-onion/)

[V1](lab47notes-c9-m3-mcp-security-permissions-elicitation-v1.md) is the lab as written, with 18 findings. This is the same system as two separate onion-layered apps on the latest versions, with those findings fixed from the start: a FastMCP server that enforces the policy itself and asks the user through MCP elicitation, and a client with a Gradio page and a Gradio chat host for `gpt-4o-mini`. 507 tests pass without a key or a network: 241 on the server (242 collected, one skipped) and 266 on the client.

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
| 3 approval not tied to what was shown | The question names the tool and arguments; an accept needs a single-use token the server issued for them (F-L47-21). Seen in the real host run and over the wire. |
| 4 a deny cannot be approved | Kept on purpose, now pinned by tests. Set a tool to `ask` to get a confirmation. |
| 5 bad policy values, argument-level rules | A bad value, a bad file or an argument-level key invalidates the file; every call is refused. |
| 6 no path containment | `LocalFileStore` resolves and checks every path. Seen: `../state/audit.log` and `C:/Windows/win.ini` refused. |
| 7 audit log gaps | One writer, JSON lines, the arguments and an event per step. Seen in the real trail. |
| 8 policy and log reachable, forged lines | They live in `state/`, outside the tools' `workspace/`; JSON escapes newlines. |
| 9 resources unchecked | Both resources go through the gate. The audit log, which lists every call's arguments, asks by default; the policy view is allowed. |
| 10 risk assessment discarded | Dropped. Risk is each tool's MCP annotations plus a "(Risk: ...)" label. |
| 11 `OpenAI()` ignores `.env` | The host entry point calls `load_dotenv()`. |
| 12 UTF-16 from `echo >` | UTF-8 throughout; a UTF-16 policy file is refused. `read_file` on one was not run. |
| 17, 18 two policy copies, shared pending approval | The clients cannot write the policy; each Gradio session has its own approval desk. |

Other differences: the GUI's "Approve & Execute" is a panel showing the waiting question with Approve and Decline; the Permissions tab is read-only; the host reads a bare yes, y, approve, ok or confirm as accept, no, n, deny or decline as decline, and cancel as cancel, so a sentence containing "yes" approves nothing; refused calls come back as MCP error results with the lab's wording. Kept from the lab: the four tools and defaults, the helper tools in the host, `execute_command` only simulating. No `Any`, `cast` or `type: ignore` in either app; Pyright reports 0 errors.

## Findings

**F-L47-19 - `ctx.elicit` is unavailable on the 2026-07-28 protocol.** `fastmcp` 4.1.3's client negotiates it by default, and `ctx.elicit()` then fails with "elicitation via server-initiated requests is unavailable on 2026-07-28 connections" (the string is in the library). The modern pattern is a tool returning an `InputRequiredResult`; the client answers and retries with the answer and an opaque `request_state`. I read this from the installed source and ran it, not from the protocol text. A client on the old handshake cannot be asked at all: the call fails and nothing runs (Seen, pinned by a test). This confirms Lab 44's note (F-L44-9).

**F-L47-20 - the official `mcp` 2.3.0 `Client` runs the retry loop; the low-level `ClientSession` negotiated the old handshake.** The `Client` negotiated 2026-07-28 and answered the question over stdio (Seen). My first spike used `ClientSession`, and `ctx.elicit` worked there, which implies the old handshake (inferred; I did not print that version).

**F-L47-21 - an approval is a single-use token the server issued.** My first version was a SHA-256 of the tool and arguments, which any client can compute, so a stolen or replayed accept worked (Copilot flagged it; I confirmed the replay over stdio: the same accepted state wrote twice). The gate now issues a random token per question, remembers the tool and arguments it was for, and accepts it once, within 300 seconds, for those exact arguments. Any use spends it, and at most 100 questions are held. Seen over stdio: a forged state, no state, a replayed state and a state used with changed arguments were all refused. For changed arguments the framework's own check refused first ("request binding") and the token was still usable afterwards. A token shows the server asked; it does not show a person answered, so a client that holds one can accept for the user (see the end).

**F-L47-22 - the SDK client must be closed by the task that opened it.** Lab 46 left this open. `McpSdkSession` now runs the client in one dedicated task, so `connect()` and `close()` work from any task (Seen in tests).

**F-L47-23 - tool errors leaked the server's absolute path.** Found in the real GUI run: a missing file returned `[Errno 2] ... 'C:\\...\\server\\workspace\\missing.txt'`. Errors now name the relative path only.

**F-L47-24 - in Gradio, a second message waits behind the turn that is waiting for it.** Seen on 6.30.0: with the default concurrency limit it blocked until it timed out; with `concurrency_limit=None` it ran. The host's chat event sets it, so a "yes" can arrive while the turn is still waiting.

**F-L47-25 - an unanswered question is cancelled after 300 seconds, never approved.** Unit-tested. It also happened by accident in a real run: the audit log shows `approval_requested` then `cancelled`.

**F-L47-26 - three of my tests passed when they should have failed.** Mutation checks found the connect-lock test, the abandoned-question test and the path-leak tests (Windows doubles backslashes inside an `OSError` message, so a full-path comparison could not see the leak). All three were strengthened and re-checked. A mutation helper of mine also silently did nothing twice; it now stops if the text is not found.

**F-L47-27 - a closed page left its tool call waiting.** Found in Copilot's review. Closing a tab left the question pending for the full 300 seconds. Each page now drops its desk when it unloads, and that cancels what was waiting. Tested at the broker, the desks and both pages; the unload event itself was not triggered by closing a real tab.

**F-L47-28 - smaller review fixes.** A policy path that is a folder, or a file that vanishes mid-read, now refuses or returns no rules instead of crashing, and the error no longer carries the server's absolute path. The server refuses to start if its state folder is inside the folder the tools can reach. The code imports `mcp.types`, not the `mcp_types` package it re-exports.

**F-L47-29 - Gradio answers arrive on worker threads, and the approval wait did not wake.** Found in Copilot's second review and reproduced: `respond` called from a thread left the waiting call asleep until its timeout, because an asyncio future cannot be completed from another thread. The broker now claims a question under a lock and hands the answer to the waiting loop with `call_soon_threadsafe`, so a question is answered once and the call wakes at once. Tested with a real worker thread. The same fix covers the page-close cancel. The audit log's read errors also now name the file only, as the policy store's do, and so do its write errors (F-L47-37).

**F-L47-30 - three more review fixes.** (a) The audit log was readable without approval by default, although it holds every call's arguments, including written file content. It now defaults to `ask`; set `read_audit_log` to `allow` to open it. (b) The audit line saying how a tool ended is written after the tool has run, so a full disk made the client see an error for a change that happened. The record of intent (allowed, asked, approved, refused) is still written first and a failure there stops the call; a failure writing the ending record now returns the result and is logged to the server's own log. The gap in the audit file is real in that case. (c) The host turned a model's non-text prompt argument into text (1 became "1"); it now returns an error to the model, as the GUI does.

**F-L47-31 - the desk registry and the session hash.** Gradio callbacks run on different threads, so two first uses of a session could each make a desk and the answer would go to the wrong one; the registry now has a lock. A request with no session hash used to share one process-wide desk, so it could see and answer another caller's question; it is now refused.

**F-L47-32 - two ordinary messages could run at once in the host chat.** Both read the same history and one overwrote the other. The chat event has no concurrency limit on purpose (a "yes" must arrive while a turn waits), so the guard is per session: a message that arrives while a turn runs, and no question is waiting, is not started and the page shows a warning. A reply to a waiting question is never blocked. Tested at the guard; the warning in a running page was not exercised.

**F-L47-33 - the MCP session was never closed at exit.** `serve` now launches without blocking, waits, and on a stop closes the session and then the page, because the session lives on the page's event loop. Seen in a real run (real server over stdio, stop simulated by raising `KeyboardInterrupt` in the wait): two server processes while running, none after. A real Ctrl+C, a terminal close and a kill were not tried. The host also rejects a non-object `arguments` field for the prompt helper instead of replacing it with an empty one.

**F-L47-34 - parent folders were created after the root check.** Another process could swap a new folder for a link in between. `write_text` now checks the root again once the folders exist and writes through the checked path. Tested by simulating the swap, since links cannot be made here without privileges. The write itself still follows whatever is at the final path when it runs, so a link swapped in during that instant is not caught. Closing that needs no-follow file opens, which Windows does not offer and which I could not test here. The stated threat model is instead that no other process writes inside the tool root while the server runs; the root is a folder the server owns.

**F-L47-35 - a refused call left its approval token alive.** The policy is read on every call. A token presented while the rule was `deny` (or the policy file was unreadable) was not spent, so changing the rule back to `ask` before it expired let the same token be accepted, even after a decline. Any presented token is now spent before the policy is looked at. Three tests failed first.

**F-L47-36 - Clear left a hidden question answerable.** Clearing the host chat emptied the page but not the session's waiting question, so the next bare "yes" could approve a question the user could no longer see. Clear now cancels the session's waiting question first, and only that session's. Tested at the handler and on the built page.

**F-L47-37 - audit log write errors leaked the server path.** If the log or its folder could not be opened, the OS error, with the absolute path in it, went back to the client from the pre-run record. Writing the log now reports only the file name and the kind of failure, as reading it does. Two tests failed first.

## Tests

Names read `test_given_..._when_..._then_...`, written test first, with a docstring on each design pin. The server tests drive the real server with an in-memory client and a real elicitation handler. The client tests run the official client against an in-process server, plus one real stdio subprocess. Mutation checks on scratchpad copies covered the policy, the gate, the audit log, the file store, the container, the session, the broker and the host page; all were caught after F-L47-26.

## Real runs

Seen here, Windows and PowerShell; the model's replies vary.

- **Server over stdio:** `write_file` asked once with the arguments in the question, then wrote; `delete_file` denied; outside paths refused; the audit trail was complete.
- **GUI app** (a real process, two `gradio_client` sessions): session B could not see or answer A's question; A's approve wrote the file, A's decline wrote nothing (file absent on disk).
- **Host app with `gpt-4o-mini`** (driven through `gradio_client`): A's chat showed the exact question; a "yes" from session B was an ordinary message and A's turn kept waiting; A's own "yes" wrote the file. A declined write wrote nothing. "Delete the file" came back as `Permission denied for tool: delete_file`.

Python from my shell could not verify `api.openai.com` (the same TLS inspection), so the host ran through a wrapper outside the repo that uses the system certificate store. That is my environment, not the app.

## What this doesn't cover

- `read_file` on a UTF-16 file, a second real browser tab, closing a real tab, and running from Git Bash.
- Stopping the page apps: only a simulated stop was run (F-L47-33), not a real Ctrl+C, a closed terminal or a kill. The host's "still working" warning was not seen in a running page.
- A link swapped in at the final path during a write, by another process with write access inside the tool root (F-L47-34).
- Client trust: the server cannot tell a person's answer from a client's. Anything that can talk to the server and holds a live token can accept.
- Stopping a standalone server: Ctrl+C prints a traceback, and a stop signal sent while the client's pipe is still open did not end the process on Windows in my probe. The client stops the server by closing the pipe, which exits cleanly. I tried handling stop signals and dropped it; it is not resolved.
- Authentication and HTTPS: the server is a local stdio subprocess for one user.
- The 2026-07-28 protocol text; the notes rely on the installed libraries' source and my runs. Both major versions are new, and the pins are exact.
- Other platforms or Python versions; one machine. The symlink test is skipped here.

---

**Completed:** 24 October 2026
