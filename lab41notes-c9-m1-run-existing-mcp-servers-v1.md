# Course 9 - Lab 41: Run Existing MCP Servers (V1)

> Code: [`course9-module1-lab1/`](course9-module1-lab1/)

A Python client connects to Context7, a documentation server for LLMs, over the two MCP transports the lab covers: STDIO (a subprocess started with `npx`) and Streamable HTTP (a hosted endpoint). Using the standalone `fastmcp` package, the client lists the server's tools, then calls `resolve-library-id` to find a library and `query-docs` to fetch its documentation. Two exercises repeat the pair for scikit-learn. There is no model in the loop: the script calls the tools directly, so what is exercised is the protocol, the tool contracts and the two transports.

This is a script port of IBM Skills Network's "Run Existing MCP Servers" notebook (the notebook is not included here), on Python 3.14 and Windows. The notebook's stdin/stdout/stderr and HTTP background sections are ported too, as `standard_streams.py` and the first half of `http_transport.py`.

The most useful measurements concern the contract a server publishes. Context7's tool descriptions state ordering and call-count rules that nothing enforces, its schema is enforced for required arguments, and a failure inside the server's own backend is delivered as ordinary result text.

## Run it

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python standard_streams.py     # optional background, standard library only
python stdio_transport.py      # needs Node.js (npx) and network access
python http_transport.py       # needs network access
```

`requirements.txt` pins `fastmcp` and `requests` (the notebook installs both unpinned). Versions in use, resolved on 4 October 2026 and then pinned:

| | |
|---|---|
| Python | 3.14.0 |
| `fastmcp` / `fastmcp-slim` | 4.0.10 / 4.0.10 |
| `mcp` / `mcp-types` | 2.3.0 / 2.3.0 |
| `requests` | 2.34.2 |
| Node.js / npx | v26.5.1 / 12.0.2 |
| Context7 (`npx -y`, unpinned) | 4.1.1 |

## File layout

```
course9-module1-lab1/
├── helpers/
│   ├── __init__.py
│   └── common.py             # Context7 constants and the steps both transports run
├── standard_streams.py       # stdout / stdin / stderr, plus two happy paths
├── stdio_transport.py        # Context7 over stdio; the steps plus the scikit-learn exercises
├── http_transport.py         # HTTP preface (requests) and Context7 over Streamable HTTP
└── requirements.txt
```

## Differences from the notebook

- The notebook's cells become `async def` functions run with `asyncio.run`; bare expressions such as `len(tools)` are printed.
- Each step (`show_tools`, `show_library_id`, `show_docs`) opens its own `async with` connection. The notebook does the same for stdio, but its HTTP cell makes all three calls inside one connection.
- Tool schemas are read as `.input_schema`, not `.inputSchema` (F-L41-3), and `content[0]` is narrowed with `isinstance` before `.text` is read (F-L41-4).
- Docs output is truncated to 1,000 characters on both transports; the notebook uses 1,000 for stdio and 500 for HTTP.
- `requests.get` has `timeout=10`; the notebook has none.
- `tokens: 5000` is no longer passed to `query-docs`, since the tool does not declare it (F-L41-6).
- `fastmcp`, `requests` and the `npx` package (`@upstash/context7-mcp@4.1.1`) are pinned; the notebook installs all three unpinned.
- `Cookie`, `Authorization` and `Proxy-Authorization` values are redacted when the HTTP preface prints request headers.
- The library ID passed to `query-docs` is the notebook's fixed value, as in the notebook; it is not parsed from the `resolve-library-id` output.

---

## Key concepts

### One client, two transports

`Client(transport)` is the same object whichever transport builds it. `StdioTransport(command="npx", args=["-y", "@upstash/context7-mcp"])` starts Context7 as a subprocess; `StreamableHttpTransport(url="https://mcp.context7.com/mcp")` points at the hosted copy. Everything after construction is identical, so each script builds its client and calls the same `run_lab_steps`. The notebook's own remark is that once you have the connection the process is the same.

### stdio is a subprocess with its streams wired up

`standard_streams.py` shows the three streams the stdio transport depends on. STDOUT and STDERR are separate, so errors can be redirected independently of output (`2> stderr.txt`); `input()` writes its prompt to STDOUT. Its second happy path is the shape of the transport: a parent starts a child process, writes one JSON line to its STDIN and reads one JSON line from its STDOUT. The real transport keeps the pipes open and exchanges many messages, and a server must keep STDOUT clean for protocol messages.

### Discover, then call

`list_tools()` returns each tool's name, description and input schema. A model host would read exactly those three things to decide what to call. Here nothing reads them except the script, which makes the lab a view of what a model would be given.

### HTTP request headers

The HTTP preface sends a plain `GET` to ibm.com and prints what the client sent: status 200, `User-Agent: python-requests/2.34.2`, `Accept-Encoding: gzip, deflate, zstd`, and no body. The printed headers also included a `Cookie` header the script never set; `r.request` is the request behind the final response, so cookies set earlier in a redirect chain would appear on it (not checked with `r.history`).

---

## Findings

**F-L41-1 — the notebook's prose and code disagree in places.** The text says the chosen ID (`/llmstxt/gofastmcp_llms-full_txt`) has a "Trust Score of 9.6", while the example listing shows a Benchmark Score of 79 and no trust score; it says the printed documentation belongs to `/punkpeye/fastmcp` while the code fetches the `llms-full_txt` ID; it says the tool requires a `libraryName` argument, while the schema requires `query` as well; and it says that for stdio "everything runs on your system", while the tool's own schema says the query is sent to the Context7 API (F-L41-5). The notebook installs both `fastmcp` and the `npx` package unpinned and needs network access throughout. The standard-streams and HTTP-preface sections contain no MCP code.

**F-L41-2 — the unpinned stack ran the notebook's code on Windows with no changes beyond one deprecation, and both transports negotiated the newest protocol revision.** `pip install fastmcp` resolved to `fastmcp` 4.0.10 on `mcp` 2.3.0. `Client`, both transports, `list_tools()` and `call_tool()` work as the notebook writes them. `StdioTransport(command="npx")` started Context7 on Windows with no `.cmd` workaround (PowerShell resolves `npx` to `npx.ps1`), and the script exited cleanly. `Client` defaults to `mode="auto"`; over stdio and over HTTP it reported `protocol_version=2026-07-28` and the same server (`Context7`, version `4.1.1`). Tool lists and schemas were identical across the two transports.

**F-L41-3 — `Tool.inputSchema` is a deprecated alias.** Reading it raises `FastMCPDeprecationWarning` ("MCP SDK v2 renamed this field to `input_schema`"), and Pylance reports `inputSchema` as an unknown attribute because only `input_schema` is a declared field. It still works at runtime through a compatibility shim. The notebook reads `.inputSchema` in three places.

**F-L41-4 — `response.content[0].text` is not type-safe.** `content` holds a union of five block types (`TextContent`, `ImageContent`, `AudioContent`, `ResourceLink`, `EmbeddedResource`) and only `TextContent` has `.text`; Pylance flags the other four. It works because Context7 returns text and nothing checks that. `TextContent` is imported from `mcp.types`; `fastmcp.content` does not exist and `fastmcp.types` does not export it.

**F-L41-5 — the tool descriptions carry rules the schema does not enforce.** `resolve-library-id` says it must be called before the documentation tool unless the caller already has an ID; each tool says not to call it more than three times per question; and `resolve-library-id` tells the caller to choose by name match, source reputation, snippet count and Benchmark Score. Each tool names the other by a display title ("Resolve Context7 Library ID", "Query Documentation"), not by its tool name. The schema enforces only that the arguments are present and typed. Both `query` arguments state that the query is sent to the Context7 API, which means the stdio server is local but its queries are not. Same shape as the Course 8 declared-versus-enforced findings, on a third-party server.

**F-L41-6 — required arguments are enforced server-side; extra and empty ones are not.** Calling `resolve-library-id` without `query` was rejected: `ToolError: Input validation error: Invalid arguments for tool resolve-library-id: query: Invalid input: expected string, received undefined`. An empty string for `query` was accepted. `query-docs` declares only `libraryId` and `query`, yet the notebook's `"tokens": 5000` was accepted without error; whether it has any effect was not tested.

**F-L41-7 — a failure inside the server's backend is returned as ordinary result text.** On one run all four calls (both tools, both exercises) returned `Error searching libraries: TypeError: fetch failed` or `Error fetching library context. Please try again later. TypeError: fetch failed` as content, and `call_tool` raised nothing, while the validation failure in F-L41-6 raised `ToolError`. A caller that prints `content[0].text` presents the failure as output. The failure was transient and the next run succeeded. `call_tool` raises `ToolError` on error results by default and did not raise here, so `is_error` was not set: the failure was ordinary content.

**F-L41-8 — the library results are neither stable between identical calls nor ordered by the score the notebook tells you to use.** With identical arguments the returned set, order and scores changed from run to run: `/websites/gofastmcp` scored 84.88 in some runs and 84.81 in others; `/stevens507/fastmcp` and `/llmstxt/gofastmcp_llms-full_txt` (the notebook's pick) appeared in some runs and not others, always last by score (52.6) when present; `/websites/gofastmcp_v2_getting-started` appeared once; and two back-to-back calls had unequal `content`. Lists were not in Benchmark Score order (a 73 listed above a 79.17). `query=""` returned a different set (`/prefecthq/fastmcp-ts` and `/llmstxt/gofastmcp_llms_txt` in, two entries out), which cannot be attributed to the empty query given that identical calls also vary. For scikit-learn, the notebook's hardcoded ID `/scikit-learn/scikit-learn` was listed first with a score of 74.8, below `/websites/scikit-learn_stable` (86.4) and `/websites/scikit-learn_dev` (78.96), so the notebook's own rule of taking the highest score would not select it.

**F-L41-9 — `query-docs` returned contributor documentation, not usage examples, for both of the lab's queries.** For the notebook's `fastmcp` ID the snippets covered generating API reference docs (`just api-ref-all`), the contributing guide and a documentation index; for scikit-learn they covered installing the docs build dependencies and `make html`. Each snippet has a title, a `Source:` URL, text and code, separated by rules. The tool's schema asks for a query "scoped to a single concept" and lists vague queries as bad; both lab queries ask for "the documentation" in general, which is what came back.

---

## Prior art carried across

- **Declared versus enforced (Course 8: CrewAI task-tool override, BeeAI requirement gate, AG2 graph).** F-L41-5: ordering and call-count rules live in a description; the schema enforces only the arguments.
- **L39/F-L39-13: guardrail placement.** A constraint stated in prose is not a constraint. The server enforces `required` in code, and enforces nothing about order.
- **L21 and L37: the description is runtime content.** The text a tool publishes is what a caller acts on, so its wording and its accuracy are part of the contract.
- **L39/F-L39-18: fluent output under a wrong structure.** F-L41-7 and F-L41-9 are the same shape: the server answers fluently and the answer is an error or off-target.

---

## What this doesn't cover

- **A model in the loop.** Whether an LLM host would follow the ordering and three-call rules, or choose a library by the published score, was not tested.
- **The `tokens` argument.** Accepted without error; whether it changes the output was not tested.
- **`is_error` on the backend-failure result** (F-L41-7).
- **Whether `StdioTransport` reuses one Context7 subprocess** across the steps' separate connections. Not measured.
- **`mode="legacy"` against Context7.** Only the default `mode="auto"` was used.
- **Pinning `mcp`.** `fastmcp`, `requests` and the `npx` package are pinned; `mcp` is not and floats within `>=2,<3`. `r.history` was not read for the cookie question.
- **Security.** Context7's advisories and data handling were not reviewed.
- **One machine, one Node version.** The Windows `npx` result is from a single environment.

---

**Completed:** 4 October 2026