# Course 7 — Lab 31: DocChat — Multi-Agent RAG

> Code: not committed here. This is IBM's Apache-2.0 lab, ported and run locally against my own stack. Upstream: [ibm-developer-skills-network/zzpwx-docchat](https://github.com/ibm-developer-skills-network/zzpwx-docchat) (branch `2-final`) — follow that repo's README to run it. This note is the port record and the findings, not a re-publication of the lab code.

Multi-agent RAG over uploaded documents, and the first Module 3 lab that isn't a variation on a single-agent reasoning loop. A LangGraph `StateGraph` routes a query through three agents with distinct prompts, temperatures, and roles — relevance check → research (draft answer) → verification (fact-check against retrieved context) — with a re-research loop on failed verification. Retrieval is hybrid (BM25 + vector via `EnsembleRetriever`) over Docling-parsed documents in ChromaDB. No onion port at V1 — this is the canonical build run locally; the onion refactor is the planned V2, where the three-agent decomposition, the hybrid retriever, and the verification loop are the patterns worth banking (or explicitly rejecting) for Hooperman.

Built as part of the IBM RAG and Agentic AI Professional Certificate — Course 7, Module 3, Section 2 (ungraded app item). The IBM lab runs in a Cloud IDE against watsonX with implicit `skills-network` auth, pins `langchain==0.3.16` / `langgraph==0.2.68` (pre-1.0), and embeds with `slate-125m-english-rtrvr-v2`. This implementation runs on my own watsonX API key + project, on `langchain==1.3.11` / `langgraph` 1.x (forcing the retriever import migrations below), and swaps the deprecated slate embedding model for `granite-embedding-278m-multilingual`. The one substitution is credentials-only — the model-provider code stays canonical watsonX `ModelInference`, closest possible fidelity to the lab.

---

## What It Does

Upload documents (`.pdf`, `.docx`, `.txt`, `.md`), ask a question, get an answer plus a verification report.

**Ingestion** — `DocumentProcessor` converts each file to Markdown with Docling, splits on headers (`MarkdownHeaderTextSplitter`), SHA-256-hashes for a pickle cache, and dedupes chunks across files.

**Retrieval** — `RetrieverBuilder` builds a hybrid `EnsembleRetriever`: a Chroma vector store (granite embeddings) weighted 0.6 and a BM25 lexical retriever weighted 0.4.

**Graph** — `AgentWorkflow` compiles a `StateGraph`: entry `check_relevance` → conditional (`relevant` → `research`, `irrelevant` → END) → `research` → `verify` → conditional (`re_research` → `research`, `end` → END).

- `RelevanceChecker` (granite, temp 0) retrieves, then classifies `CAN_ANSWER` / `PARTIAL` / `NO_MATCH`.
- `ResearchAgent` (granite, temp 0.3) drafts an answer from the retrieved context.
- `VerificationAgent` (granite, temp 0) fact-checks the draft against that same context and emits a structured `Supported / Unsupported Claims / Contradictions / Relevant / Additional Details` report.

**UI** — Gradio Blocks: upload, question box, example loader, answer + verification panes.

---

## Stack

| Component | Implementation |
|---|---|
| LLM (all three agents) | `ibm/granite-4-h-small` via watsonX `ModelInference` — relevance temp 0 / 10 tok, research temp 0.3 / 300 tok, verify temp 0 / 200 tok |
| Auth | Own watsonX API key + project via `.env`; the lab assumes Cloud IDE implicit `skills-network` auth |
| Embeddings | `ibm/granite-embedding-278m-multilingual` (slate-125m deprecated, withdrawn Aug 2026), `TRUNCATE_INPUT_TOKENS=512` |
| Parsing | Docling `DocumentConverter` → Markdown + `MarkdownHeaderTextSplitter` |
| Vector store | ChromaDB via `langchain_chroma.Chroma` (moved out of `langchain_community` in 1.x) |
| Hybrid retrieval | `langchain_classic.retrievers.EnsembleRetriever` (moved from `langchain.retrievers` in 1.x) over `langchain_community.retrievers.BM25Retriever` (k raised 4→20) + vector (k 10→20) |
| Graph | `langgraph` 1.x `StateGraph`; `Document` from `langchain_core.documents` |
| Architecture | Canonical, no onion port (V2) |
| UI | Gradio 6.x (theme/css/js moved to `launch()`) |

---

## Setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt   # lean set; watsonX packages kept, CUDA pins + 0.3.x langchain pins dropped
```

`.env` at the lab root:

```
WATSONX_API_KEY=...
WATSONX_PROJECT_ID=...
```

```powershell
python app.py   # http://localhost:5000
```

Example documents (Google 2024 Environmental Report, DeepSeek-R1 Technical Report) aren't shipped in the repo — download the two public PDFs into `examples/` with the names the `EXAMPLES` dict expects, or upload any document manually.

---

## Key Concepts

### The port's one substitution is credentials, not the model

The lab fails locally only because it uses `Credentials(url=...)` with no key and `project_id="skills-network"` — the Cloud IDE's implicit auth. Supplying my own `api_key` + `project_id` keeps every agent byte-identical to canonical: same `ModelInference.chat()`, same `response['choices'][0]['message']['content']` dict shape, same models. Credentials-only substitution is the maximally faithful V1 — the same discipline as Ollama-for-watsonX in Lab 27, here own-creds-for-skills-network.

### The 1.x retriever migration is the framework-volatility surface

The genuinely-new component of the lab — the hybrid retriever — is exactly what drifted. In LangChain 1.x, `EnsembleRetriever` was demoted out of `langchain.retrievers` into the `langchain_classic` compatibility package, `Chroma` moved to `langchain_chroma`, and `Document` to `langchain_core.documents`. `BM25Retriever` stayed in `langchain_community`. Verified against the installed packages, not guessed.

### Three agents, one degenerate edge

The three-agent decomposition maps onto real bounded contexts (classification, generation, verification), each with its own prompt and temperature — defensible, unlike the intro video's contrived examples. But the relevance edge collapses the tri-state classifier to a binary: `CAN_ANSWER` and `PARTIAL` both route to `research`, only `NO_MATCH` ends. Two-thirds of the classifier's signal is discarded the moment it hits the graph.

---

## Findings

**Docling vs PyPDFLoader, reproduced with a sharper failure than the lab narrates.** On the comparison script, PyPDFLoader returned *empty* on the image-PDF (loaded fine, extracted nothing — no OCR) and *crashed* on the "scanned PDF" with `invalid pdf header: b'\x89PNG'` — the fixture is a PNG with a `.pdf` extension. Docling OCR'd both, table and all. The lab just says "LangChain fails"; the honest version is that PyPDFLoader is a digital-text reader, not a document parser. Also had to `pip install pypdf` to even see the real limitation — the first failure was a missing dep, not the OCR gap.

**`TRUNCATE_INPUT_TOKENS=3` silently guts vector retrieval.** IBM's `builder.py` truncates every chunk *and* every query to 3 tokens before embedding. The app still runs and returns a confident, well-formatted wrong answer (`NO_MATCH` on a clearly-answerable question). Nothing surfaces the bug except the answer being wrong. Raised to 512 (the embedding model's context limit). This is the "graph correctness ≠ system correctness" thread as a one-line default.

**The relevance gate produces false negatives, and it's a recall problem not a classification one.** After fixing truncation, the specific PUE query still returned `NO_MATCH` — but a broad "any information on google" query ran the full pipeline cleanly (`Supported: YES`). The checker printed a clean `NO_MATCH` label (granite wasn't malformed), so retrieval was surfacing the wrong chunks. Cause: `BM25Retriever` defaults to **k=4**, and `VECTOR_SEARCH_K` was 10 — ~14 of 327 chunks retrieved, and the specific Singapore-PUE table row wasn't in the window. Raised both to 20; the specific query then answered. Broad-query-passes / specific-query-fails is the exhibit: a hard gate that rejects answerable questions when recall is tuned too low.

**Verification manufactures false confidence — the capstone.** On the lab's own demo question, the verification agent returned `Supported: YES / Contradictions: None / Relevant: YES` for an answer that (a) misread the source table — reported Singapore 2nd-facility 2022 as 1.19 (the 2023 figure; 2022 is 1.21) and fabricated a 2019 value the facility never reported, (b) contradicted itself between summary and body on the 2019 figure, and (c) silently dropped half the query (the Asia-Pacific CFE 2023 part). The verifier checks the draft against the *same retrieved context the draft came from*, at temp 0 with regex parsing — a weak generation-fidelity check that here failed even at fidelity. DocChat's headline selling point ("fact-checked, hallucination-free") rubber-stamped the exact ChatGPT-hallucinates-the-numbers failure the lab opens by mocking. Direct evidence for the Hooperman position: adding a verification agent adds latency and cost *and can actively produce false assurance* without demonstrated recall improvement.

**The "granite is prompt-brittle" hypothesis was tested and did not hold.** The kickoff expected granite to be more prompt-brittle than local alternatives and to fail the strict label/format contracts. It didn't — granite returned clean, valid labels throughout; every failure traced to retrieval or to the verifier's design, not to classification format. Worth recording as a hypothesis rejected.

**The re-research loop is non-convergent, not just uncapped.** IBM's loop sends a failed verification back to `research` with identical inputs — same question, same documents; the verification report is never fed back into the research prompt. It cannot converge regardless of iteration count. Added a hard cap (3 passes) as a deliberate cost guard, flagged as a deviation. The real fix — feed the verifier's unsupported-claims into the re-research prompt — is a V2 target.

**Production line:** every failure in this lab is a retrieval or evaluation failure wearing a working-pipeline costume. The graph executes flawlessly; the multi-agent structure looks like rigour. But the relevance gate rejects answerable questions on low recall, and the verification agent green-lights wrong answers because it has no ground truth to check against — only the same context that produced the draft. External evaluation against a golden set (Hooperman's shape) remains the only mechanism that would answer "was the answer real." Multi-agent orchestration is not a substitute for eval.

---

## What This Doesn't Cover

- **Onion port.** V1 is canonical. The V2 refactor puts Docling ingestion, the hybrid retriever, and the three agents behind interfaces per the agentic-lab conventions.
- **Verification feedback into re-research.** The cap stops the bleeding; feeding the verifier's unsupported-claims back into the research prompt is the actual convergence fix (V2).
- **Retrieval-quality eval on the hybrid weights.** The `EnsembleRetriever` weights ([0.4, 0.6]) and the k values were tuned by hand to pass one query. A recall/precision harness over a golden set is the V3 candidate and the reference implementation for the SRMC-1012 BM25-hybrid ticket.
- **Verification against ground truth.** Checking the draft against the retrieved context can only catch fabrication-beyond-context, never context-is-wrong. Truth-checking needs an external authority.
- **Embedding model longevity.** `granite-embedding-278m-multilingual` is current; `slate-125m` (the lab's choice) is withdrawn Aug 2026. Any long-lived deployment re-checks the supported-models list.
- **Double retrieval.** Both `full_pipeline` and the relevance checker call `retriever.invoke(question)` — wasteful, a V2 cleanup.

---

**Completed:** 2 July 2026