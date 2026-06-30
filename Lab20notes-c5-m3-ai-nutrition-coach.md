# Course 5 — Lab 20: AI Nutrition Coach (Vision QA)

> Code: [`course5-module3-lab2/`](course5-module3-lab2/)

Flask web app for nutritional analysis of food images. Image in, structured nutrition breakdown out. Pure vision QA — no retrieval, no embeddings, no vector store. A vision LLM receives a base64-encoded image and a structured system prompt, and returns a 6-section nutritional assessment. Two backends: Meta's Llama 4 Maverick via IBM watsonx, and LLaVA locally via Ollama. Built as part of the IBM RAG and Agentic AI Professional Certificate — Course 5, Module 3.

**Not MM-RAG.** Style Finder (Lab 19) was MM-RAG: encode → retrieve → augment → generate. This lab is vision QA: encode → generate. Same "multimodal" umbrella, different architecture. Worth being clear on the distinction.

---

## What It Does

- **Encode** — uploads image through Flask form, encodes to base64 string
- **Generate** — vision LLM receives base64 image plus a structured nutritionist system prompt and produces a 6-section response (Identification, Portion Size & Calories, Total Calories, Nutrient Breakdown, Health Evaluation, Disclaimer)
- **Post-process** — regex converts Markdown `**bold**` and `*bullets*` to HTML `<strong>` and `<ul><li>` for browser rendering
- **Render** — Jinja template displays formatted response inline

No retrieval step. No vector store. The system prompt is the only "augmentation" in this pipeline.

---

## Stack

| Component | IBM Lab | Local Swap |
|---|---|---|
| Vision LLM | `meta-llama/llama-4-maverick-17b-128e-instruct-fp8` via watsonx | `llava:latest` via Ollama |
| Web framework | Flask | Same |
| Image handling | base64 encoding via `base64.b64encode()` | Same |
| Response formatting | regex-based Markdown → HTML | Same |
| Credentials | python-dotenv + .env with `WATSONX_API_KEY` and `WATSONX_PROJECT_ID` | None — no credentials needed |

Backend selected via `config.USE_LOCAL` (True = Ollama, False = IBM watsonx).

---

## File Layout

```
course5-module3-lab2/
├── app.py                            — Flask routes, composition root
├── config.py                         — model IDs, USE_LOCAL flag, generation params
├── requirements.txt
├── .env                              — WATSONX_API_KEY, WATSONX_PROJECT_ID
├── .gitignore
├── interfaces/
│   ├── __init__.py
│   └── llm_service_interface.py      — Protocol contract for LLM services
├── infra/
│   ├── __init__.py
│   ├── llm_service.py                — IBM watsonx concrete implementation
│   └── llm_service_local.py          — Ollama local concrete implementation
├── templates/
│   └── index.html                    — Jinja form + result display
└── static/
    └── style.css
```

Same architectural pattern as Lab 19 — Protocol interface in `interfaces/`, concrete implementations in `infra/`, composition root in `app.py`.

---

## Vision QA Pipeline

```
User uploads image
        │
        ▼
input_image_setup()
  → base64 string
        │
        ▼
generate_model_response()
  → builds messages with system prompt + user query + base64 image
  → sends to vision LLM via configured backend
  → returns raw Markdown response
        │
        ▼
format_response()
  → regex converts **bold** → <strong>
  → regex converts * bullets → <ul><li>
  → adds <br> between paragraphs
        │
        ▼
Jinja renders HTML inline in form
```

Note the absence of any retrieval step. Compared to Lab 19's pipeline, the entire offline indexing + online similarity search stage is removed.

---

## Core Pattern

### IBM watsonx

```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": assistant_prompt + "\n\n" + user_query},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + encoded_image}}
        ]
    }
]
response = self.model.chat(messages=messages)
return response['choices'][0]['message']['content']
```

### Ollama local

```python
messages = [
    {
        "role": "user",
        "content": assistant_prompt + "\n\n" + user_query,
        "images": [encoded_image]   # base64 string, top-level field
    }
]
response = ollama.chat(model=self.model_id, messages=messages, options=self.params)
return response['message']['content']
```

Same wire format pattern as Lab 19 — the message structure differs between the two backends in the same way (nested `content` array with typed elements for watsonx, top-level `images` field for Ollama).

---

## Key Decisions

**Lab credentials replaced with local `.env` pattern.** IBM's lab uses hardcoded `project_id = "skills-network"` and a commented-out API key, both injected by IBM's Cloud IDE. Running locally requires real credentials from a personal watsonx account, loaded via `python-dotenv`. Same fix as Lab 19.

**`ibm-watsonx-ai==1.1.20` not installable on Python 3.13.** Recurring issue — drop the pin to install latest (1.5.12). The lab's pin works on the Cloud IDE's Python 3.11 but not on Windows local Python 3.13.

**`image==1.5.33` package dropped.** The lab installs it but never uses it — `from PIL import Image` comes from Pillow, not the `image` package. Pillow is a transitive dependency of other packages; explicit install isn't needed for this lab.

**Flask `secret_key` added.** The lab code uses `flash()` for error messages but never sets `app.secret_key`. Without it, the unhappy path (no file uploaded) crashes with a Flask error. Added `app.secret_key = os.urandom(24)` immediately after `app = Flask(__name__)`.

**Protocol-based interface introduced.** Lab as written uses a single `generate_model_response()` function with the IBM SDK calls embedded. Refactored to mirror Lab 19's architecture: `LLMServiceInterface` Protocol in `interfaces/`, IBM and Ollama concretes in `infra/`, `create_llm_service()` composition root selecting via `config.USE_LOCAL`. The refactor took ~20 minutes and made the model-comparison exercises (Maverick → Llama 3.2 → LLaVA) one-line changes instead of full code edits.

**Generation params exposed in config.** Lab code uses bare `TextChatParameters()`. Replaced with config-driven `WATSONX_TEMPERATURE`, `WATSONX_TOP_P`, `WATSONX_MAX_TOKENS` so the practice exercises can be done by editing one file. Pattern matches Ollama's existing `OLLAMA_PARAMS` dict.

**No streaming.** Both Ollama and watsonx support streaming responses; this lab doesn't use it. The Flask UI blocks for the full Maverick response (~7s) or LLaVA response (~2m 15s). Production UX would need streaming or a job queue.

---

## Latency Comparison

Same image (fish burger), same prompt, same backends as Lab 19:

| Backend | Model | Generation time |
|---|---|---|
| IBM watsonx | Llama 4 Maverick 17B (400B total / 17B active MoE) | ~7 seconds |
| IBM watsonx | Llama 3.2 11B Vision Instruct | ~5–10 seconds |
| Ollama local (RTX 4070 8GB) | LLaVA 7B | ~2 minutes 15 seconds |

Consistent with Lab 19's finding: ~20x latency difference for managed API vs local 4070 inference. Managed wins for fan-facing production workloads, local wins for dev iteration with no per-call cost.

---

## Model Comparison — Same Image, Three Models

Three-way comparison run on the same fish burger image with the same system prompt:

| Model | Visual identification | Structure adherence | Macro estimates |
|---|---|---|---|
| Llama 4 Maverick (400B MoE) | ✅ Fish sandwich with tartar sauce, all components correct | ✅ Full 6-section structure | Reasonable, consistent |
| Llama 3.2 11B Vision | ❌ One sentence: "approximately 550 calories" | ❌ Ignored structure entirely | None — refused breakdown |
| LLaVA 7B (local) | ❌ Called it a chicken burger with cheese | ✅ Followed 6-section structure | Vague, hallucinated cheese |

**Key observation: model size doesn't predict capability across dimensions.** Llama 3.2 11B Vision sits between LLaVA 7B and Maverick in parameter count, yet underperformed both — Maverick on accuracy *and* LLaVA on instruction following. Capability is multi-dimensional: visual reasoning, instruction following, latency, cost, deployability. They don't correlate cleanly.

**Practical implication for production.** "Use the biggest model" is not a strategy; nor is "the cheapest that fits." The discipline is characterising the capability axes that matter for the workload and picking accordingly. The interface abstraction makes testing alternatives cheap.

---

## Temperature Determinism Experiment

Practice exercise: ran the same image and prompt twice with `temperature=0.0` on Maverick. Expected identical output (greedy decoding should be deterministic). Actually got different outputs:

| Field | Run 1 | Run 2 |
|---|---|---|
| Fish patty | 350 cal | 300 cal |
| Total calories | 590 | 540 |
| Sauce naming | "Mayonnaise or Sauce" | "Mayonnaise or Tartar Sauce" |
| Tomato vitamins | Vitamin C, Vitamin A | Vitamin C, Lycopene |
| Fish minerals | Selenium, Phosphorus | Selenium, Omega-3 |
| Health evaluation | Mentions whole-grain substitute | Mentions deep-frying |

**`temperature=0.0` is not actually deterministic on hosted LLM APIs.** This is structural, not a bug:
- **Kernel-level FP non-determinism** on parallel GPU hardware
- **Request batching** on shared infrastructure — your request batches with other tenants', and batch composition affects numerical precision
- **Greedy decoding tie-breaking** can flip when top-2 token probabilities are very close

**Production implication.** Eval pipelines (e.g. Hooperman's 33 golden questions, scaling to 200+) cannot rely on exact string matching even at `temperature=0.0`. The required techniques:
- Semantic similarity scoring (embedding distance)
- LLM-as-judge evaluation (another model grades the answer)
- Multiple runs per golden question (N=3 or N=5, score the distribution)
- Tolerance bands on numeric answers (e.g. within 10%)

---

## System Prompt as Production Guardrail

The lab's assistant prompt is genuinely well-constructed and worth retaining as a pattern reference:

```
1. **Identification** — list items, one per line
2. **Portion Size & Calorie Estimation** — bullets with structured format
3. **Total Calories** — single number
4. **Nutrient Breakdown** — bullets for Protein, Carbs, Fats, Vitamins, Minerals
5. **Health Evaluation** — one paragraph
6. **Disclaimer** — exact required text, verbatim
```

The patterns demonstrated:

- **Role assignment** ("expert nutritionist") — one line, sets persona
- **Numbered output contract** — structured output via prompting, the poor-man's JSON mode
- **Inline format examples** (`*Salmon*: 6 ounces, 210 calories`) — few-shot formatting inside a zero-shot prompt
- **Verbatim required text** for the disclaimer — production guardrail; LLM can't omit it because the prompt specifies exact text
- **Instruction reinforcement at the end** ("Format your response exactly like the template above") — LLMs follow instructions placed late in long prompts more reliably

Worth saving as a reusable structure for any vision QA task requiring consistent output format.

---

## IBM Swap Reference

| IBM | Local |
|---|---|
| `from ibm_watsonx_ai import Credentials, APIClient` | Delete |
| `from ibm_watsonx_ai.foundation_models import ModelInference` | `import ollama` |
| `from ibm_watsonx_ai.foundation_models.schema import TextChatParameters` | `params = {"temperature": 0.2, "top_p": 0.6, "num_predict": 2000}` |
| `ModelInference(model_id=..., credentials=..., project_id=..., params=...)` | Store `model_id` and `params` directly on `self` |
| `self.model.chat(messages=messages)` | `ollama.chat(model=self.model_id, messages=messages, options=self.params)` |
| `response['choices'][0]['message']['content']` | `response['message']['content']` |
| Nested content array with `type: image_url` and `data:image/jpeg;base64,` URI | `"images": [encoded_image]` as top-level field |
| `model_id=config.WATSONX_MODEL_ID` in app | `model_id=config.OLLAMA_MODEL_ID` in app |

Identical pattern to Lab 19's swap reference — the wire format differences are constant across watsonx vision tasks.

---

## Patterns Worth Retaining

**1. Vision QA is not MM-RAG.** Vision QA = image + text → vision LLM → text. MM-RAG = image → encoder → retrieval → augmentation → vision LLM → text. The course conflates them under "multimodal," but they have different infrastructure, different cost profiles, and different failure modes. Vision QA is simpler but doesn't scale to grounded factual answers from a knowledge base.

**2. The system prompt does the structural work.** With no retrieval to augment the answer, the only way to get consistent output is heavy prompt engineering. The 6-section contract, inline examples, and verbatim disclaimer are all pulling that weight.

**3. Capability is multi-dimensional.** Parameter count is a single axis. Visual accuracy, instruction following, latency, cost, and deployability are independent axes. A larger model can be worse on a given axis than a smaller one.

**4. `temperature=0.0` does not guarantee determinism on hosted inference.** Production eval pipelines must assume stochastic output and use semantic comparison, not exact matching.

**5. The Protocol-based interface is the right place for the model abstraction.** Same shape as Lab 19. Swapping models becomes a one-line config change. Swapping providers (watsonx → Ollama, or hypothetically watsonx → Bedrock) is a new file in `infra/`. Without the abstraction, every model comparison is a code edit, which means you don't do them.

**6. Regex-based output formatting is fragile.** The lab's `format_response()` breaks on the visible rendering — section headers like `**Identification**:` get converted to `<p><strong>Identification</strong></p>:`, which puts the colon orphaned on its own line. Cosmetic but reveals the brittleness of regex post-processing. Production would either use a proper Markdown parser or require structured (JSON) output from the model.

---

## Rovers Connection

**Vision QA is what Hooperman would need for image-based fan questions.** Fan uploads a photo of a player's boots / kit / stadium section and asks a question. No retrieval needed if the model can answer from visual reasoning alone. The Maverick API call pattern from this lab maps directly — same `image_url` message format, same Flask handler pattern.

**Where vision QA isn't enough: grounded factual answers.** "What was the score of this match?" can't be answered from a photo alone — needs retrieval against match data. That's MM-RAG territory (Lab 19), not vision QA.

**The eval determinism finding directly applies.** Hooperman's golden question pipeline (currently 33 questions, scaling to 200+) cannot rely on exact string matching. The temperature experiment in this lab is the concrete evidence for why semantic similarity / LLM-as-judge / multi-run averaging is required.

**The system prompt pattern transfers.** Hooperman's responses could benefit from the same "numbered sections + inline examples + required disclaimer" structure. Currently the bot's tone is freer; a more structured prompt would tighten consistency and make eval scoring easier.

---

## What This Doesn't Cover

- **Multimodal RAG** — no retrieval step. For grounded factual answers from an image, you'd combine this lab's vision QA with Lab 19's retrieval pipeline.
- **Production input validation** — no file size limits, no MIME type checks, no malicious upload protection. The lab accepts anything from `request.files.get("file")` and base64-encodes it blindly.
- **Streaming responses** — both Ollama and watsonx support streaming; lab blocks for the full response. Production UX needs streaming or async job handling.
- **Structured output (JSON mode)** — the lab uses prompt-engineered Markdown output and regex post-processing. Production would request JSON directly from the model.
- **Observability** — no logging, no tracing, no token cost tracking, no latency measurement at the application layer. Production needs all of these.
- **Eval pipeline** — no golden questions, no automated regression testing, no model A/B comparison framework. The three-way comparison in this lab was done manually.
- **Caching** — same image uploaded twice triggers two separate Maverick calls. Production would cache by image hash.