# Course 5 — Lab 18: Image Captioning System (LLaVA + Llama 4 Maverick)

Image in, caption out. Demonstrates multimodal image captioning and visual question answering using two backends: LLaVA locally via Ollama, and Llama 4 Maverick via IBM watsonx. Built as part of the IBM RAG and Agentic AI Professional Certificate — Course 5, Module 2.

---

## What It Does

- **Encode** — downloads images from URLs and encodes them to base64 strings
- **Caption** — sends each image with a text query to a vision model and returns a description
- **Query** — demonstrates visual question answering: object detection, damage assessment, label reading
- **Compare** — same queries run against LLaVA (local) and Llama 4 Maverick (watsonx) for direct comparison

---

## Stack

| Component | IBM Lab | Local Swap |
|---|---|---|
| Vision model | `meta-llama/llama-3-2-11b-vision-instruct` via watsonx | `llava:latest` via Ollama |
| Comparison model | N/A | `meta-llama/llama-4-maverick-17b-128e-instruct-fp8` via watsonx |
| Image encoding | `base64.b64encode` + `requests` | Same — no swap needed |
| Credentials | IBM SN Labs injected credentials | `python-dotenv` + `.env` with `WATSONX_API_KEY` and `WATSONX_PROJECT_ID` |
| Parameters | `TextChatParameters(temperature=0.2, top_p=0.5)` | `{"temperature": 0.2, "top_p": 0.5}` dict for Ollama |

Two scripts produced:
- `image_captioning.py` — LLaVA via Ollama (fully local, no API key needed)
- `image_captioning_ibm.py` — Llama 4 Maverick via watsonx (IBM API key required)

---

## Setup

### Local (LLaVA)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install ollama requests
ollama pull llava
```

### IBM watsonx (Llama 4 Maverick)

```powershell
pip install ibm-watsonx-ai requests python-dotenv
```

`.env` in the lab directory:

```
WATSONX_API_KEY=your_key_here
WATSONX_PROJECT_ID=your_project_id_here
```

Note: `ibm-watsonx-ai==1.1.20` (IBM's prescribed version) requires Python < 3.13. Use `ibm-watsonx-ai` (latest, currently 1.5.12) for Python 3.13.

---

## File Layout

```
course5-module2-lab2/
├── image_captioning.py           — LLaVA local version
├── image_captioning_ibm.py       — Llama 4 Maverick via watsonx
├── .env                          — WATSONX_API_KEY, WATSONX_PROJECT_ID
├── .gitignore
└── venv/
```

---

## Core Pattern

### Local (Ollama)

```python
messages = [
    {
        "role": "user",
        "content": assistant_prompt + user_query,
        "images": [encoded_image]   # base64 string, top-level field
    }
]
response = ollama.chat(model=model_id, messages=messages, options=params)
return response['message']['content']
```

### IBM watsonx

```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": assistant_prompt + user_query},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + encoded_image}}
        ]
    }
]
response = model.chat(messages=messages)
return response['choices'][0]['message']['content']
```

---

## Key Decisions

**IBM lab prescribed `llama-3-2-11b-vision-instruct` — deprecated May 5 2026.** Swapped to `llama-4-maverick-17b-128e-instruct-fp8`, which is the current vision-capable model in the SN Labs supported model list and the same model referenced in the Module 2 image captioning video.

**Ollama message structure differs from OpenAI/watsonx format.** Ollama uses `"images": [encoded_image]` as a top-level field alongside `"content"` as a plain string. The IBM/OpenAI format uses a nested `"content"` array with `"type": "text"` and `"type": "image_url"` objects. Mixing these formats silently fails — the function appears to run but the model never sees the image.

**Default argument string caused silent failure in IBM SN Labs kernel.** `assistant_prompt="You are a helpful assistant..."` in the function signature caused the cell to fail silently without error — the function was never defined. Fixed by extracting the string to a module-level constant (`DEFAULT_PROMPT`) and referencing it as the default. Root cause likely an encoding or parsing issue in the IBM Jupyter environment.

**`ibm-watsonx-ai` version constraint.** IBM prescribes `==1.1.20` which requires Python < 3.13. Python 3.13 requires `>=1.3.18`. Use unpinned `ibm-watsonx-ai` to get the latest compatible version.

---

## Model Comparison: LLaVA vs Llama 4 Maverick

Same 4 images, same queries.

### Image Captioning ("Describe the photo")

| Image | LLaVA | Llama 4 Maverick |
|---|---|---|
| City street | "tall buildings, a crosswalk, busy urban area" | Identified taxis, buses, trees, streetlights, likely NYC |
| Runner | Hallucinated "airport tarmac", got jacket colour wrong | Correctly described industrial setting, accurate clothing detail |
| Flood | Generic flooding description | Added farmhouse context, agricultural buildings, crop loss inference |
| Nutrition label | Hallucinated smartphone screen displaying a nutrition app | Correctly identified physical label on a purple box, finger pointing |

### Visual QA

| Query | LLaVA | Llama 4 Maverick |
|---|---|---|
| How many cars? (image 2) | "No cars" — hallucinated a truck | Correctly located one dark hatchback on the right |
| Flood damage severity? | Generic response | Added farmhouse, agricultural context, "major flood event" |
| How much sodium? | Not tested (IBM env) | 640mg, 27% daily value — correct |
| How much cholesterol? | Not tested (IBM env) | 20mg, 7% daily value — correct |
| Jacket colour? | Not tested (IBM env) | Yellow — correct and concise |

**Pattern:** Llama 4 Maverick (17B) consistently outperforms LLaVA on grounding, object localisation, and label reading. LLaVA hallucination rate is notably higher — it invents context (airport, smartphone) when uncertain rather than describing what's visible.

---

## IBM Swap Reference

| IBM | Local |
|---|---|
| `from ibm_watsonx_ai import Credentials, APIClient` | Delete |
| `from ibm_watsonx_ai.foundation_models import ModelInference` | `import ollama` |
| `from ibm_watsonx_ai.foundation_models.schema import TextChatParameters` | `params = {"temperature": 0.2, "top_p": 0.5}` |
| `ModelInference(model_id=..., credentials=..., project_id=..., params=...)` | `ollama.chat(model=model_id, messages=messages, options=params)` |
| `model.chat(messages=messages)` | `ollama.chat(model=model_id, messages=messages, options=params)` |
| `response['choices'][0]['message']['content']` | `response['message']['content']` |
| Nested content array with `type: image_url` | `"images": [encoded_image]` as top-level field |

---

## Patterns Worth Retaining

**1. Base64 is the universal transport format for images in LLM APIs.** Download image bytes → `base64.b64encode(response.content).decode("utf-8")` → pass to model. Same pattern regardless of provider. The encoding step is mandatory because JSON is text-only.

**2. Message structure varies by provider — don't assume OpenAI format is universal.** Ollama, watsonx, and OpenAI all accept multimodal messages but with different schemas. Always check the provider's specific format. Silent failures (no error, no image in context) are the symptom of a format mismatch.

**3. Visual QA and image captioning are the same pipeline, different prompts.** The `generate_model_response` function handles both — swap the `user_query` string. "Describe the photo" → captioning. "How many cars?" → object detection. "How much sodium?" → document/label reading. The model handles the task switch implicitly.

**4. Hallucination patterns differ by model size.** LLaVA (7B) hallucinates context when uncertain — invents plausible but wrong surroundings. Maverick (17B) grounds more accurately and is more likely to say what's actually visible. Larger models reduce but don't eliminate hallucination.

**5. Model deprecation on managed platforms moves fast.** IBM deprecated `llama-3-2-11b-vision-instruct` on May 5 2026 — weeks after this lab was published. Always check the supported model list at runtime, not just at development time.

---

## Rovers Connection

**Match photo captioning pipeline** — this lab maps directly to a practical Rovers use case. Feed match photographs into the captioning pipeline, generate descriptions, embed the captions for retrieval. Fan asks "show me photos of goals against Bohemians" → retrieve by caption similarity rather than filename. Same `encode_images_to_base64` + `generate_model_response` pattern, wrapped in a batch processing loop.

**Visual QA for damage/safety assessment** — the flood damage query demonstrates a pattern applicable to stadium infrastructure: upload a photo, ask "is there any visible damage?", route to maintenance if yes. Low-cost triage before human review.

---

## What This Doesn't Cover

- Streaming responses — `model.chat()` and `ollama.chat()` both support streaming; not used here
- Batch inference — images processed sequentially; production would parallelise
- Error handling for None encoded images — the loop doesn't skip `None` entries from failed downloads
- Prompt engineering for structured output — responses are free text; production would use JSON output format for downstream processing