# Course 5 — Lab 19: Style Finder (MM-RAG Computer Vision)

> Code: [`course5-module3-lab1/`](course5-module3-lab1/)

Complete multimodal RAG pipeline for fashion analysis. Image in, fashion analysis out. Combines ResNet50 image encoding, cosine similarity search against a pre-computed vector index, and a vision LLM to generate catalog-style fashion descriptions with retrieved product metadata. Two backends: LLaVA locally via Ollama, and Llama 4 Maverick via IBM watsonx. Built as part of the IBM RAG and Agentic AI Professional Certificate — Course 5, Module 3.

---

## What It Does

- **Encode** — uploads image through Gradio, encodes to base64 and ResNet50 feature vector
- **Retrieve** — cosine similarity search against pre-computed outfit embeddings in a pickle dataset
- **Augment** — retrieves all items from the matched outfit (name, price, link) and injects into the LLM prompt
- **Generate** — vision LLM receives the uploaded image (base64) plus structured item metadata and produces a fashion analysis
- **Post-process** — escapes Markdown characters, normalises section headers, handles model refusals

---

## Stack

| Component | IBM Lab | Local Swap |
|---|---|---|
| Vision LLM | `meta-llama/llama-4-maverick-17b-128e-instruct-fp8` via watsonx | `llava:latest` via Ollama |
| Image encoder | ResNet50 (`torchvision.models.resnet50`) | Same — no swap needed |
| Similarity search | `sklearn.metrics.pairwise.cosine_similarity` | Same — no swap needed |
| Vector index | `swift-style-embeddings.pkl` (pre-computed, Taylor Swift outfit dataset) | Same |
| Web UI | Gradio 5 (`gr.Blocks`) | Same |
| Credentials | python-dotenv + .env with WATSONX_API_KEY and WATSONX_PROJECT_ID | None — no credentials needed |

Backend selected via `config.USE_LOCAL` (True = Ollama, False = IBM watsonx).

---

## File Layout

```
course5-module3-lab1/
├── app.py                            — single entry point, composition root
├── config.py                         — model IDs, USE_LOCAL flag, image settings
├── requirements.txt
├── swift-style-embeddings.pkl        — pre-computed dataset (gitignored)
├── .env                              — WATSONX_API_KEY, WATSONX_PROJECT_ID
├── .gitignore
├── examples/
│   ├── test-6.png
│   └── test-7.png
├── interfaces/
│   ├── __init__.py
│   └── llm_service_interface.py      — Protocol contract for LLM services
├── infra/
│   ├── __init__.py
│   ├── llm_service.py                — IBM watsonx concrete implementation
│   └── llm_service_local.py          — Ollama local concrete implementation
├── models/
│   └── image_processor.py            — ResNet50 encoding + cosine similarity
├── services/
│   └── search_service.py             — SerpAPI product search (optional)
└── utils/
    └── helpers.py                    — dataset helpers, response formatting
```

---

## MM-RAG Pipeline

```
User uploads image
        │
        ▼
ImageProcessor.encode_image()
  → base64 string (for LLM)
  → ResNet50 feature vector (for retrieval)
        │
        ▼
ImageProcessor.find_closest_match()
  → cosine_similarity(user_vector, dataset_vectors)
  → closest_row + similarity_score
        │
        ▼
get_all_items_for_image(closest_row['Image URL'])
  → all items from the matched outfit (DataFrame filter)
        │
        ▼
LlamaVisionService.generate_fashion_response()
  → builds prompt with item metadata injected
  → sends base64 image + prompt to vision LLM
  → returns fashion analysis text
        │
        ▼
process_response()
  → escapes $, normalises headers, handles refusals
        │
        ▼
Gradio UI renders Markdown output
```

---

## Core Pattern

### IBM watsonx

```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
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
        "content": prompt,
        "images": [encoded_image]   # base64 string, top-level field
    }
]
response = ollama.chat(model=self.model_id, messages=messages, options=self.params)
return response['message']['content']
```

---

## Key Decisions

**`ResNet50_Weights.DEFAULT` over `pretrained=True`.** `pretrained=True` is deprecated in newer torchvision. Correct API:
```python
from torchvision.models import resnet50, ResNet50_Weights
self.model = resnet50(weights=ResNet50_Weights.DEFAULT).to(self.device)
```

**The classification layer is not stripped.** Production would use `torch.nn.Sequential(*list(model.children())[:-1])` to get clean 2048-dim feature vectors instead of 1000-dim classifier outputs. The `.pkl` dataset was generated the same way so results are consistent, but worth noting for production index builds.

**Set vs list bug in items join.** Original code used a set comprehension `{f"- {item}" for item in items_list}` — sets are unordered, items came out in random order every run. Fixed to list comprehension `[f"- {item}" for item in items_list]`.

**Duplicate items section.** LLM included `SIMILAR ITEMS` in its response but without the colon, so the fallback check `"SIMILAR ITEMS:" not in response` always triggered and appended items a second time. Fixed by dropping colon from the check:
```python
if "ITEM DETAILS" not in response and "SIMILAR ITEMS" not in response and "Similar Items" not in response:
```

**Gradio 5 theme import change.** `gr.themes.Soft()` raises a Pylance error in Gradio 5. Fix:
```python
from gradio.themes import Soft
with gr.Blocks(theme=Soft(), ...) as demo:
```

**`ibm-watsonx-ai==1.1.20` not installable on Python 3.13.** Same issue as Lab 18 — use unpinned `ibm-watsonx-ai` to get latest (1.5.12).

**Single app.py with Protocol interface and infra layer.** `LLMServiceInterface` Protocol in `interfaces/` defines the contract. IBM and Ollama concretes in `infra/` satisfy it structurally — no explicit inheritance. `create_llm_service()` in `app.py` is the composition root, wired via `config.USE_LOCAL`.

**`.pkl` gitignored.** The embeddings file is large binary data — same principle as not committing model weights. Add to `.gitignore` before first commit.

---

## Latency Comparison

Same image, same prompt, same pipeline:

| Backend | Model | Generation time |
|---|---|---|
| IBM watsonx | Llama 4 Maverick 17B | ~7 seconds |
| Ollama local (RTX 4070 8GB) | LLaVA 7B | ~2 minutes 19 seconds |

20x latency difference. Managed API wins for production fan-facing workloads. Local wins for dev iteration with no API cost per run.

---

## Model Comparison: LLaVA vs Llama 4 Maverick (this lab)

Same test image (blue shirt, khaki shorts, beret, brown boots):

| Aspect | LLaVA (local) | Llama 4 Maverick (IBM) |
|---|---|---|
| Visual description accuracy | Good — correctly identified all major items | Good — similar accuracy |
| Item metadata integration | Added contextual reasoning ("earrings that match the socks") | Listed items verbatim |
| Response style | More conversational, integrated | More clinical, structured |
| Prompt instruction following | Partial — included section headers inconsistently | Partial — same issue |

LLaVA's contextual reasoning about the retrieved items was unexpectedly better in this lab — it connected the metadata to the image rather than just appending it. Consistent with Lab 18 finding that smaller models hallucinate more but can surprise on reasoning tasks.

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
| Nested content array with `type: image_url` | `"images": [encoded_image]` as top-level field |
| `model_id=config.LLAMA_MODEL_ID` in app | `model_id=config.OLLAMA_MODEL_ID` in app |

---

## Patterns Worth Retaining

**1. The vector store is modality-agnostic.** FAISS, ChromaDB, and raw cosine similarity all operate on float arrays — they have no concept of images vs text. The intelligence about what makes two images "similar" lives entirely in the encoder (ResNet50, CLIP), not the store. Swap the encoder, same retrieval infrastructure.

**2. Offline/online split is the same as text RAG.** Offline: encode all images → store vectors in index. Online: encode query image with same model → similarity search → retrieve metadata → generate. The separation is identical. The online pipeline never touches the raw dataset images — only the vectors and metadata.

**3. The uploaded image is used twice, in two different forms.** As a feature vector for retrieval (finding the closest match), and as base64 for generation (LLM sees the actual image). Both come from the same `encode_image()` call returning a dict with both keys.

**4. The index stores vectors and metadata, not images.** The vector store holds embeddings + references (URLs, IDs). Actual image bytes are fetched separately at generation time. Same as Hooperman's retriever returning chunk text plus source metadata, not a copy of the raw document.

**5. Prompt engineering is the augmentation layer.** The items list (name, price, link) is injected as structured text into the prompt alongside the image. The LLM has no other way to know about the dataset — this is the RAG augmentation step made explicit.

**6. Fallback logic should be specific, not broad.** Checking for exact strings like `"ITEM DETAILS:"` breaks when the model paraphrases the header. Check for substrings without punctuation, or use lowercase normalisation. LLMs don't follow formatting instructions precisely.

---

## Rovers Connection

**Kit and matchday image search** — fans could upload a photo of a kit or a stadium section and retrieve relevant information. Same ResNet50 + cosine similarity pattern against a Rovers image index. "What kit is this?" answered visually without metadata.

**Richer context injection** — Hooperman currently retrieves text chunks. Could augment with structured metadata (match results, fixture dates, ticket prices) injected as a structured block alongside retrieved chunks — same pattern as the items list in this lab. More reliable than hoping the right chunk contains the price.

**Caption-then-index pattern for match photos** — generate captions for match photos using a vision LLM, embed the captions for text retrieval. Fan asks "show me photos of goals against Bohs" → retrieve by caption similarity, not filename. Lab 18 pattern applied to Rovers media library.

**The fragility question** — current Hooperman answers can be fragile with only 33 golden questions. Scaling to 200+ questions with real fan queries covers the full question space and catches retrieval failures that the current golden set doesn't surface. The eval pipeline tickets is the right fix.

---

## What This Doesn't Cover

- CLIP-style cross-modal embeddings — ResNet50 produces image-only vectors, not a joint text/image space. Text queries cannot retrieve images directly with this encoder
- Fine-tuning the encoder on domain-specific imagery — general ResNet50 embeddings work for broad visual similarity but miss fine-grained distinctions (fabric textures, specific patterns)
- Production vector store (FAISS/ChromaDB) — lab uses brute-force cosine similarity over a small dataset. Production would index into FAISS for scale
- The offline index build — the `.pkl` was pre-built. Building it yourself: loop over image URLs, call `encode_image()`, store vectors + metadata in DataFrame, `df.to_pickle()`
- Streaming responses — both Ollama and watsonx support streaming; not used here
- SerpAPI product search integration — `SearchService` is implemented but requires a paid API key; disabled in this lab