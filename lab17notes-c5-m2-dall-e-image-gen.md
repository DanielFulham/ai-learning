# Course 5 — Lab 17: DALL-E Image Generation (GPT Image API)

Text prompt in, generated image out. Demonstrates text-to-image generation using the OpenAI Images API, comparing output quality across model generations. Built as part of the IBM RAG and Agentic AI Professional Certificate — Course 5, Module 2.

---

## What It Does

- **Generate** — calls the OpenAI Images API with a text prompt and returns a generated image
- **Compare** — runs the same prompt through two model generations to compare output quality
- **Save** — decodes base64 response and writes PNG to disk
- **Display** — opens generated image in the default browser

---

## Stack

| Component | IBM Lab | Local Swap |
|---|---|---|
| Image generation (v1) | `dall-e-2` via OpenAI API | `gpt-image-1` — dall-e-2 deprecated May 12 2026 |
| Image generation (v2) | `dall-e-3` via OpenAI API | `gpt-image-2` — dall-e-3 deprecated May 12 2026 |
| Display | `IPython.display.Image()` | `base64` decode → write PNG → `webbrowser.open()` |

No IBM SN Labs environment. Runs locally against the OpenAI API with personal API credits (~$0.02-0.04 per image).

---

## Setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install openai==1.64.0 python-dotenv
```

`.env` in the lab directory:

```
OPENAI_API_KEY=sk-...
```

---

## File Layout

```
course5-module2-lab1/
├── gpt_image_1_generation_cat.py
├── gpt_image_2_generation_cat.py
├── gpt_image_1_generation_beautiful_lake.py
├── gpt_image_2_generation_beautiful_lake.py
├── .env
├── .gitignore                 — ignores *.png
└── venv/
```

---

## Core Pattern

```python
import base64
import webbrowser
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

response = client.images.generate(
    model="gpt-image-1",   # or gpt-image-2
    prompt="a white siamese cat",
    size="1024x1024",
    n=1,
)

image_base64 = response.data[0].b64_json
if image_base64:
    image_bytes = base64.b64decode(image_base64)
    output_path = Path("siamese_cat.png")
    output_path.write_bytes(image_bytes)
    webbrowser.open(str(output_path.resolve()))
else:
    print("No image returned")
```

---

## Key Decisions

**DALL-E 2/3 replaced with GPT Image series.** IBM's lab prescribes `dall-e-2` and `dall-e-3` — both deprecated and removed on May 12, 2026. `gpt-image-1` and `gpt-image-2` are the current equivalents. The API call structure is unchanged.

**Base64 over URL.** GPT Image models return base64 by default. IBM's lab uses `response.data[0].url` with `IPython.display.Image()` — neither works locally or with the new models. Decode base64, write to PNG, open in browser.

**`response.data[0].url` is typed `str | None`.** Always null-check before use.

**`IPython.display` is Jupyter-only.** Replace with `webbrowser.open()` for standalone scripts.

---

## Model Comparison

Same prompt: `"a white siamese cat"`

| Model | Observation |
|---|---|
| `gpt-image-1` | Naturalistic, photorealistic, front-facing portrait |
| `gpt-image-2` | Sharper, better fur detail, more natural pose, cleaner background, more compositional ambition |

Same prompt: `"a beautiful lake with a sunset"`

| Model | Observation |
|---|---|
| `gpt-image-1` | Moody, naturalistic — feels like a real photograph |
| `gpt-image-2` | Cinematic — added mountains, rocks, pine trees, mirror reflection, vivid colour range |

**Pattern:** gpt-image-2 doesn't just produce higher quality — it makes more ambitious compositional decisions from the same input. In production pipelines where consistency matters, this is a variable to account for.

---

## IBM Swap Reference

| IBM | Local |
|---|---|
| `model="dall-e-2"` | `model="gpt-image-1"` |
| `model="dall-e-3"` | `model="gpt-image-2"` |
| `display.Image(url=response.data[0].url)` | `base64.b64decode(response.data[0].b64_json)` → write PNG → `webbrowser.open()` |
| IBM SN Labs environment with injected credentials | `python-dotenv` + `.env` with `OPENAI_API_KEY` |

---

## Patterns Worth Retaining

**1. API deprecation moves faster than course material.** Always verify model strings against the provider's changelog before running labs. `platform.openai.com/docs/changelog` is authoritative for OpenAI.

**2. Text-to-image is the inverse of image captioning.** Same module, opposite direction — captioning is image → text, generation is text → image. Different architectures: CLIP-based encoders for captioning, autoregressive/diffusion for generation.

**3. Prompt structure matters.** Cinematic vocabulary (lighting, camera angle, mood) produces better results than plain description.

**4. Base64 is the standard transport format for binary data in API responses.** Same pattern appears in image captioning (encoding images for the model) and here (decoding generated images from the model). `base64.b64decode()` → `bytes` → write to file.

---

## What This Doesn't Cover

- Image editing (inpainting) — GPT Image 1/2 support multi-image editing but with a different API shape
- Variations — DALL-E 2 supported generating variations of an existing image
- Prompt engineering depth — production image generation benefits significantly from structured prompt templates