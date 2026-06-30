# Course 5 — Lab 16: AI Meeting Assistant

> Code: [`course5-module1-lab2/`](course5-module1-lab2/)

Audio → transcript → cleanup → meeting minutes. A meeting recording goes in, structured minutes and a task list come out. ~80 lines of glue between Whisper, two Ollama calls, and a Gradio UI.

Built as part of the IBM RAG and Agentic AI Professional Certificate — Course 5, Module 1.

---

## What It Does

- **Transcribe** — converts uploaded audio to raw text via Whisper
- **Clean** — strips non-ASCII characters, then runs a second LLM call to normalise financial terminology
- **Generate** — passes the cleaned transcript through a LangChain chain to produce structured meeting minutes and a task list
- **Serve** — Gradio UI for upload, display, and download of the output

Pipeline: `audio → Whisper → remove_non_ascii → product_assistant → chain → Gradio`

---

## Stack

| Component | IBM Lab | Local Swap |
|---|---|---|
| STT | `openai/whisper-medium` via HuggingFace `transformers` | Same — no swap needed |
| Transcript cleanup LLM | `meta-llama/llama-3-2-11b-vision-instruct` via `ModelInference` | `OllamaLLM(model="llama3.2", temperature=0.2)` |
| Meeting minutes LLM | `ibm/granite-3-3-8b-instruct` via `WatsonxLLM` | `OllamaLLM(model="llama3.2", temperature=0.5)` |
| Chain | LCEL with `RunnablePassthrough` | LCEL `prompt | llm | StrOutputParser()` |
| UI | Gradio `gr.Interface` on port 5000 | Gradio `gr.Interface`, `iface.launch()` |

No IBM SDK, no Watsonx credentials. Whisper runs locally via HuggingFace. Both LLM calls use Ollama.

---

## Setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
ollama pull llama3.2
```

ffmpeg required for Whisper audio loading — install via winget if not present:

```powershell
winget install ffmpeg
```

Restart terminal after ffmpeg install — PATH update requires new shell.

First run downloads Whisper model to HuggingFace cache (`~/.cache/huggingface/hub/`). Subsequent runs are instant. `whisper-tiny.en` (~150MB) for development, `whisper-medium` (~3GB) for production quality.

Run:

```powershell
python speech_analyzer.py
```

Opens Gradio at `http://127.0.0.1:7860`. Upload any `.wav` or `.mp3` file.

---

## File Layout

```
course5-module1-lab2/
├── simple_speech2text.py     — Step 1: Whisper standalone test + audio download
├── speech2text_app.py        — Step 2: Whisper + Gradio UI
├── simple_llm.py             — Steps 3-4: Ollama LLM test + product_assistant
├── speech_analyzer.py        — Step 6: Full pipeline (Whisper + cleanup + chain + Gradio)
├── sample-meeting.wav        — IBM sample audio (earnings call)
├── meeting_minutes_and_tasks.txt — Generated output (gitignored)
├── requirements.txt
└── venv/
```

---

## Key Decisions

**Two LLM calls, different jobs.** `product_assistant` uses a low-temperature (0.2) call for deterministic terminology cleanup. The main chain uses a higher temperature (0.5) for generative meeting minutes. Same model, different configs, different purposes — the cleanup call needs consistency, the generation call benefits from some variation.

**`RunnablePassthrough` wrapper dropped.** IBM's solution uses `{"context": RunnablePassthrough()} | prompt | llm`. Since `chain.invoke({"context": transcript})` passes the dict directly, the wrapper is redundant. `prompt | llm | StrOutputParser()` is equivalent and cleaner.

**`product_assistant` — LLM over rules engine.** Financial terminology is contextual — "LTV" means Loan-to-Value in a mortgage context but Lifetime Value in a SaaS context. A rules-based find-and-replace can't handle ambiguity. A small LLM with a detailed system prompt can. Adding "Produce the adjusted transcript only, no commentary" prevents model preamble from polluting the downstream prompt.

**`whisper-tiny.en` for development, `whisper-medium` for production.** Tiny is already cached and loads instantly. Medium produces noticeably better transcription — fewer domain-specific errors — but is 3GB and slow to download. The LLM's implicit error correction partially compensates for tiny's weaknesses (e.g. "cutting its blockchain" → LLM inferred "cutting-edge blockchain" from context).

**Gradio `gr.Interface` over `gr.Blocks`.** Single function, two outputs (text + file download). `gr.Interface` is sufficient — `gr.Blocks` overhead not justified for a linear pipeline. Note: `gr.Interface` is single-turn only, no conversation history.

---

## Pipeline Walkthrough

```python
# 1. Whisper transcribes audio
pipe = pipeline("automatic-speech-recognition", model="openai/whisper-tiny.en", chunk_length_s=30)
raw_transcript = pipe(audio_file, batch_size=8)["text"]

# 2. Strip non-ASCII
ascii_transcript = remove_non_ascii(raw_transcript)

# 3. Terminology cleanup via second LLM call
adjusted_transcript = product_assistant(ascii_transcript)

# 4. Generate meeting minutes via chain
result = chain.invoke({"context": adjusted_transcript})

# 5. Write to file and return to Gradio
```

Each stage is independent — swap Whisper for a different STT model, swap the cleanup LLM, swap the generation model. None of the downstream stages know or care about the upstream implementation.

---

## IBM Swap Reference

| IBM | Local |
|---|---|
| `from ibm_watsonx_ai import Credentials, APIClient` | Delete |
| `from langchain_ibm import WatsonxLLM` | `from langchain_ollama import OllamaLLM` |
| `from ibm_watsonx_ai.foundation_models import ModelInference` | Delete |
| `from ibm_watsonx_ai.foundation_models.schema import TextChatParameters` | Delete |
| `WatsonxLLM(model_id=..., url=..., project_id=..., params=...)` | `OllamaLLM(model="llama3.2", temperature=0.5, num_predict=512)` |
| `ModelInference(...).chat(messages=messages)` | `OllamaLLM(...).invoke(prompt_string)` |
| `response['choices'][0]['message']['content']` | Return value directly from `.invoke()` |
| `iface.launch(server_name="0.0.0.0", server_port=5000)` | `iface.launch()` |

---

## Patterns Worth Retaining

**1. Two-LLM pipeline — cheap model cleans, expensive model generates.** Pre-processing with a small, low-temperature LLM before the main generation call is a real production pattern. The cleanup call is cheap and deterministic; the generation call is expensive and creative. Don't conflate them into one prompt.

**2. Domain-specific pre-processing is prompt-engineered, not hardcoded.** `product_assistant` uses a detailed system prompt to handle contextual disambiguation (LTV = Loan-to-Value vs Lifetime Value). A rules engine can't do this. The system prompt is the business logic — keep it in source control and version it like code.

**3. HuggingFace `pipeline()` abstracts model loading, tokenisation, chunking.** For production you'd use `model.generate()` directly (Whisper's own chunking is more accurate than the pipeline wrapper's `chunk_length_s` approach). The pipeline is the right abstraction for a lab; the generate method is the right abstraction for production.

**4. `chunk_length_s` with seq2seq models is experimental.** HuggingFace warns about this — Whisper has its own internal chunking mechanism. For long-form transcription in production, call `model.generate()` directly rather than using the pipeline with `chunk_length_s`.

**5. ffmpeg is a hidden Whisper dependency.** Not in any pip install, not in requirements.txt — it's a system binary that Whisper calls under the hood to decode audio files. Easy to miss in local setup, critical for deployment. Always include in your deployment checklist and Dockerfile.

**6. HuggingFace model cache.** Models download to `~/.cache/huggingface/hub/` on first use and are reused automatically. No explicit caching code needed. For containerised deployments, mount this directory as a volume or pre-download models in the Docker build step to avoid cold-start delays.

---

## Rovers Connection

**Match report generator** — this pipeline maps directly to a Rovers backlog idea: Whisper transcribes post-match audio (manager press conference, commentary), `product_assistant`-equivalent normalises football terminology (player names, formations, competition names), main LLM generates a structured match report. Human reviews before publish.

**Query pre-processing** — `product_assistant` pattern applied to fan queries before retrieval: normalise abbreviations ("SRFC" → "Shamrock Rovers FC", "Tallaght" → "Tallaght Stadium"), resolve temporal references ("next game" → specific fixture). Small cheap LLM call before retrieval improves precision on ambiguous queries. Jira ticket raised.

---

## What This Doesn't Cover

- Speaker diarisation — who said what. Whisper transcribes but doesn't identify speakers.
- Real-time transcription — pipeline is batch only. Streaming STT requires a different architecture.
- Conversation history — `gr.Interface` is stateless. Multi-turn requires `gr.ChatInterface` or `gr.Blocks`.
- Audio quality handling — background noise, multiple speakers, accents degrade Whisper accuracy. `whisper-medium` or `whisper-large` recommended for noisy real-world audio.