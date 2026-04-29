# Course 5 — Lab 1 (Module 1): Personal Storyteller

LLM → text → TTS → audio. A topic goes in, an MP3 comes out. ~50 lines of glue between Mistral (via Ollama) and gTTS.

Built as part of the IBM RAG and Agentic AI Professional Certificate — Course 5, Module 1.

---

## What It Does

- **Generate** — takes a topic string, prompts Mistral to write a 200–300 word beginner-friendly educational story
- **Speak** — converts the generated text to MP3 via Google's TTS endpoint
- **Save** — writes the audio to `output/story_<slug>.mp3`

Pipeline is strictly procedural — no retrieval, no agents, no chaining. The interesting work happens at the seams.

---

## Stack

| Component | IBM Lab | Local Swap |
|---|---|---|
| LLM | Watsonx-hosted `mistralai/mistral-small-3-1-24b-instruct-2503` | Ollama `mistral` (7B) |
| Generation API | `ibm_watsonx_ai.ModelInference.generate_text()` | `langchain_ollama.OllamaLLM.invoke()` |
| Decoding | `DECODING_METHOD: "greedy"` | `temperature=0.0` |
| Max tokens | `MAX_NEW_TOKENS: 1000` | `num_predict=1000` |
| TTS | gTTS (unchanged) | gTTS (unchanged) |
| Playback | `IPython.display.Audio` | Filesystem MP3, any player |

No IBM SDK, no Watsonx credentials, no Claude key for this lab. Ollama is local; gTTS is keyless.

---

## Setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
ollama pull mistral  # if not already pulled
```

For full reproducibility (exact transitive versions), use the lock file instead:

```powershell
pip install -r requirements-lock.txt
```

Note: `requirements-lock.txt` includes Jupyter dependencies from running the original notebook. The script and tests don't need them — `requirements.txt` (3 direct deps) is sufficient if you only want to run `storyteller.py` and the test suite.

Run:

```powershell
python storyteller.py "the life cycle of butterflies"
python storyteller.py  # uses default topic
```

MP3 lands in `output/`.

---

## File Layout

```
course5-module1-lab1/
├── storyteller.py            — generate_story, text_to_speech, slugify, main
├── conftest.py               — sys.path fix for pytest discovery
├── requirements.txt          — direct deps (gTTS, langchain-ollama, pytest)
├── requirements-lock.txt     — full pip freeze for reproducibility
├── tests/
│   └── test_storyteller.py   — 2 tests, fully mocked
├── output/                   — generated MP3s (gitignored)
└── venv/
```

Flat. Single script doesn't earn an Onion split.

---

## Key Decisions

**Greedy decoding kept on the swap.** Setting `temperature=0.0` in Ollama is the algorithmic equivalent of IBM's `DECODING_METHOD: "greedy"`. Same prompt, same output, every run. Useful property for a lab — isolates "did the swap work?" from "is sampling random?". Bump to 0.7 for actual creative variety.

**File I/O over `BytesIO`.** The IBM solution writes MP3 bytes to an in-memory buffer for `IPython.display.Audio`. Outside Jupyter, the buffer is overhead — saving directly to disk costs the same one Google call and gives you a file you can replay without re-hitting the API. Choose ephemeral (BytesIO) vs. durable (file) based on whether you'll want the artifact later.

**Notebook → script refactor.** Module-level constants (`MODEL_ID`, `OUTPUT_DIR`, `DEFAULT_TOPIC`), three small functions, `pathlib.Path` not string concatenation, CLI arg with default fallback. `Audio()` widget dropped — pointless outside Jupyter.

**Slugified filenames.** The notebook hardcoded one filename, so re-running with a different topic silently overwrote. `output/story_<slug>.mp3` makes each run distinct.

---

## Test Suite

2 tests, both mocked. Run with:

```powershell
python -m pytest tests/ -v
```

| Test | What it verifies |
|---|---|
| `test_main_runs_full_pipeline` | argv parsed, `OllamaLLM` constructed with correct config, prompt mentions topic, gTTS receives model output, save called with `.mp3` path inside tmp dir |
| `test_main_uses_default_topic_when_no_argv` | falls back to `DEFAULT_TOPIC` when no CLI arg provided |

**Patterns:**

- `@patch("storyteller.OllamaLLM")` — patch where used, not where defined
- Decorator order is bottom-up — closest decorator → first parameter
- `tmp_path` for ephemeral filesystem, `monkeypatch` for `sys.argv` and module constants
- `mock.call_args[0][0]` for first positional arg of most recent call
- Tests verify *your* code's contracts, not dependencies' behaviour — no assertion on prompt quality, story length, or MP3 validity

Total runtime ~0.5s. No Ollama load, no Google round-trip, no MP3 written.

---

## Patterns Worth Retaining

**1. Construct ≠ execute.** `gTTS(text)` is inert. `tts.write_to_fp()` and `tts.save()` are where the network call happens. Same shape as `Path("foo.mp3")` not creating a file. Worth internalising — saves you from imagined side effects when reading code.

**2. Pipeline decoupling.** LLM generation and TTS are independent stages. Cache the LLM output, swap the TTS provider, skip TTS for text-only consumers — none of the downstream changes touch upstream code. Same instinct as retrieval ≠ generation in RAG (Course 4).

**3. Pure functions test cheap, I/O functions test expensive.** `slugify` is trivially testable; `text_to_speech` requires mocking gTTS and the filesystem. Separating pure logic from I/O at the function boundary is what makes code testable in the first place — a senior-engineering instinct that pays off everywhere.

**4. `python -m pytest` over bare `pytest`.** The `-m pytest` form uses whichever Python is first on PATH to run pytest as a module — predictable across venv-active vs. venv-inactive states. Bare `pytest` resolves via PATH and can silently invoke system Python with the wrong site-packages.

**5. `conftest.py` needs the explicit `sys.path` insert.** Empty `conftest.py` is unreliable on modern pytest (7+). Make it explicit:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
```

**6. gTTS is fragile production-wise.** Hits an undocumented Google Translate endpoint, no SLA, no auth, IP-based rate limiting. Fine for labs and personal projects; for production you want ElevenLabs / Azure Speech / AWS Polly.

---

## What This Doesn't Cover

- Streaming TTS — gTTS only does full-text-then-audio
- Voice/style control — gTTS exposes language and slow-flag, nothing else
- Multimodal input — text only; this is the upstream half of a multimodal pipeline
- Audio playback in scripts — MP3s exist on disk, opening them is the OS's problem

Module 1's STT and audio-RAG concepts (Whisper, MFCCs, mel-spectrograms) aren't touched here. Lab 2 onwards.