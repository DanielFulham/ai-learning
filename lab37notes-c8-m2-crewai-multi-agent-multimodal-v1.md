# Course 8 - Lab 37: CrewAI Multi-Agent + Multimodal (NourishBot Nutrition Coach)

> Code: [`course8-module2-lab4-v1/`](course8-module2-lab4-v1/)

Modernised port of IBM's NourishBot lab: multi-agent CrewAI + Anthropic Haiku 4.5 vision + Gradio UI. Two workflows dispatched via a Gradio Dropdown + Radio — Recipe (three-agent sequential pipeline: `ingredient_detection → dietary_filtering → recipe_suggestion`) and Analysis (single-agent crew wrapping a vision tool that collapses vision + structured output into one `messages.parse()` call). Module 2 capstone for the CrewAI section.

Modernised against notebook's shape: watsonx-hosted Llama-4-Maverick swapped for Anthropic Claude Haiku 4.5 via CrewAI's native provider; `ibm_watsonx_ai` / `langchain` / `fastapi` all dropped; direct `anthropic` SDK calls in tools with the `{"type": "image", "source": {"type": "base64", ...}}` content-block shape (not OpenAI/watsonx's `{"type": "image_url", ...}`); `output_pydantic` replaces IBM's legacy `output_json`; `context=[]` replaces `depends_on=[]` + `input_data=lambda outputs:` (which isn't a real CrewAI Task API); `from crewai.tools import tool` replaces legacy `from langchain.tools import tool`; Gradio 6.19's post-migration API (`css`/`js`/`theme` on `.launch()` not `Blocks()`); `gr.Image(type="filepath")` replaces IBM's `type="pil"` with intermediate `image.save()` write hazard; `gr.Dropdown` on StrEnum values replaces IBM's `gr.Textbox` free-text field.

## Run it

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env  # populate ANTHROPIC_API_KEY
python nutricoach.py
# http://127.0.0.1:5000
```

Pinned:
```
crewai==1.15.2
crewai-tools==1.15.2
anthropic==0.117.0
python-dotenv==1.2.2
gradio==6.19.0
pillow==12.3.0
```

## File layout

```
course8-module2-lab4-v1/
├── config/
│   ├── agents.yaml
│   └── tasks.yaml
├── examples/
│   └── food-1.jpg … food-4.jpg
├── llm.py               # MODEL constant, direct-SDK client factory, CrewAI LLM factory
├── models.py            # Pydantic wire format + DietaryRestriction StrEnum
├── tools.py             # 4 tools; vision tools call Anthropic SDK directly
├── crew.py              # 2 crew builders (recipe + analysis), plain-Python
├── nutricoach.py        # Gradio entry point + format functions + dispatch
├── requirements.txt
├── pyproject.toml       # Ruff config
├── .env.example
└── .gitignore
```

## Key concepts

### Multimodal is pushed into tools, not surfaced via CrewAI's LLM abstraction

`Agent(multimodal=True)` is documented but broken across multiple CrewAI versions (issues #4016, #2565, #2475, #2541, #2642). The framework treats image URLs/paths/base64 as plain text at the LLM boundary; the auto-added `AddImageTool` doesn't fetch and inline the image. Direct SDK calls in custom tools are the correct production workaround, not a stylistic choice. Two vision tools (`extract_ingredients`, `analyse_nutrition`) each call `client.messages.create/parse` with the Anthropic content-block shape; the framework LLM abstraction only ever sees text. Zero image bytes cross the agent seam in either workflow.

### Vision + structured output collapse in one `messages.parse()` call

Anthropic structured outputs went GA on Haiku 4.5 (Nov 2025 beta → 2026 GA). `client.messages.parse(model=..., messages=[vision_message], output_format=NutrientAnalysisOutput)` accepts vision content blocks and returns a typed Pydantic instance on `response.parsed_output`. Collapses IBM's vision-in-tool → prose → agent-restructures-to-JSON two-call flow into one. Confirmed empirically at 3-level Pydantic nesting (`NutrientAnalysisOutput` → `NutrientBreakdown` → `VitaminInfo` / `MineralInfo`) — every kickoff produced complete, valid instances first attempt.

### Framework-boundary TypedDict construction concentrated in one helper

Pyright can't verify raw dict-literal shapes against Anthropic's `ImageBlockParam` / `TextBlockParam` / `MessageParam` TypedDicts. `_build_vision_message()` concentrates the typing plumbing in one place; each dict literal is annotated with its target TypedDict (`Base64ImageSourceParam`, `ImageBlockParam`, `TextBlockParam`), and `_MediaType = Literal["image/jpeg", "image/png", "image/gif", "image/webp"]` flows honestly through `_encode_image`'s return type. No `# type: ignore`, no `cast()`.

### Content-block shape differs by provider

Anthropic uses `{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": ...}}`. OpenAI and watsonx use `{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}`. The difference is contained in one helper — if we ever needed to swap providers, one function changes.

### Kickoff-time inputs, not construction-time

Both crews are constructed once at module scope in `nutricoach.py`, reused across kickoffs. `crew.kickoff(inputs=RecipeInputs(uploaded_image=path, dietary_restrictions="vegan"))` interpolates into `{uploaded_image}` and `{dietary_restrictions}` template slots in `tasks.yaml`. Matches L35's TypedDict + kickoff-inputs seam pattern.

## Findings

**Vision + structured output composition works cleanly on Haiku 4.5.** The full `NutrientAnalysisOutput` came back typed and complete on first attempt in every analysis run — dish name, portion size, calories, macros, 3-5 vitamins with %DV, 3-5 minerals with amounts, 2-3 sentence health evaluation. All fields populated, no null gaps in required nesting, no schema violations. Direct answer to a load-bearing question named at kickoff. Not evidence of composition at 5+ levels or with discriminated unions — the schema tested was three levels deep with `str | None` unions and typed lists.

**Framework un-collapse cost is real but banked-and-accepted.** `analyse_nutrition` returns a typed `NutrientAnalysisOutput` from one LLM call (vision + parse). Wrapping it in a single-agent CrewAI task forces the agent's ReAct loop to run its own final-answer restructuring pass — two LLM calls per kickoff where one would suffice. The framework wrapping cost is real; V1 preserves it for consistency with the recipe workflow's shape. Bank as a finding about where the structure-guarantee actually pays off — pay the framework tax when you need the framework's coordination machinery, not when you don't.

**CrewAI `@tool` decorator hard-requires a docstring at decoration time.** Surfaced during a Claude Code review pass after a docstring-tidy sweep had cut `extract_ingredients`'s docstring as signature restatement — the earlier smoke tests ran against a build with all four tool docstrings present, and the crash appeared only when the review re-ran the app against the swept code. `ValueError: Function must have a docstring` at import time — before the app could start. The docstring becomes the agent-facing tool description and rides into every kickoff. Load-bearing at two layers: framework contract at decoration, prompt content at runtime. Bank as a discipline rule — `@tool`-decorated functions get docstring exemption from the "cut restatement" rule.

**Multimodal-in-tool is the correct architectural shape, not a workaround.** The pattern that emerges — vision provider-specific code confined to one tool per vision task, text flowing across the agent seam, framework LLM abstraction never touching image bytes — is cleaner from an onion-architecture perspective than the framework's own `multimodal=True` promise. Even if `Agent(multimodal=True)` were fixed tomorrow, pushing multimodal into tools would still be defensible: it isolates the provider's content-block shape to one function, it makes the seam between agents cheaper (text vs image bytes), and it lets `output_pydantic` compose with vision at the tool boundary via `messages.parse()` rather than fighting the framework's own structuring pass.

**Cross-run variance is real under CrewAI same as under LangGraph.** Recipe workflow rotated recipe ordering and ingredient quantity specificity across runs at the same input. Same L33 finding on a different framework. Median-of-N is where "we ran it three times" stops being defensible if we ever want to compare configurations.

**Prompt-embedded constraints hold under load.** `expected_output` biases like "2-3 recipes", "3-5 key vitamins", "3-5 key minerals", "2-3 sentence health evaluation", "4-8 numbered steps" held across every run. Pantry-staples carve-out in the recipe task's `expected_output` prevented ingredient hallucination (recipes used salt, pepper, water, cooking oil without inventing dairy or protein). L34's finding sharpened: `expected_output` is a real lever, not just a hint.

**OTEL tracing timeout resolves via CrewAI's own preference dialog.** After first kickoff, CrewAI prompts about traces at end of run; timing out or answering `N` saves "Tracing has been disabled" as a persistent preference. Cleaner than the `.env`-level `CREWAI_TRACING_ENABLED=false` route which requires shell-time env dance to reach CrewAI before its config layer loads.

## Prior art carried across

- **L33: cross-run variance is real; median-of-N is the discipline point measurements can't survive.** Confirmed on CrewAI's recipe pipeline (order rotation, quantity specificity varied across runs).
- **L34: `allow_delegation=False` on every agent.** Held. IBM's default `True` on `dietary_filtering_agent` is the anti-pattern.
- **L34: `expected_output` is the strongest single lever on token spend.** Held. Tightened `expected_output` on all four tasks; prompt-embedded constraints biased effectively.
- **L34: native Anthropic provider requires explicit `anthropic` pin.** Held. `crewai==1.15.2` doesn't install `anthropic` transitively.
- **L35: `output_pydantic` unlocks typed inter-task propagation via `context=[]`.** Sharpened — `output_pydantic` on task N + `context=[task_N]` on task N+1 propagates typed Pydantic instances (not stringified prose) across the agent seam.
- **L35: `context=[]` is canonical; `depends_on` + `input_data=lambda outputs` is not a real CrewAI Task API.** Confirmed against IBM's shape and current 1.15.x docs.
- **L35: prompt caching definitively zero on CrewAI 1.15.2 native Anthropic.** Not re-tested. Bank as reliable framework property.
- **L35: StrEnum picklist discipline at the boundary.** Applied at the Gradio Dropdown surface (`DietaryRestriction` values) — no free-text drift into the prompt template.
- **L36: `from crewai.tools import tool` (not `from langchain.tools import tool`).** Held.
- **L36: tool descriptions ride in every prompt regardless of use.** Sharpened — CrewAI's `@tool` decorator uses the Python docstring AND the decorator argument as agent-facing description; both are prompt content.

New to L37:

- Multimodal-in-tool as the correct production shape (framework abstraction broken; workaround is the right architecture).
- Vision + structured output composition confirmed at 3-level Pydantic nesting on Haiku 4.5 in one `messages.parse()` call.
- Framework un-collapse cost real (single-agent single-tool crew runs two LLM calls when tool already returns typed output).
- Content-block shape difference Anthropic vs OpenAI/watsonx contained in one helper (`_build_vision_message`).
- CrewAI `@tool` docstring is framework contract at decoration time — not signature restatement, always keep.

## What this doesn't cover

- **Onion port.** V1 is flat — models, tools, crew builders, entry point, all at module scope. Deferred to V2. Onion shape would extract domain (Pydantic wire format + `DietaryRestriction` as pure types), application (crew factories as use-case orchestrations), infrastructure (Anthropic SDK adapter, Gradio adapter, CrewAI adapter).
- **Test coverage.** No pytest suite for V1. Highest-value tests would be `_encode_image` format-detection table (each supported format → correct media_type, unsupported → `ValueError`), `filter_by_dietary_restriction` short-circuit on `"none"`, and `analyse_food`'s dispatch/narrowing branches. Pure, deterministic, guard the seams most likely to drift before any LLM is involved.
- **Windows path round-trip via LLM tool-arg.** Paths flow through `{uploaded_image}` interpolation into the task description, then the agent re-emits them verbatim in a JSON tool call. Empirically holds on Windows across smoke tests, but inherent to the vision-in-tool + path-in-description pattern. Correct V2 fix is binding the path outside the LLM's chosen args (closure/context), not string round-trip.
- **Median-of-N timing.** Four wall-clock smoke tests, no timing measurements. Same discipline line as L33/L34/L35 — directional not comparative.
- **Recipe `instructions: str` vs `list[str]`.** Model produces step-numbered prose in one string; Gradio Markdown doesn't render inline `2.` `3.` `4.` as list continuations. Cosmetic rendering issue. Fix would be `list[str]` in the Pydantic model + render as proper Markdown list in `format_recipe_output`. Deferred to V2.
- **Framework un-collapse for single-agent workflows.** Analysis workflow runs two LLM calls (tool + agent) when one suffices. V2 shape would bypass the CrewAI wrapper for single-agent single-tool cases and call the tool directly from `nutricoach.py`'s dispatcher. Preserves framework consistency for recipe workflow, drops the tax where it doesn't earn its cost.

---

**Completed:** 21 July 2026