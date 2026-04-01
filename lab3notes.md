# Course 1 — Lab 3: Build Your First GenAI Application The Right Way

## Lab Overview

**Course:** IBM RAG and Agentic AI Professional Certificate — Course 1  
**Lab:** Build Your First GenAI Application The Right Way  
**Completed:** 1 April 2026  
**Time taken:** ~4 hours

Built a Flask API integrating three LLM backends (Llama, Granite, Mistral) via IBM Watsonx,
using LangChain for prompt templating and structured JSON output via Pydantic models.

---

## Key Concepts

### Special Tokens & Prompt Formatting

Each model family uses different special tokens to define turn structure. Without them
the model has no boundary signal and will keep generating indefinitely.

| Model | System | User | End of turn |
|---|---|---|---|
| Llama | `<\|start_header_id\|>system<\|end_header_id\|>` | `<\|start_header_id\|>user<\|end_header_id\|>` | `<\|eot_id\|>` |
| Granite | `<\|system\|>` | `<\|user\|>` | `<\|assistant\|>` |
| Mistral | `<s>[INST]` | inline | `[/INST]` |

**Production insight:** `ChatWatsonx` and `ChatPromptTemplate` handle these automatically.
Never manage special tokens manually in production code.

---

### Multi-Model Architecture
```python
# config.py — single source of truth for model IDs
LLAMA_MODEL_ID = "meta-llama/llama-4-maverick-17b-128e-instruct-fp8"
GRANITE_MODEL_ID = "ibm/granite-4-h-small"
MISTRAL_MODEL_ID = "mistralai/mistral-small-3-1-24b-instruct-2503"

# model.py — factory function, one line to add a new model
def initialize_model(model_id):
    return ChatWatsonx(
        model_id=model_id,
        url=CREDENTIALS["url"],
        project_id=CREDENTIALS["project_id"],
        params=PARAMETERS
    )
```

---

### JSON Output Parser + Pydantic

Pydantic schema enforces structured output from the LLM. Parser fails if the model
returns any text outside the JSON structure — prompt must include format instructions.
```python
class AIResponse(BaseModel):
    summary: str = Field(description="Summary of the AI's response")
    sentiment: int = Field(description="Sentiment score 0 (negative) to 100 (positive)")
    category: str = Field(description="Category of the inquiry")
    action: str = Field(description="Recommended action for the support rep")

json_parser = JsonOutputParser(pydantic_object=AIResponse)

# {format_prompt} MUST be in the template system block
def get_ai_response(model, template, system_prompt, user_prompt):
    chain = template | model | json_parser
    return chain.invoke({
        'system_prompt': system_prompt,
        'user_prompt': user_prompt,
        'format_prompt': json_parser.get_format_instructions()
    })
```

**Key lesson:** parser returns a Python `dict`, not an `AIMessage` object.
Access fields directly — `result['summary']`, not `result.content`.

---

### Flask Integration
```python
@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    user_message = data.get('message')
    model = data.get('model')

    if not user_message or not model:
        return jsonify({"error": "Missing message or model selection"}), 400

    start_time = time.time()

    try:
        if model == 'llama':
            result = llama_response(system_prompt, user_message)
        elif model == 'granite':
            result = granite_response(system_prompt, user_message)
        elif model == 'mistral':
            result = mistral_response(system_prompt, user_message)
        else:
            return jsonify({"error": "Invalid model selection"}), 400

        result['duration'] = time.time() - start_time
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
```

---

## Bugs Hit & Fixed

| Bug | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: langchain.prompts` | Old import path | Use `langchain_core.prompts` |
| `JSONDecodeError: Expecting value` | `{format_prompt}` missing from Llama template | Add `\n{format_prompt}` to system block |
| `AttributeError: 'dict' has no attribute 'content'` | Parser returns dict not AIMessage | Access fields directly, no `.content` |
| UI showing "undefined" | Frontend reading `response` field, removed in exercise | Add `response` field back to `AIResponse` |
| Model not supported error | Course model IDs outdated | Use supported models list from error output |

---

## Model Comparison (same prompt, three models)

**Prompt:** "What is the capital of Canada? Tell me a cool fact about it."

| Model | Output quality | Structured JSON | Notes |
|---|---|---|---|
| Llama | Good | ✅ | Summarised the question not the answer |
| Granite | Best | ✅ | Most accurate summary, honest action field |
| Mistral | Good | ✅ | Adds unrequested markdown formatting |

**Conclusion:** Granite is the best default for structured output tasks — clean,
accurate, no formatting noise.

---

## Temperature & Sampling Notes

- `greedy` decoding (temperature 0) = always picks highest probability token
- Good for structured JSON output — deterministic, consistent
- Higher temperature needed for creative tasks (campaign copy generation)
- Monte Carlo parallel: greedy = running one simulation, sampling = running many

---

## Rovers Connection

- **Granite** is the default model choice for the FAQ chatbot
- **Caching** is production-critical — same question 1000 times on match day
  should not hit the LLM 1000 times
- **`AIResponse` pattern** maps directly to Rovers structured outputs:
  - Fan segmentation results
  - Propensity scores
  - Campaign copy with metadata
- **Flask `/generate` pattern** maps to Lambda handler in Rovers architecture

---

## Files

| File | Purpose |
|---|---|
| `config.py` | Model IDs, Watsonx credentials, generation parameters |
| `model.py` | Model init, prompt templates, chains, response functions |
| `app.py` | Flask routes, error handling, model routing |
| `llm_test.py` | Direct model testing without Flask |
| `templates/index.html` | Chat UI |
| `static/script.js` | Frontend — calls `/generate` endpoint |

---

## Key Classes Reference
```python
# Model wrapper
from langchain_ibm import ChatWatsonx

# Prompt templating
from langchain_core.prompts import PromptTemplate

# Structured output
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser

# Flask
from flask import Flask, request, jsonify, render_template
```
