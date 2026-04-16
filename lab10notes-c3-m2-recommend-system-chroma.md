
# Lab 10 Notes — Food Recommendation System Using Chroma DB
## Course 3, Module 2 Lab 2

---

## What This Lab Covers

Three distinct approaches to similarity search and conversational AI over a 
food dataset with nutritional information, ingredients, cooking methods, and 
taste profiles. Covers interactive CLI search, metadata filtering, and a 
RAG chatbot combining ChromaDB retrieval with IBM Granite generation.

Also used as the basis for practical exercises extending the lab code with 
search history tracking, calorie budget checking, and result count comparison.

---

## Key Concepts

**Rich document construction from structured data:**

Same dual-layer pattern as the employee lab, applied to a richer dataset. 
Structured food fields are serialised into a natural language string for 
embedding, while filterable attributes go into metadata:

```python
text = f"Name: {food['food_name']}. "
text += f"Description: {food.get('food_description', '')}. "
text += f"Ingredients: {', '.join(food.get('food_ingredients', []))}. "
text += f"Cuisine: {food.get('cuisine_type', 'Unknown')}. "
text += f"Cooking method: {food.get('cooking_method', '')}. "
text += f"Taste and features: {food.get('taste_profile', '')}. "
```

**Metadata pre-processing in load step:**

The `load_food_data()` function does double duty — reads the JSON file and 
normalises the data structure. The nested `food_features` dict is flattened 
into a `taste_profile` string at load time, before any ChromaDB interaction:

```python
if 'food_features' in item and isinstance(item['food_features'], dict):
    taste_features = []
    for key, value in item['food_features'].items():
        if value:
            taste_features.append(str(value))
    item['taste_profile'] = ', '.join(taste_features)
else:
    item['taste_profile'] = ''
```

Python dicts are dynamic — assigning to a new key creates it in place. 
The `else` branch ensures `taste_profile` exists on every item regardless 
of whether `food_features` was present in the source JSON.

**Similarity score conversion:**

ChromaDB cosine space returns distances where 0 = identical, 1 = completely 
different. Convert to similarity score for display:

```python
similarity_score = 1 - results['distances'][0][i]
```

**Numeric range filters:**

Metadata filters work on both string exact match and numeric range operators:

```python
# String exact match
where={"cuisine_type": "Italian"}

# Numeric range
where={"calories": {"$lte": 300}}

# Combined
where={"$and": [
    {"cuisine_type": "Italian"},
    {"calories": {"$lte": 600}}
]}
```

**RAG pipeline structure:**

```
user query
    → perform_similarity_search()   — ChromaDB retrieval
    → prepare_context_for_llm()     — formats retrieved docs for prompt
    → generate_llm_rag_response()   — LLM generates natural language answer
    → fallback_response()           — rule-based fallback if LLM fails
```

Embedding (all-MiniLM-L6-v2) and generation (IBM Granite) are completely 
separate models with separate jobs. The embedding model is never swapped 
without rebuilding the index. The generation model can be swapped freely.

---

## Observations

**Vector search always returns a result:**

Searching "blah blah blah" returned 5 results with scores of 8-10%. 
Without a similarity threshold guard, gibberish queries return confidently 
presented noise. The quality gate is code, not ChromaDB:

```python
if not results or results[0]['similarity_score'] < 0.35:
    return fallback_message
```

Threshold value should be calibrated from real query data, not guessed. 
The 8% vs 62% spread seen in this lab gives empirical grounding for 
where the cutoff should sit.

**Metadata filters reduce candidate pool — affects recall:**

Calorie-filtered search for "healthy light meal" under 300 calories returned 
top score of 34.5% — significantly lower than unfiltered searches hitting 62%. 
When the filter is too restrictive, you get the best of a bad pool rather 
than the best overall.

Production pattern — decouple result count from quality threshold:

```python
MAX_RESULTS = 5
QUALITY_THRESHOLD = 0.40

results = search_collection(query, n_results=MAX_RESULTS)
quality_results = [r for r in results if r['similarity_score'] >= QUALITY_THRESHOLD]
```

**Metadata filters are exact string match, case sensitive:**

`cuisine_type = "american"` does not match stored value `"American"`. 
Normalise at ingest time, not query time:

```python
"cuisine_type": food.get("cuisine_type", "unknown").lower().strip()
```

Or use uppercase controlled values for enum-like fields:

```python
"category": "WOMENS_TICKETS"  # visually distinct, harder to accidentally mix case
```

**Raw user input should never go directly into a metadata filter:**

Fan types "I fancy something Italian" — exact match on `cuisine_type` fails. 
Intent must be extracted and mapped to a controlled value before filtering:

```
Raw input: "I fancy something Italian"
    → intent extraction (keyword rules or LLM classifier)
    → controlled value: "ITALIAN"
    → where={"cuisine_type": "ITALIAN"}
```

**Conversation history tracked but not used:**

The lab's `enhanced_rag_food_chatbot` maintains a `conversation_history` list 
but never passes it to the LLM prompt. Each query is independent — no memory 
between turns despite the list existing.

For a genuine conversational experience, pass the last N turns into the prompt:

```python
def generate_response(query: str, results: list, history: list) -> str:
    history_text = ""
    if history:
        history_text = "Previous conversation:\n"
        history_text += "\n".join([f"Fan: {q}" for q in history[-3:]])
        history_text += "\n\n"
    
    prompt = f"""{history_text}
Current question: {query}

Relevant information:
{prepare_context(results)}

Answer:"""
```

Three previous turns in-memory is the right V1.5 pattern — no DynamoDB 
required, covers the most common fan follow-up question pattern.

---

## Production Notes

- Similarity threshold is a quality gate, not a result count cap — apply 
  post-retrieval, not as n_results reduction
- Two-query pattern for split UX (within budget vs over budget): run filtered 
  and unfiltered searches in parallel, deduplicate by ID before display
- Metadata schema design is an upfront architectural decision — you can only 
  filter on what you stored at index time
- Metadata values are controlled enums, not free text — normalise at ingest, 
  use uppercase for enum-like fields
- Embedding model and generation model are independent choices — swap the 
  generation model freely, never swap the embedding model without rebuilding
- For Lambda deployment: use external embedding API (Bedrock Titan or 
  Cloudflare Workers AI) — local SentenceTransformer exceeds Lambda layer limit
- Pin embedding model to a specific version — floating aliases can change 
  silently and cause retrieval regression with no error thrown

---

## Recommended Architecture for Rovers Chatbot (from this lab)

Clean separation of concerns — one job per file:

```
main.py           — composition root, wires offline and online phases
data_loader.py    — load + validate source documents (Pydantic models)
vector_store.py   — create collection, populate, rebuild on high delete ratio
retriever.py      — semantic search, filtered search, similarity threshold gate
generator.py      — prepare context, call LLM, fallback response
chatbot.py        — conversation loop, session history (last 3 turns)
config.py         — environment variables via dotenv
```

**Offline phase (runs once):**
```
data_loader → vector_store
```

**Online phase (per fan query):**
```
chatbot → retriever → generator
```

---

## Exercises Completed

**Exercise 1 — Search history tracking:**
Added `search_history` global, `show_search_history()` display function, 
and `history` command handler to `interactive_search.py`. History appended 
only on successful searches — gibberish and failed queries not tracked.

**Exercise 2 — Calorie budget checker (`calorie_checker.py`):**
Two-query pattern — filtered search for within-budget results, unfiltered 
search for over-budget alternatives. Deduplication by food_id prevents 
items appearing in both lists.

**Exercise 3 — Result count comparison (`result_limiter.py`):**
Compared n_results of 1, 3, 5, 10 across multiple queries. Score spread 
(top result minus bottom result) is the key metric — wide spread indicates 
the quality threshold should cut well before the max result count.

---

## Files

- `interactive_search.py` — CLI food search with history tracking
- `advanced_search.py` — filtered similarity search with cuisine and calorie filters
- `enhanced_rag_chatbot.py` — RAG chatbot with IBM Granite integration
- `system_comparison.py` — side-by-side comparison of all three approaches
- `shared_functions.py` — shared ChromaDB utilities
- `calorie_checker.py` — exercise 1: budget-based food search
- `result_limiter.py` — exercise 3: result count comparison
- `config.py` — environment variable loader
- `FoodDataSet.json` — 185 food items with nutritional and taste data