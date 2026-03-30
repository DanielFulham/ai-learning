# Lab 1: Prompt Engineering and LangChain — My Solutions

## Exercise 1: Basic Prompting with Parameter Tuning

Experimented with temperature, top_k and max_new_tokens to observe how
output changes with different parameter combinations.

```python
params = {
    "max_new_tokens": 1000,
    "min_new_tokens": 100,
    "temperature": 0.3,
    "top_p": 0.2,
    "top_k": 40
}

prompt = "The future of artificial intelligence is "
response = llm_model(prompt, params)
```

## Exercise 2: Zero-Shot Prompting

Three zero-shot prompts covering classification, fact-checking, and translation.

```python
movie_review_prompt = """Classify if this movie is good or bad: 
    'Shrek was terrible it is about dinosaurs.'
    Answer:
"""

climate_change_prompt = """Is this climate change question correct:
    'We will all live in the sea next year'
    Answer:
"""

translation_prompt = """Convert english text to spanish:
    'I like apples'
    Answer:
"""

responses = {}
responses["movie_review"] = llm_model(movie_review_prompt)
responses["climate_change"] = llm_model(climate_change_prompt)
responses["translation"] = llm_model(translation_prompt)

for prompt_type, response in responses.items():
    print(f"=== {prompt_type.upper()} RESPONSE ===")
    print(response)
```

## Exercise 3: One-Shot Prompting

Used English to Irish translation as a one-shot example — chose Irish
instead of French to make the example my own.

```python
prompt = """Here is an example of translating a sentence from English to Irish:

    English: "How are you?"
    Irish: "Conas atá tú?"
    
    Now, translate the following word from English to Irish:
    
    English: "Hello"
"""
response = llm_model(prompt, params)
```

## Exercise 4: Chain-of-Thought Prompting

Two CoT prompts — a decision-making scenario and a process breakdown.

```python
decision_making_prompt = """Consider the problem: 
    'A student is considering studying tonight or going to the movies 
    with a friend, they have a test in two days'
    Break down each step of your thinking
"""

sandwich_making_prompt = """Consider the problem: 
    'a student has bread, a butter knife, ham slices and cheese slices'
    Break down each step of your process
"""

responses["decision_making"] = llm_model(decision_making_prompt, params)
responses["sandwich_making"] = llm_model(sandwich_making_prompt, params)
```

## Exercise 5: LCEL Chain with JsonOutputParser

Built a product review analyser using LCEL and structured JSON output.
Key lesson: prompt and parser must agree on format — JsonOutputParser
requires the prompt to explicitly request JSON.

```python
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import JsonOutputParser

model_id = "meta-llama/llama-3-405b-instruct"

parameters = {
    GenParams.MAX_NEW_TOKENS: 1024,
    GenParams.TEMPERATURE: 0.1,
}

llm = WatsonxLLM(
    model_id=model_id,
    url="https://us-south.ml.cloud.ibm.com",
    project_id="skills-network",
    params=parameters
)

template = """
Analyze the following product review:
"{review}"

You must respond with ONLY a JSON object. No introduction, no explanation,
no examples. Start your response with {{ and end with }}.

{{
    "sentiment": "positive, negative, or neutral",
    "key_features": ["feature1", "feature2"],
    "summary": "one-sentence summary"
}}
"""

product_review_prompt = PromptTemplate.from_template(template)

def format_review_prompt(variables):
    return product_review_prompt.format(**variables)

review_analysis_chain = (
    RunnableLambda(format_review_prompt)
    | llm
    | JsonOutputParser()
)

reviews = [
    "I love this smartphone! The camera quality is exceptional and the battery \
lasts all day. The only downside is that it heats up a bit during gaming.",
    "This laptop is terrible. It's slow, crashes frequently, and the keyboard \
stopped working after just two months. Customer service was unhelpful."
]

for review in reviews:
    print(f"==== Review ====")
    result = review_analysis_chain.invoke({"review": review})
    print(result)
    print()
```

**Output:**
```
==== Review ====
{'sentiment': 'positive', 'key_features': ['camera quality', 'battery life', 
'heating issue'], 'summary': 'The reviewer loves the smartphone, praising its 
camera and battery, but notes a heating issue during gaming.'}

==== Review ====
{'sentiment': 'negative', 'key_features': ['performance', 'reliability', 
'customer service'], 'summary': 'The reviewer had a very poor experience with 
the laptop, citing issues with performance, reliability, and customer service.'}
```

## Key Learnings

- Model versions deprecate — had to update `granite-3-2-8b` to `granite-3-3-8b`
  mid-lab when hitting WMLClientError
- `JsonOutputParser` fails silently if the model doesn't return valid JSON —
  the prompt must explicitly instruct JSON-only output
- Model choice matters for structured output — Llama 3 405b followed JSON
  instructions more reliably than Granite for this task
- Temperature 0.1 + explicit JSON instructions = consistent structured output
