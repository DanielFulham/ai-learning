from operator import add
from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

from shared import llm, extract_content

TRANSLATE_FRENCH = "translate_french"
TRANSLATE_SPANISH = "translate_spanish"
TRANSLATE_JAPANESE = "translate_japanese"
AGGREGATOR = "aggregator"


class Translation(TypedDict):
    language: str
    text: str


class ParallelState(TypedDict):
    text: str
    translations: Annotated[list[Translation], add]
    combined_output: str


async def translate_french(state: ParallelState) -> dict[str, list[Translation]]:
    response = await llm.ainvoke(f"Translate the following text to French:\n\n{state['text']}")
    content = extract_content(response)
    return {"translations": [{"language": "french", "text": content.strip()}]}


async def translate_spanish(state: ParallelState) -> dict[str, list[Translation]]:
    response = await llm.ainvoke(f"Translate the following text to Spanish:\n\n{state['text']}")
    content = extract_content(response)
    return {"translations": [{"language": "spanish", "text": content.strip()}]}


async def translate_japanese(state: ParallelState) -> dict[str, list[Translation]]:
    response = await llm.ainvoke(f"Translate the following text to Japanese:\n\n{state['text']}")
    content = extract_content(response)
    return {"translations": [{"language": "japanese", "text": content.strip()}]}


def aggregator(state: ParallelState) -> dict[str, str]:
    parts = [f"Original Text: {state['text']}\n"]
    for translation in state["translations"]:
        parts.append(f"{translation['language'].capitalize()}: {translation['text']}")
    combined = "\n\n".join(parts) + "\n"
    return {"combined_output": combined}


workflow = StateGraph(ParallelState)
workflow.add_node(TRANSLATE_FRENCH, translate_french)
workflow.add_node(TRANSLATE_SPANISH, translate_spanish)
workflow.add_node(TRANSLATE_JAPANESE, translate_japanese)
workflow.add_node(AGGREGATOR, aggregator)
workflow.add_edge(START, TRANSLATE_FRENCH)
workflow.add_edge(START, TRANSLATE_SPANISH)
workflow.add_edge(START, TRANSLATE_JAPANESE)
workflow.add_edge(TRANSLATE_FRENCH, AGGREGATOR)
workflow.add_edge(TRANSLATE_SPANISH, AGGREGATOR)
workflow.add_edge(TRANSLATE_JAPANESE, AGGREGATOR)
workflow.add_edge(AGGREGATOR, END)

app = workflow.compile(checkpointer=InMemorySaver())