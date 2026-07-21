"""Tools for the NourishBot dietary crew.

Four tools split by capability:

- `extract_ingredients` — vision call, returns comma-separated ingredient text.
- `filter_ingredients` — pure Python string splitting, no LLM.
- `filter_by_dietary_restriction` — text call. Short-circuits on the
  DietaryRestriction.NONE picklist value ("none").
- `analyse_nutrition` — vision + structured output in ONE `messages.parse()`
  call, returns a typed NutrientAnalysisOutput. Collapses the IBM shape's
  vision-in-tool → prose → agent-restructures-to-JSON two-call flow into one.

Design notes:

- Multimodal is pushed into tools, not surfaced via CrewAI's LLM abstraction,
  because `Agent(multimodal=True)` is documented but broken across CrewAI
  versions (issues #4016, #2565, #2475, #2541, #2642). Direct SDK calls are
  the correct production workaround.
- Content-block shape is Anthropic-native: `{"type": "image", "source": ...}`,
  not the OpenAI/watsonx `{"type": "image_url", ...}` shape.
- media_type is detected from the file via Pillow rather than hardcoded to
  image/jpeg. Anthropic accepts jpeg, png, gif, webp; anything else fails
  loud.
- Framework-boundary TypedDict construction (ImageBlockParam, TextBlockParam,
  MessageParam) is concentrated in `_build_vision_message` so pyright verifies
  each dict-literal against its target TypedDict at one seam rather than
  repeated at every call site (L34 boundary-narrowing discipline).
- Model string and Anthropic client factory live in `llm.py` — same source
  of truth used by `crew.py` for CrewAI's native Anthropic provider, so the
  model version can't drift between direct-SDK calls (this file) and the
  framework-abstracted calls (agents).
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Final, Literal

from anthropic.types import (
    Base64ImageSourceParam,
    ImageBlockParam,
    MessageParam,
    TextBlockParam,
)
from crewai.tools import tool
from PIL import Image

from llm import MODEL, get_client
from models import DietaryRestriction, NutrientAnalysisOutput

_MediaType = Literal["image/jpeg", "image/png", "image/gif", "image/webp"]

_PILLOW_FORMAT_TO_MEDIA_TYPE: Final[dict[str, _MediaType]] = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "GIF": "image/gif",
    "WEBP": "image/webp",
}


def _encode_image(image_path: str) -> tuple[str, _MediaType]:
    """Raises FileNotFoundError if the path is missing; ValueError if the
    image format isn't one Anthropic accepts."""
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"No image file at path: {image_path}")

    with Image.open(path) as img:
        pillow_format = img.format
    if pillow_format not in _PILLOW_FORMAT_TO_MEDIA_TYPE:
        raise ValueError(
            f"Unsupported image format {pillow_format!r} at {image_path}. "
            f"Supported: {sorted(_PILLOW_FORMAT_TO_MEDIA_TYPE)}"
        )
    media_type = _PILLOW_FORMAT_TO_MEDIA_TYPE[pillow_format]

    encoded = base64.standard_b64encode(path.read_bytes()).decode("utf-8")
    return encoded, media_type


def _build_vision_message(
    encoded_image: str, media_type: _MediaType, prompt: str
) -> list[MessageParam]:
    """Concentrates Anthropic SDK TypedDict construction in one helper so
    the framework-boundary typing plumbing is confined here rather than
    repeated at each tool call site (L34 boundary-narrowing discipline).
    """
    source: Base64ImageSourceParam = {
        "type": "base64",
        "media_type": media_type,
        "data": encoded_image,
    }
    image_block: ImageBlockParam = {"type": "image", "source": source}
    text_block: TextBlockParam = {"type": "text", "text": prompt}
    return [{"role": "user", "content": [image_block, text_block]}]


_EXTRACT_PROMPT: Final[str] = (
    "Identify each distinct food ingredient visible in this image. "
    "Return them as a comma-separated list, lowercased, singular where possible. "
    "Include ingredients that are clearly identifiable; skip garnishes and "
    "seasonings unless they are prominent. Do not include any preamble, "
    "commentary, or formatting — just the comma-separated list."
)


@tool("Extract ingredients from a food image")
def extract_ingredients(image_path: str) -> str:
    encoded, media_type = _encode_image(image_path)
    client = get_client()

    response = client.messages.create(
        model=MODEL,
        max_tokens=300,
        messages=_build_vision_message(encoded, media_type, _EXTRACT_PROMPT),
    )

    first_block = response.content[0]
    if first_block.type != "text":
        raise ValueError(
            f"Expected text response from extract_ingredients, got {first_block.type!r}"
        )
    return first_block.text


@tool("Clean and normalise a raw ingredient list")
def filter_ingredients(raw_ingredients: str) -> list[str]:
    """Pure Python; no LLM call."""
    return [
        ingredient.strip().lower()
        for ingredient in raw_ingredients.split(",")
        if ingredient.strip()
    ]


_DIETARY_FILTER_PROMPT_TEMPLATE: Final[str] = (
    "You are an AI nutritionist specialised in dietary restrictions. "
    "Given the following list of ingredients: {ingredients_csv}, "
    "and the dietary restriction: {dietary_restriction}, "
    "remove any ingredient that does not comply with this restriction. "
    "Return only the compliant ingredients as a comma-separated list "
    "with no additional commentary."
)


@tool("Filter an ingredient list to comply with a dietary restriction")
def filter_by_dietary_restriction(
    ingredients: list[str], dietary_restriction: str
) -> list[str]:
    """Short-circuits and returns `ingredients` unchanged when
    `dietary_restriction` is empty or DietaryRestriction.NONE."""
    if not dietary_restriction or dietary_restriction == DietaryRestriction.NONE:
        return ingredients

    client = get_client()
    prompt = _DIETARY_FILTER_PROMPT_TEMPLATE.format(
        ingredients_csv=", ".join(ingredients),
        dietary_restriction=dietary_restriction,
    )

    messages: list[MessageParam] = [{"role": "user", "content": prompt}]
    response = client.messages.create(
        model=MODEL,
        max_tokens=150,
        messages=messages,
    )

    first_block = response.content[0]
    if first_block.type != "text":
        raise ValueError(
            f"Expected text response from filter_by_dietary_restriction, "
            f"got {first_block.type!r}"
        )

    return [
        item.strip().lower()
        for item in first_block.text.split(",")
        if item.strip()
    ]


_ANALYSE_PROMPT: Final[str] = (
    "You are an expert nutritionist. Analyse the food items in this image and "
    "produce a structured nutritional assessment. Identify the dish, estimate "
    "portion size and per-portion calories, provide macronutrient totals "
    "(protein, carbohydrates, fats) with units, list 3-5 key vitamins with "
    "percentage of Daily Value, list 3-5 key minerals with amounts, and give a "
    "2-3 sentence health evaluation of the meal. All estimates are approximate."
)


@tool("Analyse the nutrition of a meal from a food image")
def analyse_nutrition(image_path: str) -> NutrientAnalysisOutput:
    """Vision + structured output collapsed into a single `messages.parse()`
    call. Returns a typed instance directly — the agent's LLM doesn't need
    to restructure prose into JSON downstream."""
    encoded, media_type = _encode_image(image_path)
    client = get_client()

    response = client.messages.parse(
        model=MODEL,
        max_tokens=1024,
        messages=_build_vision_message(encoded, media_type, _ANALYSE_PROMPT),
        output_format=NutrientAnalysisOutput,
    )

    parsed = response.parsed_output
    if parsed is None:
        raise ValueError(
            "analyse_nutrition: messages.parse returned no parsed_output "
            "(possible refusal or max_tokens truncation)"
        )
    return parsed
