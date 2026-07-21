"""Gradio entry point for the NourishBot dietary crew.

Wires the Gradio UI to `build_recipe_crew()` and `build_analysis_crew()` from
crew.py, dispatching per workflow radio value. Typed `.pydantic` access on
CrewOutput propagates type safety from the tool layer through the crew layer
into the presentation layer (L35 discipline extended to the UI boundary).

Design notes:

- `gr.Image(type="filepath")` — Gradio hands us a temp file path directly,
  which is exactly the shape our vision tools accept. IBM's `type="pil"`
  needed an intermediate `image.save("uploaded_image.jpg")` write with
  concurrency and filename hazards.
- `gr.Dropdown` on `DietaryRestriction` values — L35 StrEnum picklist
  discipline applied at the UI boundary. No free-text drift into the prompt
  template.
- CSS/JS on `demo.launch()`, not on `gr.Blocks()`. Gradio 6 moved these
  parameters (Gradio 6 migration guide, changelog #12217). IBM's shape
  fires deprecation warnings and will eventually error.
- Crews constructed once at module scope, reused across kickoffs. Two
  Crew objects held as module-scope constants; kickoff-inputs vary per
  request (L35 kickoff-time inputs pattern).
- No try/except around crew.kickoff — Gradio surfaces exceptions in the UI
  natively, which is honest fail-loud behaviour matching the tools' posture.
- Windows-path round-trip via LLM tool-arg: paths flow through
  {uploaded_image} interpolation into the task description, then the agent
  re-emits them verbatim in a JSON tool call. Empirically holds on Windows
  across smoke tests; correct V2 fix is binding the path outside the LLM's
  chosen args (closure/context), not string-round-trip.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import gradio as gr
from crewai import CrewOutput
from dotenv import load_dotenv
from gradio.themes import Citrus

from crew import (
    AnalysisInputs,
    RecipeInputs,
    build_analysis_crew,
    build_recipe_crew,
)
from models import (
    DietaryRestriction,
    NutrientAnalysisOutput,
    RecipeSuggestionOutput,
)

load_dotenv()

_recipe_crew: Final = build_recipe_crew()
_analysis_crew: Final = build_analysis_crew()

_EXAMPLES_DIR: Final[Path] = Path(__file__).parent / "examples"

_DIETARY_CHOICES: Final[list[str]] = [e.value for e in DietaryRestriction]


def format_recipe_output(output: RecipeSuggestionOutput) -> str:
    md = "## 🍽 Recipe Ideas\n\n"

    if not output.recipes:
        return md + "No recipes could be generated."

    for idx, recipe in enumerate(output.recipes, 1):
        md += f"### {idx}. {recipe.title}\n\n"
        md += "**Ingredients:**\n\n"
        md += "| Ingredient |\n|------------|\n"
        for ingredient in recipe.ingredients:
            md += f"| {ingredient} |\n"
        md += "\n"
        md += f"**Instructions:**\n\n{recipe.instructions}\n\n"
        md += f"**Calorie estimate:** {recipe.calorie_estimate} kcal\n\n"
        md += "---\n\n"

    return md


def format_analysis_output(output: NutrientAnalysisOutput) -> str:
    md = "## 🥗 Nutritional Analysis\n\n"

    if output.dish:
        md += f"**Dish:** {output.dish}\n\n"
    if output.portion_size:
        md += f"**Portion size:** {output.portion_size}\n\n"
    if output.estimated_calories is not None:
        md += f"**Estimated calories:** {output.estimated_calories} kcal\n\n"

    md += "**Nutrient breakdown:**\n\n"
    md += "| Nutrient | Amount |\n|----------|--------|\n"
    if output.nutrients.protein:
        md += f"| Protein | {output.nutrients.protein} |\n"
    if output.nutrients.carbohydrates:
        md += f"| Carbohydrates | {output.nutrients.carbohydrates} |\n"
    if output.nutrients.fats:
        md += f"| Fats | {output.nutrients.fats} |\n"

    if output.nutrients.vitamins:
        md += "\n**Vitamins:**\n\n"
        md += "| Vitamin | %DV |\n|---------|-----|\n"
        for v in output.nutrients.vitamins:
            md += f"| {v.name} | {v.percentage_dv} |\n"

    if output.nutrients.minerals:
        md += "\n**Minerals:**\n\n"
        md += "| Mineral | Amount |\n|---------|--------|\n"
        for m in output.nutrients.minerals:
            md += f"| {m.name} | {m.amount} |\n"

    if output.health_evaluation:
        md += f"\n**Health evaluation:**\n\n{output.health_evaluation}\n"

    return md


def analyse_food(
    image_path: str | None,
    dietary_restriction: str,
    workflow_type: str,
) -> str:
    if image_path is None:
        return "Please upload an image first."
    if workflow_type not in ("recipe", "analysis"):
        return (
            f"Invalid workflow type: {workflow_type!r}. "
            f"Choose 'recipe' or 'analysis'."
        )

    if workflow_type == "recipe":
        recipe_inputs: RecipeInputs = {
            "uploaded_image": image_path,
            "dietary_restrictions": dietary_restriction,
        }
        result = _recipe_crew.kickoff(inputs=dict(recipe_inputs))
    else:  # workflow_type == "analysis"
        analysis_inputs: AnalysisInputs = {"uploaded_image": image_path}
        result = _analysis_crew.kickoff(inputs=dict(analysis_inputs))

    # CrewAI kickoff returns CrewOutput | CrewStreamingOutput (L34 finding).
    # V1 doesn't stream so CrewOutput is expected; narrow at the seam.
    if not isinstance(result, CrewOutput):
        return f"Unexpected result type from crew.kickoff: {type(result).__name__}"

    if result.pydantic is None:
        return "Crew produced no structured output. Check the terminal logs."

    if workflow_type == "recipe" and isinstance(result.pydantic, RecipeSuggestionOutput):
        return format_recipe_output(result.pydantic)
    if workflow_type == "analysis" and isinstance(result.pydantic, NutrientAnalysisOutput):
        return format_analysis_output(result.pydantic)

    return f"Unexpected structured output type: {type(result.pydantic).__name__}"


_CSS: Final[str] = """
.title {
    font-size: 1.5em !important;
    text-align: center !important;
    color: #FFD700;
}

.text {
    text-align: center;
}
"""

_JS: Final[str] = """
function createGradioAnimation() {
    var container = document.createElement('div');
    container.id = 'gradio-animation';
    container.style.fontSize = '2em';
    container.style.fontWeight = 'bold';
    container.style.textAlign = 'center';
    container.style.marginBottom = '20px';
    container.style.color = '#eba93f';

    var text = 'Welcome to your AI NourishBot!';
    for (var i = 0; i < text.length; i++) {
        (function(i){
            setTimeout(function(){
                var letter = document.createElement('span');
                letter.style.opacity = '0';
                letter.style.transition = 'opacity 0.1s';
                letter.innerText = text[i];

                container.appendChild(letter);

                setTimeout(function() {
                    letter.style.opacity = '0.9';
                }, 50);
            }, i * 250);
        })(i);
    }

    var gradioContainer = document.querySelector('.gradio-container');
    gradioContainer.insertBefore(container, gradioContainer.firstChild);

    return 'Animation created';
}
"""

with gr.Blocks() as demo:
    gr.Markdown("# How it works", elem_classes="title")
    gr.Markdown(
        "Upload a fridge or pantry photo, pick a dietary preference, "
        "and choose 'recipe' to get recipe ideas from what you have.",
        elem_classes="text",
    )
    gr.Markdown(
        "Upload a meal photo, leave dietary preference on 'none', "
        "and choose 'analysis' to get nutritional insights.",
        elem_classes="text",
    )
    gr.Markdown(
        "Or select one of the examples below to autofill the inputs "
        "and click Analyze.",
        elem_classes="text",
    )

    with gr.Row():
        with gr.Column(scale=1, min_width=400):
            gr.Markdown("## Inputs", elem_classes="title")
            image_input = gr.Image(type="filepath", label="Upload image")
            dietary_input = gr.Dropdown(
                choices=_DIETARY_CHOICES,
                value=DietaryRestriction.NONE.value,
                label="Dietary restriction",
            )
            workflow_radio = gr.Radio(
                choices=["recipe", "analysis"], label="Workflow type"
            )
            submit_btn = gr.Button("Analyze")

        with gr.Column(scale=2, min_width=600):
            gr.Examples(
                examples=[
                    [str(_EXAMPLES_DIR / "food-1.jpg"), DietaryRestriction.VEGAN.value, "recipe"],
                    [str(_EXAMPLES_DIR / "food-2.jpg"), DietaryRestriction.NONE.value, "analysis"],
                    [str(_EXAMPLES_DIR / "food-3.jpg"), DietaryRestriction.KETO.value, "recipe"],
                    [str(_EXAMPLES_DIR / "food-4.jpg"), DietaryRestriction.NONE.value, "analysis"],
                ],
                inputs=[image_input, dietary_input, workflow_radio],
                label="Try an example",
            )
            gr.Markdown("## Results", elem_classes="title")
            result_display = gr.Markdown(
                "<div style='border: 1px solid #ccc; padding: 1rem; "
                "text-align: center; color: #666;'>No results yet</div>",
                height=500,
            )

    submit_btn.click(
        fn=analyse_food,
        inputs=[image_input, dietary_input, workflow_radio],
        outputs=result_display,
    )


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=5000,
        theme=Citrus(),
        css=_CSS,
        js=_JS,
    )
