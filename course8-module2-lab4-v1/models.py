"""Pydantic wire-format models for the NourishBot dietary crew.

Typed contracts between the vision tools, the CrewAI tasks, and the Gradio
UI. Each output model binds to a task via `output_pydantic=` and propagates
as a typed instance across `context=[]` seams — type safety compounds
across the CrewAI pipeline (L35 finding).

`DietaryRestriction` is the picklist for the recipe workflow's Gradio input;
threaded through kickoff_inputs as its str value and substituted into the
`{dietary_restrictions}` prompt template.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class DietaryRestriction(StrEnum):
    """NONE is the 'no filter' semantic; `filter_by_dietary_restriction`
    short-circuits on it and returns the ingredient list unchanged."""

    NONE = "none"
    VEGAN = "vegan"
    VEGETARIAN = "vegetarian"
    GLUTEN_FREE = "gluten-free"
    KETO = "keto"


class Recipe(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str
    ingredients: list[str] = Field(
        ...,
        min_length=1,
        description=(
            "Ingredients drawn from the filtered ingredient list plus common "
            "pantry staples (salt, pepper, water, cooking oil)"
        ),
    )
    instructions: str = Field(
        ...,
        description="Step-by-step preparation instructions, 4-8 numbered steps",
    )
    calorie_estimate: int = Field(..., ge=0, description="Per serving")


class RecipeSuggestionOutput(BaseModel):
    """Recipes are biased to 2-3 by the task expected_output, not schema."""

    model_config = ConfigDict(extra="forbid")

    recipes: list[Recipe]


class VitaminInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    percentage_dv: str = Field(..., description="e.g. '45%'")


class MineralInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    amount: str = Field(..., description="With unit, e.g. '100mg'")


class NutrientBreakdown(BaseModel):
    model_config = ConfigDict(extra="forbid")

    protein: str | None = Field(None, description="With unit, e.g. '35g'")
    carbohydrates: str | None = Field(None, description="With unit")
    fats: str | None = Field(None, description="With unit")
    vitamins: list[VitaminInfo] = Field(
        default_factory=list,
        description="3-5 key vitamins per task expected_output",
    )
    minerals: list[MineralInfo] = Field(
        default_factory=list,
        description="3-5 key minerals per task expected_output",
    )


class NutrientAnalysisOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dish: str | None = None
    portion_size: str | None = Field(None, description="e.g. '1 cup'")
    estimated_calories: int | None = Field(None, ge=0, description="Per portion")
    nutrients: NutrientBreakdown
    health_evaluation: str | None = Field(
        None,
        description="2-3 sentence evaluation of the meal's healthiness",
    )
