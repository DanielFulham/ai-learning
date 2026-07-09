import os
from enum import StrEnum
from pathlib import Path
from typing import TypedDict, Unpack

from crewai import LLM, Agent, Crew, CrewOutput, Process, Task
from crewai_tools import SerperDevTool
from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator

from leftover import LeftoversCrew

OUTPUT_DIR = Path(__file__).parent / "outputs"


class StoreSection(StrEnum):
    PRODUCE = "Produce"
    MEAT = "Meat"
    DAIRY = "Dairy"
    PANTRY = "Pantry"
    FROZEN = "Frozen"
    BAKERY = "Bakery"
    OTHER = "Other"


class DifficultyLevel(StrEnum):
    EASY = "Easy"
    MEDIUM = "Medium"
    HARD = "Hard"


class PriceEstimate(BaseModel):
    """Structured price range in USD."""

    min_dollars: float = Field(ge=0, description="Lower bound of the estimated price in USD")
    max_dollars: float = Field(ge=0, description="Upper bound of the estimated price in USD")

    @model_validator(mode="after")
    def _check_range(self) -> "PriceEstimate":
        if self.max_dollars < self.min_dollars:
            raise ValueError(
                f"max_dollars ({self.max_dollars}) must be >= min_dollars ({self.min_dollars})"
            )
        return self


class GroceryItem(BaseModel):
    """Individual grocery item."""

    name: str = Field(description="Name of the grocery item")
    quantity: str = Field(description="Quantity needed (for example, '2 lbs', '1 gallon')")
    estimated_price: PriceEstimate = Field(description="Estimated price range in USD")
    category: StoreSection = Field(description="Store section")


class MealPlan(BaseModel):
    """Simple meal plan."""

    meal_name: str = Field(description="Name of the meal")
    difficulty_level: DifficultyLevel = Field(description="Cooking difficulty")
    servings: int = Field(gt=0, description="Number of people it serves")
    researched_ingredients: list[str] = Field(description="Ingredients found through research")


class ShoppingCategory(BaseModel):
    """Store section with items."""

    section_name: StoreSection = Field(description="Store section")
    items: list[GroceryItem] = Field(description="Items in this section")
    estimated_total: PriceEstimate = Field(
        description="Estimated cost range for this section in USD"
    )


class GroceryShoppingPlan(BaseModel):
    """Complete simplified shopping plan."""

    total_budget: PriceEstimate = Field(description="Total planned budget range in USD")
    meal_plans: list[MealPlan] = Field(description="Planned meals")
    shopping_sections: list[ShoppingCategory] = Field(description="Organized by store sections")
    shopping_tips: list[str] = Field(description="Money-saving and efficiency tips")


class MealPlanningInputs(TypedDict):
    meal_name: str
    servings: int
    budget: str
    dietary_restrictions: list[str]
    cooking_skill: str


def build_meal_planning_inputs(
    **inputs: Unpack[MealPlanningInputs],
) -> dict[str, str | int | list[str]]:
    return {
        "meal_name": inputs["meal_name"],
        "servings": inputs["servings"],
        "budget": inputs["budget"],
        "dietary_restrictions": inputs["dietary_restrictions"],
        "cooking_skill": inputs["cooking_skill"],
    }


def _init_runtime() -> None:
    """Load env, validate required keys, create outputs dir."""
    load_dotenv()
    for var in ("ANTHROPIC_API_KEY", "SERPER_API_KEY"):
        if not os.environ.get(var):
            raise RuntimeError(
                f"{var} not set. Copy .env.example to .env and fill it in."
            )
    OUTPUT_DIR.mkdir(exist_ok=True)


def _build_llm() -> LLM:
    return LLM(
        model="anthropic/claude-haiku-4-5",
        max_tokens=2000,
        temperature=0.0,
    )


def _build_agents(
    llm: LLM,
) -> tuple[Agent, Agent, Agent, Agent, Agent, Task]:
    """Construct all five agents plus the YAML-loaded leftover task.

    LeftoversCrew is instantiated once; both the agent and task are extracted here
    so downstream construction doesn't need to re-load the YAML.
    """
    meal_planner = Agent(
        role="Meal Planner & Recipe Researcher",
        goal="Search for optimal recipes and create detailed meal plans",
        backstory=(
            "A skilled meal planner who researches the best recipes online, "
            "considering dietary needs, cooking skill levels, and budget constraints."
        ),
        tools=[SerperDevTool()],
        llm=llm,
        max_iter=1,
        allow_delegation=False,
        verbose=False,
    )

    shopping_organizer = Agent(
        role="Shopping Organizer",
        goal="Organize grocery lists by store sections efficiently",
        backstory=(
            "An experienced shopper who knows how to organize lists for quick "
            "store trips and considers dietary restrictions."
        ),
        tools=[],
        llm=llm,
        max_iter=3,
        allow_delegation=False,
        verbose=False,
    )

    budget_advisor = Agent(
        role="Budget Advisor",
        goal="Provide cost estimates and money-saving tips",
        backstory=(
            "A budget-conscious shopper who helps families save money on groceries "
            "while respecting dietary needs."
        ),
        tools=[SerperDevTool()],
        llm=llm,
        max_iter=1,
        allow_delegation=False,
        verbose=False,
    )

    summary_agent = Agent(
        role="Report Compiler",
        goal="Compile comprehensive meal planning reports from all team outputs",
        backstory=(
            "A skilled coordinator who organizes information from multiple specialists "
            "into comprehensive, easy-to-follow reports."
        ),
        tools=[],
        llm=llm,
        max_iter=3,
        allow_delegation=False,
        verbose=False,
    )

    leftovers_cb = LeftoversCrew(llm=llm)  # type: ignore[call-arg]
    yaml_leftover_manager = leftovers_cb.leftover_manager()  # type: ignore[attr-defined]
    yaml_leftover_task = leftovers_cb.leftover_task()  # type: ignore[attr-defined]

    return (
        meal_planner,
        shopping_organizer,
        budget_advisor,
        yaml_leftover_manager,
        summary_agent,
        yaml_leftover_task,
    )


def _build_tasks(
    meal_planner: Agent,
    shopping_organizer: Agent,
    budget_advisor: Agent,
    summary_agent: Agent,
    yaml_leftover_task: Task,
) -> tuple[Task, Task, Task, Task, Task]:
    """Return all five tasks, wired with explicit context declarations."""
    meal_planning_task = Task(
        description=(
            "Search for the best '{meal_name}' recipe for {servings} people within a {budget} budget. "
            "Consider dietary restrictions: {dietary_restrictions} and "
            "cooking skill level: {cooking_skill}. "
            "Find recipes that match the skill level and provide complete ingredient lists "
            "with quantities."
        ),
        expected_output=(
            "A MealPlan for one recipe with 6-10 researched ingredient names "
            "(no prices, no store sections — those are downstream), "
            "difficulty matching the requested skill level, and servings matching input."
        ),
        agent=meal_planner,
        output_pydantic=MealPlan,
        output_file=str(OUTPUT_DIR / "meals.json"),
    )

    shopping_task = Task(
        description=(
            "Organize the ingredients from the '{meal_name}' meal plan into a grocery shopping list. "
            "Group items by store sections and estimate quantities for {servings} people. "
            "Consider dietary restrictions: {dietary_restrictions} and cooking skill: {cooking_skill}. "
            "Stay within budget: {budget}."
        ),
        expected_output=(
            "A GroceryShoppingPlan with total_budget matching input, one MealPlan carried from upstream, "
            "shopping_sections grouped by StoreSection enum (Produce/Meat/Dairy/Pantry/Frozen/Bakery/Other) "
            "with each GroceryItem carrying name (bare ingredient, no prep), quantity (2 lbs / 1 gallon shape), "
            "estimated_price as PriceEstimate {min_dollars, max_dollars}, and category matching its section. "
            "3-5 shopping_tips."
        ),
        agent=shopping_organizer,
        context=[meal_planning_task],
        output_pydantic=GroceryShoppingPlan,
        output_file=str(OUTPUT_DIR / "shopping_list.json"),
    )

    budget_task = Task(
        description=(
            "Analyze the shopping plan for '{meal_name}' serving {servings} people. "
            "Ensure total cost stays within {budget}. "
            "Consider dietary restrictions: {dietary_restrictions}. "
            "Provide practical money-saving tips and alternative ingredients if needed to meet budget."
        ),
        expected_output=(
            "A ~400-word markdown-formatted shopping guide with these sections: "
            "## Price Analysis — table of items with unit prices and section subtotals. "
            "## Budget Verification — computed total range from the shopping list vs the requested budget, "
            "explicit statement of whether the plan fits, exceeds, or has margin. "
            "## Money-Saving Tips — 3-5 tips specific to this recipe and price landscape. "
            "## Substitutions — 2-3 ingredient swaps if the plan exceeds budget, otherwise omit this section. "
            "Assume upstream shopping list is authoritative; do not re-price items unless suggesting a substitution."
        ),
        agent=budget_advisor,
        context=[meal_planning_task, shopping_task],
        output_file=str(OUTPUT_DIR / "shopping_guide.md"),
    )

    summary_task = Task(
        description=(
            "Compile a comprehensive meal planning report that includes:\n"
            "1. The complete recipe and cooking instructions from the meal planner\n"
            "2. The organized shopping list with prices from the shopping organizer\n"
            "3. The budget analysis and money-saving tips from the budget advisor\n"
            "4. The leftover management suggestions from the waste reduction specialist\n"
            "Format this as a complete, user-friendly meal planning guide."
        ),
        expected_output=(
            "A ~700-word markdown-formatted final guide with sections:\n"
            "## Recipe — meal name, difficulty, servings, ingredient list from the meal planner.\n"
            "## Shopping List — grouped by StoreSection, with quantities and price ranges from the shopping organizer.\n"
            "## Budget Notes — total estimate, fit-vs-budget statement, 2-3 money-saving tips from the budget advisor.\n"
            "## Leftover Ideas — 2-3 bonus recipe ideas from the leftover manager.\n"
            "Preserve the shape of each upstream section; do not re-analyze or re-price. "
            "This is a compilation, not new analysis."
        ),
        agent=summary_agent,
        context=[meal_planning_task, shopping_task, budget_task, yaml_leftover_task],
        output_file=str(OUTPUT_DIR / "final_guide.md"),
    )

    return meal_planning_task, shopping_task, budget_task, yaml_leftover_task, summary_task


def main() -> None:
    _init_runtime()
    llm = _build_llm()

    (
        meal_planner,
        shopping_organizer,
        budget_advisor,
        yaml_leftover_manager,
        summary_agent,
        yaml_leftover_task,
    ) = _build_agents(llm)

    tasks = _build_tasks(
        meal_planner,
        shopping_organizer,
        budget_advisor,
        summary_agent,
        yaml_leftover_task,
    )

    complete_grocery_crew = Crew(
        agents=[
            meal_planner,
            shopping_organizer,
            budget_advisor,
            yaml_leftover_manager,
            summary_agent,
        ],
        tasks=list(tasks),
        process=Process.sequential,
        verbose=True,
    )

    complete_result = complete_grocery_crew.kickoff(
        inputs=build_meal_planning_inputs(
            meal_name="Chicken Stir Fry",
            servings=4,
            budget="$25",
            dietary_restrictions=["no nuts", "low sodium"],
            cooking_skill="beginner",
        )
    )
    assert isinstance(complete_result, CrewOutput), (
        f"Expected CrewOutput, got {type(complete_result).__name__}"
    )

    print("\nComplete five-agent grocery planning completed.")
    print(f"meal task pydantic populated:     {complete_result.tasks_output[0].pydantic is not None}")
    print(f"shopping task pydantic populated: {complete_result.tasks_output[1].pydantic is not None}")
    print(f"budget task pydantic populated:   {complete_result.tasks_output[2].pydantic is not None}")
    print(f"leftover task pydantic populated: {complete_result.tasks_output[3].pydantic is not None}")
    print(f"summary task pydantic populated:  {complete_result.tasks_output[4].pydantic is not None}")

    usage = complete_result.token_usage
    print(
        f"\nUsage — prompt={usage.prompt_tokens} "
        f"cached={usage.cached_prompt_tokens} "
        f"cache_creation={usage.cache_creation_tokens} "
        f"completion={usage.completion_tokens} "
        f"requests={usage.successful_requests}"
    )


if __name__ == "__main__":
    main()
