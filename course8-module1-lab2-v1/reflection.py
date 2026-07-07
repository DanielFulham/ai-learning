"""Reflection pattern: iterative investment plan refinement.

Demonstrates AI_PATTERNS.md §7 with a Cathie Wood generator (initial plan),
a Ray Dalio generator (refinement with feedback), a Warren Buffett evaluator,
and a router that either accepts the plan or loops back for another pass.
A finalize node populates terminated_reason with 'converged' or 'iteration_cap'
so the caller can distinguish successful convergence from cap-hit fallback
(§7 fail-loudly discipline).
"""

import asyncio
import time
import uuid
from pathlib import Path
from typing import Literal, NotRequired, TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from shared import describe_graph, get_llm, render_graph_png

# ---------------------------------------------------------------------------
# Node-name constants
# ---------------------------------------------------------------------------

DETERMINE_TARGET_GRADE_NODE = "determine_target_grade"
GENERATE_PLAN_NODE = "generate_plan"
EVALUATE_PLAN_NODE = "evaluate_plan"
FINALIZE_NODE = "finalize"

MAX_REFLECTION_ITERATIONS = 5

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

Grade = Literal[
    "ultra-conservative",
    "conservative",
    "moderate",
    "aggressive",
    "high risk",
]

TerminatedReason = Literal["converged", "iteration_cap"]


class TargetGradeDecision(BaseModel):
    """Structured output for the target-grade classifier."""

    grade: Grade = Field(
        description=(
            "The risk classification that best fits the investor profile. "
            "Choose 'ultra-conservative' or 'conservative' for capital-preservation profiles, "
            "'moderate' for balanced risk-reward tolerance, "
            "'aggressive' for growth-oriented profiles accepting volatility, "
            "'high risk' for speculative profiles targeting outsized returns."
        )
    )

class InvestmentState(TypedDict):
    investor_profile: str
    target_grade: NotRequired[Grade]
    investment_plan: NotRequired[str]
    grade: NotRequired[Grade]
    feedback: NotRequired[str]
    n: NotRequired[int]
    terminated_reason: NotRequired[TerminatedReason]


# ---------------------------------------------------------------------------
# Setup: determine target grade
# ---------------------------------------------------------------------------

grade_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an investment advisor. Given an investor's profile, choose the "
            "single risk classification that best matches their situation and goals.",
        ),
        ("human", "Investor profile:\n\n{investor_profile}"),
    ]
)

grade_pipe = grade_prompt | get_llm(temperature=0.0).with_structured_output(
    TargetGradeDecision
)


def determine_target_grade(state: InvestmentState) -> dict[str, Grade]:
    """Classify the investor profile into a target risk grade."""
    result = grade_pipe.invoke({"investor_profile": state["investor_profile"]})
    if not isinstance(result, TargetGradeDecision):
        raise TypeError(
            f"grade_pipe returned {type(result).__name__}, expected TargetGradeDecision"
        )
    return {"target_grade": result.grade}

# ---------------------------------------------------------------------------
# Generator: Cathie Wood (initial) and Ray Dalio (refinement)
# ---------------------------------------------------------------------------

cathie_wood_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a bold, innovation-driven investment advisor inspired by Cathie Wood.\n\n"
            "Generate a high-conviction, forward-looking investment plan that embraces "
            "disruptive technologies, emerging markets, and long-term growth potential. "
            "You are not afraid of short-term volatility as long as the upside is "
            "transformational.\n\n"
            "Prioritize innovation and high-reward opportunities such as artificial "
            "intelligence, biotechnology, blockchain, or renewable energy.\n\n"
            "Respond with a concise investment plan in paragraph form.",
        ),
        ("human", "Investor profile:\n\n{investor_profile}"),
    ]
)

cathie_wood_pipe = cathie_wood_prompt | get_llm()

ray_dalio_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an investment advisor inspired by Ray Dalio's principles but with "
            "adaptive strategy generation. Create scenario-aware investment plans that "
            "respond to economic conditions and evaluator feedback.\n\n"
            "CORE PRINCIPLES:\n"
            "- Environmental diversification across economic regimes\n"
            "- Risk parity weighting by volatility, not just dollar amounts\n"
            "- Inflation-aware asset selection with real-return focus\n"
            "- Macroeconomic scenario planning\n\n"
            "ADAPTATION RULES based on feedback:\n"
            "- 'too conservative' → increase growth equity, add emerging markets\n"
            "- 'too aggressive' → add defensive assets, increase bond allocation\n"
            "- 'lacks inflation protection' → emphasize TIPS, commodities, REITs\n"
            "- 'too complex' → simplify to core ETF strategy\n"
            "- 'insufficient diversification' → add geographic/sector exposure\n\n"
            "Make significant adjustments from any previous approach based on the "
            "feedback provided.",
        ),
        (
            "human",
            "Investor profile:\n{investor_profile}\n\n"
            "Previous plan grade: {grade}\n\n"
            "Evaluator feedback: {feedback}\n\n"
            "Create a NEW investment strategy that addresses the concerns raised.",
        ),
    ]
)

ray_dalio_pipe = ray_dalio_prompt | get_llm()

def generate_plan(state: InvestmentState) -> dict[str, str]:
    """Generate an investment plan.

    First iteration (no feedback yet) uses the Cathie Wood generator.
    Subsequent iterations use the Ray Dalio generator with feedback loop.
    """
    if state.get("feedback"):
        response = ray_dalio_pipe.invoke(
            {
                "investor_profile": state["investor_profile"],
                "grade": state.get("grade"),
                "feedback": state.get("feedback"),
            }
        )
    else:
        response = cathie_wood_pipe.invoke(
            {"investor_profile": state["investor_profile"]}
        )

    if not isinstance(response.content, str):
        raise TypeError(
            f"generator returned content of type {type(response.content).__name__}, "
            "expected str"
        )
    return {"investment_plan": response.content}

# ---------------------------------------------------------------------------
# Evaluator: Warren Buffett critic with structured Feedback output
# ---------------------------------------------------------------------------

class Feedback(BaseModel):
    """Structured evaluator output: grade + reasoning."""

    grade: Grade = Field(
        description=(
            "The risk classification of the proposed plan. "
            "'ultra-conservative' for capital-preservation portfolios with minimal volatility, "
            "'conservative' for low-risk plans prioritising stability, "
            "'moderate' for balanced risk-reward strategies, "
            "'aggressive' for growth-oriented plans accepting significant volatility, "
            "'high risk' for speculative plans with meaningful downside exposure."
        )
    )
    feedback: str = Field(
        description=(
            "Concise reasoning for the assigned grade. Name the specific plan elements "
            "that drove the classification. Written to inform a refinement iteration."
        )
    )

evaluator_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an investment risk evaluator inspired by Warren Buffett's "
            "value-investing philosophy. Assess whether a proposed strategy aligns "
            "with capital preservation, long-term stability, and sound fundamentals. "
            "Be skeptical of speculative or high-volatility assets.\n\n"
            "EVALUATION CRITERIA:\n"
            "- Business clarity: transparent cash flows, understandable holdings\n"
            "- Margin of safety: downside protection in the entry price\n"
            "- Capital preservation: wealth protected over the long term\n"
            "- Investor alignment: matches the investor's risk tolerance and goals\n"
            "- Quality fundamentals: financially sound assets with competitive moats",
        ),
        (
            "human",
            "Investor profile:\n{investor_profile}\n\n"
            "Target risk level: {target_grade}\n\n"
            "Investment plan to evaluate:\n{investment_plan}",
        ),
    ]
)

buffett_evaluator_pipe = evaluator_prompt | get_llm(
    temperature=0.0
).with_structured_output(Feedback)

def evaluate_plan(state: InvestmentState) -> dict[str, Grade | str | int]:
    """Evaluate the current investment plan against the target grade."""
    result = buffett_evaluator_pipe.invoke(
        {
            "investment_plan": state.get("investment_plan"),
            "investor_profile": state.get("investor_profile"),
            "target_grade": state.get("target_grade"),
        }
    )
    if not isinstance(result, Feedback):
        raise TypeError(
            f"buffett_evaluator_pipe returned {type(result).__name__}, expected Feedback"
        )
    return {
        "grade": result.grade,
        "feedback": result.feedback,
        "n": state.get("n", 0) + 1,
    }

# ---------------------------------------------------------------------------
# Router: decide whether to accept the plan or loop back for refinement
# ---------------------------------------------------------------------------

ACCEPTED = "accepted"
REJECTED = "rejected"


def route_investment(state: InvestmentState) -> str:
    """Return the routing key based on convergence or iteration cap.

    'accepted' means terminate; 'rejected' means loop back to generate_plan.
    Distinguishing convergence from cap-hit happens in the finalize node —
    this router only returns the routing key.
    """
    current_grade = state.get("grade")
    target_grade = state.get("target_grade")
    iteration = state.get("n", 0)

    if current_grade == target_grade:
        return ACCEPTED
    if iteration >= MAX_REFLECTION_ITERATIONS:
        return ACCEPTED
    return REJECTED

def finalize(state: InvestmentState) -> dict[str, TerminatedReason]:
    """Set terminated_reason on the way out so the caller can distinguish
    convergence from iteration-cap bailout."""
    if state.get("grade") == state.get("target_grade"):
        return {"terminated_reason": "converged"}
    return {"terminated_reason": "iteration_cap"}

# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

_builder = StateGraph(InvestmentState)

_builder.add_node(DETERMINE_TARGET_GRADE_NODE, determine_target_grade)
_builder.add_node(GENERATE_PLAN_NODE, generate_plan)
_builder.add_node(EVALUATE_PLAN_NODE, evaluate_plan)
_builder.add_node(FINALIZE_NODE, finalize)

_builder.add_edge(START, DETERMINE_TARGET_GRADE_NODE)
_builder.add_edge(DETERMINE_TARGET_GRADE_NODE, GENERATE_PLAN_NODE)
_builder.add_edge(GENERATE_PLAN_NODE, EVALUATE_PLAN_NODE)

_builder.add_conditional_edges(
    EVALUATE_PLAN_NODE,
    route_investment,
    {
        ACCEPTED: FINALIZE_NODE,
        REJECTED: GENERATE_PLAN_NODE,
    },
)

_builder.add_edge(FINALIZE_NODE, END)

app = _builder.compile(checkpointer=InMemorySaver())

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    async def _run() -> None:
        describe_graph(app)
        png_path = Path(__file__).parent / "graph_reflection.png"
        render_graph_png(app, str(png_path))
        print(f"Graph rendered to {png_path.name}\n")

        profile = (
            "Age: 29\n"
            "Salary: $110,000\n"
            "Assets: $40,000\n"
            "Goal: Achieve financial independence by age 45\n"
            "Risk tolerance: High"
        )

        start = time.perf_counter()
        result = await app.ainvoke(
            {"investor_profile": profile},
            config={"configurable": {"thread_id": str(uuid.uuid4())}},
        )
        elapsed = time.perf_counter() - start

        print(f"\nWall-clock: {elapsed:.1f}s")

        print("=" * 60)
        print("FINAL INVESTMENT PLAN")
        print("=" * 60)
        print(f"Investor profile:\n{result['investor_profile']}\n")
        print(f"Target risk grade:      {result['target_grade']}")
        print(f"Final assigned grade:   {result['grade']}")
        print(f"Iterations:             {result['n']}")
        print(f"Terminated reason:      {result['terminated_reason']}")
        print()
        print("Evaluator feedback:")
        print("-" * 60)
        print(result["feedback"])
        print()
        print("Final investment plan:")
        print("-" * 60)
        print(result["investment_plan"])

    asyncio.run(_run())
