import os
import json
from pathlib import Path
from typing import Annotated, List, Sequence, TypedDict

from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, ToolMessage, BaseMessage, AIMessage
from langchain_tavily import TavilySearch
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

load_dotenv()

assert os.environ.get("TAVILY_API_KEY"), "TAVILY_API_KEY missing from .env"
assert os.environ.get("OPENAI_API_KEY"), "OPENAI_API_KEY missing from .env"

MAX_ITERATIONS = 4


class Reflection(BaseModel):
    missing: str = Field(description="What information is missing")
    superfluous: str = Field(description="What information is unnecessary")


class AnswerQuestion(BaseModel):
    answer: str = Field(description="Main response to the question")
    reflection: Reflection = Field(description="Self-critique of the answer")
    search_queries: List[str] = Field(description="Queries for additional research")


class ReviseAnswer(AnswerQuestion):
    """Revise your original answer to your question."""
    references: List[str] = Field(description="Citations motivating your updated answer.")


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


def build_chains():
    llm = ChatOpenAI(model="gpt-4.1-mini")
    tavily_tool = TavilySearch(max_results=3)

    prompt_template = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are Dr. Marcus Vale, a fictional nutrition researcher advocating for animal-based nutrition and challenging plant-centric dietary dogma. Focus on the superior bioavailability of animal foods and the potential toxicity of plant compounds such as oxalates, lectins, and phytates.
            Your response must follow these steps:
            1. {first_instruction}
            2. Present the evolutionary and biochemical rationale for animal-based nutrition, emphasizing nutrient density, bioavailability, and the absence of plant antinutrients.
            3. Challenge conventional "plants are healthy" narratives with mechanistic evidence about plant defense compounds and their impact on human physiology.
            4. Reflect and critique your answer. Be rigorous about evolutionary consistency and nutrient science.
            5. After the reflection, **list 1-3 search queries separately** for researching antinutrients, bioavailability studies, or evolutionary nutrition. Do not include them inside the reflection.
            Focus on: organ meats as superfoods, the carnivore elimination protocol, and why "fiber is unnecessary" based on human evolutionary biology.
            """
        ),
        MessagesPlaceholder(variable_name="messages"),
        (
            "system",
            "Answer the user's question above using the required format, emphasizing the superiority of animal-based nutrition."
        ),
    ])

    first_responder_prompt = prompt_template.partial(
        first_instruction="Provide a detailed ~250 word answer"
    )

    revise_instructions = """Revise your previous answer using the new information, applying a rigorous, evidence-based clinical-research approach.
    - Incorporate the previous critique to add clinically relevant information, focusing on mechanistic understanding and individual variability.
    - You MUST include numerical citations referencing peer-reviewed research, randomized controlled trials, or meta-analyses to ensure medical accuracy.
    - Distinguish between correlation and causation, and acknowledge limitations in current research.
    - Address potential biomarker considerations (lipid panels, inflammatory markers, and so on) when relevant.
    - Add a "References" section to the bottom of your answer (which does not count towards the word limit) in the form of:
    - [1] https://example.com
    - [2] https://example.com
    - Use the previous critique to remove speculation and ensure claims are supported by high-quality evidence. Keep response under 250 words with precision over volume.
    - When discussing nutritional interventions, consider metabolic flexibility, insulin sensitivity, and individual response variability.
    """
    revisor_prompt = prompt_template.partial(first_instruction=revise_instructions)

    initial_chain = first_responder_prompt | llm.bind_tools(tools=[AnswerQuestion])
    revisor_chain = revisor_prompt | llm.bind_tools(tools=[ReviseAnswer])

    return tavily_tool, initial_chain, revisor_chain


def build_graph(tavily_tool, initial_chain, revisor_chain):

    def execute_tools(state: List[BaseMessage]) -> List[BaseMessage]:
        last_message = state[-1]
        if not isinstance(last_message, AIMessage):
            return []

        tool_messages: List[BaseMessage] = []
        for tool_call in last_message.tool_calls:
            if tool_call["name"] in ["AnswerQuestion", "ReviseAnswer"]:
                call_id = tool_call["id"]
                search_queries = tool_call["args"].get("search_queries", [])
                query_results = {}
                for query in search_queries:
                    query_results[query] = tavily_tool.invoke(query)
                tool_messages.append(ToolMessage(
                    content=json.dumps(query_results),
                    tool_call_id=call_id,
                ))
        return tool_messages

    def respond_node(state: AgentState) -> dict:
        ai_msg = initial_chain.invoke({"messages": state["messages"]})
        return {"messages": [ai_msg]}

    def revisor_node(state: AgentState) -> dict:
        ai_msg = revisor_chain.invoke({"messages": state["messages"]})
        return {"messages": [ai_msg]}

    def execute_tools_node(state: AgentState) -> dict:
        return {"messages": execute_tools(list(state["messages"]))}

    def event_loop(state: AgentState) -> str:
        count_tool_visits = sum(isinstance(m, ToolMessage) for m in state["messages"])
        if count_tool_visits >= MAX_ITERATIONS:
            return END
        return "execute_tools"

    graph = StateGraph(AgentState)
    graph.add_node("respond", respond_node)
    graph.add_node("execute_tools", execute_tools_node)
    graph.add_node("revisor", revisor_node)

    graph.add_edge(START, "respond")
    graph.add_edge("respond", "execute_tools")
    graph.add_edge("execute_tools", "revisor")
    graph.add_conditional_edges(
        "revisor",
        event_loop,
        {"execute_tools": "execute_tools", END: END},
    )
    return graph.compile()


def export_graph_artefacts(
    app,
    mmd_path: Path = Path("graph.mmd"),
    png_path: Path = Path("graph.png"),
) -> None:
    graph = app.get_graph()

    mermaid_src = graph.draw_mermaid()
    mmd_path.write_text(mermaid_src, encoding="utf-8")
    print(f"--- Mermaid source written to {mmd_path} ---")

    png_bytes = graph.draw_mermaid_png()
    png_path.write_bytes(png_bytes)
    print(f"--- Mermaid PNG written to {png_path} ---")


def main() -> None:
    tavily_tool, initial_chain, revisor_chain = build_chains()
    app = build_graph(tavily_tool, initial_chain, revisor_chain)
    export_graph_artefacts(app)

    question = """I'm pre-diabetic and need to lower my blood sugar, and I have heart issues.
    What breakfast foods should I eat and avoid"""

    responses = app.invoke({"messages": [HumanMessage(content=question)]})

    print("--- Initial Draft Answer ---")
    initial_answer = responses["messages"][1].tool_calls[0]["args"]["answer"]
    print(initial_answer)
    print()

    print("--- Intermediate and Final Revised Answers ---")
    answers = []
    for msg in reversed(responses["messages"]):
        if isinstance(msg, AIMessage) and msg.tool_calls:
            answer = msg.tool_calls[0].get("args", {}).get("answer")
            if answer:
                answers.append(answer)

    for i, ans in enumerate(answers):
        label = "Final Revised Answer" if i == 0 else f"Intermediate Step {len(answers) - i}"
        print(f"{label}:\n{ans}\n")


if __name__ == "__main__":
    main()