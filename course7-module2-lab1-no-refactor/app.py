from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing import Annotated, Sequence
from typing_extensions import TypedDict


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


def generation_node(state: AgentState) -> AgentState:
    generated_post = generate_chain.invoke({"messages": state["messages"]})
    return {"messages": [generated_post]}


def reflection_node(state: AgentState) -> AgentState:
    res = reflect_chain.invoke({"messages": state["messages"]})
    return {"messages": [HumanMessage(content=res.content)]}


def should_continue(state: AgentState) -> str:
    messages = state["messages"]
    if len(messages) > 6:
        return END
    return "reflect"


load_dotenv()

llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.0)

generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a professional LinkedIn content assistant tasked with crafting engaging, insightful, and well-structured LinkedIn posts."
            " Generate the best LinkedIn post possible for the user's request."
            " If the user provides feedback or critique, respond with a refined version of your previous attempts, improving clarity, tone, or engagement as needed.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

generate_chain = generation_prompt | llm

reflection_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a professional LinkedIn content strategist and thought leadership expert. Your task is to critically evaluate the given LinkedIn post and provide a comprehensive critique. Follow these guidelines:

        1. Assess the post's overall quality, professionalism, and alignment with LinkedIn best practices.
        2. Evaluate the structure, tone, clarity, and readability of the post.
        3. Analyze the post's potential for engagement (likes, comments, shares) and its effectiveness in building professional credibility.
        4. Consider the post's relevance to the author's industry, audience, or current trends.
        5. Examine the use of formatting (e.g., line breaks, bullet points), hashtags, mentions, and media (if any).
        6. Evaluate the effectiveness of any call-to-action or takeaway.

        Provide a detailed critique that includes:
        - A brief explanation of the post's strengths and weaknesses.
        - Specific areas that could be improved.
        - Actionable suggestions for enhancing clarity, engagement, and professionalism.

        Your critique will be used to improve the post in the next revision step, so ensure your feedback is thoughtful, constructive, and practical.
        """
    ),
    MessagesPlaceholder(variable_name="messages")
])

reflect_chain = reflection_prompt | llm

graph = StateGraph(AgentState)

graph.add_node("generate", generation_node)
graph.add_node("reflect", reflection_node)

graph.add_edge(START, "generate")
graph.add_conditional_edges("generate", should_continue, {"reflect": "reflect", END: END})
graph.add_edge("reflect", "generate")

workflow = graph.compile()


def main() -> None:
    inputs = HumanMessage(content="""Write a linkedin post on getting a software developer job at IBM under 160 characters""")

    response = workflow.invoke({"messages": [inputs]})

    for i, msg in enumerate(response["messages"]):
        print(f"--- [{i}] {msg.__class__.__name__} ---")
        print(msg.content)
        print()

    print(response["messages"][-1].content)

    png_bytes = workflow.get_graph().draw_mermaid_png()
    with open("graph.png", "wb") as f:
        f.write(png_bytes)

    print(workflow.get_graph().draw_mermaid())


if __name__ == "__main__":
    load_dotenv()
    main()