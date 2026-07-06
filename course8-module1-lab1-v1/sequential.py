from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

from shared import llm, extract_content

GENERATE_RESUME_SUMMARY = "generate_resume_summary"
GENERATE_COVER_LETTER = "generate_cover_letter"


class ChainState(TypedDict):
    job_description: str
    resume_summary: str
    cover_letter: str


def generate_resume_summary(state: ChainState) -> dict[str, str]:
    prompt = f"""
You're a resume assistant. Read the following job description and summarize the key qualifications and experience the ideal candidate should have, phrased as if from the perspective of a strong applicant's resume summary.

Job Description:
{state['job_description']}
"""
    response = llm.invoke(prompt)
    content = extract_content(response)
    return {"resume_summary": content}


def generate_cover_letter(state: ChainState) -> dict[str, str]:
    prompt = f"""
You're a cover letter writing assistant. Using the resume summary below, write a professional and personalized cover letter for the following job.

Resume Summary:
{state['resume_summary']}

Job Description:
{state['job_description']}
"""
    response = llm.invoke(prompt)
    content = extract_content(response)
    return {"cover_letter": content}


workflow = StateGraph(ChainState)
workflow.add_node(GENERATE_RESUME_SUMMARY, generate_resume_summary)
workflow.add_node(GENERATE_COVER_LETTER, generate_cover_letter)
workflow.add_edge(START, GENERATE_RESUME_SUMMARY)
workflow.add_edge(GENERATE_RESUME_SUMMARY, GENERATE_COVER_LETTER)
workflow.add_edge(GENERATE_COVER_LETTER, END)

app = workflow.compile(checkpointer=InMemorySaver())