from typing import TypedDict, Optional

from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama


llm = ChatOllama(model="llama3.2:latest", temperature=0)


# ============================================================
# Workflow 1: Authentication (conditional + loop)
# ============================================================

class AuthState(TypedDict, total=False):
    username: Optional[str]
    password: Optional[str]
    is_authenticated: Optional[bool]
    output: Optional[str]


def input_node(state):
    print(state)
    if state.get("username", "") == "":
        username = input("What is your username?")

    password = input("Enter your password: ")

    if state.get("username", "") == "":
        return {"username": username, "password": password}
    else:
        return {"password": password}


def validate_credentials_node(state):
    username = state.get("username", "")
    password = state.get("password", "")

    print("Username :", username, "Password :", password)

    if username == "test_user" and password == "secure_password":
        is_authenticated = True
    else:
        is_authenticated = False

    return {"is_authenticated": is_authenticated}


def success_node(state):
    return {"output": "Authentication successful! Welcome."}


def failure_node(state):
    return {"output": "Not Successful, please try again!"}


def router(state):
    if state["is_authenticated"]:
        return "success_node"
    else:
        return "failure_node"


workflow = StateGraph(AuthState)
workflow.add_node("InputNode", input_node)
workflow.add_node("ValidateCredential", validate_credentials_node)
workflow.add_node("Success", success_node)
workflow.add_node("Failure", failure_node)
workflow.add_edge("InputNode", "ValidateCredential")
workflow.add_edge("Success", END)
workflow.add_edge("Failure", "InputNode")
workflow.add_conditional_edges(
    "ValidateCredential",
    router,
    {"success_node": "Success", "failure_node": "Failure"},
)
workflow.set_entry_point("InputNode")
app = workflow.compile()


# ============================================================
# Workflow 2: QA (linear, LLM-driven)
# ============================================================

class QAState(TypedDict, total=False):
    question: Optional[str]
    context: Optional[str]
    answer: Optional[str]


def input_validation_node(state):
    question = (state.get("question") or "").strip()

    if not question:
        return {"valid": False, "error": "Question cannot be empty."}

    return {"valid": True}


def context_provider_node(state):
    question = (state.get("question") or "").lower()
    if "langgraph" in question or "guided project" in question:
        context = (
            "This guided project is about using LangGraph, a Python library to design state-based workflows. "
            "LangGraph simplifies building complex applications by connecting modular nodes with conditional edges."
        )
        return {"context": context}
    return {"context": None}


def llm_qa_node(state):
    question = state.get("question", "")
    context = state.get("context", None)

    if not context:
        return {"answer": "I don't have enough context to answer your question."}

    prompt = f"Context: {context}\nQuestion: {question}\nAnswer the question based on the provided context."

    try:
        response = llm.invoke(prompt)
        if not isinstance(response.content, str):
            raise TypeError(
                f"Expected str content from text-only prompt, got {type(response.content).__name__}"
            )
        return {"answer": response.content.strip()}
    except Exception as e:
        return {"answer": f"An error occurred: {str(e)}"}


qa_workflow = StateGraph(QAState)
qa_workflow.add_node("InputNode", input_validation_node)
qa_workflow.add_node("ContextNode", context_provider_node)
qa_workflow.add_node("QANode", llm_qa_node)
qa_workflow.set_entry_point("InputNode")
qa_workflow.add_edge("InputNode", "ContextNode")
qa_workflow.add_edge("ContextNode", "QANode")
qa_workflow.add_edge("QANode", END)
qa_app = qa_workflow.compile()


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    # Workflow 1: Auth (interactive — prompts for password)
    auth_inputs: AuthState = {"username": "test_user"}
    auth_result = app.invoke(auth_inputs)
    print(auth_result)
    print(auth_result["output"])

    # Workflow 2: QA (three calls — no-context, grounded, hallucination)
    qa_result_0 = qa_app.invoke({"question": "What is the weather today?"})
    print(qa_result_0)

    qa_result_1 = qa_app.invoke({"question": "What is LangGraph?"})
    print(qa_result_1)

    qa_result_2 = qa_app.invoke({"question": "What is the best guided project?"})
    print(qa_result_2)