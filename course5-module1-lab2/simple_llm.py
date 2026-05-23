from langchain_ollama import OllamaLLM

def product_assistant(ascii_transcript):
    system_prompt = """You are an intelligent assistant specializing in financial products.
    Your task is to process transcripts of earnings calls, ensuring that all references to
    financial products and common financial terms are in the correct format. For each
    financial product or common term that is typically abbreviated as an acronym, the full term
    should be spelled out followed by the acronym in parentheses. For example, '401k' should be
    transformed to '401(k) retirement savings plan', 'HSA' should be transformed to 
    'Health Savings Account (HSA)', 'ROA' should be transformed to 'Return on Assets (ROA)'.
    In cases where numerical figures do not represent specific financial products, leave them as is.
    Produce the adjusted transcript only, no commentary."""

    prompt_input = system_prompt + "\n" + ascii_transcript

    cleanup_llm = OllamaLLM(
        model="mistral",
        temperature=0.2,
        num_predict=1024,
    )

    return cleanup_llm.invoke(prompt_input)
    

if __name__ == "__main__":
    test_transcript = "We have a healthy tier 1 capital ratio and our LTV is improving."
    result = product_assistant(test_transcript)
    print(result)