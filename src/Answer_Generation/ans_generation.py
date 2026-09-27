from LLM_Module import llm_response, llm_router
from Answer_Generation import prompt


# Wiki Agent and RAG both return very messy nested dictionaries/lists and evidence
# Therefore we clean it up to pass the original user query and only textual evidence 
def evidence_preprocessing(
    wiki_evidence: dict | None = None,
    rag_evidence: list[dict] | None = None
) -> list[str]:

    final_evidence = []

    if rag_evidence:
        for evidence in rag_evidence:
            final_evidence.append(evidence["chunk"]["text"])

    if wiki_evidence:
        for evidence in wiki_evidence["evidence"]:
            final_evidence.append(evidence["text"])

    if not final_evidence:
        print("Invalid or no evidence received! Terminating process...")
        return []

    return final_evidence

# -----------------------------------------------------------------------------

def generate_answer(query: str,evidence: list[str]) -> llm_response.LLMResponse:

    if not query.strip():
        raise ValueError("Query cannot be empty.")

    if not evidence:
        raise ValueError("Evidence cannot be empty.")


    hyperparameters = {
            "temperature": 0.2,
            "top_p": 0.9,
            "top_k": 40,
            "repetition_penalty": 1.05}
    messages = prompt.build_ans_gen_prompt(query=query, evidence=evidence)

    return llm_router.generate(messages, hyperparameters=hyperparameters)


# -----------------------------------------------------------------------------
def request_clarification(qh_response)-> llm_response.LLMResponse:

    if not qh_response:
        raise ValueError("Invalid request from Query Handler")
    
    if not isinstance(qh_response, dict):
        raise ValueError("LLM returned invalid data")

    query = qh_response.get("query")
    clarification = qh_response.get("clarification")
    status = qh_response.get("status")

    if qh_response.get("status") != "clarification_required":
        raise ValueError("Invalid Query Handler status for clarification request")
    
    if not query:
        raise ValueError("LLM returned invalid data for rewrite request")

    if not clarification:
        raise ValueError("LLM returned invalid justification for rewrite request")

    hyperparameters = {
        "temperature": 0.1,
        "top_p": 0.9,
        "top_k": 20,
        "repetition_penalty": 1.05}
    
    messages = prompt.build_query_rewrite_prompt(qh_response)
    return llm_router.generate(messages, hyperparameters=hyperparameters)
# -----------------------------------------------------------------------------

def generate_intermediate_result(
    objective: str,
    evidence: list[str]
) -> llm_response.LLMResponse:

    if not objective.strip():
        raise ValueError("Objective cannot be empty.")

    if not evidence:
        raise ValueError("Evidence cannot be empty.")

    hyperparameters = {
    "temperature": 0.1,
    "top_p": 0.9,
    "top_k": 20,
    "repetition_penalty": 1.05}

    messages = prompt.build_intermediate_result_prompt(
        objective=objective,
        evidence=evidence,
    )

    return llm_router.generate(messages, hyperparameters=hyperparameters)




