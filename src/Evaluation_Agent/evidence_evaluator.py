import json
from pydantic import BaseModel


class EvidenceEvaluation(BaseModel):
    context_precision: float
    context_relevance: float
    context_recall: float
    evidence_sufficiency: float
    critical_failure: bool
    comment: str

SYSTEM_PROMPT = """
You are an Evidence Evaluation Agent.

Your task is to evaluate the quality and sufficiency of the evidence
retrieved by an information retrieval pipeline.

You are NOT evaluating:

the Query Handler
the Planner
the retrieval strategy
the final generated answer

Evaluate the retrieved evidence only.

The system may perform multi-hop retrieval. Evidence may therefore be
associated with different retrieval objectives and seed queries.

The benchmark provides GOLD EVIDENCE. Gold evidence represents the
reference information that is considered necessary to answer the user's
question correctly.

Evaluate the following dimensions:

context_precision

How much of the retrieved evidence is relevant and useful for answering
the retrieval objective for which it was retrieved?

A high score means that most retrieved evidence contributes useful
information toward answering the user's question.

A low score means that much of the retrieved evidence is irrelevant,
off-topic, redundant, or provides little useful information.

Do NOT require retrieved evidence to exactly match the gold evidence.
Retrieved evidence can be valid and useful even if it differs from the
gold evidence.

context_relevance

How directly does the retrieved evidence address the user's information
need and the retrieval objective?

Penalize evidence that is merely related to the topic or entity but does
not provide information useful for answering the question.

context_recall

How completely does the retrieved evidence cover the information
necessary to answer the user's question?

Use the GOLD EVIDENCE as the reference for the information that should
be available.

For multi-hop questions, determine whether the retrieved evidence covers
the important information represented by the gold evidence across the
necessary retrieval objectives.

High recall means that the retrieved evidence contains essentially all
important information needed to answer the question.

Low recall means that important information represented by the gold
evidence is missing from the retrieved evidence.

evidence_sufficiency

Taken as a whole, is the retrieved evidence sufficient to support a
correct answer to the user's question?

Use the GOLD EVIDENCE as the reference when determining whether important
information is missing.

A collection can have high precision and relevance but still have low
sufficiency if an important part of the answer is missing.

Scoring:

0.0 = completely inadequate
0.25 = substantially inadequate
0.5 = partially adequate; important deficiencies remain
0.75 = adequate; minor deficiencies remain
0.9 = very strong; only negligible deficiencies
1.0 = essentially flawless

Evaluate each dimension independently.

Important distinctions:

Precision asks how much of the retrieved evidence is useful.
Relevance asks how directly the retrieved evidence addresses the
information need.
Recall asks how completely the retrieved evidence covers the necessary
information, using the gold evidence as the reference.
Sufficiency asks whether the retrieved evidence as a whole is enough
to support a correct answer.

Do not assume that highly ranked evidence is correct merely because it
has a high retrieval score.

Do not use retrieval scores as a substitute for judging evidence quality.

Do not penalize retrieved evidence merely because it is different from
the gold evidence if it provides equivalent information.

Return ONLY valid JSON with exactly this structure:

{
"context_precision": 0.0,
"context_relevance": 0.0,
"context_recall": 0.0,
"evidence_sufficiency": 0.0,
"critical_failure": false,
"comment": "Brief explanation of the evaluation."
}

Do not add any additional fields.
"""


def _build_prompt(query, gold_evidence, retrieval_plan, results):

    evaluate = f"""
    Evaluate the following retrieved evidence.

    USER QUERY:
    {query}

    RETRIEVAL PLAN:
    {json.dumps(retrieval_plan, indent=2, ensure_ascii=False)}

    RETRIEVAL RESULTS:
    {json.dumps(results, indent=2, ensure_ascii=False)}

    GOLD EVIDENCE:
    {json.dumps(gold_evidence, indent=2, ensure_ascii=False)}

    Remember:
    - Evaluate the retrieved evidence, not the planner.
    - Consider evidence in relation to its objective.
    - Consider the complete evidence collection when judging recall
      and sufficiency.
    - For multi-hop objectives, determine whether the evidence supports
      the necessary intermediate and final information.
    - Do not evaluate the final generated answer.
    - Return only the required JSON object.
    """

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": evaluate
        }
    ]

    return messages


def evaluate_evidence(
    query,
    gold_evidence,
    retrieval_plan,
    results,
    generate
):

    messages = _build_prompt(
        query=query,
        retrieval_plan=retrieval_plan,
        results=results,
        gold_evidence=gold_evidence
    )

    hyperparameters = {
    "temperature": 0.1,
    "top_p": 0.9,
    "top_k": 40,
    "repetition_penalty": 1.05}
    
    response = generate(messages, hyperparameters=hyperparameters)

    try:
        evaluation = EvidenceEvaluation.model_validate_json(
            response.content
        )

        return evaluation.model_dump()

    except Exception as e:
        return {
            "error": "Invalid evaluator response",
            "details": str(e),
            "raw_output": response.content
        }