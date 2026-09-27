import json
from LLM_Module import llm_router


SYSTEM_PROMPT = """
You are a Query and Planning Evaluation Agent.

Your task is to evaluate the quality of:
1. the Query Handler's interpretation of the user's query
2. the Planner's retrieval plan
3. the Planner's execution of that plan

You are NOT evaluating the quality of the retrieved evidence itself.
You are NOT evaluating the final generated answer.

Evaluate whether the query was correctly understood and whether the
retrieval plan and its execution are logically appropriate for answering
the user's information need.

The system may use multi-hop retrieval. An objective may depend on the
result of an earlier objective. In such cases, evaluate whether:
- the dependency is justified
- the earlier objective produces the information required by the later one
- the later objective correctly uses that intermediate result
- the overall execution progresses logically toward answering the query

Scoring:

0.0 = completely inadequate
0.25 = mostly inadequate
0.5 = partially adequate
0.75 = mostly adequate
1.0 = fully adequate

Evaluate each dimension independently.

Query quality:
- intent: Does the stated intent correctly represent the user's information need?
- entities: Are the important entities correctly identified?
- concepts: Are the important concepts correctly identified?
- relationships: Are the relevant relationships correctly identified?
- completeness: Does the QueryObject capture the important aspects of the query?

Planning quality:
- objective_decomposition: Are the objectives an appropriate decomposition of the information need?
- dependencies: Are objective dependencies logically correct and sufficient?
- seed_queries: Are the seed queries appropriate for their objectives?
- execution: Does the actual execution follow the plan correctly, including
  intermediate results and dependent objectives?

A score should reflect the quality of the operation, not whether the operation
happened to produce the correct final answer.

Return ONLY valid JSON with exactly this structure:

{
    "query_quality": {
        "intent": 0.0,
        "entities": 0.0,
        "concepts": 0.0,
        "relationships": 0.0,
        "completeness": 0.0
    },
    "planning_quality": {
        "objective_decomposition": 0.0,
        "dependencies": 0.0,
        "seed_queries": 0.0,
        "execution": 0.0
    },
    "critical_failure": false,
    "comment": "Brief explanation of the evaluation."
}

Do not add any additional fields.
"""


def _build_prompt(query,query_object,retrieval_plan,results):


    evaluate = f"""
    Evaluate the following retrieval pipeline execution.

    USER QUERY:
    {query}

    QUERY HANDLER OUTPUT:
    {query_object}

    RETRIEVAL PLAN:
    {json.dumps(retrieval_plan, indent=2, ensure_ascii=False)}

    EXECUTION RESULTS:
    {json.dumps(results, indent=2, ensure_ascii=False)}

    Remember:
    - Evaluate the Query Handler and Planner.
    - Do not evaluate whether the retrieved evidence is factually sufficient.
    - Do not evaluate the final answer.
    - For multi-hop objectives, inspect dependencies and intermediate results.
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


def evaluate_query_planning(query,query_object,retrieval_plan,results,generate):

    messages = _build_prompt(
        query=query,
        query_object=query_object,
        retrieval_plan=retrieval_plan,
        results=results
    )

    hyperparameters = {
        "temperature": 0.1,
        "top_p": 0.9,
        "top_k": 40,
        "repetition_penalty": 1.05}

    response = generate(
        messages=messages,
        hyperparameters=hyperparameters
    )

    content = response.content.strip()

    try:
        return json.loads(content)

    except json.JSONDecodeError:
        return {
            "error": "Evaluator returned invalid JSON",
            "raw_output": content
        }



if __name__ == "__main__":

    query = "Who succeeded Steve Jobs as CEO and what do they do now?"

    query_object = {
        "status": "ready",
        "query": "Who succeeded Steve Jobs as CEO and what do they do now?",
        "intent": (
            "Find out who became CEO of Apple after Steve Jobs' departure "
            "and identify their current role or activities"
        ),
        "entities": [
            "Steve Jobs",
            "Apple"
        ],
        "concepts": [
            "CEO succession",
            "corporate leadership",
            "executive roles"
        ],
        "relationships": [
            "successor relationship between Steve Jobs and his successor at Apple"
        ],
        "clarification": None
    }

    retrieval_plan = {
    "objectives": [
        {
            "objective": "Identify who succeeded Steve Jobs as CEO of Apple.",
            "seed_queries": [
                "Steve Jobs biography"
            ],
            "depends_on": [],
            "intermediate_result": None
        },
        {
            "objective": "Determine the current role of the successor.",
            "seed_queries": [
                "Steve Jobs current position"
            ],
            "depends_on": [1],
            "intermediate_result": None
        }
    ]
    }

    results = [
        {
            "objective": "Identify who succeeded Steve Jobs as CEO of Apple.",
            "depends_on": [],
            "seed_queries": [
                "who succeeded Steve Jobs as CEO",
                "Apple CEO after Steve Jobs"
            ],
            "evidence": [
                {
                    "text": (
                        "Tim Cook succeeded Steve Jobs as CEO of Apple "
                        "in August 2011."
                    ),
                    "metadata": {
                        "title": "Tim Cook",
                        "sub_heading": "Apple CEO",
                        "source": "wikipedia"
                    },
                    "score": 9.42
                },
                {
                    "text": (
                        "Steve Jobs resigned as chief executive officer "
                        "of Apple in August 2011 and was succeeded by "
                        "Tim Cook."
                    ),
                    "metadata": {
                        "title": "Steve Jobs",
                        "sub_heading": "Resignation",
                        "source": "wikipedia"
                    },
                    "score": 8.87
                }
            ],
            "answer": "Tim Cook"
        },
        {
            "objective": (
                "Determine the current role and activities of the "
                "person who succeeded Steve Jobs."
            ),
            "depends_on": [1],
            "seed_queries": [
                "Tim Cook current position",
                "Tim Cook what do they do now"
            ],
            "evidence": [
                {
                    "text": (
                        "Tim Cook is the chief executive officer of Apple "
                        "and has served as CEO since August 2011."
                    ),
                    "metadata": {
                        "title": "Tim Cook",
                        "sub_heading": "CEO of Apple",
                        "source": "wikipedia"
                    },
                    "score": 9.31
                },
                {
                    "text": (
                        "Cook has advocated for privacy, environmental "
                        "protection, and other public policy issues during "
                        "his tenure at Apple."
                    ),
                    "metadata": {
                        "title": "Tim Cook",
                        "sub_heading": "Public policy",
                        "source": "wikipedia"
                    },
                    "score": 7.14
                }
            ],
            "answer": None
        }
    ]

    evaluation = evaluate_query_planning(
        query=query,
        query_object=query_object,
        retrieval_plan=retrieval_plan,
        results=results,
        generate=llm_router.generate
    )

    print(json.dumps(evaluation, indent=2, ensure_ascii=False))