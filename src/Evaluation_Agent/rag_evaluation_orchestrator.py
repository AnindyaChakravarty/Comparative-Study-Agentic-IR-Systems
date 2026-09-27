import json

from Evaluation_Agent.answer_evaluator import evaluate_answer
from Evaluation_Agent.evidence_evaluator import evaluate_evidence
from Evaluation_Agent.query_planner_evaluator import evaluate_query_planning

from LLM_Module import llm_router


def evaluate_benchmark(
    input_file="rag_test_results.jsonl",
    output_file="rag_evaluation_results.jsonl",
    start_from=1
):

    with open(
        input_file,
        "r",
        encoding="utf-8"
    ) as input_handle, open(
        output_file,
        "a",
        encoding="utf-8"
    ) as output_handle:

        for index, line in enumerate(
            input_handle,
            start=1
        ):

            if index < start_from:
                continue

            record = json.loads(line)

            print(
                f"\n========== "
                f"RAG Evaluation {index} "
                f"=========="
            )

            print(
                f"Question: {record['question']}"
            )

            # -------------------------------------------------
            # Skip failed pipeline executions
            # -------------------------------------------------

            if not record["success"]:

                print("Skipping failed pipeline execution.")

                evaluation_result = {
                    "id": record["id"],
                    "question": record["question"],
                    "status": "pipeline_failed",
                    "error": record["error"]
                }

                output_handle.write(
                    json.dumps(
                        evaluation_result,
                        ensure_ascii=False
                    ) + "\n"
                )

                output_handle.flush()

                continue

            # -------------------------------------------------
            # Extract system result
            # -------------------------------------------------

            system_result = record.get("system_result")

            if not system_result:

                evaluation_result = {
                    "id": record["id"],
                    "question": record["question"],
                    "status": "pipeline_failed",
                    "error": record.get("error")
                }

                output_handle.write(
                    json.dumps(
                        evaluation_result,
                        ensure_ascii=False
                    ) + "\n"
                )

                output_handle.flush()

                print(
                    "Pipeline failed. Skipping evaluation."
                )

                continue

            # -------------------------------------------------
            # Check required pipeline fields
            # -------------------------------------------------

            required_fields = {
                "query_object",
                "retrieval_plan",
                "results",
                "answer"
            }

            missing_fields = (
                required_fields
                - system_result.keys()
            )

            if missing_fields:

                evaluation_result = {
                    "id": record["id"],
                    "question": record["question"],
                    "status": "pipeline_incomplete",
                    "pipeline_status": system_result.get("status"),
                    "missing_fields": sorted(missing_fields),
                    "error": record.get("error")
                }

                output_handle.write(
                    json.dumps(
                        evaluation_result,
                        ensure_ascii=False
                    ) + "\n"
                )

                output_handle.flush()

                print(
                    f"Pipeline incomplete. "
                    f"Missing fields: {sorted(missing_fields)}"
                )

                continue

            # -------------------------------------------------
            # Extract pipeline components
            # -------------------------------------------------

            query_object = system_result["query_object"]
            retrieval_plan = system_result["retrieval_plan"]
            results = system_result["results"]
            answer = system_result["answer"]

            if isinstance(answer, dict):
                answer = answer.get("content", "")

            # -------------------------------------------------
            # Query / Planning Evaluation
            # -------------------------------------------------

            print("Evaluating query and planning...")

            query_planning_evaluation = evaluate_query_planning(
                query=record["question"],
                query_object=query_object,
                retrieval_plan=retrieval_plan,
                results=results,
                generate=llm_router.generate
            )

            # -------------------------------------------------
            # Evidence Evaluation
            # -------------------------------------------------

            print("Evaluating evidence...")

            evidence_evaluation = evaluate_evidence(
                query=record["question"],
                gold_evidence=record["gold_evidence"],
                retrieval_plan=retrieval_plan,
                results=results,
                generate=llm_router.generate
            )

            # -------------------------------------------------
            # Answer Evaluation
            # -------------------------------------------------

            print("Evaluating answer...")

            answer_evaluation = evaluate_answer(
                query=record["question"],
                gold_answer=record["gold_answer"],
                evidence=results,
                answer=answer,
                generate=llm_router.generate
            )

            # -------------------------------------------------
            # Store evaluation
            # -------------------------------------------------

            evaluation_result = {
                "id": record["id"],
                "question": record["question"],
                "gold_answer": record["gold_answer"],
                "evaluations": {
                    "query_planning": query_planning_evaluation,
                    "evidence": evidence_evaluation,
                    "answer": answer_evaluation
                }
            }

            output_handle.write(
                json.dumps(
                    evaluation_result,
                    ensure_ascii=False
                ) + "\n"
            )

            output_handle.flush()

            print("RAG evaluation complete.")


if __name__ == "__main__":

    evaluate_benchmark(
        input_file="rag_test_results.jsonl",
        output_file="rag_evaluation_results.jsonl",
        start_from=1
    )