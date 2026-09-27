from Query_Planner.query_handler import query_handler
from Query_Planner.planner import create_retrieval_plan

from RAG.orchestration import RAGTool

from Answer_Generation.ans_generation import (
    evidence_preprocessing,
    generate_answer,
    generate_intermediate_result
)
import time
import main_logger

pipeline_start = time.perf_counter()

rag = RAGTool()
main_logger.log_event("Initializing RAG tool.")
print("\nLoading RAG...")
main_logger.log_event("Loading RAG.")
rag.load()
main_logger.log_event("RAG loaded successfully.")
print("RAG loaded.")

def get_total_time():
    return time.perf_counter() - pipeline_start

def run_pipeline(original_query):

    MAX_CLARIFICATIONS = 3

    clarification_history = []
    query_object = None
    query_stats = None

    main_logger.log_event(
        f"Pipeline started for query: {original_query}"
    )
    main_logger.log_data(
        "Original user query",
        original_query
    )

    # ---------------------------------------------------------
    # 1. Understand the user's query
    # ---------------------------------------------------------

    main_logger.log_event(
        "Starting query understanding and clarification phase."
    )

    for attempt in range(MAX_CLARIFICATIONS + 1):

        main_logger.log_event(
            f"Query handler attempt {attempt + 1}/{MAX_CLARIFICATIONS + 1}."
        )

        query = original_query

        for clarification in clarification_history:
            query += f"\nAdditional context: {clarification}"

        main_logger.log_data(
            "Query submitted to query handler",
            query
        )

        query_object, history, query_stats = query_handler(query)

        main_logger.log_data(
            "Query handler output",
            query_object
        )
        main_logger.log_data(
            "Query handler statistics",
            query_stats
        )

        if query_object.status == "ready":

            main_logger.log_event(
                "Query handler determined that the information need is ready."
            )

            break

        if attempt == MAX_CLARIFICATIONS:

            main_logger.log_event(
                "Maximum clarification limit reached."
            )
            main_logger.log_data(
                "Final query object after clarification limit",
                query_object
            )

            print("\nUnable to determine the user's information need.")
            return {
                "status": "max_clarifications_limit_exceeded",
                "query": original_query,
                "query_object": query_object,
                "llm_stats": {
                    "query_stats": query_stats
                },
                "total_time": get_total_time()
            }

        main_logger.log_event(
            "Query handler requested clarification."
        )
        main_logger.log_data(
            "Clarification requested",
            query_object.clarification
        )

        print(f"\nAssistant: {query_object.clarification}")

        clarification = input("User: ")

        main_logger.log_data(
            "User clarification",
            clarification
        )

        clarification_history.append(clarification)


    if query_object is None:

        main_logger.log_event(
            "Query handler failed to produce a query object."
        )

        return {
            "status": "query_handler_failed",
            "query": original_query,
            "llm_stats": {
                "query_stats": query_stats
            },
            "total_time": get_total_time()
        }

    # ---------------------------------------------------------
    # 2. Create retrieval plan
    # ---------------------------------------------------------

    main_logger.log_event(
        "Starting retrieval plan creation."
    )

    retrieval_plan, planner_stats = create_retrieval_plan(
        query=query_object.query
    )

    main_logger.log_data(
        "Retrieval plan",
        retrieval_plan
    )
    main_logger.log_data(
        "Planner statistics",
        planner_stats
    )

    print("\nRetrieval Plan:")
    print(retrieval_plan)

    if not retrieval_plan["objectives"]:

        main_logger.log_event(
            "No valid retrieval objectives were created."
        )

        return {
            "status": "no_valid_retrieval_objectives",
            "query": original_query,
            "query_object": query_object,
            "llm_stats": {
                "query_stats": query_stats,
                "planner_stats": planner_stats
            },
            "total_time": get_total_time()
        }

    main_logger.log_event(
        f"Retrieval plan created with "
        f"{len(retrieval_plan['objectives'])} objective(s)."
    )


    

    # ---------------------------------------------------------
    # 4. Execute retrieval objectives
    # ---------------------------------------------------------

    main_logger.log_event(
        "Starting retrieval objective execution."
    )
    retrieval_time = 0.0
    results = []

    for i, objective in enumerate(
        retrieval_plan["objectives"],
        start=1
    ):

        main_logger.log_event(
            f"Starting retrieval objective {i}."
        )
        main_logger.log_data(
            f"Retrieval objective {i}",
            objective
        )

        print(f"\nObjective {i}:")
        print(objective["objective"])

        # Resolve dependencies
        for dependency in objective["depends_on"]:

            main_logger.log_event(
                f"Resolving dependency for objective {i}: "
                f"objective {dependency}."
            )

            previous_result = results[dependency - 1]

            # Use the generated answer from the previous objective
            answer = previous_result["answer"]

            main_logger.log_data(
                f"Dependency answer used by objective {i}",
                answer
            )

            objective["seed_queries"] = [
                seed_query.replace(
                    "{previous_result}",
                    answer
                )
                for seed_query in objective["seed_queries"]
            ]

            main_logger.log_data(
                f"Seed queries after dependency resolution for objective {i}",
                objective["seed_queries"]
            )

        # Remove unresolved placeholder queries
        valid_seed_queries = []

        for seed_query in objective["seed_queries"]:

            if "{" in seed_query or "}" in seed_query:

                main_logger.log_event(
                    f"Skipping unresolved seed query for objective {i}: "
                    f"{seed_query}"
                )

                print(
                    f"Warning: skipping unresolved seed query: {seed_query}"
                )
                continue

            valid_seed_queries.append(seed_query)

        objective["seed_queries"] = valid_seed_queries

        main_logger.log_data(
            f"Valid seed queries for objective {i}",
            objective["seed_queries"]
        )

        objective_evidence = []

        # -----------------------------------------------------
        # Execute only the first seed query
        # -----------------------------------------------------
        retrieval_start = time.perf_counter()

        for seed_query in objective["seed_queries"][:1]:

            main_logger.log_event(
                f"Executing RAG retrieval for objective {i}."
            )
            main_logger.log_data(
                f"Seed query for objective {i}",
                seed_query
            )

            print(f"\nSeed query: {seed_query}")

            rag_evidence = rag.query(seed_query)

            main_logger.log_event(
                f"RAG retrieval completed for objective {i}. "
                f"Retrieved {len(rag_evidence)} evidence documents."
            )
            main_logger.log_data(
                f"RAG evidence for objective {i}",
                rag_evidence
            )

            print(
                f"Retrieved {len(rag_evidence)} evidence documents."
            )

            objective_evidence.extend(rag_evidence)

        retrieval_time += time.perf_counter() - retrieval_start

        main_logger.log_data(
            f"Total evidence collected for objective {i}",
            objective_evidence
        )
        main_logger.log_event(
            f"Retrieval for objective {i} completed."
        )

        # -----------------------------------------------------
        # Generate intermediate result if another objective
        # depends on this one
        # -----------------------------------------------------

        has_dependents = any(
            i in other["depends_on"]
            for other in retrieval_plan["objectives"]
        )

        if has_dependents:

            main_logger.log_event(
                f"Objective {i} has dependent objectives. "
                "Generating intermediate result."
            )

            evidence = evidence_preprocessing(
                rag_evidence=objective_evidence
            )

            main_logger.log_data(
                f"Preprocessed evidence for objective {i}",
                evidence
            )

            intermediate_response = generate_intermediate_result(
                objective=objective["objective"],
                evidence=evidence
            )

            main_logger.log_event(
                f"Intermediate result generated for objective {i}."
            )
            main_logger.log_data(
                f"Intermediate response for objective {i}",
                intermediate_response
            )

            answer = intermediate_response.content.strip()

            main_logger.log_data(
                f"Intermediate answer for objective {i}",
                answer
            )

            print("\nIntermediate result:")
            print(answer)

            if answer == "NOT_FOUND":

                main_logger.log_event(
                    f"Intermediate result NOT_FOUND for objective {i}. "
                    "Stopping retrieval pipeline."
                )

                print(
                    "\nUnable to continue retrieval: "
                    "required intermediate information was not found."
                )

                return {
                    "status": "intermediate_result_not_found",
                    "query": original_query,
                    "query_object": query_object,
                    "retrieval_plan": retrieval_plan,
                    "results": results,
                    "llm_stats": {
                        "query_stats": query_stats,
                        "planner_stats": planner_stats
                    },
                    "total_time": get_total_time()
                }

        else:

            main_logger.log_event(
                f"Objective {i} has no dependent objectives. "
                "Skipping intermediate result generation."
            )

            answer = None

        results.append(
            {
                "evidence": objective_evidence,
                "answer": answer
            }
        )

        main_logger.log_data(
            f"Completed result for objective {i}",
            results[-1]
        )

        main_logger.log_event(
            f"Retrieval objective {i} completed successfully."
        )

    # ---------------------------------------------------------
    # 5. Collect final evidence
    # ---------------------------------------------------------

    main_logger.log_event(
        "Collecting evidence from all retrieval objectives."
    )

    final_evidence = []

    for result in results:
        final_evidence.extend(result["evidence"])

    main_logger.log_data(
        "Final collected evidence",
        final_evidence
    )

    evidence = evidence_preprocessing(
        rag_evidence=final_evidence
    )

    main_logger.log_data(
        "Preprocessed final evidence",
        evidence
    )

    main_logger.log_event(
        "Final evidence preprocessing completed."
    )

    # ---------------------------------------------------------
    # 6. Generate final answer
    # ---------------------------------------------------------

    main_logger.log_event(
        "Starting final answer generation."
    )

    answer_response = generate_answer(
        query=query_object.query,
        evidence=evidence
    )

    main_logger.log_event(
        "Final answer generated successfully."
    )
    main_logger.log_data(
        "Final answer response",
        answer_response
    )
    main_logger.log_data(
        "Final answer text",
        answer_response.content
    )

    print("\nAnswer:")
    print(answer_response.content)

    total_time = get_total_time()

    main_logger.log_event(
        f"Pipeline completed successfully in {total_time:.4f} seconds."
    )

    main_logger.log_data(
        "Final pipeline result",
        {
            "query": original_query,
            "query_object": query_object,
            "retrieval_plan": retrieval_plan,
            "results": results,
            "answer": answer_response,
            "total_time": total_time,
            "status": "success"
        }
    )

    return {
        "query": original_query,
        "query_object": query_object,
        "retrieval_plan": retrieval_plan,
        "results": results,
        "answer": answer_response,
        "llm_stats": {
            "query_stats": query_stats,
            "planner_stats": planner_stats,
            "answer_generation": answer_response.stats
        },
        "total_time": get_total_time(),
        "status": "success"
    }


def main():

    main_logger.log_event(
        "Main RAG application started."
    )

    original_query = input("User: ")

    main_logger.log_event(
        "User query received. Starting RAG pipeline."
    )

    result = run_pipeline(original_query)

    main_logger.log_event(
        f"Pipeline returned with status: {result.get('status')}"
    )

    # print(result["answer"])

    print(result)

    main_logger.log_event(
        "Main RAG application finished."
    )


if __name__ == "__main__":
    main()