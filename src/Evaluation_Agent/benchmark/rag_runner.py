import json
import time

from Evaluation_Agent.benchmark.dataset import load_benchmark_examples
from main_rag import run_pipeline
from Evaluation_Agent import rag_evaluation_orchestrator


def _serialize(value):

    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    if hasattr(value, "model_dump"):
        return _serialize(value.model_dump())

    if isinstance(value, dict):
        return {
            key: _serialize(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [
            _serialize(item)
            for item in value
        ]

    if isinstance(value, tuple):
        return [
            _serialize(item)
            for item in value
        ]

    if hasattr(value, "__dict__"):
        return _serialize(vars(value))

    return str(value)


def run_benchmark(
    split="validation",
    limit=None,
    output_file="rag_test_results.jsonl"
):

    examples = load_benchmark_examples(
        split=split,
        limit=limit
    )

    total = len(examples)

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        for index, example in enumerate(examples, start=1):

            print(
                f"\n[{index}/{total}] "
                f"{example['question']}"
            )

            start_time = time.perf_counter()

            try:
                result = run_pipeline(
                    example["question"]
                )

                success = True
                error = None

            except Exception as exc:
                result = None
                success = False
                error = repr(exc)

            latency = time.perf_counter() - start_time

            benchmark_result = {
                "id": example["id"],
                "question": example["question"],
                "gold_answer": example["gold_answer"],
                "gold_evidence": example["gold_evidence"],
                "type": example["type"],
                "system_result": _serialize(result),
                "success": success,
                "error": error,
                "latency": latency
            }

            file.write(
                json.dumps(
                    benchmark_result,
                    ensure_ascii=False
                ) + "\n"
            )

            file.flush()

            print(
                f"Completed in {latency:.2f}s"
            )

            if not success:
                print(
                    f"Pipeline failed: {error}"
                )


if __name__ == "__main__":

    run_benchmark(
        split="validation",
        limit=100,
        output_file="rag_test_results.jsonl"
    )

    rag_evaluation_orchestrator.evaluate_benchmark()