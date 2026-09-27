import json
import time

from Evaluation_Agent.benchmark.dataset import load_benchmark_examples
from main import run_pipeline, global_llm_stats
from Evaluation_Agent import evaluation_orchestrator
from dataclasses import asdict


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
    output_file="benchmark_results.jsonl"
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

            # Reset LLM statistics for this question
            global_llm_stats.clear()

            start_time = time.perf_counter()

            try:
                result = run_pipeline(example["question"])
                success = True
                error = None

            except Exception as exc:
                result = None
                success = False
                error = repr(exc)

            latency = time.perf_counter() - start_time

            # Preserve all LLM calls made during this question
            question_llm_stats = [
                asdict(stats)
                for stats in global_llm_stats
            ]

            # Aggregate statistics
            llm_usage = {
                "num_calls": len(question_llm_stats),
                "input_tokens": sum(
                    stats["input_tokens"]
                    for stats in question_llm_stats
                ),
                "output_tokens": sum(
                    stats["output_tokens"]
                    for stats in question_llm_stats
                ),
                "reasoning_tokens": sum(
                    stats["reasoning_tokens"]
                    for stats in question_llm_stats
                ),
                "total_tokens": sum(
                    stats["total_tokens"]
                    for stats in question_llm_stats
                ),
                "avg_input_tokens_per_call": (
                    sum(stats["input_tokens"] for stats in question_llm_stats)
                    / len(question_llm_stats)
                    if question_llm_stats else 0
                ),
                "avg_output_tokens_per_call": (
                    sum(stats["output_tokens"] for stats in question_llm_stats)
                    / len(question_llm_stats)
                    if question_llm_stats else 0
                ),
                "avg_total_tokens_per_call": (
                    sum(stats["total_tokens"] for stats in question_llm_stats)
                    / len(question_llm_stats)
                    if question_llm_stats else 0
                ),
                "calls": question_llm_stats
            }

            benchmark_result = {
                "id": example["id"],
                "question": example["question"],
                "gold_answer": example["gold_answer"],
                "gold_evidence": example["gold_evidence"],
                "type": example["type"],
                "system_result": _serialize(result),
                "success": success,
                "error": error,
                "latency": latency,
                "llm_usage": llm_usage
            }

            file.write(
                json.dumps(
                    benchmark_result,
                    ensure_ascii=False
                ) + "\n"
            )

            file.flush()


if __name__ == "__main__":
    run_benchmark(
                split="validation",
                limit=100,
                output_file="test_results.jsonl"
            )

    evaluation_orchestrator.evaluate_benchmark()