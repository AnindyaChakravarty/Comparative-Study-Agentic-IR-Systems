from LLM_Module.llm_response import LLMStats


class LLMStatsTracker:

    def __init__(self):
        self.calls = []

    def record(self, step: str, stats: LLMStats):
        self.calls.append(
            {
                "step": step,
                "input_tokens": stats.input_tokens,
                "output_tokens": stats.output_tokens,
                "reasoning_tokens": stats.reasoning_tokens,
                "total_tokens": stats.total_tokens,
                "ttft": stats.ttft,
                "tokens_per_second": stats.tokens_per_second
            }
        )

    def summary(self) -> dict:

        return {
            "calls": len(self.calls),
            "input_tokens": self._sum("input_tokens"),
            "output_tokens": self._sum("output_tokens"),
            "reasoning_tokens": self._sum("reasoning_tokens"),
            "total_tokens": self._sum("total_tokens"),
            "ttft": None,
            "tokens_per_second": None,
            "details": self.calls
        }

    def _sum(self, field: str):

        values = [
            call[field]
            for call in self.calls
            if call[field] is not None
        ]

        if not values:
            return None

        return sum(values)