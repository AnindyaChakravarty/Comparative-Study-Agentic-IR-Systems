from dataclasses import dataclass


@dataclass
class LLMStats:
    input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None
    total_tokens: int | None = None
    ttft: float | None = None
    tokens_per_second: float | None = None


@dataclass
class LLMResponse:
    content: str
    stats: LLMStats