from LLM_Module import lm_studio_server
from LLM_Module import open_router_server
from LLM_Module.llm_response import LLMResponse


# ---------------------------------------------------------
# Provider configuration
# ---------------------------------------------------------

GLOBAL_PROVIDER = "LMS"             #LMS for LM Studio or OR for OPENROUTER


# ---------------------------------------------------------
# LLM Router
# ---------------------------------------------------------

def generate(
    messages: list[dict],
    hyperparameters: dict | None = None,
    response_format: dict | None = None
) -> LLMResponse:

    if GLOBAL_PROVIDER == "LMS":

        return lm_studio_server.generate(
            messages=messages,
            hyperparameters=hyperparameters,
            response_format=response_format
        )

    elif GLOBAL_PROVIDER == "OR":

        return open_router_server.generate(
            messages=messages,
            hyperparameters=hyperparameters,
            response_format=response_format
        )

    else:
        raise ValueError(
            f"Unsupported LLM provider: {GLOBAL_PROVIDER}"
        )

if __name__ == "__main__":

    messages = [
        {
            "role": "user",
            "content": "What is the capital of France?"
        }
    ]

    response = generate(
        messages=messages,
        hyperparameters={
            "temperature": 0.1,
            "top_p": 0.9,
            "top_k": 40,
            "repetition_penalty": 1.05,
        }
    )

    print("CONTENT:")
    print(response.content)

    print("\nSTATS:")
    print(response.stats)