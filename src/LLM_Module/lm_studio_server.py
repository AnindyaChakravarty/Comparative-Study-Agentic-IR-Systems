import requests

from LLM_Module import llm_response


LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"


def generate(
    messages: list[dict],
    hyperparameters: dict | None = None,
    response_format: dict | None = None
) -> llm_response.LLMResponse:

    payload:dict = {
        "messages": messages,
    }

    # ---------------------------------------------------------
    # Add hyperparameters
    # ---------------------------------------------------------

    if hyperparameters:

        for key, value in hyperparameters.items():

            # LM Studio uses "repeat_penalty"
            # while the common interface uses
            # "repetition_penalty".
            if key == "repetition_penalty":
                payload["repeat_penalty"] = value
            else:
                payload[key] = value

    # ---------------------------------------------------------
    # Response format
    # ---------------------------------------------------------

    if response_format:
        payload["response_format"] = response_format

    # ---------------------------------------------------------
    # Request
    # ---------------------------------------------------------

    response = requests.post(
        LM_STUDIO_URL,
        json=payload
    )

    response.raise_for_status()

    data = response.json()

    # ---------------------------------------------------------
    # Extract response
    # ---------------------------------------------------------

    content = data["choices"][0]["message"]["content"]

    # ---------------------------------------------------------
    # Extract statistics
    # ---------------------------------------------------------

    usage = data.get("usage", {})

    completion_details = usage.get(
        "completion_tokens_details",
        {}
    )

    stats = llm_response.LLMStats(
        input_tokens=usage.get("prompt_tokens"),
        output_tokens=usage.get("completion_tokens"),
        reasoning_tokens=completion_details.get("reasoning_tokens"),
        total_tokens=usage.get("total_tokens")
    )

    # ---------------------------------------------------------
    # Return
    # ---------------------------------------------------------

    return llm_response.LLMResponse(
        content=content,
        stats=stats
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