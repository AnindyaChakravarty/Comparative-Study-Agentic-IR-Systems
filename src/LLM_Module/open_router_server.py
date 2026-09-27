import os
import requests
import os
from dotenv import load_dotenv
from LLM_Module import llm_response

load_dotenv()

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"#"https://openrouter.ai/api/v1"

OPENROUTER_MODEL = (
    "nvidia/nemotron-3-ultra-550b-a55b:free"
)


def generate(
    messages: list[dict],
    hyperparameters: dict | None = None,
    response_format: dict | None = None
) -> llm_response.LLMResponse:
    
    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY environment variable is not set."
        )

    # ---------------------------------------------------------
    # Build payload
    # ---------------------------------------------------------

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": messages,
    }

    # ---------------------------------------------------------
    # Add hyperparameters
    # ---------------------------------------------------------

    if hyperparameters:
        payload.update(hyperparameters)

    # ---------------------------------------------------------
    # Response format
    # ---------------------------------------------------------

    if response_format:
        payload["response_format"] = response_format

    # ---------------------------------------------------------
    # Headers
    # ---------------------------------------------------------

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # ---------------------------------------------------------
    # Request
    # ---------------------------------------------------------

    response = requests.post(
        OPENROUTER_URL,
        headers=headers,
        json=payload
    )

    response.raise_for_status()

    try:
        data = response.json()
    except requests.exceptions.JSONDecodeError as exc:
        raise RuntimeError(
            "OpenRouter returned a non-JSON response.\n"
            f"Status code: {response.status_code}\n"
            f"Response body: {response.text!r}"
        ) from exc

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
        }
    )

    print("CONTENT:")
    print(response.content)

    print("\nSTATS:")
    print(response.stats)

