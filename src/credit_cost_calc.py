"""
OpenAI Cost Calculation Helper.
"""

# Rates per 1,000,000 tokens (USD)
OPENAI_PRICING = {
    "LUNA": {
        "model": "gpt-4o-mini",
        "input_per_1m": 0.15,
        "output_per_1m": 0.60,
    },
    "TERRA": {
        "model": "gpt-4o",
        "input_per_1m": 2.50,
        "output_per_1m": 10.00,
    }
}


def get_credits_cost(
    doc_tokens: int,
    nr_pages: int,
    prompt_tokens: int = 550,
    tokens_output_app: int = 500,
    nr_of_calls: int = 1,
    model_choice: str = "Luna"
) -> float:
    """
    Calculate estimated OpenAI API cost based on input and output tokens.
    """
    key = model_choice.upper() if model_choice else "LUNA"
    if key not in OPENAI_PRICING:
        key = "LUNA"

    pricing = OPENAI_PRICING[key]

    total_input_tokens = (doc_tokens + prompt_tokens) * nr_of_calls
    total_output_tokens = tokens_output_app * nr_of_calls

    input_cost = (total_input_tokens / 1_000_000) * pricing["input_per_1m"]
    output_cost = (total_output_tokens / 1_000_000) * pricing["output_per_1m"]

    return input_cost + output_cost