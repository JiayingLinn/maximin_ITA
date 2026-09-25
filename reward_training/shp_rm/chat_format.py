"""Chat formatting with response-preserving left truncation."""

from typing import Any


def build_messages(prompt: str, response: str) -> list[dict[str, str]]:
    """Use exactly the same two-message format for every domain model."""
    return [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]


def render_chat(tokenizer: Any, prompt: str, response: str) -> str:
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError("The tokenizer has no chat_template")
    return tokenizer.apply_chat_template(
        build_messages(prompt, response),
        tokenize=False,
        add_generation_prompt=False,
    )


def tokenize_pair_text(
    tokenizer: Any,
    prompt: str,
    response: str,
    max_length: int,
) -> dict[str, list[int]]:
    """Tokenize a conversation, truncating from the history side first.

    The assistant response is the suffix of the rendered chat. Keeping the
    final ``max_length`` tokens therefore retains the complete response whenever
    that response fits. Only an individually over-length response is truncated.
    """
    if max_length <= 0:
        raise ValueError("max_length must be positive")
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError("The tokenizer has no chat_template")

    input_ids = tokenizer.apply_chat_template(
        build_messages(prompt, response),
        tokenize=True,
        add_generation_prompt=False,
    )
    if hasattr(input_ids, "tolist"):
        input_ids = input_ids.tolist()
    if input_ids and isinstance(input_ids[0], list):
        if len(input_ids) != 1:
            raise ValueError("Expected one tokenized conversation")
        input_ids = input_ids[0]
    input_ids = list(input_ids[-max_length:])
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
    }
