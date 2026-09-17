"""Reject instruction-truncating DiCo-NLI inputs before loading policy weights."""
from rlcr.inference.prompts import render_prompt


def check_prompt_lengths(tokenizer, splits, maximum):
    if maximum is not None and maximum < 1:
        raise ValueError("max_prompt_length must be positive or null.")
    lengths = {}
    for name, dataset in splits.items():
        if dataset is None:
            continue
        largest = 0
        for index, row in enumerate(dataset):
            rendered = render_prompt(tokenizer, row["prompt"])
            size = len(tokenizer.encode(rendered, add_special_tokens=False))
            if maximum is not None and size > maximum:
                raise ValueError(
                    f"{name} row {index}: prompt needs {size} tokens, exceeding max_prompt_length "
                    f"{maximum}. DiCo-NLI instructions must not be truncated; increase the budget "
                    "within the model context limit or revise/version the prompt."
                )
            largest = max(largest, size)
        lengths[name] = largest
    return lengths
