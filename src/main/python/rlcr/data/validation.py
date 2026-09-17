"""Validate prepared prompts without dataset-specific transformations."""
from datasets import Dataset, DatasetDict


def validate_prompt_dataset(
    dataset, prompt_column="prompt", require_labels=False, allowed_labels=None
):
    """Preserve input columns; reject missing or malformed prompts/labels early."""
    if isinstance(dataset, DatasetDict):
        for split in dataset.values():
            validate_prompt_dataset(split, prompt_column, require_labels, allowed_labels)
        return dataset
    if not isinstance(dataset, Dataset):
        raise ValueError("Expected a Hugging Face Dataset or DatasetDict.")
    required = {prompt_column} | ({"label"} if require_labels else set())
    missing = required - set(dataset.column_names)
    if missing:
        raise ValueError(f"Prepared dataset is missing columns: {', '.join(sorted(missing))}")
    for index, row in enumerate(dataset):
        prompt = row[prompt_column]
        if isinstance(prompt, str):
            valid = bool(prompt.strip())
        else:
            valid = (
                isinstance(prompt, list)
                and bool(prompt)
                and all(
                    isinstance(message, dict)
                    and message.get("role") in {"system", "user", "assistant"}
                    and isinstance(message.get("content"), str)
                    and bool(message["content"].strip())
                    for message in prompt
                )
            )
        if not valid:
            raise ValueError(f"Invalid prepared prompt at row {index} in {prompt_column}.")
        if require_labels and (not isinstance(row["label"], str) or not row["label"].strip()):
            raise ValueError(f"Expected a nonempty string label at row {index}.")
        if require_labels and allowed_labels is not None and row["label"] not in allowed_labels:
            raise ValueError(f"Gold label at row {index} is outside the allowed vocabulary.")
    return dataset
