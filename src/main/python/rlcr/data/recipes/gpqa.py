"""Prepare shuffled multiple-choice GPQA evaluation questions."""
import random
from datasets import DatasetDict, load_dataset


def map_example(example, index, seed=42):
    options = [example["Correct Answer"], *(example[f"Incorrect Answer {i}"] for i in range(1, 4))]
    random.Random(seed + index).shuffle(options)
    label = "ABCD"[options.index(example["Correct Answer"])]
    difficulty = example["Writer's Difficulty Estimate"]
    difficulty = (
        1
        if difficulty is None or "undergraduate" in difficulty
        else 3
        if "Post-graduate" in difficulty
        else 2
    )
    return {
        "problem": example["Question"]
        + "\n"
        + "".join(f"{letter}) {option}\n" for letter, option in zip("ABCD", options)),
        "answer": f"Option {label} which is [{example['Correct Answer']}]",
        "difficulty": difficulty,
    }


def build_dataset(seed=42):
    dataset = load_dataset("Idavidrein/gpqa", "gpqa_main", split="train")
    dataset = dataset.map(map_example, with_indices=True, fn_kwargs={"seed": seed})
    keep = {
        "problem",
        "answer",
        "option",
        "difficulty",
        "Correct Answer",
        "Incorrect Answer 1",
        "Incorrect Answer 2",
        "Incorrect Answer 3",
    }
    dataset = dataset.remove_columns(
        [column for column in dataset.column_names if column not in keep]
    )
    return DatasetDict({"test": dataset})
