"""Construct the modified HotpotQA context-removal dataset."""
import random
from datasets import DatasetDict, load_dataset


def map_example(example, index, seed=42, choices=(0, 1, 2)):
    rng = random.Random(seed + index)
    choice = rng.choice(choices)
    supporting = example["supporting_facts"]["title"]
    distractors = sorted(set(example["context"]["title"]) - set(supporting))
    try:
        if choice == 1:
            removed = [rng.choice(supporting), rng.choice(distractors)]
        elif choice == 2:
            removed = supporting
        else:
            removed = rng.sample(distractors, 2)
    except (IndexError, ValueError):
        choice, removed = 0, []
    paragraphs = []
    for title, sentences in zip(example["context"]["title"], example["context"]["sentences"]):
        if title not in removed:
            text = "\n".join(sentences)
            paragraphs.append(
                f"Paragraph {len(paragraphs)} \n\n {text} \n\n--------------------------------\n\n"
            )
    instruction = (
        " Your answer will be verified with exact match score. To ensure correct verification, "
        "only provide the answer within the <answer> </answer> tags. Do not put any sentences "
        "or reasoning process within the <answer> </answer> tags."
    )
    return {
        "problem": f"Question: {example['question']} \n\n{instruction} \n\n"
        f"Supporting Information: {''.join(paragraphs)} \n\n",
        "source": "hotpot",
        "gold_removed": choice,
        "removed_titles": removed,
    }


def build_dataset(seed=42):
    dataset = load_dataset("hotpotqa/hotpot_qa", "distractor")
    dataset = dataset.map(map_example, with_indices=True, fn_kwargs={"seed": seed})
    dataset = dataset.filter(lambda example: len(example["problem"]) < 6000)
    return DatasetDict(
        {
            "train": dataset["train"].select(range(20000)).remove_columns("question"),
            "test": dataset["validation"].select(range(500)).remove_columns("question"),
        }
    )
