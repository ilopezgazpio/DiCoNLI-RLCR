"""Select numeric-answer Big-Math problems with intermediate solve rates."""
from datasets import DatasetDict, load_dataset


def keep_example(example):
    answer, rate = example["answer"], example["llama8b_solve_rate"]
    if rate is None or not (0 < rate < 0.75) or len(answer) >= 20 or "text" in answer:
        return False
    try:
        float(answer)
        return True
    except ValueError:
        return False


def build_dataset(seed=42):
    dataset = load_dataset("SynthLabsAI/Big-Math-RL-Verified", split="train")
    dataset = dataset.filter(keep_example).shuffle(seed=seed).select(range(40000))
    return DatasetDict(
        {
            "train": dataset.select(range(30000)),
            "test": dataset.select(range(30000, 31000)),
        }
    )
