"""Prepare TriviaQA evaluation answers as alias lists."""
from datasets import DatasetDict, load_dataset


def keep_example(example):
    return len(example["answer"]["aliases"]) <= 5


def map_example(example):
    return {"answer": example["answer"]["aliases"]}


def build_dataset(seed=42):
    dataset = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")
    dataset = dataset.filter(keep_example).map(map_example)
    dataset = dataset.remove_columns(["question_source", "entity_pages", "search_results"])
    dataset = dataset.shuffle(seed=seed).select(range(2000))
    return DatasetDict({"test": dataset})
