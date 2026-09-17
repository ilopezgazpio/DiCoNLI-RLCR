"""Prepared data preserves identifiers/metadata and never invents task prompts."""
from types import SimpleNamespace

from datasets import Dataset, DatasetDict
import pytest

from rlcr.data.validation import validate_prompt_dataset
from rlcr.evaluation.storage import load_evaluation_dataset
from rlcr.arguments.evaluation_global_args import GlobalArgs
from rlcr.arguments.evaluation_model_config import LocalConfig
from rlcr.inference.completion import Completion
from rlcr.inference.generation import Generation
from rlcr.inference.prompts import render_prompt
from rlcr.training.datasets import load_training_datasets


@pytest.mark.parametrize("chat", [False, True])
def test_prepared_prompts_preserve_all_metadata(chat):
    prompt = [{"role": "user", "content": "Compare two sentences."}] if chat else "Input text"
    dataset = Dataset.from_dict(
        {
            "prompt": [prompt],
            "label": ["ENTAILMENT"],
            "instance_id": ["example-1"],
            "pair_id": ["pair-1"],
            "direction": ["forward"],
        }
    )
    for value in [dataset, DatasetDict({"train": dataset})]:
        assert validate_prompt_dataset(value, require_labels=True) is value


@pytest.mark.parametrize("prompt", ["", " ", [], None, [{"role": "tool", "content": "x"}], [{}]])
def test_invalid_prompts_are_rejected(prompt):
    dataset = Dataset.from_dict({"prompt": [prompt]})
    with pytest.raises(ValueError, match="Invalid prepared prompt"):
        validate_prompt_dataset(dataset)


def test_no_implicit_task_column_conversion():
    with pytest.raises(ValueError, match="missing columns"):
        validate_prompt_dataset(Dataset.from_dict({"text": ["input"]}))
    with pytest.raises(ValueError, match="missing columns"):
        validate_prompt_dataset(Dataset.from_dict({"prompt": ["input"]}), require_labels=True)
    with pytest.raises(ValueError, match="string label"):
        validate_prompt_dataset(
            Dataset.from_dict({"prompt": ["input"], "label": [1]}), require_labels=True
        )


def test_training_loader_preserves_prepared_data(tmp_path):
    dataset = Dataset.from_dict({"prompt": ["input"], "label": ["ENTAILMENT"], "pair_id": ["p1"]})
    DatasetDict({"train": dataset, "test": dataset}).save_to_disk(tmp_path)
    script = SimpleNamespace(
        dataset_name=str(tmp_path),
        dataset_config=None,
        dataset_train_split="train",
        dataset_test_split="test",
        train_subset_size=None,
        eval_subset_size=None,
        reward_funcs=["accuracy", "brier"],
    )
    train, evaluation = load_training_datasets(script, SimpleNamespace(eval_strategy="steps"))
    assert train.to_dict() == dataset.to_dict()
    assert evaluation.to_dict() == dataset.to_dict()


@pytest.mark.parametrize("ids", [["x", "x"], ["", "x"], [None, "x"], [1, 2]])
def test_evaluation_requires_unique_string_ids(tmp_path, ids):
    Dataset.from_dict({"instance_id": ids, "prompt": ["one", "two"]}).save_to_disk(tmp_path)
    with pytest.raises(ValueError, match="identifiers"):
        load_evaluation_dataset(GlobalArgs(dataset_name=str(tmp_path), output_dir="unused"))


def test_missing_evaluation_ids_are_not_hashed(tmp_path):
    Dataset.from_dict({"prompt": ["input"]}).save_to_disk(tmp_path)
    with pytest.raises(ValueError, match="instance_id"):
        load_evaluation_dataset(GlobalArgs(dataset_name=str(tmp_path), output_dir="unused"))


def test_rendering_does_not_wrap_plain_text(tiny_model):
    _, tokenizer = tiny_model
    assert render_prompt(tokenizer, "literal input") == "literal input"
    messages = [{"role": "user", "content": "literal input"}]
    assert render_prompt(tokenizer, messages) == tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


@pytest.mark.parametrize("chat", [False, True])
def test_batch_generation_preserves_raw_responses(tiny_model, monkeypatch, chat):
    from rlcr.evaluation import generation

    model, tokenizer = tiny_model
    calls = []
    raw = ["malformed", "<answer>ENTAILMENT</answer>"]
    prompt = [{"role": "user", "content": "input"}] if chat else "input"
    monkeypatch.setattr(generation, "load_tokenizer", lambda _: tokenizer)
    monkeypatch.setattr(generation, "load_hf_generator", lambda _: model)

    def generate(model, tokenizer, texts, **kwargs):
        calls.append(texts)
        assert kwargs["n"] == 2
        assert not kwargs.get("return_logprobs", False)
        return [Generation([Completion(text, []) for text in raw])]

    monkeypatch.setattr(generation, "hf_generate", generate)
    columns, stats = generation.generate_columns(
        Dataset.from_dict({"prompt": [prompt]}), LocalConfig(name="test", model="unused", n=2)
    )
    assert len(calls) == 1
    assert columns == {"test-output_0": [raw[0]], "test-output_1": [raw[1]]}
    assert stats == {"examples": 1, "completions": 2}
