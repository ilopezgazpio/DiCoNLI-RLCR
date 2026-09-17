"""Incremental evaluation must preserve predictions, metrics, and input data."""
import json

from datasets import Dataset, DatasetDict, load_from_disk
import pytest

from rlcr.arguments.evaluation_global_args import GlobalArgs
from rlcr.arguments.evaluation_model_config import LocalConfig
from rlcr.evaluation import runner, storage


@pytest.fixture
def evaluation_run(tmp_path, monkeypatch):
    dataset_dir = tmp_path / "input"
    DatasetDict(
        {
            "test": Dataset.from_dict(
                {"instance_id": ["1", "2"], "prompt": ["one", "two"], "label": ["a", "b"]}
            )
        }
    ).save_to_disk(dataset_dir)
    args = GlobalArgs(dataset_name=str(dataset_dir), output_dir=str(tmp_path / "output"))
    model = LocalConfig(name="first", model="fixture")
    calls = []

    def generate(dataset, config):
        calls.append(config.name)
        return {f"{config.name}-output_0": ["raw prediction"] * len(dataset)}, {
            "examples": len(dataset),
            "completions": len(dataset),
        }

    monkeypatch.setattr(runner, "generate_columns", generate)
    return args, model, calls, tmp_path / "output"


def test_rerun_preserves_metrics_and_adds_models(evaluation_run):
    args, model, calls, output = evaluation_run
    expected = {"first": {"examples": 2, "completions": 2}}
    assert runner.run_evaluation(args, [model]) == expected
    assert runner.run_evaluation(args, [model]) == expected
    assert calls == ["first"]
    assert json.loads((output / "metrics.json").read_text()) == expected
    second = LocalConfig(name="second", model="fixture")
    expected["second"] = {"examples": 2, "completions": 2}
    assert runner.run_evaluation(args, [model, second]) == expected
    assert calls == ["first", "second"]
    saved = load_from_disk(output / "predictions")
    assert {"first-output_0", "second-output_0"} <= set(saved.column_names)
    assert "first-output_0" not in load_from_disk(args.dataset_name)["test"].column_names


def test_fresh_rerun_can_replace_saved_arrow_data(evaluation_run):
    args, model, calls, output = evaluation_run
    runner.run_evaluation(args, [model])
    args.fresh = True
    runner.run_evaluation(args, [model])
    assert calls == ["first", "first"]
    assert len(load_from_disk(output / "predictions")) == 2


def test_different_input_subset_does_not_overwrite_existing_run(evaluation_run):
    args, model, calls, output = evaluation_run
    runner.run_evaluation(args, [model])
    before = (output / "metrics.json").read_bytes()
    args.sample_size = 1
    with pytest.raises(ValueError, match="different examples"):
        runner.run_evaluation(args, [model])
    assert (output / "metrics.json").read_bytes() == before
    assert calls == ["first"]


def test_missing_output_never_queries_the_hub(tmp_path, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("Output directories must never be interpreted as Hub dataset IDs")

    monkeypatch.setattr(storage, "load_dataset", unexpected)
    args = GlobalArgs(dataset_name="unused", output_dir=str(tmp_path / "missing"))
    assert storage.load_existing_results(args) is None
