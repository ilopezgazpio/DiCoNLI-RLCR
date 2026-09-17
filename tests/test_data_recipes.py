"""Dataset recipe transforms are testable without downloading the source data."""
from importlib import import_module
from types import SimpleNamespace

from datasets import Dataset, DatasetDict, load_from_disk
import pytest

from rlcr.cli import main
from rlcr.data import preparation
from rlcr.data.recipes import RECIPES
from rlcr.data.recipes.big_math_digits import keep_example
from rlcr.data.recipes.gpqa import map_example as map_gpqa
from rlcr.data.recipes.hotpotqa import map_example as map_hotpot
from rlcr.data.recipes.trivia import map_example as map_trivia


@pytest.mark.parametrize(
    "answer,rate,keep",
    [("0.25", 0.5, True), ("2", 0, False), ("2", None, False), ("text", 0.5, False)],
)
def test_math_filter(answer, rate, keep):
    assert keep_example({"answer": answer, "llama8b_solve_rate": rate}) is keep


def test_hotpot_transform_is_seeded_and_keeps_expected_columns():
    example = {
        "question": "question",
        "supporting_facts": {"title": ["a", "b"]},
        "context": {
            "title": ["a", "b", "c", "d"],
            "sentences": [["sentence a"], ["sentence b"], ["sentence c"], ["sentence d"]],
        },
    }
    first = map_hotpot(example, 0, seed=42)
    assert first == map_hotpot(example, 0, seed=42)
    assert first["source"] == "hotpot"
    for title in first["removed_titles"]:
        assert f"sentence {title}" not in first["problem"]


def test_gpqa_transform_is_seeded_and_keeps_correct_label():
    example = {
        "Question": "question",
        "Correct Answer": "yes",
        "Incorrect Answer 1": "no",
        "Incorrect Answer 2": "maybe",
        "Incorrect Answer 3": "unknown",
        "Writer's Difficulty Estimate": "Post-graduate",
    }
    result = map_gpqa(example, 0, seed=42)
    assert result == map_gpqa(example, 0, seed=42)
    label = result["answer"].split()[1]
    assert f"{label}) yes" in result["problem"]
    assert result["difficulty"] == 3


def test_trivia_transform_retains_answer_aliases():
    assert map_trivia({"answer": {"aliases": ["yes", "y"]}}) == {"answer": ["yes", "y"]}


def test_preparation_cli_saves_local_dataset_without_hub_upload(tmp_path, monkeypatch):
    calls = []

    def load_builder(module):
        calls.append(module)

        def build_dataset(seed):
            assert seed == 17
            return DatasetDict(
                {"test": Dataset.from_dict({"question": ["question"], "answer": ["yes"]})}
            )

        return SimpleNamespace(build_dataset=build_dataset)

    monkeypatch.setattr(preparation, "import_module", load_builder)
    output = tmp_path / "prepared"
    assert (
        main(["prepare-data", "--recipe", "trivia", "--output", str(output), "--seed", "17"]) == 0
    )
    assert calls == ["rlcr.data.recipes.trivia"]
    assert load_from_disk(output)["test"][0]["answer"] == "yes"
    with pytest.raises(FileExistsError):
        preparation.prepare_dataset("trivia", output)
    assert len(calls) == 1


def test_recipe_modules_do_not_download_datasets_on_import(monkeypatch):
    import datasets
    import importlib

    def unexpected_download(*args, **kwargs):
        raise AssertionError("A recipe must not download data during import")

    monkeypatch.setattr(datasets, "load_dataset", unexpected_download)
    for module in RECIPES.values():
        imported = import_module(f"rlcr.data.recipes.{module}")
        importlib.reload(imported)
