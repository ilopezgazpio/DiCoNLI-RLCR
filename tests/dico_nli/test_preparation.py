"""Data recipe, offline CLI, provenance, and non-destructive persistence checks."""
import json
import os
from pathlib import Path
import subprocess
import sys

from datasets import load_from_disk
import pytest
import yaml

from rlcr.data.dico_nli.configuration import SOURCE_REPOSITORY, load_preparation_config
from rlcr.data.dico_nli.preparation import prepare_dico_data


@pytest.fixture
def recipe(tmp_path, split_spec):
    config = {
        "source": {"repository": SOURCE_REPOSITORY, "revision": "a" * 40},
        "output_dir": str(tmp_path / "prepared"),
        "seed": 42,
        "splits": {"train": split_spec},
    }
    path = tmp_path / "data.yaml"
    path.write_text(yaml.safe_dump(config))
    return path, config


def test_roundtrip_preserves_metadata_not_prompts(recipe):
    path, config = recipe
    report = prepare_dico_data(path)
    output = Path(config["output_dir"])
    dataset = load_from_disk(output)["train"]
    assert set(dataset.column_names) == {
        "instance_id",
        "pair_id",
        "text1_lang",
        "text2_lang",
        "text1",
        "text2",
        "label",
        "reverse_pair_id",
        "pairing_available",
    }
    assert len(dataset) == 5
    assert list(dataset["instance_id"]) == sorted(dataset["instance_id"])
    assert all(dataset["pairing_available"])
    assert (output / "resolved-config.yaml").is_file()
    assert json.loads((output / "audit.json").read_text()) == report
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["prompt_ready"] is False
    assert manifest["source"] == config["source"]
    assert manifest["selected_instance_ids"]["train"] == list(dataset["instance_id"])
    assert manifest["inputs"]["train"]["participant"]["sha256"] == (
        config["splits"]["train"]["participant"]["sha256"]
    )
    assert manifest["sampling"]["algorithm"] == "source-label-signature-sha256-v1"


def test_unlabeled_saved_columns_have_explicit_nullable_types(recipe, task_rows, write_csv):
    path, config = recipe
    rows = [{key: value for key, value in row.items() if key != "label"} for row in task_rows]
    config["splits"] = {
        "test": {
            "participant": write_csv("test.csv", rows),
            "labeled": False,
        }
    }
    path.write_text(yaml.safe_dump(config))
    report = prepare_dico_data(path)
    dataset = load_from_disk(config["output_dir"])["test"]
    assert dataset.features["label"].dtype == "string"
    assert dataset.features["reverse_pair_id"].dtype == "string"
    assert list(dataset["label"]) == [None] * 5
    assert list(dataset["pairing_available"]) == [False] * 5
    assert report["selected"]["splits"]["test"]["pairing_unknown_instances"] == 5


@pytest.mark.parametrize("preexisting", ["empty-directory", "file", "completed-run", "symlink"])
def test_output_is_never_overwritten(recipe, preexisting, tmp_path):
    path, config = recipe
    output = Path(config["output_dir"])
    if preexisting == "completed-run":
        prepare_dico_data(path)
        before = (output / "manifest.json").read_bytes()
    elif preexisting == "empty-directory":
        output.mkdir()
    elif preexisting == "file":
        output.write_text("keep")
    else:
        output.symlink_to(tmp_path / "nonexistent", target_is_directory=True)
    with pytest.raises(ValueError, match="already exists"):
        prepare_dico_data(path)
    if preexisting == "completed-run":
        assert (output / "manifest.json").read_bytes() == before
    if preexisting == "file":
        assert output.read_text() == "keep"


def test_full_input_leakage_rejected_before_sampling_or_output(recipe, write_csv, task_rows):
    path, config = recipe
    config["splits"]["dev"] = {
        "participant": write_csv("dev.csv", task_rows),
        "labeled": True,
        "max_source_pairs": 3,
    }
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="Split leakage"):
        prepare_dico_data(path)
    assert not Path(config["output_dir"]).exists()


def test_hash_failure_leaves_no_output(recipe):
    path, config = recipe
    config["splits"]["train"]["participant"]["sha256"] = "0" * 64
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        prepare_dico_data(path)
    assert not Path(config["output_dir"]).exists()


@pytest.mark.parametrize(
    "mutation",
    [
        "unknown-top",
        "unknown-split",
        "unknown-file",
        "no-hash",
        "wrong-hash",
        "no-labeled",
        "bad-bool",
        "wrong-repository",
        "branch-revision",
        "bad-seed",
        "no-splits",
        "bad-split-name",
        "bad-limit",
        "unlabeled-reference",
    ],
)
def test_recipe_validation(recipe, mutation):
    path, config = recipe
    train = config["splits"]["train"]
    if mutation == "unknown-top":
        config["typo"] = True
    elif mutation == "unknown-split":
        train["typo"] = True
    elif mutation == "unknown-file":
        train["participant"]["typo"] = True
    elif mutation == "no-hash":
        train["participant"].pop("sha256")
    elif mutation == "wrong-hash":
        train["participant"]["sha256"] = "typo"
    elif mutation == "no-labeled":
        train.pop("labeled")
    elif mutation == "bad-bool":
        train["labeled"] = "yes"
    elif mutation == "wrong-repository":
        config["source"]["repository"] = "unknown"
    elif mutation == "branch-revision":
        config["source"]["revision"] = "main"
    elif mutation == "bad-seed":
        config["seed"] = True
    elif mutation == "no-splits":
        config["splits"] = {}
    elif mutation == "bad-split-name":
        config["splits"] = {"../train": train}
    elif mutation == "bad-limit":
        train["max_source_pairs"] = True
    else:
        train["labeled"] = False
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError):
        prepare_dico_data(path)
    assert not Path(config["output_dir"]).exists()


@pytest.mark.parametrize(
    "text",
    [
        "",
        "[]",
        "source: [",
        "source: {}\nsource: {}\n",
        "a: &a\n  self: *a\n",
        "a: &a {b: 1}\nc: *a\n",
        "1: value\n",
    ],
)
def test_malformed_duplicate_and_recursive_yaml_fail(tmp_path, text):
    path = tmp_path / "bad.yaml"
    path.write_text(text)
    with pytest.raises(ValueError):
        load_preparation_config(path)


def test_cli_runs_offline_outside_checkout(recipe, tmp_path):
    path, config = recipe
    output = tmp_path / "override"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "rlcr",
            "prepare-data",
            "--config",
            str(path),
            "--output-dir",
            str(output),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
        env={**os.environ, "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"},
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["selected"]["splits"]["train"]["instances"] == 5
    assert (output / "manifest.json").is_file()
    assert not Path(config["output_dir"]).exists()


def test_cli_validation_errors_are_actionable(recipe, tmp_path):
    path, config = recipe
    path.write_text("source: [")
    result = subprocess.run(
        [sys.executable, "-m", "rlcr", "prepare-data", "--config", str(path)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 2
    assert "Invalid data YAML" in result.stderr
    assert "Traceback" not in result.stderr
    assert not Path(config["output_dir"]).exists()


def test_data_import_has_no_model_or_download_side_effects(tmp_path):
    code = (
        "import sys; from rlcr.data.dico_nli.preparation import prepare_dico_data; "
        "assert not {'torch', 'transformers', 'datasets', 'trl'} & sys.modules.keys()"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_checked_in_data_recipe_is_valid():
    root = Path(__file__).resolve().parents[2]
    config = load_preparation_config(root / "configs/data/dico-nli-en.yaml")
    assert set(config["splits"]) == {"train", "dev"}
    assert all(spec["labeled"] for spec in config["splits"].values())


def test_interrupted_write_has_no_success_manifest(recipe, monkeypatch):
    from datasets import DatasetDict

    def fail(*args, **kwargs):
        raise OSError("Simulated storage failure")

    path, config = recipe
    monkeypatch.setattr(DatasetDict, "save_to_disk", fail)
    with pytest.raises(OSError, match="Simulated storage failure"):
        prepare_dico_data(path)
    output = Path(config["output_dir"])
    assert output.is_dir()
    assert not (output / "manifest.json").exists()
    with pytest.raises(ValueError, match="already exists"):
        prepare_dico_data(path)
