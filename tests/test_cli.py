"""Exercise the installed entry point with local models and datasets only."""
import json
import os
from pathlib import Path
import subprocess
import sys

from datasets import Dataset, DatasetDict, load_from_disk
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def cli(*args, cwd):
    env = {
        **os.environ,
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "CUDA_VISIBLE_DEVICES": "",
    }
    result = subprocess.run(
        [sys.executable, "-m", "rlcr", *args],
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result


@pytest.mark.parametrize(
    "command",
    [
        [],
        ["train"],
        ["evaluate"],
        ["infer"],
        ["prepare-data"],
        ["prepare-prompts"],
        ["fetch-scorer"],
        ["export-submission"],
        ["score"],
    ],
)
def test_cli_help_outside_checkout(command, tmp_path):
    assert "usage:" in cli(*command, "--help", cwd=tmp_path).stdout


def test_cli_help_does_not_import_ml_libraries(tmp_path):
    program = (
        "import sys; from rlcr.cli import build_parser; build_parser().print_help(); "
        "assert not {'torch', 'transformers', 'datasets', 'trl'} & sys.modules.keys()"
    )
    result = subprocess.run(
        [sys.executable, "-c", program], cwd=tmp_path, text=True, capture_output=True
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("peft", [False, True], ids=["full", "lora"])
def test_cli_training_runs_and_honors_overrides(tiny_model, tmp_path, peft):
    model, tokenizer = tiny_model
    base = tmp_path / "base"
    model.save_pretrained(base)
    tokenizer.save_pretrained(base)
    dataset_dir = tmp_path / "dataset"
    DatasetDict(
        {
            "train": Dataset.from_dict(
                {"prompt": ["question", "long question"], "label": ["yes", "no"]}
            )
        }
    ).save_to_disk(dataset_dir)
    config = tmp_path / "training.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "model_name_or_path": str(base),
                "dataset_name": str(dataset_dir),
                "output_dir": str(tmp_path / "unused"),
                "torch_dtype": "float32",
                "model_init_kwargs": {"local_files_only": True},
                "use_cpu": True,
                "use_peft": peft,
                "lora_r": 2,
                "lora_target_modules": ["q_proj", "v_proj"],
                "reward_funcs": ["format"],
                "beta": 0,
                "per_device_train_batch_size": 1,
                "gradient_accumulation_steps": 4,
                "num_generations": 4,
                "max_prompt_length": 32,
                "max_completion_length": 3,
                "max_steps": 3,
                "eval_strategy": "no",
                "save_strategy": "no",
                "report_to": [],
                "log_completions": False,
                "push_to_hub": False,
            }
        )
    )
    output = tmp_path / "trained"
    cli(
        "train",
        "--config",
        str(config),
        "--max_steps",
        "1",
        "--output_dir",
        str(output),
        cwd=tmp_path,
    )
    state = json.loads((output / "trainer_state.json").read_text())
    assert state["global_step"] == 1
    assert (output / ("adapter_model.safetensors" if peft else "model.safetensors")).exists()
    assert (output / "README.md").exists()
    assert not (tmp_path / "unused").exists()
    resolved = yaml.safe_load((output / "resolved-config.yaml").read_text())
    assert resolved["max_steps"] == 1
    assert resolved["output_dir"] == str(output)
    assert resolved["use_peft"] is peft
    assert resolved["model_init_kwargs"] == {"local_files_only": True}
    assert "hub_token" not in resolved


def test_cli_inference_returns_grouped_json(tiny_model, tmp_path):
    model, tokenizer = tiny_model
    model.save_pretrained(tmp_path)
    tokenizer.save_pretrained(tmp_path)
    result = cli(
        "infer",
        "--model",
        str(tmp_path),
        "--prompt",
        "question",
        "--prompt",
        "long question",
        "--system-prompt",
        "Return a structured prediction.",
        "--torch-dtype",
        "float32",
        "--max-tokens",
        "3",
        "--n",
        "2",
        cwd=tmp_path,
    )
    records = [
        json.loads(line) for line in result.stdout.splitlines() if line.startswith('{"prompt"')
    ]
    assert [record["prompt"] for record in records] == ["question", "long question"]
    assert all(len(record["completions"]) == 2 for record in records)


def test_cli_evaluation_loads_local_dataset(tiny_model, tmp_path):
    model, tokenizer = tiny_model
    base = tmp_path / "base"
    model.save_pretrained(base)
    tokenizer.save_pretrained(base)
    dataset_dir = tmp_path / "dataset"
    DatasetDict(
        {
            "test": Dataset.from_dict(
                {"instance_id": ["sample-1"], "prompt": ["question"], "label": ["yes"]}
            )
        }
    ).save_to_disk(dataset_dir)
    output = tmp_path / "results"
    config = tmp_path / "evaluation.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "dataset": {
                    "name": "unused-dataset",
                    "id_column": "instance_id",
                },
                "output_dir": str(tmp_path / "unused"),
                "models": [
                    {
                        "name": "test",
                        "model": "unused-model",
                        "torch_dtype": "float32",
                        "max_tokens": 3,
                    }
                ],
            }
        )
    )
    cli(
        "evaluate",
        "--config",
        str(config),
        "--dataset",
        str(dataset_dir),
        "--model",
        str(base),
        "--output-dir",
        str(output),
        "--sample-size",
        "1",
        cwd=tmp_path,
    )
    saved = load_from_disk(output / "predictions")
    assert len(saved) == 1
    assert "test-output_0" in saved.column_names
    assert json.loads((output / "metrics.json").read_text()) == {
        "test": {"examples": 1, "completions": 1}
    }
    resolved = yaml.safe_load((output / "resolved-config.yaml").read_text())
    assert resolved["output_dir"] == str(output)
    assert resolved["dataset"]["name"] == str(dataset_dir)
    assert resolved["dataset"]["sample_size"] == 1
    assert resolved["models"][0]["model"] == str(base)
    assert not (tmp_path / "unused").exists()


def test_cli_rejects_unknown_evaluation_flags(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "rlcr", "evaluate", "--config", "unused", "--typo"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 2
    assert "unrecognized arguments" in result.stderr
