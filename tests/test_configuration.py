"""Configuration validation, snapshots, and launcher ownership without GPU execution."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from rlcr.configuration.snapshots import save_resolved_config
from rlcr.evaluation.configuration import evaluation_config_dict, load_evaluation_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def evaluation_config():
    return {
        "dataset": {"name": "data/example", "hash_key": "question"},
        "models": [{"name": "base", "model": "org/model", "sys_prompt_name": "gen"}],
        "output_dir": "outputs/eval/example",
    }


def test_evaluation_yaml_roundtrip_and_overrides(tmp_path, evaluation_config):
    path = tmp_path / "evaluation.yaml"
    path.write_text(yaml.safe_dump(evaluation_config))
    args, models = load_evaluation_config(
        path,
        dataset_name="/datasets/example",
        output_dir="/scratch/run",
        split="validation",
        sample_size=5,
        model="/models/adapter",
        fresh=True,
    )
    assert args.dataset_name == "/datasets/example"
    assert args.output_dir == "/scratch/run"
    assert args.split == "validation"
    assert args.sample_size == 5
    assert args.fresh is True
    assert models[0].model == "/models/adapter"
    path.write_text(yaml.safe_dump(evaluation_config_dict(args, models)))
    assert load_evaluation_config(path) == (args, models)
    assert load_evaluation_config(path, fresh=False)[0].fresh is False


@pytest.mark.parametrize(
    "section,key,value,message",
    [
        (None, "typo", True, "Unknown evaluation"),
        (None, "output_dir", None, "output_dir"),
        (None, "fresh", "yes", "boolean"),
        (None, "models", [], "nonempty list"),
        (None, "models", ["invalid"], "mapping"),
        ("dataset", "typo", 1, "Unknown dataset"),
        ("dataset", "name", "", "dataset_name"),
        ("dataset", "sample_size", 0, "positive integer"),
        ("dataset", "sample_size", True, "positive integer"),
        ("model", "name", "", "nonempty name"),
        ("model", "typo", 1, "Invalid evaluation model"),
        ("model", "correctness_fn", "unused", "Invalid evaluation model"),
        ("model", "task_spec", "orm", "Only task_spec: gen"),
        ("model", "sys_prompt_name", "ver", "Invalid system prompt"),
    ],
)
def test_invalid_evaluation_settings(tmp_path, evaluation_config, section, key, value, message):
    target = evaluation_config
    if section == "dataset":
        target = target["dataset"]
    elif section == "model":
        target = target["models"][0]
    target[key] = value
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(evaluation_config))
    with pytest.raises(ValueError, match=message):
        load_evaluation_config(path)


def test_legacy_positional_list_is_rejected(tmp_path):
    path = tmp_path / "old.json"
    path.write_text('[{"dataset_name": "old"}, {"name": "test", "model": "old"}]')
    with pytest.raises(ValueError, match="dataset, models, and output_dir"):
        load_evaluation_config(path)


def test_duplicate_model_names_and_ambiguous_override_rejected(tmp_path, evaluation_config):
    path = tmp_path / "evaluation.yaml"
    evaluation_config["models"].append(deepcopy(evaluation_config["models"][0]))
    path.write_text(yaml.safe_dump(evaluation_config))
    with pytest.raises(ValueError, match="unique"):
        load_evaluation_config(path)
    with pytest.raises(ValueError, match="single-model"):
        load_evaluation_config(path, model="new/model")


def test_snapshot_preserves_previous_settings(tmp_path):
    save_resolved_config(tmp_path, {"max_steps": 1})
    save_resolved_config(tmp_path, {"max_steps": 1})
    assert not (tmp_path / "config-history").exists()
    save_resolved_config(tmp_path, {"max_steps": 2})
    assert yaml.safe_load((tmp_path / "resolved-config.yaml").read_text()) == {"max_steps": 2}
    history = list((tmp_path / "config-history").glob("*.yaml"))
    assert len(history) == 1
    assert yaml.safe_load(history[0].read_text()) == {"max_steps": 1}


@pytest.mark.parametrize("accumulation", [4, 8, 64])
def test_zero2_optimization_settings_come_from_training_arguments(accumulation):
    from accelerate import DeepSpeedPlugin
    from transformers.integrations.deepspeed import HfTrainerDeepSpeedConfig

    config = yaml.safe_load((ROOT / "configs/accelerate/zero2.yaml").read_text())
    plugin = DeepSpeedPlugin(**config["deepspeed_config"])
    assert plugin.deepspeed_config["gradient_accumulation_steps"] == "auto"
    assert plugin.deepspeed_config["gradient_clipping"] == "auto"
    resolved = HfTrainerDeepSpeedConfig(deepcopy(plugin.deepspeed_config))
    args = SimpleNamespace(
        world_size=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=accumulation,
        max_grad_norm=0.5,
        learning_rate=1e-5,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        weight_decay=0.0,
        fp16=False,
        fp16_full_eval=False,
        fp16_backend="auto",
        bf16=False,
        bf16_full_eval=False,
        save_on_each_node=False,
        fp16_opt_level="O1",
    )
    resolved.trainer_config_process(args)
    assert resolved.config["gradient_accumulation_steps"] == accumulation
    assert resolved.config["gradient_clipping"] == 0.5
    assert not resolved.mismatches


def test_configuration_layout():
    assert list((ROOT / "configs/train").glob("*.yaml"))
    assert list((ROOT / "configs/eval").glob("*.yaml"))
    assert not (ROOT / "eval_configs").exists()
    assert not (ROOT / "eval_outputs").exists()
    assert not (ROOT / "results").exists()
    assert not list((ROOT / "data").glob("RLCR-*"))
