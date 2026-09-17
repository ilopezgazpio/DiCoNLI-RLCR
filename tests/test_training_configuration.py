"""Reject misleading configuration before loading datasets or model weights."""
from dataclasses import fields
from pathlib import Path

import pytest
import yaml

from rlcr.arguments.evaluation_model_config import LocalConfig
from rlcr.arguments.grpo_config import GRPOConfig
from rlcr.arguments.script_arguments import GRPOScriptArguments
from rlcr.cli import main
from rlcr.rewards.registry import build_reward_functions
from rlcr.training.configuration import load_training_config


@pytest.fixture
def training_yaml(tmp_path):
    config = {
        "dataset_name": "unused",
        "model_name_or_path": "unused",
        "output_dir": str(tmp_path / "output"),
        "use_cpu": True,
        "report_to": [],
        "push_to_hub": False,
    }
    path = tmp_path / "training.yaml"
    path.write_text(yaml.safe_dump(config))
    return path, config


@pytest.mark.parametrize(
    "key",
    [
        "learning_rate_typo",
        "num_processes",
        "callbacks",
        "system_prompt",
        "completion_logging_steps",
        "eval_log_keys",
        "set_pad_token",
        "orm_key",
        "sys_prompt_name",
        "task_spec",
        "format_pattern",
        "gradient_checkpointing_use_reentrant",
        "ignore_bias_buffers",
    ],
)
def test_unknown_and_removed_training_keys_fail_early(training_yaml, monkeypatch, key):
    path, config = training_yaml
    monkeypatch.delenv("RLCR_TEST_REJECTED_CONFIG", raising=False)
    config.update({key: 1, "env": {"RLCR_TEST_REJECTED_CONFIG": "should-not-be-applied"}})
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match=f"Unknown training settings: {key}"):
        load_training_config(path)
    import os

    assert "RLCR_TEST_REJECTED_CONFIG" not in os.environ
    assert not Path(config["output_dir"]).exists()


def test_cli_reports_unknown_yaml_without_starting_training(training_yaml, monkeypatch, capsys):
    from rlcr.training import runner

    path, config = training_yaml
    config["callbacks"] = ["unused"]
    path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(
        runner, "run_training", lambda *args: pytest.fail("Training must not start")
    )
    with pytest.raises(SystemExit) as error:
        main(["train", "--config", str(path)])
    assert error.value.code == 2
    assert "Unknown training settings: callbacks" in capsys.readouterr().err


def test_training_overrides_and_env_are_preserved(training_yaml, monkeypatch):
    import os

    path, config = training_yaml
    monkeypatch.delenv("RLCR_TEST_ACCEPTED_CONFIG", raising=False)
    config.update({"max_steps": 100, "env": {"RLCR_TEST_ACCEPTED_CONFIG": 17}})
    path.write_text(yaml.safe_dump(config))
    script, args, model = load_training_config(path, ["--max_steps", "2", "--use_peft", "true"])
    assert script.dataset_name == "unused"
    assert args.max_steps == 2
    assert model.use_peft is True
    assert os.environ["RLCR_TEST_ACCEPTED_CONFIG"] == "17"


def test_unknown_training_cli_flag_is_rejected(training_yaml):
    with pytest.raises(ValueError, match="not used"):
        load_training_config(training_yaml[0], ["--set_pad_token", "123"])


@pytest.mark.parametrize("document", [None, [], "invalid", {1: "invalid"}])
def test_training_yaml_requires_a_mapping(training_yaml, document):
    path, _ = training_yaml
    path.write_text(yaml.safe_dump(document))
    with pytest.raises(ValueError, match="mapping with string keys"):
        load_training_config(path)


def test_removed_options_are_not_exposed_as_dataclass_fields():
    script_fields = {field.name for field in fields(GRPOScriptArguments)}
    assert not script_fields & {
        "set_pad_token",
        "orm_key",
        "ignore_bias_buffers",
        "gradient_checkpointing_use_reentrant",
    }
    trainer_fields = {field.name for field in fields(GRPOConfig)}
    assert not trainer_fields & {
        "callbacks",
        "system_prompt",
        "completion_logging_steps",
        "eval_log_keys",
    }
    assert "correctness_fn" not in {field.name for field in fields(LocalConfig)}


def test_reward_defaults_and_help():
    script = GRPOScriptArguments(dataset_name="unused")
    assert script.reward_funcs == ["accuracy", "brier"]
    assert len(build_reward_functions(script)) == 2
    help_text = GRPOScriptArguments.__dataclass_fields__["reward_funcs"].metadata["help"]
    for name in ["accuracy", "format", "brier"]:
        assert name in help_text
        script.reward_funcs = [name]
        assert len(build_reward_functions(script)) == 1


@pytest.mark.parametrize(
    "name",
    [
        "reasoning_steps",
        "cosine",
        "repetition_penalty",
        "mean_confidence",
        "confidence_one_or_zero",
    ],
)
def test_unsupported_rewards_fail_before_dataset_loading(training_yaml, monkeypatch, name):
    from rlcr.training import runner

    path, config = training_yaml
    config["reward_funcs"] = [name]
    path.write_text(yaml.safe_dump(config))
    parsed = load_training_config(path)
    monkeypatch.setattr(runner, "load_training_datasets", lambda *a: pytest.fail("No data loading"))
    with pytest.raises(ValueError, match="Unknown reward functions"):
        runner.run_training(*parsed)
    assert not Path(config["output_dir"]).exists()
