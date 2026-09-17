"""No-knowledge DiCo rewards in real tiny full-model/LoRA optimizer and CLI runs."""
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from datasets import Dataset, DatasetDict, load_from_disk
from peft import LoraConfig
import pytest
import torch
import yaml

from rlcr.arguments.grpo_config import GRPOConfig
from rlcr.arguments.script_arguments import GRPOScriptArguments
from rlcr.data.dico_nli.prompts import build_nli_prompt, INPUT_FIELDS
from rlcr.models.inference import load_hf_generator
from rlcr.rewards.registry import build_reward_functions
from rlcr.training.grpo.trainer import GRPOTrainer
from model_fixture import make_response_model


def training_data(task_rows):
    return Dataset.from_list(
        [
            {**row, "prompt": build_nli_prompt(**{key: row[key] for key in INPUT_FIELDS})}
            for row in task_rows
        ]
    )


@pytest.mark.parametrize("peft", [False, True], ids=["full", "lora"])
@pytest.mark.parametrize("calibration", [False, True], ids=["accuracy", "rlcr"])
def test_real_nli_reward_updates_reload_and_generation(task_rows, tmp_path, peft, calibration):
    model, tokenizer = make_response_model()
    base = tmp_path / "base"
    model.save_pretrained(base)
    tokenizer.save_pretrained(base)
    # Reload like the application so PEFT records a real base checkpoint path.
    model = type(model).from_pretrained(base)
    names = ["dico_accuracy", "dico_brier"] if calibration else ["dico_accuracy"]
    functions = build_reward_functions(
        GRPOScriptArguments(dataset_name="unused", reward_funcs=names)
    )
    args = GRPOConfig(
        output_dir=str(tmp_path / "trained"),
        use_cpu=True,
        beta=0,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=4,
        max_steps=3,
        max_prompt_length=256,
        max_completion_length=1,
        learning_rate=0.01,
        report_to=[],
        save_strategy="no",
        log_completions=False,
        logging_steps=1,
        scale_rewards=False,
        seed=42,
        reward_weights=[1.0, 0.5] if calibration else [1.0],
    )
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=args,
        reward_funcs=functions,
        train_dataset=training_data(task_rows),
        peft_config=LoraConfig(task_type="CAUSAL_LM", r=2, target_modules=["q_proj", "v_proj"])
        if peft
        else None,
    )
    before = {
        name: parameter.detach().clone() for name, parameter in trainer.model.named_parameters()
    }
    result = trainer.train()
    assert result.global_step == 3
    assert trainer.ref_model is None
    assert any(
        not torch.equal(before[name], parameter)
        for name, parameter in trainer.model.named_parameters()
        if parameter.requires_grad
    )
    assert all(
        torch.equal(before[name], parameter)
        for name, parameter in trainer.model.named_parameters()
        if not parameter.requires_grad
    )
    logs = [row for row in trainer.state.log_history if "dico/invalid_rate" in row]
    assert logs and all(0 <= row["dico/invalid_rate"] <= 1 for row in logs)
    assert any(row["zero_reward_std_fraction"] < 1 for row in logs)
    trainer.save_model(args.output_dir)
    reloaded = load_hf_generator(
        SimpleNamespace(model=args.output_dir, torch_dtype="float32", load_in_4bit=False)
    )
    rendered = tokenizer.apply_chat_template(
        training_data(task_rows)[0]["prompt"], tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(rendered, return_tensors="pt")
    trainer.model.eval()
    with torch.no_grad():
        torch.testing.assert_close(trainer.model(**inputs).logits, reloaded(**inputs).logits)
        old = trainer.model.generate(**inputs, max_new_tokens=1, do_sample=False)
        new = reloaded.generate(**inputs, max_new_tokens=1, do_sample=False)
    assert torch.equal(old, new)


@pytest.mark.parametrize("peft", [False, True], ids=["full", "lora"])
def test_cli_training_and_batch_inference(task_rows, tmp_path, peft):
    model, tokenizer = make_response_model()
    model.save_pretrained(tmp_path / "base")
    tokenizer.save_pretrained(tmp_path / "base")
    DatasetDict({"train": training_data(task_rows), "dev": training_data(task_rows)}).save_to_disk(
        tmp_path / "data"
    )
    root = Path(__file__).resolve().parents[2]
    config = yaml.safe_load((root / "configs/train/dico-nli-rlcr.yaml").read_text())
    config.update(
        {
            "model_name_or_path": str(tmp_path / "base"),
            "dataset_name": str(tmp_path / "data"),
            "output_dir": str(tmp_path / "trained"),
            "use_cpu": True,
            "bf16": False,
            "torch_dtype": "float32",
            "use_peft": peft,
            "load_in_4bit": False,
            "gradient_checkpointing": False,
            "lora_r": 2,
            "max_steps": 2,
            "max_completion_length": 1,
            "learning_rate": 0.01,
            "save_strategy": "no",
        }
    )
    (tmp_path / "training.yaml").write_text(yaml.safe_dump(config))
    result = subprocess.run(
        [sys.executable, "-m", "rlcr", "train", "--config", str(tmp_path / "training.yaml")],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    state = json.loads((tmp_path / "trained/trainer_state.json").read_text())
    assert state["global_step"] == 2
    assert any("dico/invalid_rate" in item for item in state["log_history"])
    generation = yaml.safe_load((root / "configs/eval/dico-nli.yaml").read_text())
    generation["dataset"]["name"] = str(tmp_path / "data")
    generation["models"][0].update(
        {
            "model": str(tmp_path / "trained"),
            "torch_dtype": "float32",
            "load_in_4bit": False,
            "max_tokens": 1,
        }
    )
    generation["output_dir"] = str(tmp_path / "generated")
    (tmp_path / "generation.yaml").write_text(yaml.safe_dump(generation))
    result = subprocess.run(
        [sys.executable, "-m", "rlcr", "evaluate", "--config", str(tmp_path / "generation.yaml")],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    predictions = load_from_disk(tmp_path / "generated/predictions")
    assert predictions["instance_id"] == [row["instance_id"] for row in task_rows]
    assert len(predictions["candidate-output_0"]) == 5


def test_invalid_gold_fails_before_model_loading_and_output(task_rows, tmp_path, monkeypatch):
    from rlcr.training.configuration import load_training_config
    from rlcr.training import runner

    dataset = training_data(task_rows).remove_columns("label").add_column("label", ["UNKNOWN"] * 5)
    DatasetDict({"train": dataset}).save_to_disk(tmp_path / "data")
    config = {
        "dataset_name": str(tmp_path / "data"),
        "model_name_or_path": "must-not-load",
        "output_dir": str(tmp_path / "out"),
        "reward_funcs": ["dico_brier"],
        "use_cpu": True,
        "report_to": [],
    }
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config))
    monkeypatch.setattr(
        runner, "load_training_tokenizer", lambda *a: pytest.fail("No model access")
    )
    with pytest.raises(ValueError, match="vocabulary"):
        runner.run_training(*load_training_config(tmp_path / "config.yaml"))
    assert not (tmp_path / "out").exists()
