"""CPU regression checks using locally constructed models; no model downloads."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from datasets import Dataset, DatasetDict, load_from_disk
from packaging.requirements import Requirement
from peft import LoraConfig
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (
    PreTrainedTokenizerFast,
    Qwen2Config,
    Qwen2ForCausalLM,
    Qwen2ForSequenceClassification,
)

from rlcr.arguments.grpo_config import GRPOConfig
from rlcr.training.configuration import load_training_config
from rlcr.arguments.evaluation_global_args import GlobalArgs
from rlcr.arguments.evaluation_model_config import LocalConfig
from rlcr.training.grpo.trainer import GRPOTrainer
from rlcr.inference.completion import Completion
from rlcr.inference.generation import Generation
from rlcr.inference.generator import hf_generate
from rlcr.models.inference import load_hf_generator
from rlcr.models.tokenizer import load_tokenizer


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("temperature", [0, 0.7])
def test_generation_groups_and_selected_token_probabilities(tiny_model, temperature):
    model, tokenizer = tiny_model
    prompts = ["question", "long question", "long long question"]
    outputs = hf_generate(
        model,
        tokenizer,
        prompts,
        n=2,
        temperature=temperature,
        max_tokens=4,
        seed=42,
        batch_size=2,
        return_logprobs=True,
    )
    assert len(outputs) == len(prompts)
    for group in outputs:
        assert len(group.outputs) == 2
        for completion in group.outputs:
            assert len(completion.token_ids) == len(completion.token_logprobs)
            assert 1 <= len(completion.token_ids) <= 4
            assert all(value <= 0 for value in completion.token_logprobs)
            if tokenizer.eos_token_id in completion.token_ids:
                assert completion.token_ids[-1] == tokenizer.eos_token_id

    if temperature == 0:
        # Compare against the model's next-token distribution independently.
        for prompt, group in zip(prompts, outputs):
            completion = group.outputs[0]
            assert completion.text == group.outputs[1].text
            assert completion is not group.outputs[1]
            ids = tokenizer.encode(prompt) + completion.token_ids
            with torch.no_grad():
                logits = model(torch.tensor([ids])).logits[0]
            start = len(ids) - len(completion.token_ids) - 1
            expected = (
                logits[start:-1]
                .log_softmax(-1)
                .gather(1, torch.tensor(completion.token_ids)[:, None])
                .flatten()
            )
            torch.testing.assert_close(torch.tensor(completion.token_logprobs), expected)


def test_generation_without_probability_collection(tiny_model):
    model, tokenizer = tiny_model
    output = hf_generate(model, tokenizer, ["question"], 1, 0, 3, 42, 1)
    assert output[0].outputs[0].token_logprobs is None


@pytest.mark.parametrize("use_peft", [False, True], ids=["full", "lora"])
def test_grpo_trains_and_reloads_model(tiny_model, tmp_path, monkeypatch, use_peft):
    model, tokenizer = tiny_model
    base_dir = tmp_path / "base"
    model.save_pretrained(base_dir)
    tokenizer.save_pretrained(base_dir)
    model = Qwen2ForCausalLM.from_pretrained(base_dir)

    def rewards(completions, answer, source, **kwargs):
        assert answer == ["yes"] * len(completions)
        assert source == ["hotpot"] * len(completions)
        return [float(i) for i in range(len(completions))]

    args = GRPOConfig(
        output_dir=str(tmp_path / "trained"),
        use_cpu=True,
        beta=0,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=4,
        max_steps=2,
        max_prompt_length=32,
        max_completion_length=4,
        learning_rate=0.01,
        report_to=[],
        save_strategy="no",
        log_completions=False,
    )
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=args,
        reward_funcs=[rewards],
        train_dataset=Dataset.from_dict(
            {
                "prompt": [[{"role": "user", "content": "question"}]] * 2,
                "answer": ["yes"] * 2,
                "source": ["hotpot"] * 2,
            }
        ),
        peft_config=(
            LoraConfig(task_type="CAUSAL_LM", r=2, target_modules=["q_proj", "v_proj"])
            if use_peft
            else None
        ),
    )
    before = {name: param.detach().clone() for name, param in trainer.model.named_parameters()}
    generate = trainer.model.generate
    calls = []

    def record_generation(*args, **kwargs):
        calls.append((trainer.model.training, torch.is_grad_enabled()))
        return generate(*args, **kwargs)

    monkeypatch.setattr(trainer.model, "generate", record_generation)
    result = trainer.train()
    assert result.global_step == 2
    assert calls == [(False, False), (False, False)]
    assert trainer.ref_model is None
    assert any(
        not torch.equal(before[name], param)
        for name, param in trainer.model.named_parameters()
        if param.requires_grad
    )
    assert all(
        torch.equal(before[name], param)
        for name, param in trainer.model.named_parameters()
        if not param.requires_grad
    )

    trainer.save_model(args.output_dir)
    reloaded = load_hf_generator(
        SimpleNamespace(model=args.output_dir, torch_dtype="float32", load_in_4bit=False)
    )
    inputs = tokenizer("question", return_tensors="pt")
    trainer.model.eval()
    with torch.no_grad():
        torch.testing.assert_close(trainer.model(**inputs).logits, reloaded(**inputs).logits)
    assert load_tokenizer(args.output_dir).padding_side == "left"


@pytest.mark.parametrize("path", sorted((ROOT / "configs/train").glob("*.yaml")))
def test_training_configs_parse(path, tmp_path):
    # CPU overrides let all checked-in GPU configurations be parsed on a laptop.
    script, training, model = load_training_config(
        path,
        [
            "--use_cpu",
            "true",
            "--bf16",
            "false",
            "--report_to",
            "none",
            "--push_to_hub",
            "false",
            "--output_dir",
            str(tmp_path),
        ],
    )
    assert model.model_name_or_path
    assert script.dataset_name
    assert training.remove_unused_columns is False
    assert training.generation_batch_size % training.num_generations == 0


@pytest.mark.parametrize("path", sorted((ROOT / "configs/eval").glob("*.yaml")))
def test_evaluation_configs_parse(path):
    from rlcr.evaluation.configuration import load_evaluation_config

    args, configs = load_evaluation_config(path)
    assert args.output_dir.startswith("outputs/eval/")
    for config in configs:
        assert config.hf_batch_size >= 1


def test_requirements_cover_full_and_adapter_training():
    requirements = [
        Requirement(line.strip())
        for line in (ROOT / "requirements.txt").read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    by_name = {requirement.name: requirement for requirement in requirements}
    assert len(by_name) == len(requirements), "Duplicate dependency definitions"
    assert {
        "torch",
        "transformers",
        "accelerate",
        "deepspeed",
        "peft",
        "bitsandbytes",
        "trl",
    } <= by_name.keys()
    assert all(requirement.specifier or requirement.url for requirement in requirements)


@pytest.mark.parametrize("task", ["confidence_at_end", "confidence_prob", "gen_then_classify"])
def test_evaluation_postprocessing_paths(tiny_model, tmp_path, monkeypatch, task):
    from rlcr.evaluation import runner, storage

    model, tokenizer = tiny_model
    model_dir = tmp_path / "model"
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    classifier_dir = tmp_path / "classifier"
    class_config = Qwen2Config(**model.config.to_dict())
    class_config.num_labels = 2
    classifier = Qwen2ForSequenceClassification(class_config)
    classifier.save_pretrained(classifier_dir)
    tokenizer.save_pretrained(classifier_dir)
    dataset = DatasetDict(
        {
            "test": Dataset.from_dict(
                {
                    "problem": ["question", "long question"],
                    "answer": ["yes", "no"],
                }
            )
        }
    )

    def load_dataset(path, **kwargs):
        if path == "fixture":
            return dataset
        raise FileNotFoundError(path)

    monkeypatch.setattr(storage, "load_dataset", load_dataset)
    store = str(tmp_path / "outputs")
    runner.run_evaluation(
        GlobalArgs(dataset_name="fixture", hash_key="problem", output_dir=store),
        [
            LocalConfig(
                name="test",
                model=str(model_dir),
                class_model=str(classifier_dir),
                torch_dtype="float32",
                sys_prompt_name="gen",
                tasks=["ans_at_end", task],
                max_tokens=4,
                hf_batch_size=2,
            )
        ],
    )
    saved = load_from_disk(str(Path(store) / "predictions"))
    assert len(saved) == 2
    if task == "gen_then_classify":
        assert all(0 <= probability <= 1 for probability in saved["test-class_output"])
    else:
        assert all("<confidence>" in output for output in saved["test-output_0"])


def test_judge_uses_transformers_and_preserves_response_groups(tiny_model, tmp_path, monkeypatch):
    from rlcr.evaluation.verifiers import judge

    model, tokenizer = tiny_model
    model.save_pretrained(tmp_path)
    tokenizer.save_pretrained(tmp_path)
    calls = []

    def judge_generate(judge, tokenizer, prompts, **kwargs):
        assert isinstance(judge, Qwen2ForCausalLM)
        assert not judge.training
        calls.append(len(prompts))
        return [Generation([Completion(text, [])]) for text in ["YES", "NO", "NO", "YES"]]

    monkeypatch.setattr(judge, "hf_generate", judge_generate)
    labels, metrics = judge.llm_confidence_verifier(
        Dataset.from_dict(
            {
                "question": ["question", "long question"],
                "answer": ["yes", "no"],
                "test-output_0": ["<answer>yes</answer><confidence>0.8</confidence>"] * 2,
                "test-output_1": ["<answer>no</answer><confidence>0.2</confidence>"] * 2,
            }
        ),
        LocalConfig(name="test", model="unused", n=2, torch_dtype="float32"),
        judge_model=str(tmp_path),
    )
    assert calls == [4]
    assert labels["test-evals"] == [[1, 0], [0, 1]]
    assert metrics["accuracy"] == 0.5
