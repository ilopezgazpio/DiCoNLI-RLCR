"""Model-loading extras have explicit ownership and reach weights and tokenizers."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from transformers import TrainerCallback, TrainerControl, TrainerState
from trl import SyncRefModelCallback, get_peft_config

from rlcr.arguments.grpo_config import GRPOConfig
from rlcr.arguments.model_config import ModelConfig
from rlcr.models.training_kwargs import training_model_kwargs
from rlcr.training.grpo.trainer import GRPOTrainer


def test_additional_loading_kwargs_are_preserved_without_mutating_input():
    extra = {"local_files_only": True, "cache_dir": "/tmp/test-cache"}
    args = SimpleNamespace(model_init_kwargs=extra, gradient_checkpointing=True)
    result = training_model_kwargs(ModelConfig(torch_dtype="float32"), args)
    assert result["local_files_only"] is True
    assert result["cache_dir"] == "/tmp/test-cache"
    assert result["torch_dtype"] == torch.float32
    assert result["use_cache"] is False
    assert extra == {"local_files_only": True, "cache_dir": "/tmp/test-cache"}


@pytest.mark.parametrize(
    "key",
    [
        "revision",
        "trust_remote_code",
        "attn_implementation",
        "torch_dtype",
        "use_cache",
        "quantization_config",
        "device_map",
        "load_in_4bit",
        "load_in_8bit",
    ],
)
def test_managed_model_settings_cannot_be_silently_overridden(key):
    args = SimpleNamespace(model_init_kwargs={key: "value"}, gradient_checkpointing=False)
    with pytest.raises(ValueError, match=f"cannot override managed settings: {key}"):
        training_model_kwargs(ModelConfig(), args)


@pytest.mark.parametrize("value", [[], "not-a-mapping", {1: "not-a-string-key"}])
def test_extra_loading_kwargs_must_be_a_mapping(value):
    with pytest.raises(ValueError, match="mapping with string keys"):
        training_model_kwargs(ModelConfig(), SimpleNamespace(model_init_kwargs=value))


@pytest.mark.parametrize("key", ["token", "use_auth_token"])
def test_credentials_are_not_accepted_in_serialized_loading_kwargs(key):
    with pytest.raises(ValueError, match="Use HF_TOKEN") as error:
        training_model_kwargs(
            ModelConfig(), SimpleNamespace(model_init_kwargs={key: "secret-value"})
        )
    assert "secret-value" not in str(error.value)


@pytest.mark.parametrize("bits", [4, 8])
def test_quantization_configuration_is_preserved(bits):
    args = SimpleNamespace(
        model_init_kwargs={"local_files_only": True}, gradient_checkpointing=True
    )
    model = ModelConfig(
        use_peft=True,
        load_in_4bit=bits == 4,
        load_in_8bit=bits == 8,
        torch_dtype="bfloat16",
        use_bnb_nested_quant=True,
    )
    kwargs = training_model_kwargs(model, args)
    quantization = kwargs["quantization_config"]
    assert quantization.load_in_4bit is (bits == 4)
    assert quantization.load_in_8bit is (bits == 8)
    assert kwargs["local_files_only"] is True
    assert quantization.bnb_4bit_compute_dtype == torch.bfloat16


def test_real_trainer_forwards_shared_model_and_tokenizer_kwargs(tiny_model, tmp_path, monkeypatch):
    from rlcr.models import policy, tokenizer as tokenizer_module

    model, tokenizer = tiny_model
    model.config._name_or_path = "fixture"
    calls = {}

    def load_model(name, **kwargs):
        calls["model"] = kwargs
        assert name == "fixture"
        return model

    def load_tokenizer(name, **kwargs):
        calls["tokenizer"] = kwargs
        assert name == "fixture"
        return tokenizer

    monkeypatch.setattr(policy.AutoModelForCausalLM, "from_pretrained", load_model)
    monkeypatch.setattr(tokenizer_module.AutoTokenizer, "from_pretrained", load_tokenizer)
    args = GRPOConfig(
        output_dir=str(tmp_path),
        use_cpu=True,
        report_to=[],
        beta=0,
        model_init_kwargs={"local_files_only": True, "cache_dir": str(tmp_path)},
    )
    args.model_init_kwargs = training_model_kwargs(
        ModelConfig(model_revision="test-revision", trust_remote_code=False, torch_dtype="float32"),
        args,
    )
    callback = TrainerCallback()
    trainer = GRPOTrainer(
        model="fixture",
        args=args,
        reward_funcs=[lambda completions, **kw: [0.0] * len(completions)],
        callbacks=[callback],
    )
    assert callback in trainer.callback_handler.callbacks
    for key in ["revision", "trust_remote_code", "local_files_only", "cache_dir"]:
        assert calls["model"][key] == calls["tokenizer"][key]
    assert calls["model"]["revision"] == "test-revision"
    assert "torch_dtype" not in calls["tokenizer"]


def test_lora_and_reference_sync_options_remain_functional(monkeypatch):
    model_args = ModelConfig(
        use_peft=True,
        lora_r=3,
        lora_alpha=7,
        lora_dropout=0.2,
        lora_target_modules=["q_proj", "v_proj"],
        lora_modules_to_save=["lm_head"],
        use_rslora=True,
    )
    config = get_peft_config(model_args)
    assert (config.r, config.lora_alpha, config.lora_dropout) == (3, 7, 0.2)
    assert config.use_rslora is True
    assert config.target_modules == {"q_proj", "v_proj"}
    assert config.modules_to_save == ["lm_head"]

    reference = torch.nn.Linear(1, 1, bias=False)
    model = deepcopy(reference)
    reference.weight.data.zero_()
    model.weight.data.fill_(4.0)
    callback = SyncRefModelCallback(reference, accelerator=None)
    monkeypatch.setattr(callback, "sync_target_model", callback._sync_target_model)
    args = SimpleNamespace(ref_model_sync_steps=2, ref_model_mixup_alpha=0.25)
    state = TrainerState(global_step=1)
    callback.on_step_end(args, state, TrainerControl(), model=model)
    assert reference.weight.item() == 0.0
    state.global_step = 2
    callback.on_step_end(args, state, TrainerControl(), model=model)
    assert reference.weight.item() == 1.0
