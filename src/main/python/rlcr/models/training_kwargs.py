import os
import torch
from transformers import BitsAndBytesConfig


def training_model_kwargs(model_args, training_args):
    extra = training_args.model_init_kwargs
    if extra is None:
        extra = {}
    if not isinstance(extra, dict) or any(not isinstance(key, str) for key in extra):
        raise ValueError("model_init_kwargs must be a mapping with string keys.")
    if {"token", "use_auth_token"} & extra.keys():
        raise ValueError("Use HF_TOKEN or Hub login for credentials, not model_init_kwargs.")
    owned = {
        "revision",
        "trust_remote_code",
        "attn_implementation",
        "torch_dtype",
        "use_cache",
        "quantization_config",
        "device_map",
        "load_in_4bit",
        "load_in_8bit",
    }
    conflicts = owned & extra.keys()
    if conflicts:
        raise ValueError(
            f"model_init_kwargs cannot override managed settings: {', '.join(sorted(conflicts))}. "
            "Use the dedicated model/quantization fields; cache and placement are managed by training."
        )
    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
    )
    if model_args.load_in_4bit or model_args.load_in_8bit:
        if not model_args.use_peft:
            raise ValueError("4-bit/8-bit loading is only supported with PEFT/LoRA training.")
        compute_dtype = torch_dtype if isinstance(torch_dtype, torch.dtype) else torch.bfloat16
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=model_args.load_in_4bit,
            load_in_8bit=model_args.load_in_8bit,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=model_args.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=model_args.use_bnb_nested_quant,
        )
        if torch.cuda.is_available():
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            model_kwargs["device_map"] = {"": local_rank}
    return {**extra, **model_kwargs}
