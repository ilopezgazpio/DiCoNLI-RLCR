import os
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig


def load_hf_generator(config):
    dtype_name = config.torch_dtype
    torch_dtype = "auto" if dtype_name in [None, "auto"] else getattr(torch, dtype_name)
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch_dtype,
        "device_map": "auto",
    }
    if config.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    if os.path.exists(os.path.join(config.model, "adapter_config.json")):
        from peft import AutoPeftModelForCausalLM

        model = AutoPeftModelForCausalLM.from_pretrained(config.model, **model_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(config.model, **model_kwargs)
    model.eval()
    return model
