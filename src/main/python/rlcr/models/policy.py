# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Construct the trainable policy, optional adapters, and KL reference model."""

import torch
from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from trl.models import create_reference_model


def enable_gradient_checkpointing(model, args):
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    kwargs = args.gradient_checkpointing_kwargs or {}
    if kwargs.get("use_reentrant", True):
        model.enable_input_require_grads()
    return model


def load_policy(model, args, peft_config=None):
    """Load one policy per worker; QLoRA freezes its quantized base weights."""
    kwargs = dict(args.model_init_kwargs or {})
    if isinstance(model, str):
        dtype = kwargs.get("torch_dtype")
        if isinstance(dtype, str) and dtype != "auto":
            kwargs["torch_dtype"] = getattr(torch, dtype)
        elif dtype is not None and dtype != "auto" and not isinstance(dtype, torch.dtype):
            raise ValueError("torch_dtype must be a torch.dtype, dtype name, 'auto', or None.")
        if args.gradient_checkpointing:
            kwargs["use_cache"] = False
        model = AutoModelForCausalLM.from_pretrained(model, **kwargs)
    elif args.model_init_kwargs is not None:
        raise ValueError("model_init_kwargs cannot be used with an instantiated model.")

    is_quantized = getattr(model, "is_loaded_in_8bit", False) or getattr(
        model, "is_loaded_in_4bit", False
    )
    if peft_config is not None:
        if is_quantized:
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=args.gradient_checkpointing,
            )
        if not isinstance(model, PeftModel):
            model = get_peft_model(model, peft_config)
    if args.gradient_checkpointing:
        if peft_config is not None and is_quantized:
            model.config.use_cache = False
        else:
            enable_gradient_checkpointing(model, args)
    return model


def load_reference_policy(model, args, accelerator):
    if args.beta == 0:
        return None
    if is_deepspeed_zero3_enabled():
        reference = AutoModelForCausalLM.from_pretrained(
            model.config._name_or_path,
            **(args.model_init_kwargs or {}),
        )
        reference = accelerator.prepare(reference)
    else:
        reference = create_reference_model(model).to(accelerator.device)
    reference.eval()
    return reference


def disable_dropout(model):
    if model is not None:
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0
