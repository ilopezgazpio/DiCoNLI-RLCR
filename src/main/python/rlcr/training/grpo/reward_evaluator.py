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

"""Load reward providers and score local completions before a global gather."""

from functools import partial

import torch
from accelerate.utils import gather
from torch import nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer, PreTrainedModel
from trl import apply_chat_template, is_conversational
from trl.trainer.utils import prepare_deepspeed


class RewardEvaluator:
    def __init__(self, functions, tokenizers=None, weights=None, model_kwargs=None):
        self.functions = list(functions) if isinstance(functions, (list, tuple)) else [functions]
        self.names = []
        for index, function in enumerate(self.functions):
            if isinstance(function, str):
                function = AutoModelForSequenceClassification.from_pretrained(
                    function,
                    num_labels=1,
                    **(model_kwargs or {}),
                )
                self.functions[index] = function
            if isinstance(function, nn.Module):
                self.names.append(function.config._name_or_path.split("/")[-1])
            else:
                self.names.append(
                    function.func.__name__ if isinstance(function, partial) else function.__name__
                )

        if weights is not None and len(weights) != len(self.functions):
            raise ValueError(
                "The number of reward weights must match the number of reward functions."
            )
        self.weights = (
            torch.tensor(weights, dtype=torch.float32)
            if weights is not None
            else torch.ones(len(self.functions)) / len(self.functions)
        )
        if tokenizers is None:
            tokenizers = [None] * len(self.functions)
        elif not isinstance(tokenizers, list):
            tokenizers = [tokenizers]
        if len(tokenizers) != len(self.functions):
            raise ValueError(
                "The number of reward tokenizers must match the number of reward functions."
            )
        self.tokenizers = tokenizers
        for index, (function, tokenizer) in enumerate(zip(self.functions, tokenizers)):
            if isinstance(function, PreTrainedModel):
                if tokenizer is None:
                    tokenizer = AutoTokenizer.from_pretrained(function.config._name_or_path)
                if tokenizer.pad_token_id is None:
                    tokenizer.pad_token = tokenizer.eos_token
                function.config.pad_token_id = tokenizer.pad_token_id
                self.tokenizers[index] = tokenizer

    def prepare_models(self, accelerator, deepspeed_enabled):
        for index, function in enumerate(self.functions):
            if isinstance(function, PreTrainedModel):
                self.functions[index] = (
                    prepare_deepspeed(function, accelerator)
                    if deepspeed_enabled
                    else accelerator.prepare_model(
                        function, evaluation_mode=True, device_placement=True
                    )
                )

    def score(self, examples, prompts, completions, completion_ids, prepare_inputs, device):
        values = torch.zeros(len(prompts), len(self.functions), device=device)
        for index, (function, tokenizer) in enumerate(zip(self.functions, self.tokenizers)):
            if isinstance(function, nn.Module):
                if is_conversational(examples[0]):
                    messages = [
                        {"messages": prompt + completion}
                        for prompt, completion in zip(prompts, completions)
                    ]
                    texts = [
                        apply_chat_template(message, tokenizer)["text"] for message in messages
                    ]
                else:
                    texts = [
                        prompt + completion for prompt, completion in zip(prompts, completions)
                    ]
                inputs = prepare_inputs(
                    tokenizer(
                        texts,
                        return_tensors="pt",
                        padding=True,
                        padding_side="right",
                        add_special_tokens=False,
                    )
                )
                with torch.inference_mode():
                    values[:, index] = function(**inputs).logits[:, 0]
            else:
                keys = [
                    key
                    for key in examples[0]
                    if key not in {"prompt", "completion", "completion_ids"}
                ]
                columns = {key: [example[key] for example in examples] for key in keys}
                rewards = function(
                    prompts=prompts,
                    completions=completions,
                    completion_ids=completion_ids,
                    **columns,
                )
                rewards = [torch.nan if reward is None else reward for reward in rewards]
                values[:, index] = torch.tensor(rewards, dtype=torch.float32, device=device)
        # Groups can span workers; normalize only after gathering all rewards.
        return gather(values)
