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

"""Generate on-policy completions and attach globally normalized rewards."""

import torch
from transformers import GenerationConfig
from trl import is_conversational, maybe_apply_chat_template

from .advantages import grouped_advantages
from .log_probs import per_token_log_probs


def generate_and_score(
    examples,
    *,
    model,
    tokenizer,
    args,
    accelerator,
    reward_evaluator,
    prepare_inputs,
    metrics,
    state,
):
    mode = "train" if model.training else "eval"
    prompts = [example["prompt"] for example in examples]
    texts = [maybe_apply_chat_template(example, tokenizer)["prompt"] for example in examples]
    encoded = prepare_inputs(
        tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
    )
    prompt_ids, prompt_mask = encoded["input_ids"], encoded["attention_mask"]
    if args.max_prompt_length is not None:
        prompt_ids = prompt_ids[:, -args.max_prompt_length :]
        prompt_mask = prompt_mask[:, -args.max_prompt_length :]

    torch.cuda.empty_cache()
    generation_config = GenerationConfig(
        max_new_tokens=args.max_completion_length,
        do_sample=True,
        temperature=args.temperature,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            generated = model.generate(
                input_ids=prompt_ids,
                attention_mask=prompt_mask,
                generation_config=generation_config,
            )
    finally:
        model.train(was_training)
    completion_ids = generated[:, prompt_ids.size(1) :]

    is_eos = completion_ids == tokenizer.eos_token_id
    eos_index = torch.full(
        (is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=accelerator.device
    )
    eos_index[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
    indices = torch.arange(is_eos.size(1), device=accelerator.device).expand(is_eos.size(0), -1)
    completion_mask = (indices <= eos_index.unsqueeze(1)).int()
    token_lists = [
        [token.item() for token, keep in zip(row, mask) if keep]
        for row, mask in zip(completion_ids, completion_mask)
    ]
    if args.mask_truncated_completions:
        completion_mask = completion_mask * is_eos.any(dim=1).unsqueeze(1).int()
    attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
    old_logps = None
    if args.num_iterations > 1 or args.steps_per_generation > args.gradient_accumulation_steps:
        with torch.no_grad():
            old_logps = per_token_log_probs(
                model,
                generated,
                attention_mask,
                completion_ids.size(1),
                args.temperature,
                args.per_device_train_batch_size if mode == "train" else 1,
            )

    completion_texts = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
    if is_conversational(examples[0]):
        completions = []
        for prompt, text in zip(prompts, completion_texts):
            bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
            completions.append([{"role": "assistant", "content": bootstrap + text}])
    else:
        completions = completion_texts

    rewards = reward_evaluator.score(
        examples,
        prompts,
        completions,
        token_lists,
        prepare_inputs,
        accelerator.device,
    )
    advantages, means, stds = grouped_advantages(
        rewards,
        reward_evaluator.weights,
        args,
        accelerator.process_index,
        len(prompts),
    )
    metrics.record_rollout(
        mode,
        state,
        attention_mask,
        completion_mask,
        is_eos,
        texts,
        completion_texts,
        rewards,
        means,
        stds,
        reward_evaluator.names,
    )
    return {
        "prompt_ids": prompt_ids,
        "prompt_mask": prompt_mask,
        "completion_ids": completion_ids,
        "completion_mask": completion_mask,
        "advantages": advantages,
        "old_per_token_logps": old_logps,
    }
