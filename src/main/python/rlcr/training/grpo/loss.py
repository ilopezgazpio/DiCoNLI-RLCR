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

"""The clipped GRPO objective, independent of launchers and model loading."""

import torch


def grpo_loss(logps, old_logps, advantages, mask, args, reference_logps=None):
    old_logps = logps.detach() if old_logps is None else old_logps
    ratio = torch.exp(logps - old_logps)
    epsilon_high = args.epsilon if args.epsilon_high is None else args.epsilon_high
    clipped_ratio = torch.clamp(ratio, 1 - args.epsilon, 1 + epsilon_high)
    bounded_ratio = ratio if args.delta is None else torch.clamp(ratio, max=args.delta)
    per_token_loss = -torch.min(
        bounded_ratio * advantages[:, None], clipped_ratio * advantages[:, None]
    )

    diagnostics = {}
    if args.beta != 0:
        if reference_logps is None:
            raise ValueError("A KL reference is required when beta is nonzero.")
        difference = reference_logps - logps
        kl = torch.exp(difference) - difference - 1
        per_token_loss = per_token_loss + args.beta * kl
        diagnostics["kl"] = (kl * mask).sum() / mask.sum()

    if args.loss_type == "grpo":
        loss = ((per_token_loss * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)).mean()
    elif args.loss_type == "bnpo":
        loss = (per_token_loss * mask).sum() / mask.sum().clamp(min=1.0)
    elif args.loss_type == "dr_grpo":
        loss = (per_token_loss * mask).sum() / (per_token_loss.size(0) * args.max_completion_length)
    else:
        raise ValueError(f"Unknown loss type: {args.loss_type}")

    low = (ratio < 1 - args.epsilon) & (advantages[:, None] < 0)
    high = (ratio > 1 + epsilon_high) & (advantages[:, None] > 0)
    diagnostics["low"] = (low * mask).sum() / mask.sum()
    diagnostics["high"] = (high * mask).sum() / mask.sum()
    diagnostics["region"] = ((low | high) * mask).sum() / mask.sum()
    return loss, diagnostics
