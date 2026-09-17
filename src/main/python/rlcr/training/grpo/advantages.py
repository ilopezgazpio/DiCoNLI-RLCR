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

"""Compute per-question advantages from rewards gathered across workers."""


def grouped_advantages(rewards_per_func, weights, args, process_index, local_count):
    rewards = (rewards_per_func * weights.to(rewards_per_func.device).unsqueeze(0)).sum(dim=1)
    groups = rewards.view(-1, args.num_generations)
    means = groups.mean(dim=1).repeat_interleave(args.num_generations)
    stds = groups.std(dim=1).repeat_interleave(args.num_generations)
    advantages = rewards - means
    if args.scale_rewards:
        advantages = advantages / (stds + 1e-4)
    start = process_index * local_count
    return advantages[start : start + local_count], means, stds
