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


"""Aggregate training diagnostics and optional completion tables."""
from collections import defaultdict, deque
import torch
from accelerate.utils import gather_object
from .tensors import nanmin, nanmax


class TrainingMetrics:
    def __init__(self, accelerator, num_completions_to_log):
        self.accelerator = accelerator
        self.num_completions_to_log = num_completions_to_log
        self.values = {"train": defaultdict(list), "eval": defaultdict(list)}
        self.textual_logs = {
            "step": deque(maxlen=50000),
            "prompt": deque(maxlen=50000),
            "completion": deque(maxlen=50000),
            "rewards": defaultdict(lambda: deque(maxlen=50000)),
        }

    def record_rollout(
        self,
        mode,
        state,
        attention_mask,
        completion_mask,
        is_eos,
        prompts_text,
        completions_text,
        rewards_per_func,
        mean_grouped_rewards,
        std_grouped_rewards,
        reward_func_names,
    ):
        device = self.accelerator.device
        # Log the metrics
        if mode == "train":
            state.num_input_tokens_seen += (
                self.accelerator.gather_for_metrics(attention_mask.sum()).sum().item()
            )
        self.values[mode]["num_tokens"] = [state.num_input_tokens_seen]

        # log completion lengths, mean, min, max
        agg_completion_mask = self.accelerator.gather_for_metrics(completion_mask.sum(1))
        self.values[mode]["completions/mean_length"].append(
            agg_completion_mask.float().mean().item()
        )
        self.values[mode]["completions/min_length"].append(agg_completion_mask.float().min().item())
        self.values[mode]["completions/max_length"].append(agg_completion_mask.float().max().item())

        agg_terminated_with_eos = self.accelerator.gather_for_metrics(is_eos.any(dim=1))
        term_completion_mask = agg_completion_mask[agg_terminated_with_eos]
        clipped_completions_ratio = 1 - len(term_completion_mask) / len(agg_completion_mask)
        self.values[mode]["completions/clipped_ratio"].append(clipped_completions_ratio)
        if len(term_completion_mask) == 0:
            # edge case where no completed sequences are found
            term_completion_mask = torch.zeros(1, device=device)
        self.values[mode]["completions/mean_terminated_length"].append(
            term_completion_mask.float().mean().item()
        )
        self.values[mode]["completions/min_terminated_length"].append(
            term_completion_mask.float().min().item()
        )
        self.values[mode]["completions/max_terminated_length"].append(
            term_completion_mask.float().max().item()
        )

        # Calculate mean reward per function, but only for samples where the function was applied (non-NaN values)
        for i, reward_func_name in enumerate(reward_func_names):
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self.values[mode][f"rewards/{reward_func_name}"].append(mean_rewards)
        self.values[mode]["reward"].append(mean_grouped_rewards.mean().item())
        self.values[mode]["reward_std"].append(std_grouped_rewards.mean().item())
        self.values[mode]["zero_reward_std_fraction"].append(
            (std_grouped_rewards == 0).float().mean().item()
        )
        invalid_penalty_rewards = {
            "dico_accuracy_reward",
            "dico_brier_reward",
            "dico_format_reward",
        }
        dico_columns = [
            i for i, name in enumerate(reward_func_names) if name in invalid_penalty_rewards
        ]
        if dico_columns:
            invalid = (rewards_per_func[:, dico_columns] < 0).any(dim=1)
            self.values[mode]["dico/invalid_rate"].append(invalid.float().mean().item())

        # Log prompt and completion texts
        num_completions_to_log = self.num_completions_to_log
        self.textual_logs["step"].extend([str(state.global_step)] * num_completions_to_log)
        self.textual_logs["prompt"].extend(gather_object(prompts_text)[0:num_completions_to_log])
        self.textual_logs["completion"].extend(
            gather_object(completions_text)[0:num_completions_to_log]
        )
        for i, name in enumerate(reward_func_names):
            self.textual_logs["rewards"][name].extend(
                rewards_per_func[:, i].tolist()[0:num_completions_to_log]
            )

    def record_loss(self, mode, diagnostics):
        values = self.values[mode]
        if "kl" in diagnostics:
            values["kl"].append(
                self.accelerator.gather_for_metrics(diagnostics["kl"]).nanmean().item()
            )
        for name in ("low", "high", "region"):
            gathered = self.accelerator.gather_for_metrics(diagnostics[name])
            values[f"clip_ratio/{name}_mean"].append(gathered.nanmean().item())
            if name == "low":
                values["clip_ratio/low_min"].append(nanmin(gathered).item())
            elif name == "high":
                values["clip_ratio/high_max"].append(nanmax(gathered).item())

    def drain(self, mode):
        values = {key: sum(items) / len(items) for key, items in self.values[mode].items()}
        self.values[mode].clear()
        if mode == "eval":
            values = {f"eval_{key}": value for key, value in values.items()}
        return values

    def log_completions(self, report_to):
        if not self.accelerator.is_main_process or "wandb" not in report_to:
            return
        import pandas as pd
        import wandb

        if wandb.run is not None:
            table = {
                "step": self.textual_logs["step"],
                "prompt": self.textual_logs["prompt"],
                "completion": self.textual_logs["completion"],
                **self.textual_logs["rewards"],
            }
            wandb.log({"completions": wandb.Table(dataframe=pd.DataFrame(table))})
