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

"""Transformers integration for GRPO; numerical work lives in focused modules."""

import torch
from accelerate.utils import set_seed
from transformers import Trainer
from trl import SyncRefModelCallback

from rlcr.arguments.grpo_config import GRPOConfig
from rlcr.models.policy import disable_dropout, load_policy, load_reference_policy
from rlcr.models.tokenizer import load_training_tokenizer
from rlcr.training.model_card import create_model_card
from rlcr.training.profiling import profiling_context, profiling_decorator
from .dataloader import build_train_dataloader
from .log_probs import per_token_log_probs
from .loss import grpo_loss
from .repeat_sampler import RepeatSampler
from .reward_evaluator import RewardEvaluator
from .rollout import generate_and_score
from .rollout_buffer import RolloutBuffer
from .training_metrics import TrainingMetrics


def identity_collator(features):
    """Keep prompt conversations and reward columns intact."""
    return features


class GRPOTrainer(Trainer):
    """Coordinate policy setup, rollouts, and the inherited training loop.

    The Transformers Trainer owns optimizer stepping, distributed wrapping,
    callbacks, and checkpoints. There is no separate rollout model or RL layer;
    a reference policy is optional when the KL penalty is enabled.
    """

    _tag_names = ["trl", "grpo"]

    def __init__(
        self,
        model,
        reward_funcs,
        args=None,
        train_dataset=None,
        eval_dataset=None,
        processing_class=None,
        reward_processing_classes=None,
        callbacks=None,
        optimizers=(None, None),
        peft_config=None,
    ):
        if args is None:
            name = model if isinstance(model, str) else model.config._name_or_path
            args = GRPOConfig(output_dir=f"{name.split('/')[-1]}-GRPO")
        model = load_policy(model, args, peft_config)
        if processing_class is None:
            processing_class = load_training_tokenizer(
                model.config._name_or_path, args.model_init_kwargs
            )
        if processing_class.pad_token is None:
            processing_class.pad_token = processing_class.eos_token

        self.reward_evaluator = RewardEvaluator(
            reward_funcs,
            reward_processing_classes,
            args.reward_weights,
            args.model_init_kwargs,
        )
        self.rollout_buffer = RolloutBuffer(args.steps_per_generation, args.num_iterations)
        model.warnings_issued["estimate_tokens"] = True
        super().__init__(
            model=model,
            args=args,
            data_collator=identity_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )
        self.ref_model = load_reference_policy(model, args, self.accelerator)
        if args.disable_dropout:
            disable_dropout(model)
            disable_dropout(self.ref_model)
        self.metrics = TrainingMetrics(self.accelerator, args.num_completions_to_log)
        set_seed(args.seed, device_specific=True)
        self.model_accepts_loss_kwargs = False
        self.model.add_model_tags(self._tag_names)
        if args.sync_ref_model:
            self.add_callback(
                SyncRefModelCallback(ref_model=self.ref_model, accelerator=self.accelerator)
            )
        self.reward_evaluator.prepare_models(self.accelerator, self.is_deepspeed_enabled)

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]

    def get_train_dataloader(self):
        return build_train_dataloader(self)

    def _get_train_sampler(self):
        return RepeatSampler(
            self.train_dataset,
            mini_repeat_count=self.args.num_generations,
            batch_size=self.args.generation_batch_size // self.args.num_generations,
            repeat_count=self.args.num_iterations * self.args.steps_per_generation,
            shuffle=self.args.shuffle_dataset,
            seed=self.args.seed,
        )

    def _get_eval_sampler(self, eval_dataset):
        return RepeatSampler(eval_dataset, mini_repeat_count=1, seed=self.args.seed)

    @profiling_decorator
    def _prepare_inputs(self, examples):
        if self.model.training:
            return self.rollout_buffer.prepare(examples, self._generate_and_score_completions)
        return self._generate_and_score_completions(examples)

    def _generate_and_score_completions(self, examples):
        with profiling_context(self, "rollout"):
            return generate_and_score(
                examples,
                model=self.model,
                tokenizer=self.processing_class,
                args=self.args,
                accelerator=self.accelerator,
                reward_evaluator=self.reward_evaluator,
                prepare_inputs=super()._prepare_inputs,
                metrics=self.metrics,
                state=self.state,
            )

    @profiling_decorator
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        mode = "train" if self.model.training else "eval"
        if mode == "train":
            torch.cuda.empty_cache()
        length = inputs["completion_mask"].sum(1).max()
        completion_mask = inputs["completion_mask"][:, :length]
        completion_ids = inputs["completion_ids"][:, :length]
        input_ids = torch.cat([inputs["prompt_ids"], completion_ids], dim=1)
        attention_mask = torch.cat([inputs["prompt_mask"], completion_mask], dim=1)
        logps = per_token_log_probs(
            model,
            input_ids,
            attention_mask,
            completion_ids.size(1),
            self.args.temperature,
            self.args.per_device_train_batch_size,
        )
        reference_logps = None
        if self.args.beta != 0:
            with torch.no_grad():
                reference_logps = per_token_log_probs(
                    self.ref_model,
                    input_ids,
                    attention_mask,
                    completion_ids.size(1),
                    self.args.temperature,
                    self.args.per_device_train_batch_size,
                )
        loss, diagnostics = grpo_loss(
            logps,
            inputs["old_per_token_logps"],
            inputs["advantages"],
            completion_mask,
            self.args,
            reference_logps,
        )
        self.metrics.record_loss(mode, diagnostics)
        return loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        self._prepare_inputs(inputs)
        return torch.tensor(0.0, device=self.accelerator.device), None, None

    def log(self, logs, start_time=None):
        mode = "train" if self.model.training else "eval"
        super().log({**logs, **self.metrics.drain(mode)}, start_time)
        if self.args.log_completions:
            self.metrics.log_completions(self.args.report_to)

    def create_model_card(self, model_name=None, dataset_name=None, tags=None):
        create_model_card(self, model_name=model_name, dataset_name=dataset_name, tags=tags)
