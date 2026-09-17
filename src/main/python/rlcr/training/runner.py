"""Run one training experiment inside a launcher-created worker process."""
import logging
import os
from dataclasses import asdict
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint
from trl import get_peft_config

from rlcr.models.training_kwargs import training_model_kwargs
from rlcr.configuration.snapshots import save_resolved_config
from rlcr.rewards.registry import build_reward_functions
from .datasets import load_training_datasets
from .grpo.trainer import GRPOTrainer
from .logging import configure_logging

logger = logging.getLogger(__name__)


def run_training(script_args, training_args, model_args):
    # Reject unsupported rewards/loading overrides before data access or output writes.
    reward_funcs = build_reward_functions(script_args)
    model_kwargs = training_model_kwargs(model_args, training_args)
    set_seed(training_args.seed)
    configure_logging(script_args, training_args, model_args)
    checkpoint = training_args.resume_from_checkpoint
    if checkpoint is None and os.path.isdir(training_args.output_dir):
        checkpoint = get_last_checkpoint(training_args.output_dir)
    if checkpoint is not None:
        logger.info("Resuming training from %s", checkpoint)

    # Capture parser defaults and CLI overrides before injecting runtime model objects.
    if training_args.process_index == 0:
        settings = {**asdict(script_args), **training_args.to_dict(), **asdict(model_args)}
        # Credential values are never persisted; obtain them from the environment instead.
        for key in ("hub_token", "push_to_hub_token"):
            settings.pop(key, None)
        settings["resume_from_checkpoint"] = checkpoint
        save_resolved_config(training_args.output_dir, settings)

    train_dataset, eval_dataset = load_training_datasets(script_args, training_args)
    training_args.model_init_kwargs = model_kwargs
    if training_args.wandb_project is not None:
        os.environ["WANDB_PROJECT"] = training_args.wandb_project
    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=get_peft_config(model_args) if model_args.use_peft else None,
    )

    logger.info("Starting training")
    result = trainer.train(resume_from_checkpoint=checkpoint)
    result.metrics["train_samples"] = len(train_dataset)
    trainer.save_state()
    trainer.save_model(training_args.output_dir)
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(dataset_name=script_args.dataset_name, tags=["rl-verify"])
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)
    logger.info("Model saved to %s", training_args.output_dir)
    return result.metrics
