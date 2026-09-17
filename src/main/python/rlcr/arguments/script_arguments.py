"""Prepared dataset selection and explicit reward-family settings."""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GRPOScriptArguments:
    """Training consumes prepared prompts; task adapters are a separate concern."""

    dataset_name: str = field(metadata={"help": "Prepared dataset directory or Hub ID."})
    dataset_config: Optional[str] = field(default=None)
    dataset_train_split: str = field(default="train")
    dataset_test_split: str = field(default="test")
    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "brier"],
        metadata={
            "help": "Rewards: accuracy, format, brier; four-label bindings: dico_accuracy, dico_format, dico_brier. Do not mix families."
        },
    )
    train_subset_size: Optional[int] = field(default=None)
    eval_subset_size: Optional[int] = field(default=None)
