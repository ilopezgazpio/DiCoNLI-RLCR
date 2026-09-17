"""Dataset selection and dataset-neutral reward settings."""
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
        metadata={"help": "Reward functions: accuracy, format, brier."},
    )
    train_subset_size: Optional[int] = field(default=None)
    eval_subset_size: Optional[int] = field(default=None)
