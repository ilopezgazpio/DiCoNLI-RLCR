from dataclasses import dataclass, field
from typing import Optional
from rlcr.text.prompts import get_sys_prompt


@dataclass
class GRPOScriptArguments:
    """Dataset selection, prompt formatting, and reward-function settings."""

    dataset_name: str = field(metadata={"help": "Dataset name."})
    dataset_config: Optional[str] = field(
        default=None,
        metadata={
            "help": "Dataset configuration name. Corresponds to the `name` argument of the `datasets.load_dataset` "
            "function."
        },
    )
    dataset_train_split: str = field(
        default="train", metadata={"help": "Dataset split to use for training."}
    )
    dataset_test_split: str = field(
        default="test", metadata={"help": "Dataset split to use for evaluation."}
    )

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={
            "help": "Reward functions: accuracy, format, brier, mean_confidence, confidence_one_or_zero."
        },
    )

    sys_prompt_name: str = field(default="gen", metadata={"help": "System prompt name."})
    task_spec: str = field(
        default="gen",
        metadata={"help": "Only generation-based GRPO is supported.", "choices": ["gen"]},
    )

    train_subset_size: Optional[int] = field(
        default=None,
        metadata={"help": "Size of the training subset."},
    )

    eval_subset_size: Optional[int] = field(
        default=None,
        metadata={"help": "Size of the evaluation subset."},
    )

    format_pattern: Optional[str] = field(
        default="ta",
        metadata={"help": "The format pattern to use for the reward function."},
    )

    def __post_init__(self):
        if self.task_spec != "gen":
            raise ValueError(
                "Only task_spec: gen is supported; SFT/ORM training is not implemented."
            )
        get_sys_prompt(self.sys_prompt_name)
