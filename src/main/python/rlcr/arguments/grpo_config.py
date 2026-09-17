from dataclasses import dataclass, field
from typing import Optional
from transformers import TrainingArguments


@dataclass
class GRPOConfig(TrainingArguments):
    """Training arguments for the local GRPO implementation and generation loop."""

    run_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional descriptor for the run. Notably used for wandb, mlflow and comet logging."
        },
    )

    model_init_kwargs: Optional[dict] = field(
        default=None,
        metadata={
            "help": "Extra model-loading kwargs, e.g. local_files_only or cache_dir. Dedicated model settings cannot be duplicated here."
        },
    )
    # Reward functions need the ground-truth answer and source columns.
    remove_unused_columns: bool = field(default=False)
    num_iterations: int = field(default=1)
    reward_weights: Optional[list[float]] = field(default=None)
    sync_ref_model: bool = field(default=False)
    ref_model_mixup_alpha: float = field(default=0.6)
    ref_model_sync_steps: int = field(default=512)

    # Parameters that control the training
    learning_rate: float = field(
        default=1e-6,
        metadata={
            "help": "Initial learning rate for `AdamW` optimizer. The default value replaces that of "
            "`transformers.TrainingArguments`."
        },
    )

    per_device_train_batch_size: int = field(
        default=8,
        metadata={"help": "Number of completion sequences in each worker's training microbatch."},
    )
    gradient_accumulation_steps: int = field(
        default=4,
        metadata={"help": "Number of microbatches to accumulate before an optimizer update."},
    )
    steps_per_generation: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of microbatches in a generation batch. Defaults to gradient_accumulation_steps."
        },
    )

    generation_batch_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "Batch size to use for generation. If `None`, it defaults to the effective training batch size: "
            "`per_device_train_batch_size * num_processes * gradient_accumulation_steps`."
        },
    )

    beta: float = field(
        default=0.04,
        metadata={"help": "KL coefficient."},
    )

    epsilon: float = field(
        default=0.2,
        metadata={"help": "Epsilon value for clipping."},
    )

    epsilon_high: Optional[float] = field(
        default=None,
        metadata={
            "help": "Upper-bound epsilon value for clipping. If not specified, it defaults to the same value as the "
            "lower-bound specified in argument `epsilon`. Paper DAPO recommends `0.28`."
        },
    )

    disable_dropout: bool = field(
        default=True,
        metadata={
            "help": "Whether to disable dropout in the model. This is useful for training with a reference model, as "
            "it prevents the model from generating different logprobs for the same input."
        },
    )

    wandb_project: Optional[str] = field(
        default="RLCR",
        metadata={"help": ("The project to store runs under.")},
    )
    max_prompt_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "Maximum length of the prompt. If the prompt is longer than this value, it will be truncated left."
        },
    )
    num_generations: Optional[int] = field(
        default=4,
        metadata={"help": "Number of generations to sample."},
    )
    temperature: Optional[float] = field(
        default=0.9,
        metadata={
            "help": "Temperature for sampling. The higher the temperature, the more random the completions."
        },
    )
    max_completion_length: Optional[int] = field(
        default=256,
        metadata={"help": "Maximum length of the generated completion."},
    )

    # Parameters that control the logging
    log_completions: bool = field(
        default=True,
        metadata={"help": "Log completion tables to WandB at the regular Trainer logging cadence."},
    )

    num_completions_to_log: Optional[int] = field(
        default=3, metadata={"help": "Number of completions to log."}
    )

    mask_truncated_completions: bool = field(
        default=False,
        metadata={
            "help": "When enabled, truncated completions are excluded from the loss calculation, preventing them from "
            "being incorrectly penalized and introducing noise during training. According to the DAPO paper, this is "
            "a good practice for training stability."
        },
    )

    loss_type: str = field(
        default="bnpo",
        metadata={
            "help": "Specifies the loss formulation to use. Supported values are `grpo`, `bnpo`, and `dr_grpo`. "
            "`'grpo'`: Aggregates token-level losses by normalizing over sequence length. Not recommended due to "
            "length bias—this approach tends to prefer shorter completions with positive advantages and longer ones "
            "with negative advantages. "
            "`'bnpo'`: Aggregates token-level losses by normalizing number of active token in the local batch. "
            "Note that normalization is performed over the local batch only, so results may slightly vary depending "
            "on the local batch size, despite a constant effective batch size. When using "
            "`per_device_train_batch_size==1`, the loss is equivalent to the GRPO loss. "
            "`'dr_grpo'`: Aggregates token-level losses by normalizing with a global constant. This method was "
            "introduced in the Dr. GRPO paper to eliminate length bias. The value of the constant corresponds to "
            "`max_completion_length`."
        },
    )

    delta: Optional[float] = field(
        default=None,
        metadata={
            "help": "If set to a float value (e.g., 2.0), enables the upper clipping bound in two-sided GRPO loss. If None (default), the standard GRPO clipping is used. Recommended to be > 1 + epsilon when enabled."
        },
    )

    scale_rewards: bool = field(
        default=True,
        metadata={
            "help": "Whether to scale the rewards by dividing them by their standard deviation. If `True` (default), "
            "the rewards are normalized by the standard deviation, ensuring they have unit variance. If `False`, no "
            "scaling is applied. The Dr. GRPO paper recommends not scaling the rewards, as scaling by the standard "
            "deviation introduces a question-level difficulty bias."
        },
    )

    shuffle_dataset: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to shuffle the training dataset."},
    )

    def __post_init__(self):
        super().__post_init__()

        num_processes = self.world_size
        # The current default effective batch size
        if self.generation_batch_size is not None and self.steps_per_generation is not None:
            raise ValueError(
                "'generation_batch_size' and 'steps_per_generation' can not be both configured at the same time"
            )

        if self.steps_per_generation is None:
            self.steps_per_generation = self.gradient_accumulation_steps

        if self.generation_batch_size is None:
            self.generation_batch_size = (
                self.per_device_train_batch_size * num_processes * self.steps_per_generation
            )

        if self.generation_batch_size % self.per_device_train_batch_size * num_processes != 0:
            raise ValueError(
                f"generation_batch_size ({self.generation_batch_size}) must be divisible by the global batch size "
                f"({self.per_device_train_batch_size * num_processes})."
            )

        self.steps_per_generation = self.generation_batch_size // (
            self.per_device_train_batch_size * num_processes
        )

        possible_values = [
            n_gen
            for n_gen in range(2, self.generation_batch_size + 1)
            if (self.generation_batch_size) % n_gen == 0
        ]

        if self.num_generations not in possible_values:
            raise ValueError(
                f"The effective train batch size ({num_processes} x {self.per_device_train_batch_size} x "
                f"{self.steps_per_generation}) must be evenly divisible by the number of generations per "
                f"prompt ({self.num_generations}). Given the current effective train batch size, the valid values for "
                f"the number of generations are: {possible_values}."
            )

        if self.eval_strategy != "no":
            global_eval_batch_size = self.per_device_eval_batch_size * num_processes
            possible_values = [
                n_gen
                for n_gen in range(2, global_eval_batch_size + 1)
                if (global_eval_batch_size) % n_gen == 0
            ]
            if self.num_generations not in possible_values:
                raise ValueError(
                    f"The global eval batch size ({num_processes} x {self.per_device_eval_batch_size}) must be "
                    f"evenly divisible by the number of generations per prompt ({self.num_generations}). Given the "
                    "current global eval batch size, the valid values for the number of generations are: "
                    f"{possible_values}."
                )
