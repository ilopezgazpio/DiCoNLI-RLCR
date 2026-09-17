from dataclasses import dataclass, field
from typing import Optional, List, Dict
from rlcr.text.prompts import get_sys_prompt


@dataclass
class LocalConfig:
    name: str = field(metadata={"help": "Name of the local configuration, used for naming outputs"})
    model: str = field(metadata={"help": "Model identifier to use for generation"})
    tokenize_key: str = field(
        default="prompt",
        metadata={"help": "Key in the dataset to tokenize before feeding into the model"},
    )
    n: int = field(default=1, metadata={"help": "Number of samples to generate per input"})
    temperature: float = field(default=0, metadata={"help": "Sampling temperature for generation"})
    max_tokens: int = field(
        default=4096, metadata={"help": "Maximum number of tokens to generate per sample"}
    )
    seed: Optional[int] = field(default=42, metadata={"help": "Random seed for reproducibility"})
    check_fn: str = field(
        default=None, metadata={"help": "Function to check the dataset post-processing"}
    )
    check_fn_args: Dict = field(
        default_factory=dict, metadata={"help": "Arguments to pass to the check function"}
    )
    task_spec: str = field(
        default="gen", metadata={"help": "Only generation-based evaluation is supported."}
    )
    sys_prompt_name: str = field(default="gen", metadata={"help": "System prompt name."})
    pass_k_vals: List = field(
        default_factory=list, metadata={"help": "List of k values to pass to the check function"}
    )
    tasks: List[str] = field(
        default_factory=lambda: ["generate"],
        metadata={"help": "Generation and post-processing tasks"},
    )
    class_model: str = field(
        default=None, metadata={"help": "Model to use for classification if using a hybrid task"}
    )
    fresh: bool = field(default=False, metadata={"help": "Whether to overwrite existing results"})
    split_at_confidence: bool = field(
        default=False, metadata={"help": "Whether to split at confidence"}
    )
    load_in_4bit: bool = field(
        default=False, metadata={"help": "Whether to load the generation model in 4-bit"}
    )
    torch_dtype: str = field(default="bfloat16", metadata={"help": "Torch dtype for HF generation"})
    hf_batch_size: int = field(
        default=1, metadata={"help": "Batch size for Transformers inference"}
    )

    def __post_init__(self):
        if self.task_spec != "gen":
            raise ValueError(
                "Only task_spec: gen is supported; SFT/ORM evaluation is not implemented."
            )
        get_sys_prompt(self.sys_prompt_name)
