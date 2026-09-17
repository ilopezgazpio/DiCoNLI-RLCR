from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GlobalArgs:
    dataset_name: str = field(metadata={"help": "Hub dataset ID or local dataset directory"})
    output_dir: str = field(metadata={"help": "Local run directory for predictions and metrics"})
    dataset_config: Optional[str] = field(
        default=None, metadata={"help": "Specific configuration for the dataset, if applicable"}
    )
    split: str = field(
        default="test", metadata={"help": "Dataset split to use (e.g., train, test, validation)"}
    )
    hash_key: str = field(
        default="prompt",
        metadata={"help": "Column key to generate a unique hash ID for each example"},
    )
    sample_size: Optional[int] = field(
        default=None, metadata={"help": "Number of samples to use for evaluation"}
    )
    fresh: bool = field(default=False, metadata={"help": "Whether to overwrite existing results"})
