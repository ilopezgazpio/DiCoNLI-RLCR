"""One model's raw prediction-generation settings."""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LocalConfig:
    name: str = field(metadata={"help": "Name used for output columns."})
    model: str = field(metadata={"help": "Model ID or local model/adapter directory."})
    tokenize_key: str = field(default="prompt")
    n: int = field(default=1)
    temperature: float = field(default=0)
    max_tokens: int = field(default=4096)
    seed: Optional[int] = field(default=42)
    fresh: bool = field(default=False)
    load_in_4bit: bool = field(default=False)
    torch_dtype: str = field(default="bfloat16")
    hf_batch_size: int = field(default=1)
