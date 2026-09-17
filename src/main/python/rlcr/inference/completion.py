from dataclasses import dataclass
from typing import Optional


@dataclass
class Completion:
    text: str
    token_ids: list[int]
    token_logprobs: Optional[list[float]] = None
