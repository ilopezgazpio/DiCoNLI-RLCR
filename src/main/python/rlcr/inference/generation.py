from dataclasses import dataclass
from .completion import Completion


@dataclass
class Generation:
    outputs: list[Completion]
