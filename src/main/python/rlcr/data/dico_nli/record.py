"""An ordered phrase instance, independent of tokenizers and training prompts."""
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class NLIRecord:
    instance_id: str
    pair_id: str
    text1_lang: str
    text2_lang: str
    text1: str
    text2: str
    label: str | None = None
    # Upstream calls this reverse_pair_id, but its value is an INSTANCE identifier.
    reverse_pair_id: str | None = None
    # Distinguish absent references from a reference-confirmed unpaired negative.
    pairing_available: bool = False
