"""Parse an exact label and a scalar correctness confidence, without answer repair."""
import math
import re


_PREDICTION = re.compile(
    r"\s*<answer>\s*([^<>]+?)\s*</answer>\s*" r"<confidence>\s*([^<>]+?)\s*</confidence>\s*"
)


def parse_prediction(completion, *, allowed_labels=None):
    """Return (label, confidence), or None for an invalid structured response."""
    if isinstance(completion, list):
        if len(completion) != 1 or not isinstance(completion[0], dict):
            return None
        if completion[0].get("role") != "assistant":
            return None
        completion = completion[0].get("content")
    if not isinstance(completion, str):
        return None
    match = _PREDICTION.fullmatch(completion)
    if match is None or not match[1].strip():
        return None
    try:
        confidence = float(match[2])
    except ValueError:
        return None
    if not math.isfinite(confidence) or not 0 <= confidence <= 1:
        return None
    label = match[1].strip()
    if allowed_labels is not None and label not in allowed_labels:
        return None
    return label, confidence
