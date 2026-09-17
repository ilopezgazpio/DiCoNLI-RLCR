"""Reward a valid exact-label/scalar-confidence response."""
from rlcr.text.prediction import parse_prediction


def format_reward(completions, *, allowed_labels=None, **kwargs):
    return [
        1.0 if parse_prediction(item, allowed_labels=allowed_labels) is not None else -1.0
        for item in completions
    ]
