"""Exact-label correctness; no free-text or symbolic answer matching."""
from rlcr.text.prediction import parse_prediction


def accuracy_reward(completions, label, *, allowed_labels=None, **kwargs):
    rewards = []
    for completion, target in zip(completions, label, strict=True):
        if allowed_labels is not None and target not in allowed_labels:
            raise ValueError(f"Gold label {target!r} is outside the allowed vocabulary.")
        prediction = parse_prediction(completion, allowed_labels=allowed_labels)
        rewards.append(-1.0 if prediction is None else float(prediction[0] == target))
    return rewards
