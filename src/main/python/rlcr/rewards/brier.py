"""Reward scalar confidence in the correctness of the emitted label."""
from rlcr.text.prediction import parse_prediction


def brier_reward(completions, label, *, allowed_labels=None, **kwargs):
    rewards = []
    for completion, target in zip(completions, label, strict=True):
        if allowed_labels is not None and target not in allowed_labels:
            raise ValueError(f"Gold label {target!r} is outside the allowed vocabulary.")
        prediction = parse_prediction(completion, allowed_labels=allowed_labels)
        if prediction is None:
            rewards.append(-1.0)
        else:
            answer, confidence = prediction
            correct = float(answer == target)
            rewards.append(1.0 - (confidence - correct) ** 2)
    return rewards
