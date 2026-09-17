"""Exact-label correctness; no free-text or symbolic answer matching."""
from rlcr.text.prediction import parse_prediction


def accuracy_reward(completions, label, **kwargs):
    rewards = []
    for completion, target in zip(completions, label, strict=True):
        prediction = parse_prediction(completion)
        rewards.append(-1.0 if prediction is None else float(prediction[0] == target))
    return rewards
