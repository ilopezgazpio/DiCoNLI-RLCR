"""Reward scalar confidence in the correctness of the emitted label."""
from rlcr.text.prediction import parse_prediction


def brier_reward(completions, label, **kwargs):
    rewards = []
    for completion, target in zip(completions, label, strict=True):
        prediction = parse_prediction(completion)
        if prediction is None:
            rewards.append(-1.0)
        else:
            answer, confidence = prediction
            correct = float(answer == target)
            rewards.append(1.0 - (confidence - correct) ** 2)
    return rewards
