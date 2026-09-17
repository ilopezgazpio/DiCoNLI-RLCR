"""Exact-label rewards retain calibration without task-specific answer matching."""
import math

import pytest

from rlcr.rewards.accuracy import accuracy_reward
from rlcr.rewards.brier import brier_reward
from rlcr.rewards.format import format_reward
from rlcr.text.prediction import parse_prediction
from rlcr.evaluation.metrics import get_brier, get_ece


def completion(label="ENTAILMENT", confidence="0.75"):
    return f"<answer>{label}</answer><confidence>{confidence}</confidence>"


@pytest.mark.parametrize("confidence", ["0", "0.75", "1", "1e-2"])
@pytest.mark.parametrize("chat", [False, True])
def test_valid_predictions(confidence, chat):
    response = completion(confidence=confidence)
    if chat:
        response = [{"role": "assistant", "content": response}]
    assert parse_prediction(response) == ("ENTAILMENT", float(confidence))


@pytest.mark.parametrize(
    "response",
    [
        None,
        "",
        "ENTAILMENT",
        "<answer>ENTAILMENT</answer>",
        completion(confidence="nan"),
        completion(confidence="inf"),
        completion(confidence="-0.1"),
        completion(confidence="1.1"),
        completion(confidence="75%"),
        completion(label=" "),
        completion() + completion(),
        "explanation " + completion(),
        completion() + " explanation",
        [{"role": "user", "content": completion()}],
        [{"role": "assistant", "content": completion()}] * 2,
    ],
)
def test_invalid_predictions_are_not_repaired(response):
    assert parse_prediction(response) is None
    assert format_reward([response]) == [-1.0]
    assert accuracy_reward([response], label=["ENTAILMENT"]) == [-1.0]
    assert brier_reward([response], label=["ENTAILMENT"]) == [-1.0]


def test_rewards_use_exact_labels_and_gold_correctness():
    responses = [completion(), completion("entailment"), completion("ENTAILMENT.", "0")]
    gold = ["ENTAILMENT"] * 3
    assert accuracy_reward(responses, label=gold) == [1.0, 0.0, 0.0]
    assert brier_reward(responses, label=gold) == [0.9375, 0.4375, 1.0]
    assert format_reward(responses) == [1.0] * 3


@pytest.mark.parametrize("reward", [accuracy_reward, brier_reward])
def test_reward_length_mismatch_is_not_silently_truncated(reward):
    with pytest.raises(ValueError):
        reward([completion()], label=[])


def test_generic_calibration_diagnostics():
    assert get_brier([1, 0], [1, 0]) == 0
    assert get_ece([1, 0], [1, 0]) == 0
    assert math.isclose(get_brier([1, 0], [0.75, 0.75]), 0.3125)
