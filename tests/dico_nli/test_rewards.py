"""Official labels and scalar correctness calibration share one strict parser."""
from types import SimpleNamespace

from datasets import Dataset, DatasetDict
import pytest

from rlcr.arguments.script_arguments import GRPOScriptArguments
from rlcr.data.dico_nli.labels import LABELS
from rlcr.rewards.registry import build_reward_functions
from rlcr.text.prediction import parse_prediction
from rlcr.training.datasets import load_training_datasets


def rewards(names=("dico_accuracy", "dico_brier", "dico_format")):
    return build_reward_functions(
        GRPOScriptArguments(dataset_name="unused", reward_funcs=list(names))
    )


@pytest.mark.parametrize("label", LABELS)
def test_every_official_label_and_correctness_confidence(label):
    response = f"<answer>{label}</answer><confidence>0.75</confidence>"
    assert parse_prediction(response, allowed_labels=LABELS) == (label, 0.75)
    assert [f([response], label=[label]) for f in rewards()] == [[1.0], [0.9375], [1.0]]
    wrong = next(item for item in LABELS if item != label)
    assert [f([response], label=[wrong]) for f in rewards()] == [[0.0], [0.4375], [1.0]]


@pytest.mark.parametrize("bad", ["unknown", "equivalence", "FORWARD_ENTAILMENT.", "ENTAILMENT"])
def test_unknown_labels_cannot_earn_low_confidence_brier(bad):
    response = f"<answer>{bad}</answer><confidence>0</confidence>"
    assert parse_prediction(response, allowed_labels=LABELS) is None
    assert [f([response], label=["EQUIVALENCE"]) for f in rewards()] == [[-1.0]] * 3


@pytest.mark.parametrize("confidence", ["nan", "inf", "-0.01", "1.01", "75%"])
def test_invalid_confidence_is_negative_in_every_component(confidence):
    response = f"<answer>EQUIVALENCE</answer><confidence>{confidence}</confidence>"
    assert [f([response], label=["EQUIVALENCE"]) for f in rewards()] == [[-1.0]] * 3


def test_known_wrong_label_low_confidence_is_valid_calibration_not_accuracy():
    response = "<answer>NEGATIVE_OTHER</answer><confidence>0</confidence>"
    assert [f([response], label=["EQUIVALENCE"]) for f in rewards()] == [[0.0], [1.0], [1.0]]


def test_gold_and_kwargs_cannot_relax_task_contract():
    for function in rewards()[:2]:
        with pytest.raises(ValueError, match="Gold label"):
            function(["invalid"], label=["UNKNOWN"])
    response = "<answer>UNKNOWN</answer><confidence>0</confidence>"
    for function in rewards():
        assert function([response], label=["EQUIVALENCE"], allowed_labels=["UNKNOWN"]) == [-1.0]


def test_unrestricted_and_task_rewards_cannot_be_mixed():
    with pytest.raises(ValueError, match="Do not mix"):
        rewards(("accuracy", "dico_brier"))


@pytest.mark.parametrize("labels", [["UNKNOWN"], [None]])
def test_invalid_training_gold_fails_during_dataset_loading(tmp_path, labels):
    DatasetDict({"train": Dataset.from_dict({"prompt": ["input"], "label": labels})}).save_to_disk(
        tmp_path
    )
    args = GRPOScriptArguments(dataset_name=str(tmp_path), reward_funcs=["dico_brier"])
    with pytest.raises(ValueError):
        load_training_datasets(args, SimpleNamespace(eval_strategy="no"))


def test_task_subset_must_preserve_source_groups():
    args = GRPOScriptArguments(
        dataset_name="unused", reward_funcs=["dico_accuracy"], train_subset_size=1
    )
    with pytest.raises(ValueError, match="source-group"):
        load_training_datasets(args, SimpleNamespace(eval_strategy="no"))


def test_all_invalid_groups_have_zero_advantage():
    import torch
    from rlcr.training.grpo.advantages import grouped_advantages

    responses = ["<answer>UNKNOWN</answer><confidence>0</confidence>"] * 4
    components = torch.tensor([f(responses, label=["EQUIVALENCE"] * 4) for f in rewards()[:2]]).T
    advantages, means, stds = grouped_advantages(
        components,
        torch.tensor([1.0, 0.5]),
        SimpleNamespace(num_generations=4, scale_rewards=False),
        0,
        4,
    )
    assert means.tolist() == [-1.5] * 4
    assert advantages.tolist() == stds.tolist() == [0.0] * 4
