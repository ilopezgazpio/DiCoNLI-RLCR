"""Task vocabulary bindings; formulas stay in the reusable reward functions."""
from rlcr.data.dico_nli.labels import LABELS
from .accuracy import accuracy_reward
from .brier import brier_reward
from .format import format_reward


def dico_accuracy_reward(completions, label, **kwargs):
    return accuracy_reward(completions, label, allowed_labels=LABELS)


def dico_brier_reward(completions, label, **kwargs):
    return brier_reward(completions, label, allowed_labels=LABELS)


def dico_format_reward(completions, **kwargs):
    return format_reward(completions, allowed_labels=LABELS)
