"""DiCo-NLI label contract; no inference is assigned to the negative class.

Contract source: SemEval-2027-Task-2-DiCo-NLI/evaluation_functions/labels.py.
These constants describe the data format, not a replacement official scorer.
"""
LABELS = ("EQUIVALENCE", "FORWARD_ENTAILMENT", "BACKWARD_ENTAILMENT", "NEGATIVE_OTHER")
REVERSE_LABELS = {
    "EQUIVALENCE": "EQUIVALENCE",
    "FORWARD_ENTAILMENT": "BACKWARD_ENTAILMENT",
    "BACKWARD_ENTAILMENT": "FORWARD_ENTAILMENT",
}


def reverse_label(label):
    if label not in REVERSE_LABELS:
        raise ValueError(f"Label {label!r} has no deterministic reverse.")
    return REVERSE_LABELS[label]
