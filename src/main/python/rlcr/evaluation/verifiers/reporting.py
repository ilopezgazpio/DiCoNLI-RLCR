"""Common labels and calibration metrics for symbolic and LLM judges."""
import numpy as np
from rlcr.evaluation.metrics import compute_pass_n, get_auroc, get_brier, get_ece
from .confidence import confidence_extractor


def build_report(dataset, config, evals):
    lengths, confidences, formats = [], [], []
    for example in dataset:
        responses = [example[f"{config.name}-output_{index}"] for index in range(config.n)]
        extracted = [confidence_extractor(response) for response in responses]
        lengths.append([len(response) for response in responses])
        formats.append([item[0] for item in extracted])
        confidences.append([item[1] for item in extracted])

    class_column = f"{config.name}-class_output"
    if class_column in dataset.column_names:
        scores = dataset[class_column]
        confidences = [[score[1]] if isinstance(score, list) else [score] for score in scores]

    metrics = {}
    for k in dict.fromkeys([*config.pass_k_vals, config.n, 1]):
        if k <= config.n:
            metrics[f"pass@{k}"] = compute_pass_n(evals, k)
    correctness = np.array(evals).flatten()
    confidence = np.array(confidences).flatten()
    metrics.update(
        {
            "brier_score": get_brier(correctness, confidence),
            "ece": get_ece(correctness, confidence),
            "auroc": get_auroc(correctness, confidence),
            "accuracy": metrics["pass@1"],
            "completion length": np.mean(np.array(lengths)),
            "confidence level": np.mean(np.array(confidences)),
            "confidence format adherence": np.mean(np.array(formats)),
        }
    )
    columns = {
        f"{config.name}-evals": evals,
        f"{config.name}-c_lengths": lengths,
        f"{config.name}-confidence_levels": confidences,
        f"{config.name}-conf_format_adherence": formats,
    }
    return columns, metrics
