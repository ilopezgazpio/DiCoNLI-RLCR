"""Dataset-neutral calibration diagnostics, not the official task scorer."""
import numpy as np


def get_brier(correctness, confidence):
    return np.mean((np.asarray(confidence) - np.asarray(correctness)) ** 2)


def get_ece(correctness, confidence):
    correctness, confidence = np.asarray(correctness), np.asarray(confidence)
    bin_edges = np.linspace(0, 1, 11)
    bin_edges[0], bin_edges[-1] = -np.inf, np.inf
    bin_indices = np.digitize(confidence, bin_edges) - 1
    ece = 0.0
    for index in range(10):
        mask = bin_indices == index
        if np.any(mask):
            ece += np.mean(mask) * abs(np.mean(confidence[mask]) - np.mean(correctness[mask]))
    return ece
