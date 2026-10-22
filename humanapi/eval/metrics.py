"""Evaluation metrics: precision, rates, reliability bins, calibration error and bits."""
from __future__ import annotations

import math
from typing import Sequence


def precision(hits: int, autos: int) -> float | None:
    """Share of auto-selected tokens that matched the true token; None when nothing was auto-selected."""
    if autos == 0:
        return None
    return hits / autos


def auto_select_rate(autos: int, total: int) -> float:
    if total == 0:
        return 0.0
    return autos / total


def asks_per_100(asks: int, total: int) -> float:
    if total == 0:
        return 0.0
    return 100.0 * asks / total


def reliability_bins(points: Sequence[tuple[float, bool, float]], n_bins: int = 10) -> list[dict]:
    """Bin (confidence, hit, weight) points into equal-width confidence bins.

    Each returned dict has lo, hi, mean_conf, accuracy and n (the weight sum). Empty bins are omitted.
    """
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1")
    sum_w = [0.0] * n_bins
    sum_conf = [0.0] * n_bins
    sum_hit = [0.0] * n_bins
    for conf, hit, weight in points:
        c = min(max(float(conf), 0.0), 1.0)
        idx = min(int(c * n_bins), n_bins - 1)
        sum_w[idx] += float(weight)
        sum_conf[idx] += float(weight) * c
        sum_hit[idx] += float(weight) if hit else 0.0
    bins: list[dict] = []
    for idx in range(n_bins):
        if sum_w[idx] <= 0.0:
            continue
        bins.append(
            {
                "lo": idx / n_bins,
                "hi": (idx + 1) / n_bins,
                "mean_conf": sum_conf[idx] / sum_w[idx],
                "accuracy": sum_hit[idx] / sum_w[idx],
                "n": sum_w[idx],
            }
        )
    return bins


def expected_calibration_error(bins: Sequence[dict]) -> float:
    """Weighted mean of |mean_conf - accuracy| over the bins; 0.0 when there are no bins."""
    total = sum(float(b["n"]) for b in bins)
    if total <= 0.0:
        return 0.0
    return sum(float(b["n"]) * abs(float(b["mean_conf"]) - float(b["accuracy"])) for b in bins) / total


def bits_contributed(probs_of_choices: Sequence[float]) -> float:
    """Sum of -log2(p) over the person's actual choices (probabilities are floored to avoid infinity)."""
    return sum(-math.log2(max(float(p), 1e-12)) for p in probs_of_choices)
