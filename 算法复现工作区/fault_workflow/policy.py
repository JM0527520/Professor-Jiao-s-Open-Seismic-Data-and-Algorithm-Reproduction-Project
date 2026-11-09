"""Explicit annotation states: padding and unknown samples never become negatives."""
from __future__ import annotations

import numpy as np

from .data import valid_trace_envelope

INVALID, UNKNOWN, NEGATIVE, POSITIVE = 0, 1, 2, 3
STATE_NAMES = {INVALID: "invalid", UNKNOWN: "unknown", NEGATIVE: "confirmed_negative", POSITIVE: "reference_positive"}


def annotation_states(seismic_raw, labels_raw, complete_annotation_raw=None) -> np.ndarray:
    seismic, labels = np.asarray(seismic_raw), np.asarray(labels_raw)
    if seismic.shape != labels.shape or seismic.ndim != 3 or not np.isfinite(seismic).all():
        raise ValueError("Expected aligned finite raw-order 3D seismic and labels")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Expected binary supplied reference labels")
    support = valid_trace_envelope(seismic)
    states = np.full(labels.shape, INVALID, dtype=np.uint8)
    states[support] = UNKNOWN
    if complete_annotation_raw is not None:
        complete = np.asarray(complete_annotation_raw)
        if complete.shape != labels.shape or complete.dtype != np.bool_:
            raise ValueError("Confirmed complete-annotation mask must be aligned boolean data")
        states[support & complete & (labels == 0)] = NEGATIVE
    states[support & (labels == 1)] = POSITIVE
    return states


class ReferenceAccumulator:
    """Aggregate counts across planes; never average section-wise Dice ratios."""

    def __init__(self, thresholds):
        self.thresholds = np.asarray(thresholds, dtype=float)
        if (self.thresholds.ndim != 1 or not len(self.thresholds)
                or not np.isfinite(self.thresholds).all()
                or np.any(np.diff(self.thresholds) <= 0)
                or self.thresholds[0] < 0 or self.thresholds[-1] > 1):
            raise ValueError("Thresholds must be finite, strictly increasing values in [0,1]")
        self.tp = np.zeros(len(self.thresholds), dtype=np.int64)
        self.fp = self.tp.copy()
        self.fn = self.tp.copy()
        self.response = self.tp.copy()
        self.support = self.positive = self.negative = self.unknown = self.invalid_positive = 0

    def update(self, states, labels, probability):
        states, labels, probability = map(np.asarray, (states, labels, probability))
        if (states.shape != labels.shape or labels.shape != probability.shape
                or not np.isin(states, list(STATE_NAMES)).all()
                or not np.isin(labels, [0, 1]).all()
                or not np.isfinite(probability).all()
                or np.any(probability < 0) or np.any(probability > 1)):
            raise ValueError("Invalid or misaligned evaluation inputs")
        if np.any((states == POSITIVE) & (labels != 1)) or np.any((states == NEGATIVE) & (labels != 0)):
            raise ValueError("Annotation states contradict reference labels")
        pos, neg, support = states == POSITIVE, states == NEGATIVE, states != INVALID
        self.support += int(support.sum())
        self.positive += int(pos.sum())
        self.negative += int(neg.sum())
        self.unknown += int((states == UNKNOWN).sum())
        self.invalid_positive += int(((states == INVALID) & (labels == 1)).sum())
        for i, threshold in enumerate(self.thresholds):
            predicted = probability >= threshold
            self.tp[i] += np.count_nonzero(predicted & pos)
            self.fp[i] += np.count_nonzero(predicted & neg)
            self.fn[i] += np.count_nonzero(~predicted & pos)
            self.response[i] += np.count_nonzero(predicted & support)

    def report(self) -> dict:
        def ratio(a, b):
            return float(a / b) if b else None

        complete = self.unknown == 0 and self.support > 0
        rows = []
        for i, threshold in enumerate(self.thresholds):
            tp, fp, fn = map(int, (self.tp[i], self.fp[i], self.fn[i]))
            rows.append({"threshold": float(threshold), "covered_reference_positive": tp,
                         "missed_reference_positive": fn,
                         "reference_positive_coverage": ratio(tp, tp + fn),
                         "response_fraction_in_amplitude_support": ratio(self.response[i], self.support),
                         "precision": ratio(tp, tp + fp) if complete else None,
                         "recall": ratio(tp, tp + fn) if complete else None,
                         "dice": ratio(2 * tp, 2 * tp + fp + fn) if complete else None})
        return {"support_voxels": self.support, "reference_positive_voxels": self.positive,
                "confirmed_negative_voxels": self.negative, "unknown_voxels": self.unknown,
                "invalid_reference_positive_voxels": self.invalid_positive,
                "complete_binary_reference_in_support": complete,
                "strict_metrics_enabled": complete,
                "threshold_sweep": rows,
                "interpretation": "Coverage of supplied positives and response area are descriptive; unknown is never a false positive."}
