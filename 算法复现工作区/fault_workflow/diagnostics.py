"""Geometric proximity diagnosis only; never edit saved predictions or gate scores."""
import numpy as np

from .policy import INVALID, POSITIVE


def neighbourhood_any(mask, radius):
    """Boolean L-infinity neighbourhood, with no wrap across array boundaries."""
    if not isinstance(radius, int) or radius < 0 or mask.ndim != 3:
        raise ValueError("Expected a 3D mask and a nonnegative integer radius")
    result = np.asarray(mask, dtype=bool).copy()
    for axis, length in enumerate(mask.shape):
        expanded = result.copy()
        for shift in range(1, min(radius, length - 1) + 1):
            target, source = [slice(None)] * 3, [slice(None)] * 3
            target[axis], source[axis] = slice(shift, None), slice(None, -shift)
            expanded[tuple(target)] |= result[tuple(source)]
            expanded[tuple(source)] |= result[tuple(target)]
        result = expanded
    return result


def proximity_diagnosis(states, probability, threshold=0.5, radii=(0, 2, 4, 8)):
    if states.shape != probability.shape or probability.ndim != 3:
        raise ValueError("Diagnosis arrays must align")
    positive = states == POSITIVE
    prediction = (probability >= threshold) & (states != INVALID)
    rows = []
    for radius in radii:
        nearby = neighbourhood_any(prediction, radius)
        rows.append({"radius_samples": radius, "reference_positive_near_prediction": int((positive & nearby).sum()),
                     "reference_positive_proximity_fraction": float(nearby[positive].mean()) if positive.any() else None})
    return {"threshold": threshold, "reference_positive_voxels": int(positive.sum()), "radii": rows,
            "distance": "L-infinity in raw array indices; maximum coordinate difference per axis",
            "usage": "diagnosis of offsets/bandwidth versus gaps only; NOT an accuracy or object-recall metric",
            "saved_probabilities_modified": False, "used_for_numeric_promotion_gate": False}
