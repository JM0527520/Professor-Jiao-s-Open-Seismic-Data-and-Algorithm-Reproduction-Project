"""Controlled single-window padding diagnosis; not a promoted inference path."""
from __future__ import annotations

import numpy as np


def support_standardize(array, support):
    """Use full-trace support supplied by the caller; keep interior zero samples."""
    array, support = np.asarray(array, dtype=np.float32), np.asarray(support)
    if (array.ndim != 3 or support.shape != array.shape or support.dtype != np.bool_
            or not np.isfinite(array).all()):
        raise ValueError("Expected finite aligned 3D amplitude and boolean full-trace support")
    valid = array[support]
    mean = float(valid.mean()) if valid.size else 0.0
    std = float(valid.std()) if valid.size else 0.0
    result = np.zeros_like(array)
    if std:
        result[support] = (valid - np.float32(mean)) / np.float32(std)
    return result, {"mean": mean, "std": std, "constant": std == 0,
                    "support_voxels": int(valid.size), "padding_filled_with_normalized_mean": True}
