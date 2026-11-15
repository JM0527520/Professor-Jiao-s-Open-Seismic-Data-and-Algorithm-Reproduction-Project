import numpy as np
import pytest

from fault_workflow.common import standardize
from fault_workflow.padding_diagnosis import support_standardize


def test_support_normalization_matches_author_when_no_padding_and_preserves_inputs():
    array = np.arange(64, dtype=np.float32).reshape(4, 4, 4)
    original = array.copy()
    expected, _ = standardize(array)
    actual, stats = support_standardize(array, np.ones(array.shape, dtype=bool))
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(array, original)
    assert stats["support_voxels"] == 64


def test_support_normalization_keeps_interior_zeros_and_excludes_only_supplied_padding():
    array = np.array([0, 2, 0, 4, 0], dtype=np.float32).reshape(1, 1, 5)
    support = np.array([False, True, True, True, False]).reshape(array.shape)
    result, stats = support_standardize(array, support)
    assert stats["mean"] == 2 and stats["support_voxels"] == 3
    assert result[0, 0, 2] < 0
    assert not result[~support].any()
    assert abs(float(result[support].mean())) < 1e-6


def test_empty_or_constant_support_is_finite_and_nonboolean_mask_is_rejected():
    array = np.ones((2, 3, 4), dtype=np.float32)
    for support in (np.zeros(array.shape, dtype=bool), np.ones(array.shape, dtype=bool)):
        result, stats = support_standardize(array, support)
        assert not result.any() and stats["constant"]
    with pytest.raises(ValueError, match="boolean"):
        support_standardize(array, np.ones(array.shape, dtype=np.uint8))
