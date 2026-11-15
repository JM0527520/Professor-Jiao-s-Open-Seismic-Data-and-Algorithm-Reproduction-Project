import numpy as np

from fault_workflow.diagnostics import neighbourhood_any, proximity_diagnosis
from fault_workflow.policy import INVALID, POSITIVE, UNKNOWN


def test_proximity_neighbourhood_is_exact_cube_without_wrapping():
    mask = np.zeros((5, 6, 7), dtype=bool)
    mask[2, 3, 4] = True
    expanded = neighbourhood_any(mask, 1)
    expected = np.zeros_like(mask)
    expected[1:4, 2:5, 3:6] = True
    np.testing.assert_array_equal(expanded, expected)
    mask[:] = False
    mask[0, 0, 0] = True
    expanded = neighbourhood_any(mask, 1)
    assert expanded.sum() == 8 and not expanded[-1].any()
    np.testing.assert_array_equal(neighbourhood_any(mask, 0), mask)


def test_proximity_is_separate_from_exact_scores_and_ignores_padding_predictions():
    states = np.full((8, 8, 8), UNKNOWN, dtype=np.uint8)
    states[2, 2, 2] = states[6, 6, 6] = POSITIVE
    probability = np.zeros(states.shape, dtype=np.float32)
    probability[2, 2, 3] = 0.9
    states[6, 6, 5] = INVALID
    probability[6, 6, 5] = 0.99
    original = probability.copy()
    result = proximity_diagnosis(states, probability, radii=(0, 1, 4))
    assert [r["reference_positive_proximity_fraction"] for r in result["radii"]] == [0, 0.5, 1]
    assert not result["used_for_numeric_promotion_gate"]
    np.testing.assert_array_equal(probability, original)
