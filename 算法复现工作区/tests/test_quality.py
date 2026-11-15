"""Independent checks for inference changes that can affect fault recognition."""
import json

import numpy as np
import pytest

from fault_workflow.common import raw_to_model
from fault_workflow.inference import (author_overlap_mask, horizontal_ensemble, predict_context_core,
                                      separable_overlap_mask, sliding_predict)
from fault_workflow.policy import INVALID, POSITIVE, UNKNOWN, ReferenceAccumulator
from fault_workflow.quality import core_slices, edge_mask, extract_contexts, score_region


def test_separable_weights_retain_all_axes_and_reflection_symmetry():
    mask = separable_overlap_mask(8, 4)
    # Each outermost face has exp(-4.5), corner has their product exp(-13.5).
    assert mask[0, 3, 3] == pytest.approx(np.exp(-4.5))
    assert mask[0, 0, 0] == pytest.approx(np.exp(-13.5))
    assert author_overlap_mask(8, 4)[0, 3, 3] == 1
    assert 0 < mask.min() <= mask.max() <= 1
    for axis in range(3):
        np.testing.assert_array_equal(mask, np.flip(mask, axis=axis))
        np.testing.assert_allclose(mask, np.swapaxes(mask, 0, axis), rtol=2e-7, atol=0)
    np.testing.assert_array_equal(separable_overlap_mask(8, 0), np.ones((8,) * 3))
    with pytest.raises(ValueError):
        separable_overlap_mask(8, 5)


def test_context_result_restores_core_coordinates_and_constant_probability():
    raw = np.arange(12 * 16 * 12, dtype=np.float32).reshape(12, 16, 12)
    seen = []

    def predictor(array):
        seen.append(array.shape)
        return np.full(array.shape, 0.7, dtype=np.float32)

    result, metadata = predict_context_core(raw, predictor, (2, 4, 2), (8, 8, 8), 8, 4)
    assert result.shape == (8, 8, 8) and result.flags.c_contiguous
    np.testing.assert_allclose(result, 0.7, atol=2e-7)
    assert len(seen) == 12
    assert metadata["halo_after_raw"] == [2, 4, 2]
    assert metadata["output_order"].startswith("raw")
    with pytest.raises(ValueError, match="contained"):
        predict_context_core(raw, predictor, (5, 4, 2), (8, 8, 8), 8, 4)


def test_valid_zero_crossings_use_full_trace_even_in_new_regions(tmp_path, monkeypatch):
    import fault_workflow.quality as quality

    seismic = np.zeros((12, 12, 12), dtype=np.float64)
    seismic[:, :, 0] = 1
    seismic[:, :, -1] = -1
    seismic[0, 0, :] = 0
    labels = np.zeros_like(seismic, dtype=bool)
    labels[3, 3, 5] = True
    np.savez_compressed(tmp_path / "seis.npz", seismic)
    np.savez_compressed(tmp_path / "fault.npz", labels)
    pair = {"split": "val", "section_start": 0, "section_stop": 12,
            "seismic": "seis.npz", "labels": "fault.npz"}
    audit = {"thebe": {"pairs": [pair]}, "source_fingerprints": []}
    protocol = {"halo_raw": [1, 1, 1], "thebe_regions": [
        {"name": "zero_crossing", "start_raw": [2, 2, 4], "shape_raw": [4, 4, 4]}]}
    out = tmp_path / "out"
    out.mkdir()
    quality.extract_contexts({"data_root": tmp_path}, audit, protocol, out)
    context = np.load(out / "zero_crossing_seismic_raw.npy")
    states = np.load(out / "zero_crossing_states_raw.npy")
    assert np.all(context == 0)  # Still valid because the original full traces extend past this crop.
    assert states[2, 2, 2] == POSITIVE
    assert np.count_nonzero(states == UNKNOWN) == states.size - 1
    assert not np.any(states == INVALID)
    # Reuse checked hashes; a corrupt array must fail, not produce a better score.
    context[0, 0, 0] = 99
    np.save(out / "zero_crossing_seismic_raw.npy", context)
    with pytest.raises(ValueError, match="cache changed"):
        extract_contexts({"data_root": tmp_path}, audit, protocol, out)
    assert json.loads((out / "context_manifest.json").read_text(encoding="utf-8"))["signature"]["protocol"] == protocol


def test_context_and_core_crop_preserve_unequal_axis_tiling():
    raw = np.arange(12 * 16 * 20, dtype=np.float32).reshape(12, 16, 20)

    def predictor(array):
        return (1 / (1 + np.exp(-array))).astype(np.float32)

    expected, _ = sliding_predict(raw_to_model(raw), predictor, 8, 4, fusion="separable")
    actual, _ = predict_context_core(raw, predictor, (2, 4, 5), (8, 8, 9), 8, 4)
    np.testing.assert_array_equal(actual, expected.transpose(2, 1, 0)[2:10, 4:12, 5:14])


def test_local_effect_scores_do_not_invent_confirmed_negatives():
    labels = np.zeros((5, 5, 5), dtype=bool)
    labels[0, 0, 0] = labels[2, 2, 2] = True
    states = np.where(labels, POSITIVE, UNKNOWN).astype(np.uint8)
    probability = np.zeros_like(labels, dtype=np.float32)
    probability[0, 0, 0] = probability[1, 1, 1] = 0.9
    result = score_region(states, labels, probability, {"threshold": 0.5, "diagnostic_thresholds": [0.5], "boundary_band_samples": 1})
    assert result["at_fixed_threshold"]["reference_positive_coverage"] == 0.5
    assert result["at_fixed_threshold"]["precision"] is None
    assert result["response_outside_supplied_reference"] == 1
    assert result["edge_and_interior"]["interior"]["positive_coverage"] == 0
    assert result["edge_and_interior"]["edge_band"]["positive_coverage"] == 1
    assert np.count_nonzero(edge_mask((5, 5, 5), 1)) == 125 - 27
    assert core_slices((5, 6, 7), (1, 2, 3)) == (slice(1, 6), slice(2, 8), slice(3, 10))


def test_thebe_quality_context_cannot_cross_into_test(tmp_path):
    protocol = {"halo_raw": [32, 32, 32], "thebe_regions": [
        {"name": "bad", "start_raw": [100, 100, 100], "shape_raw": [128, 128, 128]}]}
    with pytest.raises(ValueError, match="VALIDATION"):
        extract_contexts({"data_root": tmp_path}, {"source_fingerprints": [], "thebe": {"pairs": []}}, protocol, tmp_path)
    protocol["clip_context_to_validation"] = True
    with pytest.raises(ValueError, match="VALIDATION"):
        extract_contexts({"data_root": tmp_path}, {"source_fingerprints": [], "thebe": {"pairs": []}}, protocol, tmp_path)


def test_positive_reference_cannot_silently_become_unknown():
    accumulator = ReferenceAccumulator([0.5])
    with pytest.raises(ValueError, match="contradict"):
        accumulator.update(np.array([UNKNOWN]), np.array([1]), np.array([0.9]))


def test_unresolved_checkpoint_training_loss_stops_before_training(tmp_path, monkeypatch):
    import h5py
    import fault_workflow.baseline as baseline

    path = tmp_path / "checkpoint.hdf5"
    with h5py.File(path, "w") as archive:
        archive.attrs["training_config"] = json.dumps({"loss": "cross_entropy_balanced"})
    monkeypatch.setattr(baseline, "verified_checkpoint", lambda config: path)
    with pytest.raises(ValueError, match="recipe unresolved"):
        baseline.require_checkpoint_recipe_match({}, {"loss": "binary_crossentropy"})
    baseline.require_checkpoint_recipe_match({}, {"loss": "cross_entropy_balanced"})


def test_horizontal_ensemble_inverts_every_transform_without_vertical_reflection():
    raw = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5) / 60
    seen = []

    def equivariant_predictor(array):
        seen.append(array.copy())
        return array

    actual = horizontal_ensemble(equivariant_predictor)(raw)
    np.testing.assert_array_equal(actual, raw)
    assert len(seen) == 4
    assert all(float(v[0].mean()) < float(v[-1].mean()) for v in seen)
    # Output position bias must be undone, not averaged in the reflected coordinate system.
    def corner_predictor(array):
        result = np.zeros(array.shape, dtype=np.float32)
        result[0, 0, 0] = 1
        return result

    result = horizontal_ensemble(corner_predictor)(raw)
    assert result[0, 0, 0] == result[0, -1, -1] == 0.25
    assert result.sum() == 1 and result[-1].sum() == 0
