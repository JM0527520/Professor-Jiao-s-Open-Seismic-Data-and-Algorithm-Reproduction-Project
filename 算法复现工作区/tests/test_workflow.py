"""Regressions for actual scientific-data risks, independent of any network score."""
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest

from fault_workflow.audit import summarize_plane
from fault_workflow.common import load_config, model_to_raw, raw_to_model, relative_path, standardize
from fault_workflow.data import npz_header, plane_stream, rectangle_start, thebe_parts, valid_trace_envelope
from fault_workflow.inference import (author_overlap_mask, reference_overlap, sliding_predict,
                                      tile_starts, validate_probability)


def test_axis_coordinates_are_preserved_for_unequal_axes():
    raw = np.fromfunction(lambda section, lateral, sample: 100 * section + 10 * lateral + sample, (2, 3, 4))
    model = raw_to_model(raw)
    assert model.shape == (4, 3, 2)
    assert model[3, 1, 0] == 13
    np.testing.assert_array_equal(model_to_raw(model), raw)
    with pytest.raises(ValueError):
        raw_to_model(raw[0])


def test_zscore_matches_author_float32_calculation():
    raw = np.linspace(-2, 7, 120, dtype=np.float32).reshape(4, 5, 6)
    expected = (raw - np.mean(raw)) / np.std(raw)
    result, info = standardize(raw)
    np.testing.assert_array_equal(result, expected)
    assert not info["constant"]
    zeros, info = standardize(np.full((3, 4, 5), 7, dtype=np.float32))
    assert info["constant"] and np.count_nonzero(zeros) == 0


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_reject_nonfinite_inputs(bad):
    raw = np.zeros((2, 3, 4), dtype=np.float32)
    raw[0, 0, 0] = bad
    with pytest.raises(ValueError):
        standardize(raw)


def test_npz_stream_reconstructs_values_without_array_concatenation(tmp_path):
    raw = np.arange(120, dtype=np.float64).reshape(4, 5, 6)
    path = tmp_path / "part.npz"
    np.savez_compressed(path, raw)
    header = npz_header(path)
    assert header["shape"] == [4, 5, 6]
    with plane_stream(path) as planes:
        reconstructed = np.stack(list(planes))
    np.testing.assert_array_equal(reconstructed, raw)


def test_npz_rejects_wrong_members_and_bad_crc(tmp_path):
    path = tmp_path / "wrong.npz"
    np.savez_compressed(path, named=np.zeros((2, 3, 4)))
    with pytest.raises(ValueError, match="Unexpected NPZ"):
        npz_header(path)
    good = tmp_path / "corrupt.npz"
    np.savez(good, np.arange(120).reshape(4, 5, 6))
    data = bytearray(good.read_bytes())
    with zipfile.ZipFile(good) as archive:
        info = archive.getinfo("arr_0.npy")
        offset = info.header_offset
        name_size = int.from_bytes(data[offset + 26:offset + 28], "little")
        extra_size = int.from_bytes(data[offset + 28:offset + 30], "little")
        data[offset + 30 + name_size + extra_size + info.file_size - 1] ^= 1
    good.write_bytes(data)
    with pytest.raises(zipfile.BadZipFile):
        with plane_stream(good) as planes:
            list(planes)


def test_parts_are_numeric_and_missing_parts_fail(tmp_path):
    folder = tmp_path / "02_Thebe/data/npzfiles/seis"
    folder.mkdir(parents=True)
    for number in reversed(range(1, 10)):
        (folder / f"seistrain{number}.npz").touch()
    assert [p.stem for p in thebe_parts(tmp_path, "train", "seis")] == [f"seistrain{i}" for i in range(1, 10)]
    (folder / "seistrain3.npz").unlink()
    with pytest.raises(ValueError, match="Missing"):
        thebe_parts(tmp_path, "train", "seis")


def test_roi_selection_is_deterministic_and_within_bounds():
    projection = np.zeros((12, 15), dtype=np.int32)
    projection[6:10, 8:13] = 1
    assert rectangle_start(projection, 4, 5) == (6, 8)
    with pytest.raises(ValueError, match="No labelled"):
        rectangle_start(np.zeros_like(projection), 4, 5)


def test_zero_crossing_is_not_confused_with_empty_trace():
    raw = np.array([[[0, 2, 0, -2, 0], [0, 0, 0, 0, 0]]], dtype=float)
    expected = np.array([[[False, True, True, True, False], [False] * 5]])
    np.testing.assert_array_equal(valid_trace_envelope(raw), expected)


def test_report_counts_labels_on_padding_without_dropping_them():
    raw = np.array([[0, 1, 0, -1], [0, 0, 0, 0]], dtype=float)
    labels = np.array([[False, True, False, False], [False, True, True, False]])
    stats = summarize_plane(raw, labels)
    assert stats["fault_voxels"] == 3
    assert stats["fault_voxels_on_empty_traces"] == 2
    assert stats["empty_traces"] == 1
    assert stats["label_sample_bounds"] == [1, 3]


@pytest.mark.parametrize("length,patch,stride", [(19, 8, 4), (8, 8, 4), (3, 8, 4), (25, 8, 8)])
def test_tiling_has_no_holes_or_extra_tail(length, patch, stride):
    covered = np.zeros(max(length, patch), dtype=bool)
    starts = tile_starts(length, patch, stride)
    for start in starts:
        covered[start:start + patch] = True
    assert covered.all()
    assert starts[-1] + patch == max(length, patch)
    with pytest.raises(ValueError):
        tile_starts(length, patch, patch + 1)


def test_mask_preserves_author_overwrite_semantics():
    mask = author_overlap_mask(8, 4)
    assert mask[0, 3, 3] == 1  # The third-axis assignment overwrites the first-axis taper.
    assert mask[3, 3, 0] == pytest.approx(np.exp(-4.5))
    assert np.all(mask > 0)


def test_weighted_stitching_and_tail_restore():
    # Independent expected overlap: first tile predicts 0.2, second tile 0.8.
    raw = np.arange(8 * 12 * 8, dtype=np.float32).reshape(8, 12, 8)
    calls = []

    def predictor(array):
        calls.append(array.shape)
        return np.full(array.shape, 0.2 if len(calls) == 1 else 0.8, dtype=np.float32)

    result, info = sliding_predict(raw, predictor, patch=8, stride=4)
    assert info["tiles"] == 2 and result.shape == raw.shape
    np.testing.assert_allclose(result[:, :4, :], 0.2, atol=1e-6)
    np.testing.assert_allclose(result[:, 4:8, :], 0.5, atol=1e-6)
    np.testing.assert_allclose(result[:, 8:, :], 0.8, atol=1e-6)
    small = np.arange(3 * 5 * 7, dtype=np.float32).reshape(3, 5, 7)
    restored, _ = sliding_predict(small, lambda a: np.full(a.shape, 0.25), patch=8, stride=4)
    assert restored.shape == small.shape
    np.testing.assert_allclose(restored, 0.25, atol=1e-6)


def test_constant_padding_never_calls_model_and_bad_outputs_fail():
    def forbidden(_):
        raise AssertionError("Constant input must be skipped")

    result, _ = sliding_predict(np.zeros((8, 8, 8)), forbidden, patch=8, stride=4)
    assert np.count_nonzero(result) == 0
    with pytest.raises(ValueError):
        validate_probability(np.full((2, 3, 4), 1.2), (2, 3, 4))
    with pytest.raises(ValueError):
        validate_probability(np.zeros((2, 4, 3)), (2, 3, 4))


def test_reference_overlap_has_independently_known_counts():
    labels = np.array([True, True, False, False]).reshape(1, 2, 2)
    probs = np.array([0.9, 0.1, 0.8, 0.2]).reshape(1, 2, 2)
    result = reference_overlap(labels, probs, np.ones_like(labels), 0.5)
    assert result["tp"] == result["fp_vs_supplied_reference"] == result["fn"] == 1
    assert result["dice_vs_supplied_reference"] == 0.5


def test_config_locks_test_and_protects_original_data(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    config = {"data_root": str(source), "output_root": str(tmp_path / "out"),
              "cache_root": str(tmp_path / "cache"), "thebe_demo_split": "test", "patch_size": 128, "stride": 64}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="TEST is locked"):
        load_config(path)
    config.update(thebe_demo_split="val", output_root=str(source / "out"))
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="original dataset"):
        load_config(path)
    assert relative_path(r"data\train\0.dat") == Path("data/train/0.dat")
