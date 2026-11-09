"""Annotation semantics, source-generator parity and bounded-memory reconstruction."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from fault_workflow.baseline import AUTHOR_DIR, get_policy, training_settings, writable_path
from fault_workflow.common import load_config, model_to_raw, raw_to_model
from fault_workflow.evaluation import cache_thebe_validation, disk_sliding_predict, run_evaluation
from fault_workflow.inference import sliding_predict
from fault_workflow.policy import (INVALID, NEGATIVE, POSITIVE, UNKNOWN,
                                   ReferenceAccumulator, annotation_states)
from fault_workflow.training import author_batch, epoch_orders, run_gpu_smoke, run_training


def test_padding_unknown_and_zero_crossing_have_distinct_states():
    seismic = np.array([[[0, 1, 0, -1, 0], [0, 0, 0, 0, 0]]], dtype=float)
    labels = np.array([[[1, 1, 0, 0, 1], [0, 1, 0, 0, 0]]], dtype=bool)
    original = labels.copy()
    states = annotation_states(seismic, labels)
    np.testing.assert_array_equal(states, [[[INVALID, POSITIVE, UNKNOWN, UNKNOWN, INVALID], [INVALID] * 5]])
    np.testing.assert_array_equal(labels, original)
    complete = np.zeros_like(labels)
    complete[..., 2] = True
    states = annotation_states(seismic, labels, complete)
    assert states[0, 0, 2] == NEGATIVE and states[0, 1, 2] == INVALID


def test_full_trace_support_must_precede_crop():
    seismic = np.array([[[2, 0, 0, -2]]], dtype=float)
    labels = np.array([[[0, 1, 0, 0]]], dtype=bool)
    correct = annotation_states(seismic, labels)[..., 1:3]
    assert correct[0, 0, 0] == POSITIVE and correct[0, 0, 1] == UNKNOWN
    cropped = annotation_states(seismic[..., 1:3], labels[..., 1:3])
    assert np.all(cropped == INVALID)  # This tempting order would silently lose valid samples.


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_bad_annotation_inputs_fail(bad):
    seismic = np.full((1, 2, 3), bad)
    with pytest.raises(ValueError):
        annotation_states(seismic, np.zeros_like(seismic))


def test_unconfirmed_negatives_never_produce_precision_or_false_positives():
    states = np.array([POSITIVE, POSITIVE, UNKNOWN, INVALID])
    labels = np.array([1, 1, 0, 1])
    probability = np.array([0.9, 0.1, 0.99, 0.99])
    score = ReferenceAccumulator([0.5])
    score.update(states, labels, probability)
    report = score.report()
    row = report["threshold_sweep"][0]
    assert report["invalid_reference_positive_voxels"] == 1
    assert row["reference_positive_coverage"] == 0.5
    assert row["response_fraction_in_amplitude_support"] == pytest.approx(2 / 3)
    assert row["dice"] is row["precision"] is row["recall"] is None
    assert not report["strict_metrics_enabled"]


def test_micro_counts_and_no_positive_denominator():
    score = ReferenceAccumulator([0.5])
    score.update(np.array([POSITIVE, NEGATIVE]), np.array([1, 0]), np.array([0.9, 0.9]))
    score.update(np.array([POSITIVE, POSITIVE, POSITIVE]), np.ones(3), np.array([0.1, 0.1, 0.1]))
    row = score.report()["threshold_sweep"][0]
    assert row["dice"] == pytest.approx(2 / 6)  # Global TP=1, FP=1, FN=3.
    assert row["precision"] == 0.5 and row["recall"] == 0.25
    empty = ReferenceAccumulator([0.5])
    empty.update(np.array([INVALID]), np.array([0]), np.array([0.5]))
    row = empty.report()["threshold_sweep"][0]
    assert row["reference_positive_coverage"] is None
    assert row["response_fraction_in_amplitude_support"] is None


@pytest.mark.parametrize("thresholds", [[0.5, 0.5], [1, 0], [np.nan], [-0.1], [1.1], []])
def test_invalid_thresholds_are_rejected(thresholds):
    with pytest.raises(ValueError):
        ReferenceAccumulator(thresholds)


def test_author_batch_matches_pinned_generator_without_importing_into_source():
    config = load_config(Path("configs/baseline.json"))
    path = config["data_root"] / AUTHOR_DIR / "utils.py"
    # Execute only the author's generator class, with its inert Sequence base.
    # Reading an AST avoids legacy imports and never creates __pycache__ in original data.
    parsed = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(n for n in parsed.body if isinstance(n, ast.ClassDef) and n.name == "DataGenerator")
    namespace = {"np": np, "keras": SimpleNamespace(utils=SimpleNamespace(Sequence=object))}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), namespace)
    base = config["data_root"] / "01_FaultSeg3D/训练与验证数据/train"
    generator = namespace["DataGenerator"](str(base / "seis") + "/", str(base / "fault") + "/", [0], shuffle=False)
    expected_x, expected_y = generator[0]
    actual_x, actual_y = author_batch(config, "train", 0)
    np.testing.assert_array_equal(actual_x, expected_x)
    np.testing.assert_array_equal(actual_y, expected_y)
    assert actual_x.shape == (2, 128, 128, 128, 1)


def test_epoch_order_replay_and_all_ids_are_preserved():
    config = load_config(Path("configs/baseline.json"))
    settings = training_settings(config)
    first = list(epoch_orders(settings, 3))
    resumed = list(epoch_orders(settings, 3))[2]
    for train, val in first:
        assert sorted(train) == list(range(200)) and sorted(val) == list(range(20))
    np.testing.assert_array_equal(first[2][0], resumed[0])
    assert not np.array_equal(first[0][0], first[1][0])


def test_disk_tiling_matches_in_memory_author_order(tmp_path):
    raw = np.arange(11 * 13 * 15, dtype=np.float32).reshape(11, 13, 15)
    calls = []

    def predictor(array):
        calls.append(array.mean())
        # Input-dependent spatial response exposes transposition/weight errors.
        return (1 / (1 + np.exp(-array))).astype(np.float32)

    expected, info = sliding_predict(raw_to_model(raw), predictor, 8, 4)
    actual, disk_info = disk_sliding_predict(raw, predictor, tmp_path / "probability.npy", 8, 4)
    np.testing.assert_array_equal(actual, model_to_raw(expected))
    assert disk_info["tiles"] == info["tiles"]
    assert isinstance(actual, np.memmap)


def test_local_commands_cannot_start_deferred_work_or_unlock_test(tmp_path):
    with pytest.raises(ValueError, match="deferred"):
        run_training({}, server_work=False)
    with pytest.raises(ValueError, match="deferred"):
        run_evaluation({}, "synthetic", server_work=False)
    with pytest.raises(ValueError, match="deferred"):
        run_gpu_smoke({}, server_work=False)
    with pytest.raises(ValueError, match="TEST remains locked"):
        run_evaluation({}, "test", server_work=True)
    with pytest.raises(ValueError, match="workspace"):
        writable_path({"output_root": tmp_path, "data_root": tmp_path / "source"}, "results")
    config = load_config(Path("configs/baseline.json"))
    policy = get_policy(config)
    assert policy["thebe"]["complete_annotation_mask"] is None
    assert not policy["thebe"]["supervised_training_enabled"]
    changed = json.loads(json.dumps(config["training"]))
    changed["effective_batch_size"] = 1
    with pytest.raises(ValueError, match="settings changed"):
        training_settings({"training": changed})


def test_full_trace_cache_preserves_numeric_part_order_and_detects_tampering(tmp_path, monkeypatch):
    import fault_workflow.evaluation as evaluation

    # A miniature survey exercises the same streaming/cache code without allocating GB.
    monkeypatch.setattr(evaluation, "WORKSPACE", tmp_path)
    source = tmp_path / "source"
    source.mkdir()
    pairs = []
    for part in range(2):
        seismic = np.zeros((1, 3, 9), dtype=np.float64)
        seismic[0, 0, 1] = part + 1
        seismic[0, 0, -2] = -part - 1
        labels = np.zeros_like(seismic, dtype=bool)
        labels[0, 0, 4] = True  # Valid interior zero.
        labels[0, 1, 4] = True  # Invalid wholly empty trace.
        np.savez_compressed(source / f"seis{part}.npz", seismic)
        np.savez_compressed(source / f"fault{part}.npz", labels)
        pairs.append({"split": "val", "section_start": part, "section_stop": part + 1,
                      "shape": list(seismic.shape), "seismic": f"seis{part}.npz", "labels": f"fault{part}.npz"})
    audit = {"thebe": {"pairs": pairs},
             "source_fingerprints": [{"relative_path": p[k]} for p in pairs for k in ("seismic", "labels")]}
    config = {"data_root": source, "cache_root": tmp_path / "cache"}
    cached = cache_thebe_validation(config, audit, {"version": "test"})
    assert cached["seismic"][1, 0, 1] == 2
    assert cached["states"][0, 0, 4] == POSITIVE
    assert cached["states"][0, 1, 4] == INVALID
    del cached
    reused = cache_thebe_validation(config, audit, {"version": "test"})
    assert reused["seismic"].shape == (2, 3, 9)
    del reused
    path = tmp_path / "cache/baseline_thebe_validation/seismic_raw.npy"
    damaged = np.load(path, mmap_mode="r+")
    damaged[0, 0, 1] = 99
    damaged.flush()
    del damaged
    with pytest.raises(ValueError, match="checksum changed"):
        cache_thebe_validation(config, audit, {"version": "test"})
