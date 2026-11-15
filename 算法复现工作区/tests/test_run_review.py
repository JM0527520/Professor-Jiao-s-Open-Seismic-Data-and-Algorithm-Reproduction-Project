"""Fixed-threshold and source-only acceptance boundaries for server work."""
import json

import numpy as np
import pytest

from fault_workflow.run_review import fixed_score, unknown_response, validate_comparison_reports


def paired_reports():
    reports = {}
    for name in ("author_source", "trained_source", "author_thebe", "trained_thebe", "trained_separable"):
        source = name.endswith("source")
        kind = "author" if name.startswith("author") else "trained"
        reports[name] = {"status": "passed", "test_used": False, "split": "validation", "threshold_selected": False,
                         "dataset": "synthetic" if source else "thebe", "policy": {"fixed": True},
                         "model": {"kind": kind, "sha256": kind + "_checkpoint"},
                         "fusion": "separable" if name.endswith("separable") else "author",
                         "volumes": 20, "sections": 200,
                         "cases": ([{"id": i} for i in range(20)] if source else
                                   [{"section_in_validation": i, "global_array_section": 900 + i} for i in range(200)])}
    return reports


def test_paired_review_rejects_checkpoint_mixups_and_duplicate_sections():
    reports = paired_reports()
    validate_comparison_reports(reports, {"fixed": True})
    reports["trained_thebe"]["model"]["sha256"] = "wrong_epoch"
    with pytest.raises(ValueError, match="different checkpoints"):
        validate_comparison_reports(reports, {"fixed": True})
    reports = paired_reports()
    reports["author_thebe"]["cases"][-1] = reports["author_thebe"]["cases"][0]
    with pytest.raises(ValueError, match="200 original"):
        validate_comparison_reports(reports, {"fixed": True})
    reports = paired_reports()
    reports["trained_separable"]["policy"] = {"fixed": False}
    with pytest.raises(ValueError, match="ineligible"):
        validate_comparison_reports(reports, {"fixed": True})


def test_review_uses_fixed_threshold_not_a_list_position():
    report = {"aggregate": {"threshold_sweep": [
        {"threshold": 0.1, "dice": 0.6}, {"threshold": 0.5, "dice": 0.7}]}}
    assert fixed_score(report)["dice"] == 0.7


def test_unknown_response_is_descriptive_and_uses_integer_counts():
    report = {"aggregate": {"confirmed_negative_voxels": 0, "unknown_voxels": 80,
                             "support_voxels": 100, "threshold_sweep": [
        {"threshold": 0.5, "response_fraction_in_amplitude_support": 0.3, "covered_reference_positive": 10}]}}
    assert unknown_response(report) == 0.25
    report["aggregate"]["confirmed_negative_voxels"] = 1
    with pytest.raises(ValueError, match="semantics"):
        unknown_response(report)


def test_source_acceptance_never_certifies_untransferred_thebe(tmp_path, monkeypatch):
    import fault_workflow.source_stage as stage

    (tmp_path / "data").mkdir()
    monkeypatch.setattr(stage, "training_settings", lambda c: {"loss": "cross_entropy_balanced"})
    monkeypatch.setattr(stage, "get_policy", lambda c: {"test_locked": True})
    monkeypatch.setattr(stage, "require_checkpoint_recipe_match", lambda c, s: None)
    monkeypatch.setattr(stage, "audit_synthetic", lambda c: {"status": "passed", "rows": []})
    monkeypatch.setattr(stage, "writable_path", lambda c, name: tmp_path / "outputs" / name)
    checkpoint = tmp_path / "data/model.hdf5"
    checkpoint.write_bytes(b"fixture")
    monkeypatch.setattr(stage, "verified_checkpoint", lambda c: checkpoint)
    monkeypatch.setattr(stage, "sha256", lambda p: "fixture_hash")
    result = stage.prepare_source_training({"data_root": tmp_path / "data"})
    assert result["scope"] == "FaultSeg3D_source_only"
    assert result["full_public_data_audit"] is False
    assert result["thebe"]["pairs"] == []
    preparation = json.loads((tmp_path / "outputs/preparation/preparation_report.json").read_text(encoding="utf-8"))
    assert preparation["thebe_preparation_completed"] is False
    assert preparation["full_training_started"] is False


def test_full_evaluation_forwards_fusion_to_predictor_and_separates_outputs(tmp_path, monkeypatch):
    import fault_workflow.evaluation as evaluation

    monkeypatch.setattr(evaluation, "require_audit", lambda c: {})
    monkeypatch.setattr(evaluation, "get_policy", lambda c: {"thresholds": [0.5]})
    monkeypatch.setattr(evaluation, "require_preparation", lambda c: None)
    monkeypatch.setattr(evaluation, "configure_tensorflow", lambda c, d: None)
    monkeypatch.setattr(evaluation, "require_gpu_smoke", lambda c, tf: None)
    monkeypatch.setattr(evaluation, "_evaluation_model", lambda c, tf, k: (None, {"kind": k}))
    monkeypatch.setattr(evaluation, "runtime_info", lambda tf, d: {})
    monkeypatch.setattr(evaluation, "writable_path", lambda c, p: tmp_path / p)
    monkeypatch.setattr(evaluation, "sha256", lambda p: "fixture")
    arrays = {"seismic": np.zeros((200, 1, 1)), "labels": np.ones((200, 1, 1), dtype=bool),
              "states": np.full((200, 1, 1), 3, dtype=np.uint8)}
    monkeypatch.setattr(evaluation, "cache_thebe_validation", lambda c, a, p: arrays)
    monkeypatch.setattr(evaluation, "_section_figure", lambda *args: None)
    calls = []

    def predict(raw, predictor, path, patch, stride, *, fusion):
        calls.append((path, fusion))
        return np.full(raw.shape, 0.8), {"fusion": fusion}

    monkeypatch.setattr(evaluation, "disk_sliding_predict", predict)
    config = {"output_root": tmp_path, "training": {"run_name": "fixture"},
              "evaluation_fusion": "separable", "patch_size": 128, "stride": 64}
    report = evaluation.run_evaluation(config, "thebe", "trained", server_work=True)
    assert report["fusion"] == report["tiling"]["fusion"] == "separable"
    assert calls[0][0].parts[-3:] == ("trained_separable", "thebe", "probability_raw.npy")
    assert report["aggregate"]["threshold_sweep"][0]["reference_positive_coverage"] == 1
