"""Review saved predictions only; never train, infer, or select a threshold."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from fault_workflow.baseline import get_policy, require_audit
from fault_workflow.common import WORKSPACE, load_config, sha256, timestamp, write_json
from fault_workflow.figures import plt
from fault_workflow.initial_optimized import comparison_views
from fault_workflow.quality import score_region
from fault_workflow.run_review import fixed_score, unknown_response, validate_comparison_reports


def run():
    config = load_config(WORKSPACE / "configs/baseline_balanced.json")
    require_audit(config)
    base = config["output_root"] / "baseline" / config["training"]["run_name"]
    folder = config["output_root"] / "review"
    paths = {name: base / sub / dataset / "evaluation_report.json" for name, sub, dataset in (
        ("author_source", "author", "synthetic"), ("trained_source", "trained", "synthetic"),
        ("author_thebe", "author", "thebe"), ("trained_thebe", "trained", "thebe"),
        ("trained_separable", "trained_separable", "thebe"),
        ("author_separable", "author_separable", "thebe"))}
    reports = {k: json.loads(p.read_text(encoding="utf-8")) for k, p in paths.items()}
    policy = get_policy(config)
    validate_comparison_reports(reports, policy)
    followup = reports["author_separable"]
    if (followup["status"] != "passed" or followup["test_used"] or followup["threshold_selected"]
            or followup["split"] != "validation" or followup["dataset"] != "thebe"
            or followup["model"]["kind"] != "author" or followup["fusion"] != "separable"
            or followup["model"]["sha256"] != reports["author_thebe"]["model"]["sha256"]
            or followup["policy"] != policy or followup["sections"] != 200
            or [c["section_in_validation"] for c in followup["cases"]] != list(range(200))
            or [c["global_array_section"] for c in followup["cases"]] != list(range(900, 1100))):
        raise ValueError("Followup must cover all validation sections with unchanged author weights and policy")
    previous_path = folder / "run_review.json"
    previous = json.loads(previous_path.read_text(encoding="utf-8"))
    for k, digest in previous["report_sha256"].items():
        if sha256(paths[k]) != digest:
            raise ValueError("Historical review inputs changed")
    gate = json.loads((WORKSPACE / "configs/acceptance_v2.json").read_text(encoding="utf-8"))
    if sha256(WORKSPACE / "configs/acceptance_v2.json") != previous["gate_sha256"]:
        raise ValueError("Acceptance criteria changed")
    cache = config["cache_root"] / "baseline_thebe_validation"
    marker = json.loads((cache / "ready.json").read_text(encoding="utf-8"))
    for kind, digest in marker["sha256"].items():
        if sha256(cache / f"{kind}_raw.npy") != digest:
            raise ValueError("Validation cache changed")
    arrays = {k: np.load(cache / f"{k}_raw.npy", mmap_mode="r") for k in ("seismic", "labels", "states")}
    probs = {}
    for k in ("author_thebe", "author_separable"):
        path = paths[k].parent / "probability_raw.npy"
        if sha256(path) != reports[k]["probability_sha256"]:
            raise ValueError("Saved probability changed")
        probs[k] = np.load(path, mmap_mode="r")
        if probs[k].shape != arrays["states"].shape:
            raise ValueError("Prediction shape mismatch")
    protocol = json.loads((WORKSPACE / "configs/quality_review.json").read_text(encoding="utf-8"))
    cores = []
    for region in protocol["thebe_regions"]:
        roi = tuple(slice(s, s + n) for s, n in zip(region["start_raw"], region["shape_raw"], strict=True))
        scores = {k: score_region(arrays["states"][roi], arrays["labels"][roi], p[roi], protocol) for k, p in probs.items()}
        delta = scores["author_separable"]["at_fixed_threshold"]["reference_positive_coverage"] - scores["author_thebe"]["at_fixed_threshold"]["reference_positive_coverage"]
        cores.append({"region": region, "scores": scores, "coverage_delta": delta})
        comparison_views(folder / f"{region['name']}_author_fusion_followup.png", arrays["seismic"][roi], arrays["labels"][roi], probs["author_thebe"][roi], probs["author_separable"][roi], region,
                         original_title="作者模型：作者融合", optimized_title="作者模型：乘积融合")
    for section in gate["thebe_validation"]["fixed_review_sections"]:
        seismic, labels = arrays["seismic"][section].T, arrays["labels"][section].T
        limit = max(float(np.quantile(np.abs(seismic[seismic != 0]), .99)), 1e-6)
        fig, axes = plt.subplots(2, 3, figsize=(20, 12), constrained_layout=True)
        for row, (k, p) in zip(axes, probs.items(), strict=True):
            for ax in row:
                ax.imshow(seismic, cmap="gray", vmin=-limit, vmax=limit, aspect="auto", interpolation="nearest")
            row[0].imshow(np.ma.masked_where(~labels, labels), cmap="autumn", alpha=.65, aspect="auto")
            row[0].set_title("原始地震＋参考标签")
            row[1].imshow(p[section].T, cmap="magma", vmin=0, vmax=1, aspect="auto")
            row[1].set_title(k + " 概率")
            row[2].imshow(np.ma.masked_where(p[section].T < .5, p[section].T), cmap="cool", vmin=0, vmax=1, alpha=.75, aspect="auto")
            row[2].contour(labels, levels=[.5], colors="lime", linewidths=.45)
            row[2].set_title(k + " 阈值0.5＋参考绿线")
        fig.suptitle(f"Thebe验证剖面{section}（全局数组{section + 900}）：仅改变融合；未知响应不等于误检")
        fig.savefig(folder / f"section_{section:03}_author_fusion_followup.png", dpi=140)
        plt.close(fig)
    a, b = (fixed_score(reports[k]) for k in ("author_thebe", "author_separable"))
    delta = b["reference_positive_coverage"] - a["reference_positive_coverage"]
    unknown_delta = unknown_response(reports["author_separable"]) - unknown_response(reports["author_thebe"])
    criteria = gate["thebe_validation"]
    passed = (delta >= criteria["aggregate_reference_positive_coverage_delta_vs_author_target"]
              and unknown_delta <= criteria["unknown_response_fraction_increase_vs_author_maximum"]
              and min(c["coverage_delta"] for c in cores) >= criteria["development_region_coverage_delta_minimum"])
    result = {"status": "completed_review_only", "finished": timestamp(), "test_used": False,
              "recognition_quality_accepted": False, "user_visual_review_pending": True,
              "project_paused_by_user": True, "new_training_or_inference_started_by_review": False,
              "report_sha256": {k: sha256(p) for k, p in paths.items()},
              "previous_review_sha256": sha256(previous_path), "gate_sha256": previous["gate_sha256"],
              "thebe_2x2": {k: {"coverage": fixed_score(reports[k])["reference_positive_coverage"],
                                   "unknown_response": unknown_response(reports[k])} for k in
                             ("author_thebe", "author_separable", "trained_thebe", "trained_separable")},
              "author_fusion_followup": {"coverage_delta": delta, "unknown_response_delta": unknown_delta,
                                         "provisional_numeric_gate_passed": passed}, "development_cores": cores}
    write_json(folder / "fusion_followup_review.json", result)
    print(json.dumps({k: v for k, v in result.items() if k in ("thebe_2x2", "author_fusion_followup")}, indent=2))


if __name__ == "__main__":
    run()
