"""Compare complete validation reports without inventing real-data negatives."""
from __future__ import annotations

import argparse
import json

import numpy as np

from .baseline import get_policy, require_audit, writable_path
from .common import WORKSPACE, load_config, sha256, timestamp, write_json
from .diagnostics import proximity_diagnosis
from .initial_optimized import comparison_views
from .quality import score_region


def fixed_score(report, threshold=0.5):
    return next(row for row in report["aggregate"]["threshold_sweep"] if row["threshold"] == threshold)


def unknown_response(report, threshold=0.5):
    row = fixed_score(report, threshold)
    counts = report["aggregate"]
    if counts["confirmed_negative_voxels"] != 0 or counts["unknown_voxels"] == 0:
        raise ValueError("Expected original Thebe positive/unknown annotation semantics")
    # Recover the exact integer response count saved as count/support, not a new metric.
    total_response = round(row["response_fraction_in_amplitude_support"] * counts["support_voxels"])
    return (total_response - row["covered_reference_positive"]) / counts["unknown_voxels"]


def validate_comparison_reports(reports, policy):
    """Reject stale, mixed-model or incomplete reports before paired comparisons."""
    expected = {"author_source": ("author", "author", "synthetic"),
                "trained_source": ("trained", "author", "synthetic"),
                "author_thebe": ("author", "author", "thebe"),
                "trained_thebe": ("trained", "author", "thebe"),
                "trained_separable": ("trained", "separable", "thebe")}
    for name, (kind, fusion, dataset) in expected.items():
        report = reports[name]
        if (report["status"] != "passed" or report["test_used"] or report["split"] != "validation"
                or report["threshold_selected"] or report["dataset"] != dataset
                or report["policy"] != policy or report["model"]["kind"] != kind
                or report["fusion"] != fusion):
            raise ValueError(f"Incomplete or ineligible report: {name}")
        if dataset == "synthetic":
            if report["volumes"] != 20 or [c["id"] for c in report["cases"]] != list(range(20)):
                raise ValueError("All 20 original synthetic validation volumes are required")
        elif (report["sections"] != 200
              or [c["section_in_validation"] for c in report["cases"]] != list(range(200))
              or [c["global_array_section"] for c in report["cases"]] != list(range(900, 1100))):
            raise ValueError("All 200 original Thebe validation sections are required in order")
        source = reports[f"{kind}_source"]
        if report["model"]["sha256"] != source["model"]["sha256"]:
            raise ValueError(f"Paired evaluations use different checkpoints: {name}")


def section_comparison(path, seismic, labels, probabilities, section):
    """Same fixed section and display scale for all three full-volume predictions."""
    from .figures import plt

    nonzero = np.abs(seismic[seismic != 0])
    limit = max(float(np.quantile(nonzero, 0.99)) if nonzero.size else 1, 1e-6)
    fig, axes = plt.subplots(2, 3, figsize=(20, 12), constrained_layout=True)
    titles = ["地震振幅", "地震＋原参考标签", "作者模型：预测概率",
              "作者模型：阈值0.5", "复现第60轮：作者融合", "复现第60轮：乘积融合"]
    for ax, title in zip(axes.flat, titles, strict=True):
        ax.imshow(seismic.T, cmap="gray", vmin=-limit, vmax=limit, aspect="auto", interpolation="nearest")
        ax.set(title=title, xlabel="横向数组索引", ylabel="垂向采样索引")
    axes[0, 1].imshow(np.ma.masked_where(~labels.T, labels.T), cmap="autumn", vmin=0, vmax=1,
                      alpha=0.7, aspect="auto", interpolation="nearest")
    im = axes[0, 2].imshow(probabilities["author_thebe"].T, cmap="magma", vmin=0, vmax=1,
                            aspect="auto", interpolation="nearest")
    fig.colorbar(im, ax=axes[0, 2], shrink=0.8, label="断层概率")
    for ax, name in zip(axes[1], ("author_thebe", "trained_thebe", "trained_separable"), strict=True):
        plane = probabilities[name].T
        ax.imshow(np.ma.masked_where(plane < 0.5, plane), cmap="cool", vmin=0, vmax=1,
                  alpha=0.75, aspect="auto", interpolation="nearest")
        if labels.any() and not labels.all():
            ax.contour(labels.T, levels=[0.5], colors="lime", linewidths=0.45)
    fig.suptitle(f"Thebe验证剖面{section}（全局数组{section + 900}）：同坐标、同阈值对比\n"
                 "绿色：参考边界；紫色：预测响应；未标注区响应不能直接判为误检。")
    fig.savefig(path, dpi=140)
    plt.close(fig)


def run(config):
    require_audit(config)
    policy = get_policy(config)
    gate_path = WORKSPACE / "configs/acceptance_v2.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    base = writable_path(config, f"baseline/{config['training']['run_name']}")
    folder = writable_path(config, "review")
    folder.mkdir(parents=True, exist_ok=True)
    paths = {"author_source": base / "author/synthetic/evaluation_report.json",
             "trained_source": base / "trained/synthetic/evaluation_report.json",
             "author_thebe": base / "author/thebe/evaluation_report.json",
             "trained_thebe": base / "trained/thebe/evaluation_report.json",
             "trained_separable": base / "trained_separable/thebe/evaluation_report.json"}
    reports = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in paths.items()}
    validate_comparison_reports(reports, policy)
    source_a, source_b = (fixed_score(reports[k]) for k in ("author_source", "trained_source"))
    source_delta = source_b["dice"] - source_a["dice"]
    case_delta = [fixed_score({"aggregate": b})["dice"] - fixed_score({"aggregate": a})["dice"]
                  for a, b in zip(reports["author_source"]["cases"], reports["trained_source"]["cases"], strict=True)]
    source_ok = (source_delta >= gate["source_validation"]["aggregate_dice_delta_vs_author_minimum"]
                 and min(case_delta) >= gate["source_validation"]["individual_volume_dice_delta_vs_author_minimum"])
    cache = config["cache_root"] / "baseline_thebe_validation"
    marker = json.loads((cache / "ready.json").read_text(encoding="utf-8"))
    for kind, digest in marker["sha256"].items():
        if sha256(cache / f"{kind}_raw.npy") != digest:
            raise ValueError("Thebe validation cache changed before review")
    arrays = {name: np.load(cache / f"{name}_raw.npy", mmap_mode="r") for name in ("seismic", "labels", "states")}
    probabilities = {}
    for key, subdir in (("author_thebe", "author"), ("trained_thebe", "trained"), ("trained_separable", "trained_separable")):
        path = base / subdir / "thebe/probability_raw.npy"
        if sha256(path) != reports[key]["probability_sha256"]:
            raise ValueError("Completed probability changed before review")
        probabilities[key] = np.load(path, mmap_mode="r")
        if probabilities[key].shape != arrays["states"].shape:
            raise ValueError("Full-volume prediction and reference shapes differ")
    for section in gate["thebe_validation"]["fixed_review_sections"]:
        section_comparison(folder / f"section_{section:03}_paired.png", arrays["seismic"][section],
                           arrays["labels"][section], {k: p[section] for k, p in probabilities.items()}, section)
    protocol = json.loads((WORKSPACE / "configs/quality_review.json").read_text(encoding="utf-8"))
    core_results = []
    for region in protocol["thebe_regions"]:
        core = tuple(slice(s, s + n) for s, n in zip(region["start_raw"], region["shape_raw"], strict=True))
        scores = {name: score_region(arrays["states"][core], arrays["labels"][core], value[core], protocol)
                  for name, value in probabilities.items()}
        proximity = {name: proximity_diagnosis(arrays["states"][core], value[core]) for name, value in probabilities.items()}
        core_results.append({"region": region, "scores": scores, "proximity_diagnosis_only": proximity})
        comparison_views(folder / f"{region['name']}_author_vs_trained.png", arrays["seismic"][core],
                         arrays["labels"][core], probabilities["author_thebe"][core],
                         probabilities["trained_separable"][core], region,
                         original_title="作者模型：完整验证区推理", optimized_title="复现第60轮：完整验证区＋乘积融合")
    a = fixed_score(reports["author_thebe"])
    ua = unknown_response(reports["author_thebe"])
    results = {}
    for name in ("trained_thebe", "trained_separable"):
        b = fixed_score(reports[name])
        coverage_delta = b["reference_positive_coverage"] - a["reference_positive_coverage"]
        unknown_delta = unknown_response(reports[name]) - ua
        core_delta = {c["region"]["name"]: c["scores"][name]["at_fixed_threshold"]["reference_positive_coverage"]
                      - c["scores"]["author_thebe"]["at_fixed_threshold"]["reference_positive_coverage"] for c in core_results}
        criteria = gate["thebe_validation"]
        passed = (source_ok and coverage_delta >= criteria["aggregate_reference_positive_coverage_delta_vs_author_target"]
                  and unknown_delta <= criteria["unknown_response_fraction_increase_vs_author_maximum"]
                  and min(core_delta.values()) >= criteria["development_region_coverage_delta_minimum"])
        results[name] = {"at_threshold_0_5": b, "coverage_delta_vs_author": coverage_delta,
                         "unknown_response_fraction": unknown_response(reports[name]),
                         "unknown_response_delta_vs_author": unknown_delta, "core_coverage_deltas": core_delta,
                         "provisional_numeric_gate_passed": passed}
    result = {"status": "passed_review_execution_only", "finished": timestamp(),
              "recognition_quality_accepted": False, "user_visual_review_pending": True,
              "training": config["training"], "policy": policy, "gate": gate, "gate_sha256": sha256(gate_path),
              "report_sha256": {key: sha256(path) for key, path in paths.items()},
              "source": {"author": source_a, "trained": source_b, "dice_delta": source_delta,
                         "minimum_per_volume_delta": min(case_delta), "source_gate_passed": source_ok},
              "thebe_author": {"at_threshold_0_5": a, "unknown_response_fraction": ua},
              "candidates": results, "development_cores": core_results,
              "test_used": False, "interpretation": "Local development and full validation comparison; no independent TEST claim or field precision estimate."}
    write_json(folder / "run_review.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description="完整服务器结果配对自检；保留人工效果验收")
    parser.add_argument("--config", type=str, default="configs/baseline_balanced.json")
    args = parser.parse_args()
    print(json.dumps(run(load_config(WORKSPACE / args.config))["candidates"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
