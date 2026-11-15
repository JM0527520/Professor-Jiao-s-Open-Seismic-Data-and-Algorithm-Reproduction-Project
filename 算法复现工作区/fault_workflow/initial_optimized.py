"""Re-run the retained LOCAL inference candidate with source and result checks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from .baseline import require_audit, verified_checkpoint, writable_path
from .common import WORKSPACE, load_config, sha256, timestamp, write_json
from .figures import plt
from .inference import load_author_model, predict_context_core
from .quality import core_slices, extract_contexts, score_region


def comparison_views(path, seismic, labels, original, optimized, region, *,
                     original_title="优化前", optimized_title="优化后：真实上下文32＋乘积融合"):
    """Fixed centre slices in all three raw axes; never select the prettiest slice."""
    nonzero = np.abs(seismic[seismic != 0])
    limit = max(float(np.quantile(nonzero, 0.99)) if nonzero.size else 1, 1e-6)
    fig, axes = plt.subplots(3, 4, figsize=(16, 13), constrained_layout=True)
    for axis, row in enumerate(axes):
        index = seismic.shape[axis] // 2
        image = np.take(seismic, index, axis=axis).T
        reference = np.take(labels, index, axis=axis).T
        for ax in row:
            ax.imshow(image, cmap="gray", vmin=-limit, vmax=limit, aspect="auto")
            ax.set(xlabel="局部索引", ylabel="局部索引")
        row[0].set_title(f"原始轴{axis}中切片；索引{index}")
        row[1].imshow(np.ma.masked_where(~reference, reference), cmap="autumn",
                       vmin=0, vmax=1, alpha=0.65, aspect="auto")
        row[1].set_title("原始参考标签")
        for ax, volume, title in ((row[2], original, original_title), (row[3], optimized, optimized_title)):
            plane = np.take(volume, index, axis=axis).T
            ax.imshow(np.ma.masked_where(plane < 0.5, plane), cmap="cool", vmin=0, vmax=1,
                      alpha=0.75, aspect="auto")
            if reference.any() and not reference.all():
                ax.contour(reference, levels=[0.5], colors="lime", linewidths=0.5)
            ax.set_title(title)
    fig.suptitle(f"{region['name']}：三向同坐标对比；阈值固定0.5\n"
                 "绿色：参考边界；紫色：预测响应。未知区响应无法直接判断误检；局部效果尚未验收。")
    fig.savefig(path, dpi=130)
    plt.close(fig)


def run(config, selection_path):
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if (selection["variant"] != "halo_separable" or selection["halo_raw"] != [32, 32, 32]
            or selection["threshold"] != 0.5 or selection["checkpoint"] != "fseg-60.hdf5"
            or selection["fusion"] != "separable" or selection["horizontal_ensemble"]):
        raise ValueError("Retained initial candidate must match the reviewed configuration")
    protocol_path = (WORKSPACE / selection["review_protocol"]).resolve()
    if not protocol_path.is_relative_to(WORKSPACE):
        raise ValueError("Review protocol must stay inside the workspace")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    audit = require_audit(config)
    parent = writable_path(config, "quality_review/2026-11-14")
    inputs = {"paired": parent / "quality_report.json",
              "horizontal4": parent / "followup_report.json",
              "wide64": parent / "wide_context/wide_report.json"}
    reports = {key: json.loads(path.read_text(encoding="utf-8")) for key, path in inputs.items()}
    checkpoint_hash = sha256(verified_checkpoint(config))
    for value in reports.values():
        if (value["status"] != "passed" or value["test_used"] or value["training_performed"]
                or value["full_baseline_run"] or value["runtime"]["sha256"] != checkpoint_hash):
            raise ValueError("Finish the paired local comparison with the same unchanged checkpoint")
    if reports["paired"]["protocol"] != protocol:
        raise ValueError("Reviewed protocol changed")
    if reports["horizontal4"]["base_report_sha256"] != sha256(inputs["paired"]):
        raise ValueError("Followup report belongs to a different paired review")
    extract_contexts(config, audit, protocol, parent)
    folder = writable_path(config, "quality_review/2026-11-14/selected")
    folder.mkdir(parents=True, exist_ok=True)
    result = {"status": "running", "started": timestamp(), "selection": selection,
              "checkpoint_sha256": checkpoint_hash, "training_performed": False, "test_used": False,
              "full_baseline_run": False, "recognition_quality_accepted": False,
              "reference_predict_default_changed": False,
              "comparison_report_sha256": {key: sha256(path) for key, path in inputs.items()},
              "context_manifest_sha256": sha256(parent / "context_manifest.json"),
              "code_sha256": {p.name: sha256(p) for p in (WORKSPACE / "fault_workflow").glob("*.py")},
              "config_sha256": {str(p.relative_to(WORKSPACE)): sha256(p) for p in (selection_path, protocol_path)},
              "cases": []}
    write_json(folder / "selected_report.json", result)
    try:
        predictor, info = load_author_model(config)
        result["runtime"] = info
        for region in protocol["thebe_regions"]:
            name = region["name"]
            paired = next(c for c in reports["paired"]["thebe"] if c["region"] == region)
            source = np.load(parent / f"{name}_seismic_raw.npy", mmap_mode="r")
            core = core_slices(region["shape_raw"], selection["halo_raw"])
            labels = np.load(parent / f"{name}_labels_raw.npy", mmap_mode="r")[core]
            states = np.load(parent / f"{name}_states_raw.npy", mmap_mode="r")[core]
            comparisons = dict(paired["variants"])
            for key in ("horizontal4", "wide64"):
                comparisons[key] = next(c for c in reports[key]["cases"] if c["region"] == region)
            # Verify the reference and retained candidate before producing comparisons.
            for variant in ("legacy_core", selection["variant"]):
                path = parent / f"{name}_{variant}_probability_raw.npy"
                if sha256(path) != paired["variants"][variant]["probability_sha256"]:
                    raise ValueError("Reviewed probability changed")
            original = np.load(parent / f"{name}_legacy_core_probability_raw.npy", mmap_mode="r")
            reviewed = np.load(parent / f"{name}_{selection['variant']}_probability_raw.npy", mmap_mode="r")
            started = time.perf_counter()
            probability, tiling = predict_context_core(source, predictor, selection["halo_raw"], region["shape_raw"],
                                                       protocol["patch_size"], protocol["stride"], fusion=selection["fusion"])
            elapsed = time.perf_counter() - started
            delta = float(np.max(np.abs(reviewed - probability)))
            if delta > 1e-6:
                raise ValueError("Retained inference candidate failed the local repeat check")
            output = folder / f"{name}_probability_raw.npy"
            np.save(output, probability)
            score = score_region(states, labels, probability, protocol)
            candidates = {key: {"at_fixed_threshold": value["at_fixed_threshold"],
                                "unknown_response_fraction": value["response_outside_supplied_reference"] / value["unknown_voxels"],
                                "tiles": value["tiling"]["tiles"]}
                          for key, value in comparisons.items()}
            score.update(region=region, tiling=tiling, elapsed_seconds=round(elapsed, 3),
                         repeat_max_abs_difference=delta, probability_sha256=sha256(output),
                         comparisons=candidates)
            result["cases"].append(score)
            comparison_views(folder / f"{name}_before_after.png", source[core], labels, original, probability, region)
            write_json(folder / "selected_report.json", result)
        result.update(status="passed", interpretation="Repeat and integrity checks passed; local recognition quality remains unaccepted.",
                      retained_reason="halo32 separable increases supplied-positive coverage in all three cores; retain as local candidate only",
                      rejected={"horizontal4": "historical coverage declines below original; 4x network calls",
                                "wide64": "A and B regress relative to halo32; 3x or more tiles"})
    except Exception as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        result["finished"] = timestamp()
        write_json(folder / "selected_report.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description="复跑初版局部优化候选；不训练、不进行全区评价")
    parser.add_argument("--config", type=Path, default=WORKSPACE / "configs/baseline.json")
    parser.add_argument("--selection", type=Path, default=WORKSPACE / "configs/initial_optimized.json")
    args = parser.parse_args()
    run(load_config(args.config), args.selection.resolve())


if __name__ == "__main__":
    main()
