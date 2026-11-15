"""Small, paired effect diagnosis. No training, checkpoint search, or TEST access."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import h5py
import numpy as np

from .baseline import require_audit, verified_checkpoint, writable_path
from .common import WORKSPACE, load_config, raw_to_model, relative_path, sha256, standardize, timestamp, write_json
from .data import plane_stream, read_synthetic
from .figures import plt
from .inference import load_author_model, reference_overlap, sliding_predict, validate_probability
from .policy import INVALID, POSITIVE, UNKNOWN, ReferenceAccumulator, annotation_states


def core_slices(core_shape, halo):
    if len(core_shape) != 3 or len(halo) != 3 or any(h < 0 for h in halo):
        raise ValueError("Expected three nonnegative halo dimensions")
    return tuple(slice(h, h + n) for h, n in zip(halo, core_shape, strict=True))


def extract_contexts(config, audit, protocol, folder):
    """Stream original validation planes once; envelope BEFORE cropping, no TEST."""
    halo = protocol["halo_raw"]
    marker = folder / "context_manifest.json"
    sources = [s for s in audit["source_fingerprints"] if any(
        s["relative_path"] in (p["seismic"], p["labels"]) for p in audit["thebe"]["pairs"] if p["split"] == "val")]
    signature = {"protocol": protocol, "sources": sources, "data_root": str(config["data_root"])}
    if marker.exists():
        saved = json.loads(marker.read_text(encoding="utf-8"))
        if saved["signature"] != signature:
            raise ValueError("Quality context protocol changed; use a new output directory")
        if any(sha256(folder / n) != h for n, h in saved["files"].items()):
            raise ValueError("Quality context cache changed; cannot reuse")
        return
    buffers = {}
    for region in protocol["thebe_regions"]:
        core_start = np.array(region["start_raw"])
        core_shape = np.array(region["shape_raw"])
        bounds = np.array([200, 3174, 1537])
        if (core_start.shape != (3,) or core_shape.shape != (3,)
                or np.any(core_start < 0) or np.any(core_shape <= 0)
                or np.any(core_start + core_shape > bounds)):
            raise ValueError("Core must stay within original Thebe VALIDATION")
        start = core_start - halo
        shape = core_shape + 2 * np.array(halo)
        stop = start + shape
        if protocol.get("clip_context_to_validation", False):
            start = np.maximum(start, 0)
            stop = np.minimum(stop, [200, 3174, 1537])
            shape = stop - start
        if np.any(start < 0) or np.any(stop > np.array([200, 3174, 1537])):
            raise ValueError("Context must stay within original Thebe VALIDATION")
        buffers[region["name"]] = {"start": start, "stop": stop,
                                    "seismic": np.zeros(shape, dtype=np.float32),
                                    "labels": np.zeros(shape, dtype=bool),
                                    "states": np.zeros(shape, dtype=np.uint8),
                                    "filled": np.zeros(shape[0], dtype=bool)}
    for pair in audit["thebe"]["pairs"]:
        if pair["split"] != "val":
            continue
        with plane_stream(config["data_root"] / relative_path(pair["seismic"])) as seismic_planes, \
                plane_stream(config["data_root"] / relative_path(pair["labels"])) as label_planes:
            for local, (seismic, labels) in enumerate(zip(seismic_planes, label_planes, strict=True)):
                section = pair["section_start"] + local
                states = annotation_states(seismic[None], labels[None])[0]
                for buffer in buffers.values():
                    start, stop = buffer["start"], buffer["stop"]
                    if start[0] <= section < stop[0]:
                        destination = section - start[0]
                        roi = np.s_[start[1]:stop[1], start[2]:stop[2]]
                        for name, array in (("seismic", seismic), ("labels", labels), ("states", states)):
                            buffer[name][destination] = array[roi]
                        buffer["filled"][destination] = True
        print(f"Full-trace quality contexts: validation [{pair['section_start']},{pair['section_stop']})", flush=True)
    files = {}
    for name, buffer in buffers.items():
        if not buffer["filled"].all():
            raise ValueError("Incomplete context reconstruction")
        for kind in ("seismic", "labels", "states"):
            path = folder / f"{name}_{kind}_raw.npy"
            np.save(path, buffer[kind])
            files[path.name] = sha256(path)
    write_json(marker, {"signature": signature, "files": files,
                        "contexts": {name: {"start_raw": b["start"].tolist(), "stop_raw": b["stop"].tolist()}
                                     for name, b in buffers.items()}})


def edge_mask(shape, band):
    mask = np.zeros(shape, dtype=bool)
    for axis, length in enumerate(shape):
        first, last = [slice(None)] * 3, [slice(None)] * 3
        first[axis], last[axis] = slice(0, band), slice(max(0, length - band), length)
        mask[tuple(first)] = mask[tuple(last)] = True
    return mask


def score_region(states, labels, probability, protocol):
    validate_probability(probability, labels.shape)
    accumulator = ReferenceAccumulator(protocol["diagnostic_thresholds"])
    accumulator.update(states, labels, probability)
    result = accumulator.report()
    predicted = probability >= protocol["threshold"]
    edge = edge_mask(labels.shape, protocol["boundary_band_samples"])
    result["at_fixed_threshold"] = next(r for r in result["threshold_sweep"] if r["threshold"] == protocol["threshold"])
    result["edge_and_interior"] = {}
    for name, roi in (("edge_band", edge), ("interior", ~edge)):
        pos = (states == POSITIVE) & roi
        unknown = (states == UNKNOWN) & roi
        result["edge_and_interior"][name] = {
            "reference_positive": int(pos.sum()),
            "positive_coverage": float(predicted[pos].mean()) if pos.any() else None,
            "unknown_response": float(predicted[unknown].mean()) if unknown.any() else None}
    result["response_outside_supplied_reference"] = int((predicted & (states == UNKNOWN)).sum())
    result["reference_positive_fraction_in_support"] = float((states == POSITIVE).sum() / (states != INVALID).sum()) if (states != INVALID).any() else None
    return result


def _comparison_figure(path, seismic_raw, labels_raw, predictions, region):
    # Identical source coordinates, scales, and threshold; no smoothing/dilation.
    section = seismic_raw.shape[0] // 2
    image, labels = seismic_raw[section].T, labels_raw[section].T
    nonzero = np.abs(image[image != 0])
    limit = max(float(np.quantile(nonzero, 0.99)) if nonzero.size else 1, 1e-6)
    fig, axes = plt.subplots(2, 4, figsize=(18, 10), constrained_layout=True)
    for ax in axes.flat:
        ax.imshow(image, cmap="gray", vmin=-limit, vmax=limit, aspect="auto")
        ax.set(xlabel="另一横向轴（局部索引）", ylabel="垂向采样（局部索引）")
    axes[0, 0].set_title("地震振幅（同一剖面）")
    axes[1, 0].imshow(np.ma.masked_where(~labels, labels), cmap="autumn", vmin=0, vmax=1, alpha=0.6, aspect="auto")
    axes[1, 0].set_title("参考标签（原样，未增厚预测）")
    for column, (name, probability) in enumerate(predictions.items(), 1):
        plane = probability[section].T
        axes[0, column].imshow(plane, cmap="magma", vmin=0, vmax=1, aspect="auto")
        axes[0, column].set_title(name + "：概率")
        axes[1, column].imshow(np.ma.masked_where(plane < 0.5, plane), cmap="cool", vmin=0, vmax=1, alpha=0.75, aspect="auto")
        if labels.any() and not labels.all():
            axes[1, column].contour(labels, levels=[0.5], colors="lime", linewidths=0.5)
        axes[1, column].set_title(name + "：固定阈值0.5＋参考绿线")
    fig.suptitle(f"{region['name']}：同一核心区域；全局数组剖面{900 + region['start_raw'][0] + section}\n局部开发诊断，非独立测试；未知区响应不能直接认定为误检")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_quality_review(config, protocol_path=None):
    protocol_path = protocol_path or WORKSPACE / "configs/quality_review.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["checkpoint"] != "fseg-60.hdf5" or protocol["threshold"] != 0.5:
        raise ValueError("Quality review fixes original checkpoint and threshold")
    if protocol["variants"] != ["legacy_core", "halo_author", "halo_separable"]:
        raise ValueError("Declare the paired inference variants before running")
    audit = require_audit(config)
    folder = writable_path(config, "quality_review/2026-11-14")
    folder.mkdir(parents=True, exist_ok=True)
    write_json(folder / "protocol.json", protocol)
    report = {"status": "running", "started": timestamp(), "protocol": protocol,
              "test_used": False, "training_performed": False, "full_baseline_run": False,
              "checkpoint_sha256": sha256(verified_checkpoint(config)),
              "synthetic": [], "thebe": [], "candidate_promoted": False}
    write_json(folder / "quality_report.json", report)
    try:
        with h5py.File(verified_checkpoint(config)) as archive:
            report["checkpoint_training_config"] = json.loads(archive.attrs["training_config"])
        extract_contexts(config, audit, protocol, folder)
        predictor, info = load_author_model(config)
        report["runtime"] = info
        for index in protocol["synthetic_validation_ids"]:
            raw = read_synthetic(config["data_root"], "validation", "seis", index)
            labels = raw_to_model(read_synthetic(config["data_root"], "validation", "fault", index)).astype(bool)
            # Independent expression copied from the author notebook, raw before transpose.
            normalized = np.transpose((raw - np.mean(raw)) / np.std(raw)).copy()
            original = predictor(normalized)
            current, _ = standardize(raw_to_model(raw))
            wrapper = predictor(current)
            difference = float(np.max(np.abs(original - wrapper)))
            stats = reference_overlap(labels, original, np.ones_like(labels), 0.5)
            report["synthetic"].append({"id": index, "author_notebook_vs_wrapper_max_abs_difference": difference,
                                         "reference_overlap": stats})
            print(f"Synthetic diagnostic {index}: Dice={stats['dice_vs_supplied_reference']:.4f}; notebook delta={difference:.2g}", flush=True)
        halo = protocol["halo_raw"]
        for region in protocol["thebe_regions"]:
            name = region["name"]
            parent = np.load(folder / f"{name}_seismic_raw.npy", mmap_mode="r")
            parent_labels = np.load(folder / f"{name}_labels_raw.npy", mmap_mode="r")
            parent_states = np.load(folder / f"{name}_states_raw.npy", mmap_mode="r")
            core = core_slices(region["shape_raw"], halo)
            seismic, labels, states = parent[core], parent_labels[core], parent_states[core]
            case = {"region": region, "variants": {}}
            predictions = {}
            for variant in protocol["variants"]:
                begin = time.perf_counter()
                source = seismic if variant == "legacy_core" else parent
                probability, tiling = sliding_predict(raw_to_model(source), predictor, 128, 64,
                                                      fusion="separable" if variant == "halo_separable" else "author")
                probability_raw = probability.transpose(2, 1, 0)
                if variant != "legacy_core":
                    probability_raw = probability_raw[core]
                probability_raw = np.ascontiguousarray(probability_raw)
                path = folder / f"{name}_{variant}_probability_raw.npy"
                np.save(path, probability_raw)
                predictions[variant] = probability_raw
                result = score_region(states, labels, probability_raw, protocol)
                result.update(probability_sha256=sha256(path), tiling=tiling,
                              elapsed_seconds=round(time.perf_counter() - begin, 3))
                case["variants"][variant] = result
                print(f"{name} {variant}: {result['at_fixed_threshold']}", flush=True)
            if name == "historical_demo":
                historical_path = config["output_root"] / "predictions/thebe_validation_demo_probability_raw.npy"
                if historical_path.exists():
                    previous = np.load(historical_path, allow_pickle=False)
                    case["historical_reproduction_max_abs_difference"] = float(np.max(np.abs(previous - predictions["legacy_core"])))
            _comparison_figure(folder / f"{name}_comparison.png", seismic, labels, predictions, region)
            report["thebe"].append(case)
            write_json(folder / "quality_report.json", report)
        report.update(status="passed", interpretation="execution passed only; inspect paired effects before accepting a variant")
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["finished"] = timestamp()
        write_json(folder / "quality_report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description="初版断层识别小规模配对诊断，不训练、不读取TEST")
    parser.add_argument("--config", type=Path, default=WORKSPACE / "configs/baseline.json")
    args = parser.parse_args()
    run_quality_review(load_config(args.config))


if __name__ == "__main__":
    main()
