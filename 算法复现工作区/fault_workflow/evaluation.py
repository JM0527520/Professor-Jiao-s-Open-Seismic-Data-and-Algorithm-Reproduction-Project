"""Server evaluation, with disk-backed full-volume prediction and explicit unknowns."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .baseline import (configure_tensorflow, get_policy, require_audit, require_gpu_smoke, require_preparation, runtime_info,
                       training_settings, verified_checkpoint, writable_path)
from .common import WORKSPACE, model_to_raw, raw_to_model, relative_path, sha256, standardize, timestamp, write_json
from .data import plane_stream, read_synthetic
from .inference import author_overlap_mask, tile_starts, validate_probability
from .policy import NEGATIVE, POSITIVE, ReferenceAccumulator, annotation_states


def save_curve(path, report):
    rows = report["threshold_sweep"]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def disk_sliding_predict(seismic_raw, predictor, output_path, patch=128, stride=64):
    """Same author overlap and model-axis tile order as the local in-memory path.

    Accumulators reside in NPY memmaps, not whole-survey RAM arrays. Scratch files
    are retained with explicit names for inspection; they can be removed after a
    completed report without touching source data.
    """
    if seismic_raw.ndim != 3 or min(seismic_raw.shape) < patch:
        raise ValueError("Disk-backed prediction requires raw-order axes >= patch size")
    shape = seismic_raw.shape
    output_path = Path(output_path)
    values = np.lib.format.open_memmap(output_path.with_name("scratch_sum_raw.npy"), mode="w+", dtype="float32", shape=shape)
    weights = np.lib.format.open_memmap(output_path.with_name("scratch_weight_raw.npy"), mode="w+", dtype="float32", shape=shape)
    for section in range(shape[0]):
        values[section] = 0
        weights[section] = 0
    mask = model_to_raw(author_overlap_mask(patch, patch - stride))
    count = skipped = 0
    for sample in tile_starts(shape[2], patch, stride):
        for lateral in tile_starts(shape[1], patch, stride):
            for section in tile_starts(shape[0], patch, stride):
                roi = np.s_[section:section + patch, lateral:lateral + patch, sample:sample + patch]
                normalized, stats = standardize(raw_to_model(seismic_raw[roi]))
                if stats["constant"]:
                    probability = np.zeros_like(normalized)
                    skipped += 1
                else:
                    probability = np.asarray(predictor(normalized), dtype=np.float32)
                validate_probability(probability, (patch,) * 3)
                values[roi] += model_to_raw(probability) * mask
                weights[roi] += mask
                count += 1
                if count % 50 == 0:
                    print(f"Thebe full-volume prediction: {count} tiles", flush=True)
    values.flush()
    weights.flush()
    result = np.lib.format.open_memmap(output_path, mode="w+", dtype="float32", shape=shape)
    for section in range(shape[0]):
        if not np.isfinite(weights[section]).all() or np.any(weights[section] <= 0):
            raise ValueError("Disk-backed tiling left uncovered samples")
        result[section] = values[section] / weights[section]
        validate_probability(result[section], shape[1:])
    result.flush()
    return result, {"tiles": count, "constant_tiles_skipped": skipped, "shape_raw": list(shape),
                    "patch": patch, "stride": stride, "grid": "model-axis order; end-anchored final tiles",
                    "scratch_files": ["scratch_sum_raw.npy", "scratch_weight_raw.npy"]}


def cache_thebe_validation(config, audit, policy):
    folder = (config["cache_root"] / "baseline_thebe_validation").resolve()
    if not folder.is_relative_to(WORKSPACE) or folder.is_relative_to(config["data_root"]):
        raise ValueError("Thebe cache must stay inside the algorithm workspace")
    folder.mkdir(parents=True, exist_ok=True)
    marker = folder / "ready.json"
    pairs = [p for p in audit["thebe"]["pairs"] if p["split"] == "val"]
    if not pairs:
        raise ValueError("No original Thebe validation pairs in acceptance report")
    shape = (sum(p["shape"][0] for p in pairs), *pairs[0]["shape"][1:])
    names = {p[k] for p in pairs for k in ("seismic", "labels")}
    protocol = {"policy": policy, "shape_raw": list(shape),
                "sources": [p for p in audit["source_fingerprints"] if p["relative_path"] in names]}
    expected = {"seismic": (np.dtype("float32"), shape), "labels": (np.dtype("bool"), shape),
                "states": (np.dtype("uint8"), shape)}
    if marker.exists():
        accepted = json.loads(marker.read_text(encoding="utf-8"))
        if accepted["protocol"] != protocol:
            raise ValueError("Existing Thebe cache protocol changed; use a fresh cache_root")
        if any(sha256(folder / f"{k}_raw.npy") != accepted["sha256"][k] for k in expected):
            raise ValueError("Thebe cache checksum changed; rebuild in a fresh cache_root")
        arrays = {k: np.load(folder / f"{k}_raw.npy", mmap_mode="r", allow_pickle=False) for k in expected}
        if any(a.dtype != expected[k][0] or a.shape != expected[k][1] for k, a in arrays.items()):
            raise ValueError("Thebe cache shape/dtype differs from its completion marker")
        return arrays
    arrays = {k: np.lib.format.open_memmap(folder / f"{k}_raw.npy", mode="w+", dtype=dtype, shape=dimensions)
              for k, (dtype, dimensions) in expected.items()}
    filled = np.zeros(shape[0], dtype=bool)
    for pair in pairs:
        with plane_stream(config["data_root"] / relative_path(pair["seismic"])) as seismics, \
                plane_stream(config["data_root"] / relative_path(pair["labels"])) as references:
            for local, (seismic, labels) in enumerate(zip(seismics, references, strict=True)):
                index = pair["section_start"] + local
                states = annotation_states(seismic[None], labels[None])[0]
                arrays["seismic"][index] = seismic
                arrays["labels"][index] = labels
                arrays["states"][index] = states
                filled[index] = True
        print(f"Full-trace Thebe validation cache: {pair['section_stop']}/200 sections", flush=True)
    if not filled.all():
        raise ValueError("Incomplete Thebe validation cache")
    for array in arrays.values():
        array.flush()
    write_json(marker, {"protocol": protocol,
                        "sha256": {k: sha256(folder / f"{k}_raw.npy") for k in expected}})
    return arrays


def _evaluation_model(config, tf, model_kind):
    if model_kind == "author":
        path = verified_checkpoint(config)
    elif model_kind == "trained":
        settings = training_settings(config)
        run = writable_path(config, f"training/{settings['run_name']}")
        training_report = json.loads((run / "training_report.json").read_text(encoding="utf-8"))
        if training_report["status"] != "passed" or training_report["completed_epoch"] != 100:
            raise ValueError("Finish the 100-epoch protocol before the trained baseline evaluation")
        path = run / "epoch-060.keras"
        record = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        if sha256(path) != record["checkpoint_sha256"]:
            raise ValueError("Trained baseline checkpoint checksum mismatch")
        protocol = json.loads((run / "protocol.json").read_text(encoding="utf-8"))
        if protocol["training"] != settings or protocol["policy"] != get_policy(config):
            raise ValueError("Trained baseline protocol differs from the current evaluation configuration")
    else:
        raise ValueError("Choose the author checkpoint or trained epoch-060 checkpoint")
    model = tf.keras.models.load_model(path, compile=False)
    if model.count_params() != 1459585:
        raise ValueError("Unexpected baseline model")

    def predictor(array):
        return model([array[None, ..., None]], training=False).numpy()[0, ..., 0]

    return predictor, {"kind": model_kind, "checkpoint": str(path), "sha256": sha256(path)}


def run_evaluation(config, dataset, model_kind="author", server_work=False):
    if not server_work:
        raise ValueError("Full evaluation is deferred; use --server-work only on the prepared server")
    if dataset not in ("synthetic", "thebe"):
        raise ValueError("Only original VALIDATION may be evaluated; TEST remains locked")
    audit, policy = require_audit(config), get_policy(config)
    require_preparation(config)
    tf = configure_tensorflow(config, "gpu")
    require_gpu_smoke(config, tf)
    predictor, model_info = _evaluation_model(config, tf, model_kind)
    out = writable_path(config, f"baseline/{config['training']['run_name']}/{model_kind}/{dataset}")
    out.mkdir(parents=True, exist_ok=True)
    if (out / "evaluation_report.json").exists():
        raise ValueError("Evaluation already has a report; preserve/archive its directory before a retry")
    report = {"status": "running", "started": timestamp(), "dataset": dataset, "split": "validation",
              "model": model_info, "runtime": runtime_info(tf, "gpu"), "policy": policy,
              "acceptance_report_sha256": sha256(config["output_root"] / "audit/audit_report.json"),
              "test_used": False, "threshold_selected": False}
    write_json(out / "evaluation_report.json", report)
    try:
        accumulator = ReferenceAccumulator(policy["thresholds"])
        if dataset == "synthetic":
            cases = []
            for index in range(20):
                # Match author utils.py normalization before transpose for these full cubes.
                raw, _ = standardize(read_synthetic(config["data_root"], "validation", "seis", index))
                labels = raw_to_model(read_synthetic(config["data_root"], "validation", "fault", index)).astype(bool)
                probability = predictor(raw_to_model(raw))
                validate_probability(probability, labels.shape)
                np.save(out / f"validation_{index:03}_probability_model.npy", probability)
                states = np.where(labels, POSITIVE, NEGATIVE).astype(np.uint8)
                one = ReferenceAccumulator(policy["thresholds"])
                one.update(states, labels, probability)
                accumulator.update(states, labels, probability)
                cases.append({"id": index,
                              "probability_sha256": sha256(out / f"validation_{index:03}_probability_model.npy"),
                              **one.report()})
                print(f"Synthetic full validation {index + 1}/20", flush=True)
            report.update(cases=cases, probability_order="(vertical,lateral2,lateral3)", volumes=20)
        else:
            arrays = cache_thebe_validation(config, audit, policy)
            probability, tiling = disk_sliding_predict(arrays["seismic"], predictor, out / "probability_raw.npy",
                                                      config["patch_size"], config["stride"])
            cases = []
            for section in range(200):
                one = ReferenceAccumulator(policy["thresholds"])
                one.update(arrays["states"][section], arrays["labels"][section], probability[section])
                accumulator.update(arrays["states"][section], arrays["labels"][section], probability[section])
                cases.append({"section_in_validation": section, "global_array_section": section + 900, **one.report()})
                if section in (0, 50, 100, 150, 199):
                    _section_figure(out / f"section_{section:03}.png", arrays["seismic"][section],
                                    arrays["labels"][section], probability[section], section)
            report.update(cases=cases, tiling=tiling, probability_order="(section,lateral,sample)", sections=200,
                          probability_sha256=sha256(out / "probability_raw.npy"),
                          region="all 200 original validation sections, full 3174x1537 planes; no label-based ROI",
                          interpretation="Real-data reference coverage; no confirmed complete-annotation negatives, so no field precision/Dice")
        report.update(status="passed", aggregate=accumulator.report())
        save_curve(out / "threshold_curve.csv", report["aggregate"])
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["finished"] = timestamp()
        write_json(out / "evaluation_report.json", report)
    return report


def _section_figure(path, seismic, labels, probability, section):
    from .figures import plt

    nonzero = np.abs(seismic[seismic != 0])
    limit = float(np.quantile(nonzero, 0.99)) if nonzero.size else 1
    fig, axes = plt.subplots(1, 3, figsize=(18, 7), constrained_layout=True)
    for ax in axes[:2]:
        ax.imshow(seismic.T, aspect="auto", cmap="gray", vmin=-limit, vmax=limit)
    axes[1].imshow(np.ma.masked_where(~labels.T, labels.T), aspect="auto", cmap="autumn", vmin=0, vmax=1, alpha=0.7)
    axes[2].imshow(probability.T, aspect="auto", cmap="magma", vmin=0, vmax=1)
    for ax, title in zip(axes, ("地震", "地震＋参考标签", "预测概率"), strict=True):
        ax.set(title=title, xlabel="另一横向数组索引", ylabel="垂向采样索引")
    fig.suptitle(f"Thebe验证区剖面{section}（全局数组{900 + section}）；未标注不等于无断层")
    fig.savefig(path, dpi=140)
    plt.close(fig)
