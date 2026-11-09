"""Local-only preparation: source recipes, explicit masks and anomaly decisions."""
from __future__ import annotations

import csv

import numpy as np

from .baseline import AUTHOR_DIR, get_policy, require_audit, training_settings, verified_checkpoint, writable_path
from .common import model_to_raw, raw_to_model, relative_path, sha256, timestamp, write_json
from .data import plane_stream
from .policy import INVALID, STATE_NAMES, ReferenceAccumulator, annotation_states


def run_preparation(config):
    audit = require_audit(config)
    settings, policy = training_settings(config), get_policy(config)
    out = writable_path(config, "preparation")
    out.mkdir(parents=True, exist_ok=True)
    report = {"status": "running", "started": timestamp(), "policy_version": policy["version"],
              "test_used": False, "source_labels_modified": False, "full_training_started": False}
    write_json(out / "preparation_report.json", report)
    try:
        sources = []
        for name in ("train.py", "utils.py", "unet3.py", "README.md"):
            path = config["data_root"] / AUTHOR_DIR / name
            sources.append({"relative_path": path.relative_to(config["data_root"]).as_posix(), "sha256": sha256(path)})
        manifest = {"status": "prepared_not_trained", "training": settings,
                    "policy": policy, "author_sources": sources,
                    "original_checkpoint_sha256": sha256(verified_checkpoint(config)),
                    "synthetic_pairs": [{"split": r["split"], "id": r["id"],
                                         "seis_sha256": r["seis"]["sha256"], "fault_sha256": r["fault"]["sha256"]}
                                        for r in audit["faultseg3d"]["rows"]],
                    "baseline_selection": "predeclared epoch 60; all 100 saved; no post-hoc best checkpoint",
                    "thebe_selection": "entire original validation split, all 200 sections; no label-density filtering",
                    "representative_validation_section_indices": [0, 50, 100, 150, 199],
                    "pending_server_work": ["GPU environment verification", "full 128^3 training",
                                            "all 20 synthetic validation volumes", "all 200 Thebe validation sections"],
                    "pending_domain_evidence": ["complete interpretation mask for any Thebe supervised fine-tuning",
                                                "confirmed physical coordinate and sample mapping"]}
        write_json(out / "baseline_manifest.json", manifest)

        anomalies_path = config["output_root"] / "audit/thebe_padding_label_anomalies.csv"
        with anomalies_path.open(encoding="utf-8-sig", newline="") as stream:
            anomalies = list(csv.DictReader(stream))
        expected = {(int(r["section_in_split"]), int(r["lateral_array_index"]), int(r["vertical_sample_index"]))
                    for r in anomalies}
        if len(expected) != 213 or any(r["split"] != "train" for r in anomalies):
            raise ValueError("Accepted anomaly list differs from the expected 213 training voxels")
        demo = audit["thebe"]["demo"]
        states_raw = np.zeros(demo["shape_raw"], dtype=np.uint8)
        filled = np.zeros(demo["shape_raw"][0], dtype=bool)
        with np.load(config["cache_root"] / "thebe_validation_demo.npz", allow_pickle=False) as cache:
            cached_seismic, cached_labels = model_to_raw(cache["seismic"]), model_to_raw(cache["labels"])
        checked = set()
        summaries = []
        preview = None
        invalid_columns = ["split", "section_in_split", "global_array_section", "lateral_array_index",
                           "vertical_sample_index", "reason", "decision"]
        invalid_path = out / "振幅支持范围外正标签.csv"
        with invalid_path.open("w", encoding="utf-8-sig", newline="") as stream:
            csv.DictWriter(stream, fieldnames=invalid_columns).writeheader()
        # Apply the same full-trace rule to ALL train/validation sections.
        # Support is calculated from the full 1537-sample trace BEFORE cropping.
        for pair in audit["thebe"]["pairs"]:
            if pair["split"] == "test":
                continue
            counts = {"split": pair["split"], "section_start": pair["section_start"],
                      "section_stop": pair["section_stop"], "states": {v: 0 for v in STATE_NAMES.values()},
                      "invalid_reference_positive": 0}
            invalid_rows = []
            with plane_stream(config["data_root"] / relative_path(pair["seismic"])) as seismics, \
                    plane_stream(config["data_root"] / relative_path(pair["labels"])) as references:
                for local, (seismic, labels) in enumerate(zip(seismics, references, strict=True)):
                    section = pair["section_start"] + local
                    states = annotation_states(seismic[None], labels[None])[0]
                    for state, name in STATE_NAMES.items():
                        counts["states"][name] += int(np.count_nonzero(states == state))
                    counts["invalid_reference_positive"] += int(np.count_nonzero((states == INVALID) & labels))
                    lateral_indices, samples = np.nonzero((states == INVALID) & labels)
                    empty = ~np.any(seismic != 0, axis=-1)
                    invalid_rows.extend({"split": pair["split"], "section_in_split": section,
                                         "global_array_section": section + (900 if pair["split"] == "val" else 0),
                                         "lateral_array_index": int(x), "vertical_sample_index": int(t),
                                         "reason": "entirely_zero_trace" if empty[x] else "outside_first_last_nonzero_envelope",
                                         "decision": "EXCLUDE_PENDING_REVIEW_SOURCE_PRESERVED"}
                                        for x, t in zip(lateral_indices, samples, strict=True))
                    if pair["split"] == "train":
                        for key in (key for key in expected if key[0] == section):
                            _, lateral, sample = key
                            if not labels[lateral, sample] or np.any(seismic[lateral]) or states[lateral, sample] != INVALID:
                                raise ValueError(f"Anomaly source verification failed: {key}")
                            checked.add(key)
                        if section == 747:
                            preview = (seismic[:12, 920:1020].copy(), labels[:12, 920:1020].copy(),
                                       states[:12, 920:1020].copy())
                    elif demo["section_start"] <= section < demo["section_stop"]:
                        dest = section - demo["section_start"]
                        roi = np.s_[demo["lateral_start"]:demo["lateral_stop"], demo["sample_start"]:demo["sample_stop"]]
                        if (not np.array_equal(seismic[roi].astype(np.float32), cached_seismic[dest])
                                or not np.array_equal(labels[roi], cached_labels[dest])):
                            raise ValueError("Demonstration cache differs from its accepted full-trace source")
                        states_raw[dest] = states[roi]
                        filled[dest] = True
            summaries.append(counts)
            with invalid_path.open("a", encoding="utf-8", newline="") as stream:
                csv.DictWriter(stream, fieldnames=invalid_columns).writerows(invalid_rows)
            print(f"Policy support/CRC verified: {pair['split']} sections [{pair['section_start']},{pair['section_stop']})", flush=True)
        if checked != expected or not filled.all():
            raise ValueError("Anomaly verification or demo support reconstruction incomplete")
        np.save(out / "thebe_demo_annotation_states_raw.npy", states_raw)
        states_model = raw_to_model(states_raw)
        np.save(out / "thebe_demo_annotation_states_model.npy", states_model)
        decisions = [{**row, "decision": "INVALID_EXCLUDE", "source_label_preserved": "true",
                      "reason": "reference_positive_on_entirely_zero_trace"} for row in anomalies]
        with (out / "异常标签处理清单.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(decisions[0]))
            writer.writeheader()
            writer.writerows(decisions)
        probability_path = config["output_root"] / "predictions/thebe_validation_demo_probability_model.npy"
        probability = np.load(probability_path, allow_pickle=False)
        score = ReferenceAccumulator(policy["thresholds"])
        score.update(states_model, raw_to_model(cached_labels), probability)
        evaluation = score.report()
        evaluation.update(purpose="policy demonstration on the previous label-rich ROI, NOT the baseline",
                          probability_sha256=sha256(probability_path),
                          state_encoding=STATE_NAMES, full_trace_support_before_crop=True,
                          source_coordinates=demo, states_model_sha256=sha256(out / "thebe_demo_annotation_states_model.npy"))
        write_json(out / "thebe_demo_policy_evaluation.json", evaluation)
        if preview is not None:
            _anomaly_figure(out / "异常标签核查图.png", *preview)
        report.update(status="passed", verified_invalid_positive_voxels=len(checked),
                      inspected_parts=summaries, state_encoding=STATE_NAMES,
                      demo_counts={name: int((states_raw == code).sum()) for code, name in STATE_NAMES.items()},
                      scope="all 900 training and 200 validation sections; full-trace envelopes computed before cropping",
                      test_used=False)
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["finished"] = timestamp()
        write_json(out / "preparation_report.json", report)
    return report


def _anomaly_figure(path, seismic, labels, states):
    # figures configures a locally available Chinese font and a headless backend.
    from .figures import plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 5), constrained_layout=True)
    for ax, data, title in zip(axes, (seismic.T, labels.T, (states != INVALID).T),
                               ("原始振幅", "参考正标签（原样保留）", "振幅支持范围"), strict=True):
        ax.imshow(data, aspect="auto", cmap="gray", interpolation="nearest", extent=(-0.5, 11.5, 1019.5, 919.5))
        ax.set(title=title, xlabel="另一横向数组索引", ylabel="垂向采样索引")
    fig.suptitle("Thebe训练区剖面747边界：标签落在全零道的位置不用于监督和评价")
    fig.savefig(path, dpi=160)
    plt.close(fig)
