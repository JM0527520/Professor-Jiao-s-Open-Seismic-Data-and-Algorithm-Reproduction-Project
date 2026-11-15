"""Bounded second inference variant on the same initial development regions."""
import json

import numpy as np

from .baseline import require_audit, writable_path
from .common import WORKSPACE, load_config, model_to_raw, sha256, timestamp, write_json
from .inference import horizontal_ensemble, load_author_model, predict_context_core
from .quality import core_slices, extract_contexts, score_region


def main():
    config = load_config(WORKSPACE / "configs/baseline.json")
    audit = require_audit(config)
    protocol = json.loads((WORKSPACE / "configs/quality_review.json").read_text(encoding="utf-8"))
    followup = json.loads((WORKSPACE / "configs/quality_followup.json").read_text(encoding="utf-8"))
    folder = writable_path(config, "quality_review/2026-11-14")
    reference = json.loads((folder / "quality_report.json").read_text(encoding="utf-8"))
    if reference["status"] != "passed" or reference["protocol"] != protocol:
        raise ValueError("Finish the declared paired review first")
    extract_contexts(config, audit, protocol, folder)
    report = {"status": "running", "started": timestamp(), "protocol": followup,
              "base_report_sha256": sha256(folder / "quality_report.json"), "cases": [],
              "training_performed": False, "test_used": False, "full_baseline_run": False}
    write_json(folder / "followup_report.json", report)
    try:
        predictor, info = load_author_model(config)
        report["runtime"] = info
        ensemble = horizontal_ensemble(predictor)
        for region in protocol["thebe_regions"]:
            name = region["name"]
            parent = np.load(folder / f"{name}_seismic_raw.npy", mmap_mode="r")
            labels = np.load(folder / f"{name}_labels_raw.npy", mmap_mode="r")
            states = np.load(folder / f"{name}_states_raw.npy", mmap_mode="r")
            core = core_slices(region["shape_raw"], protocol["halo_raw"])
            probability, tiling = predict_context_core(parent, ensemble, protocol["halo_raw"], region["shape_raw"], fusion="separable")
            path = folder / f"{name}_{followup['candidate']}_probability_raw.npy"
            np.save(path, probability)
            score = score_region(states[core], labels[core], probability, protocol)
            score.update(region=region, tiling=tiling, probability_sha256=sha256(path),
                         original=next(c for c in reference["thebe"] if c["region"]["name"] == name)["variants"]["legacy_core"]["at_fixed_threshold"])
            report["cases"].append(score)
            write_json(folder / "followup_report.json", report)
            print(f"Horizontal ensemble {name}: {score['at_fixed_threshold']}", flush=True)
        # One synthetic anchor checks whether extra averaging also changes already working inputs.
        from .common import raw_to_model, standardize
        from .data import read_synthetic
        from .inference import reference_overlap

        raw = read_synthetic(config["data_root"], "validation", "seis", 0)
        normalized, _ = standardize(raw)
        labels = read_synthetic(config["data_root"], "validation", "fault", 0).astype(bool)
        probability = model_to_raw(ensemble(raw_to_model(normalized)))
        report["synthetic_anchor_0"] = reference_overlap(labels, probability, np.ones_like(labels), 0.5)
        report.update(status="passed", candidate_promoted=False)
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["finished"] = timestamp()
        write_json(folder / "followup_report.json", report)


if __name__ == "__main__":
    main()
