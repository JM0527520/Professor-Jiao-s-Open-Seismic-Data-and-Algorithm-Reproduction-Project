"""The last bounded local comparison: wider real context, same core and weights."""
import json

import numpy as np

from .baseline import require_audit, writable_path
from .common import WORKSPACE, load_config, sha256, timestamp, write_json
from .inference import load_author_model, predict_context_core
from .quality import extract_contexts, score_region


def main():
    config = load_config(WORKSPACE / "configs/baseline.json")
    audit = require_audit(config)
    protocol = json.loads((WORKSPACE / "configs/quality_review.json").read_text(encoding="utf-8"))
    followup = json.loads((WORKSPACE / "configs/quality_wide.json").read_text(encoding="utf-8"))
    folder = writable_path(config, "quality_review/2026-11-14/wide_context")
    folder.mkdir(parents=True, exist_ok=True)
    context_protocol = {**protocol, "version": followup["version"], "halo_raw": followup["halo_raw"],
                        "clip_context_to_validation": True}
    report = {"status": "running", "started": timestamp(), "protocol": followup,
              "cases": [], "training_performed": False, "test_used": False, "full_baseline_run": False}
    write_json(folder / "wide_report.json", report)
    try:
        extract_contexts(config, audit, context_protocol, folder)
        contexts = json.loads((folder / "context_manifest.json").read_text(encoding="utf-8"))["contexts"]
        predictor, info = load_author_model(config)
        report["runtime"] = info
        for region in protocol["thebe_regions"]:
            name = region["name"]
            source = np.load(folder / f"{name}_seismic_raw.npy", mmap_mode="r")
            labels = np.load(folder / f"{name}_labels_raw.npy", mmap_mode="r")
            states = np.load(folder / f"{name}_states_raw.npy", mmap_mode="r")
            core_start = [s - p for s, p in zip(region["start_raw"], contexts[name]["start_raw"], strict=True)]
            core = tuple(slice(s, s + n) for s, n in zip(core_start, region["shape_raw"], strict=True))
            probability, tiling = predict_context_core(source, predictor, core_start, region["shape_raw"], fusion="separable")
            path = folder / f"{name}_wide_probability_raw.npy"
            np.save(path, probability)
            score = score_region(states[core], labels[core], probability, protocol)
            score.update(region=region, tiling=tiling, probability_sha256=sha256(path))
            report["cases"].append(score)
            write_json(folder / "wide_report.json", report)
            print(f"Wide context {name}: {score['at_fixed_threshold']}", flush=True)
        report.update(status="passed", candidate_promoted=False)
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["finished"] = timestamp()
        write_json(folder / "wide_report.json", report)


if __name__ == "__main__":
    main()
