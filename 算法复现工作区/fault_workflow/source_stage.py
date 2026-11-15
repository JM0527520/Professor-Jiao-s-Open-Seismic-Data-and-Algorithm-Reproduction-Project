"""Verify FaultSeg3D independently while the separate Thebe transfer completes."""
from .audit import audit_synthetic
from .baseline import (AUTHOR_DIR, get_policy, require_checkpoint_recipe_match, training_settings,
                       verified_checkpoint, writable_path)
from .common import sha256, timestamp, write_json
from .data import synthetic_path


def prepare_source_training(config):
    settings, policy = training_settings(config), get_policy(config)
    require_checkpoint_recipe_match(config, settings)
    out = writable_path(config, "audit")
    out.mkdir(parents=True, exist_ok=True)
    synthetic = audit_synthetic(config)
    fingerprints = []
    for row in synthetic["rows"]:
        for kind in ("seis", "fault"):
            path = synthetic_path(config["data_root"], row["split"], kind, row["id"])
            stat = path.stat()
            fingerprints.append({"relative_path": path.relative_to(config["data_root"]).as_posix(),
                                 "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                                 "sha256": row[kind]["sha256"]})
    checkpoint = verified_checkpoint(config)
    stat = checkpoint.stat()
    fingerprints.append({"relative_path": checkpoint.relative_to(config["data_root"]).as_posix(),
                         "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": sha256(checkpoint)})
    report = {"status": "passed", "scope": "FaultSeg3D_source_only", "finished": timestamp(),
              "data_root": str(config["data_root"]), "source_fingerprints": fingerprints,
              "faultseg3d": synthetic, "thebe": {"status": "not_audited_on_this_server_yet", "pairs": []},
              "test_used": False, "full_public_data_audit": False}
    # Keep the source-only acceptance permanently when the later full audit replaces the entry point.
    write_json(out / "source_acceptance.json", report)
    write_json(out / "audit_report.json", report)
    sources = [{"relative_path": f"{AUTHOR_DIR}/{name}", "sha256": sha256(config["data_root"] / AUTHOR_DIR / name)}
               for name in ("train.py", "utils.py", "unet3.py", "README.md")]
    manifest = {"status": "source_training_prepared_not_trained", "scope": "FaultSeg3D_source_only",
                "training": settings, "policy": policy, "author_sources": sources,
                "original_checkpoint_sha256": sha256(checkpoint),
                "loss_evidence": "checkpoint training_config cross_entropy_balanced and pinned unet3.py function; BCE script is a distinct historical recipe",
                "thebe_full_acceptance_pending": True, "test_used": False}
    write_json(writable_path(config, "preparation/baseline_manifest.json"), manifest)
    write_json(writable_path(config, "preparation/source_preparation_manifest.json"), manifest)
    write_json(writable_path(config, "preparation/preparation_report.json"),
               {"status": "passed", "scope": "FaultSeg3D_source_training_only",
                "thebe_preparation_completed": False, "full_training_started": False, "finished": timestamp()})
    return report
