"""Author training compatibility; full 128^3 training is an explicit server-only stage."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import time

import numpy as np

from .baseline import (AUTHOR_DIR, author_graph, compile_author_model, configure_tensorflow,
                       get_policy, graph_signature, require_audit, require_gpu_smoke, require_preparation, runtime_info,
                       training_settings, verified_checkpoint, writable_path)
from .common import raw_to_model, sha256, standardize, timestamp, write_json
from .data import read_synthetic


def author_batch(config, split, index, crop=None):
    """One source volume produces identity + flipud, including on validation."""
    seismic = read_synthetic(config["data_root"], split, "seis", index)
    labels = raw_to_model(read_synthetic(config["data_root"], split, "fault", index))
    seismic, stats = standardize(seismic)
    if stats["constant"] or not np.isin(labels, [0, 1]).all():
        raise ValueError("Cannot train on constant inputs or nonbinary labels")
    seismic = raw_to_model(seismic)
    if crop is not None:
        if crop <= 0 or crop > 128 or crop % 8:
            raise ValueError("Smoke crop must be a multiple of 8, no larger than 128")
        offset = (128 - crop) // 2
        roi = (slice(offset, offset + crop),) * 3
        seismic, labels = seismic[roi], labels[roi]
    x = np.stack((seismic, np.flip(seismic, axis=0)))[..., None]
    y = np.stack((labels, np.flip(labels, axis=0)))[..., None]
    return x, y


def epoch_orders(settings, epochs=None):
    """Replayable shared NumPy shuffle stream; no validation-dependent selection."""
    rng = np.random.RandomState(settings["numpy_seed"])
    for _ in range(epochs or settings["epochs"]):
        yield rng.permutation(200), rng.permutation(20)


def atomic_model_save(model, path):
    temporary = path.with_name(path.stem + ".saving.keras")
    model.save(temporary)
    temporary.replace(path)


def run_smoke(config):
    require_audit(config)
    settings = training_settings(config)
    tf = configure_tensorflow(config, "cpu")
    out = writable_path(config, "preparation/smoke")
    out.mkdir(parents=True, exist_ok=True)
    report = {"status": "running", "started": timestamp(), "runtime": runtime_info(tf, "cpu"),
              "purpose": "32^3 central crop of training volume 0, effective batch 2; engineering check only",
              "baseline_training_performed": False, "local_gradient_steps": 3,
              "validation_used_for_gradient": False, "test_used": False}
    write_json(out / "smoke_report.json", report)
    try:
        model = author_graph(tf)
        source = tf.keras.models.load_model(verified_checkpoint(config), compile=False)
        assert graph_signature(model) == graph_signature(source), "Author graph configuration differs"
        assert model.count_params() == 1459585
        random_weights = model.get_weights()
        assert any(not np.array_equal(a, b) for a, b in zip(random_weights, source.get_weights(), strict=True))
        probe = np.linspace(-1, 1, 32**3, dtype=np.float32).reshape(1, 32, 32, 32, 1)
        model.set_weights(source.get_weights())
        expected = source([probe], training=False).numpy()
        actual = model([probe], training=False).numpy()
        difference = float(np.max(np.abs(expected - actual)))
        if difference > 1e-6:
            raise ValueError("Modern author graph does not reproduce checkpoint forward pass")
        model.set_weights(random_weights)
        del source
        compile_author_model(tf, model, settings)
        x, y = author_batch(config, "train", 0, crop=32)
        before = float(model.test_on_batch([x], y, return_dict=True)["loss"])
        first_weight = model.get_weights()[0].copy()
        losses = []
        for _ in range(3):
            model.reset_metrics()
            losses.append(float(model.train_on_batch([x], y, return_dict=True)["loss"]))
        model.reset_metrics()
        after = float(model.test_on_batch([x], y, return_dict=True)["loss"])
        if not np.isfinite([before, after, *losses]).all() or not after < before:
            raise ValueError("Tiny training smoke did not reduce the finite training loss")
        if np.array_equal(first_weight, model.get_weights()[0]):
            raise ValueError("Training failed to update network weights")
        prediction = model([x], training=False).numpy()
        path = out / "smoke_only.keras"
        atomic_model_save(model, path)
        restored = tf.keras.models.load_model(path)
        np.testing.assert_array_equal(prediction, restored([x], training=False).numpy())
        assert int(restored.optimizer.iterations.numpy()) == 3
        model.reset_metrics()
        restored.reset_metrics()
        model.train_on_batch([x], y)
        restored.train_on_batch([x], y)
        resume_delta = max(float(np.max(np.abs(a - b))) for a, b in zip(model.get_weights(), restored.get_weights(), strict=True))
        if resume_delta > 1e-6:
            raise ValueError("Optimizer restore failed the identical next-update check")
        # The validation reader is exercised separately; it never supplies a gradient.
        vx, vy = author_batch(config, "validation", 0, crop=32)
        restored.reset_metrics()
        validation = restored.test_on_batch([vx], vy, return_dict=True)
        if not all(np.isfinite(v) for v in validation.values()):
            raise ValueError("Non-finite validation forward/loss")
        report.update(status="passed", parameter_count=model.count_params(),
                      author_graph_forward_max_absolute_difference=difference,
                      training_loss_before=before, training_loss_after=after, step_losses=losses,
                      saved_optimizer_iterations=3, resume_next_update_max_absolute_difference=resume_delta,
                      post_restore_validation={k: float(v) for k, v in validation.items()},
                      note="3 updates plus one independent resume-verification update in memory; saved model is not a baseline")
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["finished"] = timestamp()
        write_json(out / "smoke_report.json", report)
    print(f"Local smoke passed: loss {before:.6f} -> {after:.6f}; checkpoint + optimizer restore passed", flush=True)
    return report


def run_training(config, resume=None, server_work=False):
    if not server_work:
        raise ValueError("Full training is deferred; use --server-work only on the prepared server")
    audit = require_audit(config)
    settings = training_settings(config)
    policy = get_policy(config)
    require_preparation(config)
    tf = configure_tensorflow(config, "gpu")
    require_gpu_smoke(config, tf)
    out = writable_path(config, f"training/{settings['run_name']}")
    protocol = {"training": settings, "policy": policy,
                "source_model_sha256": sha256(verified_checkpoint(config)),
                "source_code_sha256": {name: sha256(config["data_root"] / AUTHOR_DIR / name)
                                       for name in ("train.py", "utils.py", "unet3.py")},
                "synthetic_source_sha256": [{"split": r["split"], "id": r["id"],
                                             "seis": r["seis"]["sha256"], "fault": r["fault"]["sha256"]}
                                            for r in audit["faultseg3d"]["rows"]]}
    if resume is None:
        if out.exists() and any(out.iterdir()):
            raise ValueError("Run already exists; resume its completed epoch or choose a new run_name")
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / "protocol.json", protocol)
        model = author_graph(tf)
        compile_author_model(tf, model, settings)
        completed = 0
    else:
        resume = Path(resume).resolve()
        if resume.parent != out.resolve():
            raise ValueError("Resume checkpoint must belong to this configured run")
        if json.loads((out / "protocol.json").read_text(encoding="utf-8")) != protocol:
            raise ValueError("Resume protocol/source differs from the original run")
        state = json.loads(resume.with_suffix(".json").read_text(encoding="utf-8"))
        if state["checkpoint_sha256"] != sha256(resume):
            raise ValueError("Resume checkpoint checksum mismatch")
        model = tf.keras.models.load_model(resume)
        completed = state["completed_epoch"]
        if int(model.optimizer.iterations.numpy()) != completed * 200:
            raise ValueError("Resume optimizer step does not match completed epochs")
        # No later epochs may remain beside a rewound checkpoint.
        if any(int(p.stem.split("-")[1]) > completed for p in out.glob("epoch-*.json")):
            raise ValueError("Resume the latest completed checkpoint; do not overwrite later epochs")
    write_json(out / "runtime.json", runtime_info(tf, "gpu"))
    report = {"status": "running", "started": timestamp(), "completed_epoch": completed,
              "test_used": False, "thebe_supervised_training": False,
              "primary_checkpoint_rule": "fixed epoch 60, declared before training; save all 100 epochs"}
    write_json(out / "training_report.json", report)
    try:
        for epoch, (train_ids, validation_ids) in enumerate(epoch_orders(settings), 1):
            if epoch <= completed:
                continue
            start = time.perf_counter()
            row = {"epoch": epoch}
            for split, ids, prefix in (("train", train_ids, "train"), ("validation", validation_ids, "validation")):
                values = []
                for index in ids:
                    x, y = author_batch(config, split, int(index))
                    model.reset_metrics()
                    result = (model.train_on_batch([x], y, return_dict=True) if split == "train"
                              else model.test_on_batch([x], y, return_dict=True))
                    if not all(np.isfinite(v) for v in result.values()):
                        raise ValueError(f"Non-finite training statistics: epoch {epoch}, {split}/{index}")
                    values.append([float(result["loss"]), float(result["binary_accuracy"])])
                mean = np.asarray(values).mean(0)
                row.update({f"{prefix}_loss": float(mean[0]), f"{prefix}_binary_accuracy": float(mean[1])})
            row["seconds"] = round(time.perf_counter() - start, 3)
            checkpoint = out / f"epoch-{epoch:03}.keras"
            atomic_model_save(model, checkpoint)
            write_json(checkpoint.with_suffix(".json"), {"completed_epoch": epoch,
                       "checkpoint_sha256": sha256(checkpoint), "metrics": row})
            # Per-epoch JSON is authoritative; the CSV is rebuilt to avoid stale rows after interruption.
            rows = [json.loads(p.read_text(encoding="utf-8"))["metrics"] for p in sorted(out.glob("epoch-*.json"))]
            with (out / "history.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader()
                writer.writerows(rows)
            report.update(completed_epoch=epoch, latest_checkpoint=checkpoint.name)
            write_json(out / "training_report.json", report)
            print(f"Epoch {epoch}/100 train_loss={row['train_loss']:.6f} validation_loss={row['validation_loss']:.6f}", flush=True)
        report.update(status="passed", fixed_baseline_checkpoint="epoch-060.keras")
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["finished"] = timestamp()
        write_json(out / "training_report.json", report)
    return report


def run_gpu_smoke(config, server_work=False):
    if not server_work:
        raise ValueError("Full-shape GPU check is deferred until the server is available")
    require_audit(config)
    require_preparation(config)
    settings = training_settings(config)
    tf = configure_tensorflow(config, "gpu")
    path = writable_path(config, "preparation/gpu_smoke_report.json")
    report = {"status": "running", "started": timestamp(), "training": settings,
              "runtime": runtime_info(tf, "gpu"), "shape": [2, 128, 128, 128, 1],
              "baseline_training_performed": False, "gradient_steps": 1, "test_used": False}
    write_json(path, report)
    try:
        model = author_graph(tf)
        compile_author_model(tf, model, settings)
        x, y = author_batch(config, "train", 0)
        result = model.train_on_batch([x], y, return_dict=True)
        if not all(np.isfinite(v) for v in result.values()) or int(model.optimizer.iterations.numpy()) != 1:
            raise ValueError("Full-shape GPU update did not finish correctly")
        report.update(status="passed", metrics={k: float(v) for k, v in result.items()},
                      gpu_memory_bytes=tf.config.experimental.get_memory_info("GPU:0"))
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["finished"] = timestamp()
        write_json(path, report)
    return report
