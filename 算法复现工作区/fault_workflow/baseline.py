"""Baseline configuration, immutable inputs and the author's modern Keras graph."""
from __future__ import annotations

import json
import os
from pathlib import Path
import platform

import numpy as np

from .common import WORKSPACE, relative_path, sha256, timestamp, write_json

AUTHOR_DIR = "01_FaultSeg3D/作者代码/已解压/faultSeg-0ab8ba1c10cb9e5748b0129bfe1a2fe3031b80fe"
CHECKPOINT_SHA256 = "5174031cb013e6c8a311c1c74689f473157105baf4a089e8463abf01a96349da"


def writable_path(config, relative) -> Path:
    path = (config["output_root"] / relative).resolve()
    if not path.is_relative_to(WORKSPACE) or path.is_relative_to(config["data_root"]):
        raise ValueError("Generated baseline files must stay inside the algorithm workspace")
    return path


def get_policy(config):
    path = (WORKSPACE / config["policy"]).resolve()
    if not path.is_relative_to(WORKSPACE):
        raise ValueError("Policy must be in the algorithm workspace")
    policy = json.loads(path.read_text(encoding="utf-8"))
    if not policy["test_locked"] or policy["thebe"]["supervised_training_enabled"]:
        raise ValueError("This baseline keeps TEST locked and does not train on unconfirmed Thebe negatives")
    return policy


def require_audit(config) -> dict:
    path = config["output_root"] / "audit/audit_report.json"
    audit = json.loads(path.read_text(encoding="utf-8"))
    if audit["status"] != "passed" or audit["data_root"] != str(config["data_root"]):
        raise ValueError("Repeat the acceptance audit in the current data location")
    for item in audit["source_fingerprints"]:
        source = config["data_root"] / relative_path(item["relative_path"])
        stat = source.stat()
        if stat.st_size != item["bytes"] or stat.st_mtime_ns != item["mtime_ns"]:
            raise ValueError(f"Source changed since acceptance; repeat audit: {source}")
    return audit


def training_settings(config) -> dict:
    settings = config["training"]
    expected = {"epochs": 100, "fixed_baseline_epoch": 60, "learning_rate": 1e-4,
                "adam_beta_1": 0.9, "adam_beta_2": 0.999, "adam_epsilon": 1e-7,
                "numpy_seed": 12345, "tensorflow_seed": 1234,
                "source_volumes_per_step": 1, "effective_batch_size": 2,
                "augmentation": "identity_and_model_axis0_flip", "loss": "binary_crossentropy",
                "initialization": "random", "precision": "float32", "shuffle_training": True,
                "shuffle_validation": True, "save_each_epoch": True,
                "learning_rate_schedule": None, "early_stopping": False}
    if any(settings.get(k) != v for k, v in expected.items()):
        raise ValueError("Author baseline settings changed; create a separately reviewed protocol")
    name = settings["run_name"]
    if not name or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for c in name):
        raise ValueError("Use a simple, unique lowercase run name")
    return settings


def require_preparation(config):
    manifest = json.loads(writable_path(config, "preparation/baseline_manifest.json").read_text(encoding="utf-8"))
    report = json.loads(writable_path(config, "preparation/preparation_report.json").read_text(encoding="utf-8"))
    if report["status"] != "passed" or manifest["training"] != training_settings(config) or manifest["policy"] != get_policy(config):
        raise ValueError("Complete local preparation for the current training and annotation protocol")
    for source in manifest["author_sources"]:
        if sha256(config["data_root"] / relative_path(source["relative_path"])) != source["sha256"]:
            raise ValueError("Pinned author source changed after preparation")
    return manifest


def configure_tensorflow(config, device="cpu"):
    if device not in ("cpu", "gpu"):
        raise ValueError("Device must be cpu or gpu")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    import tensorflow as tf

    tf.config.threading.set_intra_op_parallelism_threads(config["tensorflow_cpu_threads"])
    tf.config.threading.set_inter_op_parallelism_threads(1)
    gpus = tf.config.list_physical_devices("GPU")
    if device == "gpu" and not gpus:
        raise RuntimeError("No TensorFlow GPU is available; full server work has not started")
    if device == "gpu":
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    tf.keras.mixed_precision.set_global_policy("float32")
    # Preserve the two explicit seeds in author train.py; historical kernels differ.
    tf.keras.utils.set_random_seed(config["training"]["tensorflow_seed"])
    np.random.seed(config["training"]["numpy_seed"])
    return tf


def author_graph(tf):
    """Same layer sequence as pinned unet3.py, expressed using current tf.keras APIs."""
    layers = tf.keras.layers
    inputs = layers.Input((None, None, None, 1))

    def block(value, channels):
        value = layers.Conv3D(channels, 3, activation="relu", padding="same")(value)
        return layers.Conv3D(channels, 3, activation="relu", padding="same")(value)

    first = block(inputs, 16)
    second = block(layers.MaxPooling3D(2)(first), 32)
    third = block(layers.MaxPooling3D(2)(second), 64)
    value = block(layers.MaxPooling3D(2)(third), 128)
    for skip, channels in ((third, 64), (second, 32), (first, 16)):
        value = layers.Concatenate(axis=-1)([layers.UpSampling3D(2)(value), skip])
        value = block(value, channels)
    output = layers.Conv3D(1, 1, activation="sigmoid")(value)
    return tf.keras.Model(inputs=[inputs], outputs=[output])


def compile_author_model(tf, model, settings):
    model.compile(optimizer=tf.keras.optimizers.Adam(
        learning_rate=settings["learning_rate"], beta_1=settings["adam_beta_1"],
        beta_2=settings["adam_beta_2"], epsilon=settings["adam_epsilon"]),
        loss="binary_crossentropy", metrics=["binary_accuracy"], jit_compile=False)


def graph_signature(model):
    """Compare operations/settings independent of automatically generated layer names."""
    keys = ("filters", "kernel_size", "strides", "padding", "activation", "use_bias",
            "pool_size", "size", "axis", "data_format", "dilation_rate")
    return [{"class": type(layer).__name__, **{k: layer.get_config()[k] for k in keys if k in layer.get_config()}}
            for layer in model.layers]


def verified_checkpoint(config) -> Path:
    path = config["data_root"] / relative_path(config["checkpoint"])
    if path.name != "fseg-60.hdf5" or sha256(path) != CHECKPOINT_SHA256:
        raise ValueError("Expected the accepted original fseg-60 checkpoint")
    return path


def runtime_info(tf, device):
    return {"python": platform.python_version(), "platform": platform.platform(),
            "tensorflow": tf.__version__, "keras": tf.keras.__version__, "numpy": np.__version__,
            "requested_device": device, "gpus": [v.name for v in tf.config.list_physical_devices("GPU")],
            "gpu_details": [tf.config.experimental.get_device_details(v) for v in tf.config.list_physical_devices("GPU")],
            "precision": tf.keras.mixed_precision.global_policy().name,
            "build": tf.sysconfig.get_build_info(), "time": timestamp()}


def record_preflight(config, device):
    require_audit(config)
    training_settings(config)
    get_policy(config)
    tf = configure_tensorflow(config, device)
    report = runtime_info(tf, device)
    report.update(status="passed", full_training_started=False)
    write_json(writable_path(config, f"preparation/preflight_{device}.json"), report)
    return report


def require_gpu_smoke(config, tf):
    report = json.loads(writable_path(config, "preparation/gpu_smoke_report.json").read_text(encoding="utf-8"))
    current = runtime_info(tf, "gpu")
    keys = ("python", "platform", "tensorflow", "keras", "numpy", "gpu_details", "build", "precision")
    if (report["status"] != "passed" or report["training"] != training_settings(config)
            or json.dumps({k: current[k] for k in keys}, sort_keys=True)
            != json.dumps({k: report["runtime"][k] for k in keys}, sort_keys=True)):
        raise ValueError("Run the full-shape GPU smoke in this server environment before full work")
