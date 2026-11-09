"""Paths and provenance. All generated artifacts must stay outside the data root."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np

WORKSPACE = Path(__file__).resolve().parents[1]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_path(value: str) -> Path:
    """Accept provenance manifests created on Windows when running later on Linux."""
    return Path(value.replace("\\", "/"))


def load_config(path: Path | None = None) -> dict:
    path = path or WORKSPACE / "configs/local.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    for name in ("data_root", "output_root", "cache_root"):
        config[name] = (WORKSPACE / config[name]).resolve()
    if not config["data_root"].is_dir():
        raise FileNotFoundError(config["data_root"])
    for name in ("output_root", "cache_root"):
        if config[name].is_relative_to(config["data_root"]):
            raise ValueError(f"{name} must not be inside the original dataset")
    if config["thebe_demo_split"] != "val":
        raise ValueError("This preparation workflow only demonstrates Thebe VALIDATION; TEST is locked")
    patch, stride = config["patch_size"], config["stride"]
    if patch != 128 or not 0 < stride <= patch:
        raise ValueError("Author-model preparation uses 128^3 patches and a covered stride")
    return config


def raw_to_model(array: np.ndarray) -> np.ndarray:
    """Author np.transpose: raw (section,lateral,sample) -> (sample,lateral,section)."""
    if array.ndim != 3:
        raise ValueError("Expected a three-dimensional volume")
    return np.ascontiguousarray(array.transpose(2, 1, 0))


def model_to_raw(array: np.ndarray) -> np.ndarray:
    return raw_to_model(array)


def standardize(array: np.ndarray) -> tuple[np.ndarray, dict]:
    """Match predNew.ipynb: float32, one mean/std per cube, with a constant guard."""
    array = np.asarray(array, dtype=np.float32)
    if array.ndim != 3 or not np.isfinite(array).all():
        raise ValueError("Expected finite 3D seismic samples")
    mean, std = float(array.mean()), float(array.std())
    if std == 0.0:
        return np.zeros_like(array), {"mean": mean, "std": std, "constant": True}
    normalized = (array - np.float32(mean)) / np.float32(std)
    if not np.isfinite(normalized).all():
        raise ValueError("Non-finite normalized samples")
    return normalized, {"mean": mean, "std": std, "constant": False}
