"""Bounded-memory readers; NPZ volumes are decompressed as planes, never mmap'ed."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import re
import zipfile

import numpy as np

SPLIT_COUNTS = {"train": 900, "val": 200, "test": 703}
SPLIT_OFFSETS = {"train": 0, "val": 900, "test": 1100}
SPLIT_PARTS = {"train": 9, "val": 2, "test": 7}


def synthetic_path(root: Path, split: str, kind: str, index: int) -> Path:
    if split not in ("train", "validation") or kind not in ("seis", "fault"):
        raise ValueError("Unknown synthetic split or array kind")
    return root / "01_FaultSeg3D/训练与验证数据" / split / kind / f"{index}.dat"


def read_synthetic(root: Path, split: str, kind: str, index: int) -> np.ndarray:
    path = synthetic_path(root, split, kind, index)
    if path.stat().st_size != 128**3 * 4:
        raise ValueError(f"Unexpected binary volume size: {path}")
    return np.fromfile(path, dtype="<f4").reshape(128, 128, 128)


def thebe_parts(root: Path, split: str, kind: str) -> list[Path]:
    if split not in SPLIT_COUNTS or kind not in ("seis", "fault"):
        raise ValueError("Unknown Thebe split or array kind")
    folder = root / "02_Thebe/data/npzfiles" / kind
    paths = list(folder.glob(f"{kind}{split}[0-9]*.npz"))
    paths.sort(key=lambda p: int(re.search(r"(\d+)\.npz$", p.name).group(1)))
    expected = [f"{kind}{split}{i}.npz" for i in range(1, SPLIT_PARTS[split] + 1)]
    if [p.name for p in paths] != expected:
        raise ValueError(f"Missing or ambiguous Thebe parts: {split}/{kind}")
    return paths


def read_header(source) -> tuple[tuple[int, ...], np.dtype, int]:
    version = np.lib.format.read_magic(source)
    if version == (1, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(source)
    elif version == (2, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_2_0(source)
    else:
        raise ValueError(f"Unsupported NPY version: {version}")
    if fortran or dtype.hasobject or len(shape) != 3:
        raise ValueError("Expected numeric C-order 3D NPY")
    return shape, dtype, source.tell()


def npz_header(path: Path) -> dict:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if names != ["arr_0.npy"]:
            raise ValueError(f"Unexpected NPZ members in {path}: {names}")
        with archive.open(names[0]) as source:
            shape, dtype, offset = read_header(source)
        if archive.getinfo(names[0]).file_size != offset + int(np.prod(shape)) * dtype.itemsize:
            raise ValueError(f"NPY header/payload size mismatch: {path}")
    return {"shape": list(shape), "dtype": dtype.str, "member": names[0]}


@contextmanager
def plane_stream(path: Path):
    """Consume all yielded planes to verify the ZIP CRC and trailing bytes."""
    with zipfile.ZipFile(path) as archive, archive.open("arr_0.npy") as source:
        shape, dtype, _ = read_header(source)
        size = int(np.prod(shape[1:])) * dtype.itemsize

        def planes():
            for _ in range(shape[0]):
                payload = source.read(size)
                if len(payload) != size:
                    raise ValueError(f"Truncated NPY payload: {path}")
                yield np.frombuffer(payload, dtype=dtype).reshape(shape[1:])
            if source.read(1):
                raise ValueError(f"Unexpected trailing NPY bytes: {path}")

        yield planes()


def valid_trace_envelope(seismic: np.ndarray) -> np.ndarray:
    """Exclude empty traces and leading/trailing zero padding, retaining interior zeros.

    This is an amplitude-support mask, NOT an annotation-completeness mask.
    Input is raw (section,lateral,sample), so the final axis is vertical.
    """
    present = seismic != 0
    first = present.argmax(axis=-1)
    last = seismic.shape[-1] - 1 - present[..., ::-1].argmax(axis=-1)
    samples = np.arange(seismic.shape[-1])
    return present.any(axis=-1)[..., None] & (samples >= first[..., None]) & (samples <= last[..., None])


def rectangle_start(projection: np.ndarray, width: int, depth: int) -> tuple[int, int]:
    """Choose a labelled demonstration ROI before looking at model predictions."""
    if projection.ndim != 2 or width > projection.shape[0] or depth > projection.shape[1]:
        raise ValueError("ROI is larger than available data")
    integral = np.pad(projection.astype(np.int64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    sums = integral[width:, depth:] - integral[:-width, depth:] - integral[width:, :-depth] + integral[:-width, :-depth]
    if sums.max() == 0:
        raise ValueError("No labelled faults in the selected validation sections")
    return tuple(int(v) for v in np.unravel_index(sums.argmax(), sums.shape))
