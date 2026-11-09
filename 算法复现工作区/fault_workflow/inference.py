"""Stage 2: load the author's unchanged checkpoint and reproduce its forward path."""
from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import time

import h5py
import numpy as np

from .common import (model_to_raw, raw_to_model, relative_path, sha256, standardize, timestamp,
                     write_json)
from .data import read_synthetic, valid_trace_envelope
from .figures import three_views


def tile_starts(length: int, patch: int, stride: int) -> list[int]:
    if min(length, patch, stride) <= 0 or stride > patch:
        raise ValueError("Positive dimensions and stride <= patch are required")
    if length <= patch:
        return [0]
    starts = list(range(0, length - patch + 1, stride))
    if starts[-1] != length - patch:
        starts.append(length - patch)
    return starts


def author_overlap_mask(size: int, overlap: int) -> np.ndarray:
    """Reproduce prediction.ipynb's getMask, including its overwrite order.

    The author assigns each axis in succession instead of multiplying 1D weights.
    This is intentionally retained for reproduction, NOT silently corrected.
    """
    if overlap < 0 or overlap > size // 2:
        raise ValueError("Expected 0 <= overlap <= half the patch size")
    mask = np.ones((size, size, size), dtype=np.float32)
    if overlap == 0:
        return mask
    sigma = overlap / 4.0
    for k in range(overlap):
        weight = np.float32(np.exp(-((k - overlap + 1)**2) * 0.5 / (sigma**2)))
        mask[k, :, :] = weight
        mask[size - k - 1, :, :] = weight
    for k in range(overlap):
        weight = np.float32(np.exp(-((k - overlap + 1)**2) * 0.5 / (sigma**2)))
        mask[:, k, :] = weight
        mask[:, size - k - 1, :] = weight
    for k in range(overlap):
        weight = np.float32(np.exp(-((k - overlap + 1)**2) * 0.5 / (sigma**2)))
        mask[:, :, k] = weight
        mask[:, :, size - k - 1] = weight
    return mask


def validate_probability(probability: np.ndarray, expected_shape: tuple[int, ...]) -> None:
    if probability.shape != expected_shape or not np.isfinite(probability).all():
        raise ValueError("Prediction shape mismatch or non-finite values")
    if probability.min() < 0 or probability.max() > 1:
        raise ValueError("Checkpoint output must be sigmoid probabilities in [0,1]")


def sliding_predict(seismic: np.ndarray, predictor, patch: int = 128, stride: int = 64) -> tuple[np.ndarray, dict]:
    """Covered model-order tiles, per-tile z-score, author weights and size restoration."""
    if seismic.ndim != 3 or not np.isfinite(seismic).all():
        raise ValueError("Expected finite 3D seismic")
    if stride > patch or stride <= 0:
        raise ValueError("Stride must be positive and no larger than patch")
    overlap = patch - stride
    if overlap > patch // 2:
        raise ValueError("This author-mask implementation supports overlap <= patch/2")
    original_shape = seismic.shape
    padding = [(0, max(0, patch - n)) for n in original_shape]
    padded = np.pad(seismic, padding, mode="constant").astype(np.float32)
    values = np.zeros(padded.shape, dtype=np.float32)
    weights = np.zeros(padded.shape, dtype=np.float32)
    starts = [tile_starts(n, patch, stride) for n in padded.shape]
    mask = author_overlap_mask(patch, overlap)
    normalization = []
    count = 0
    for i in starts[0]:
        for j in starts[1]:
            for k in starts[2]:
                roi = np.s_[i:i + patch, j:j + patch, k:k + patch]
                normalized, stats = standardize(padded[roi])
                if stats["constant"]:
                    # An empty waveform has no meaningful inference context.
                    probability = np.zeros((patch,) * 3, dtype=np.float32)
                else:
                    probability = np.asarray(predictor(normalized), dtype=np.float32)
                validate_probability(probability, (patch,) * 3)
                values[roi] += probability * mask
                weights[roi] += mask
                normalization.append({"start_model": [i, j, k], **stats})
                count += 1
                print(f"  sliding-window tile {count}: {[i,j,k]}", flush=True)
    if not np.isfinite(weights).all() or not np.all(weights > 0):
        raise ValueError("Sliding window left uncovered or invalid voxels")
    restored = (values / weights)[tuple(slice(0, n) for n in original_shape)]
    validate_probability(restored, original_shape)
    return restored, {"patch_size": patch, "stride": stride, "tiles": count,
                      "tail_padding": padding, "minimum_accumulated_weight": float(weights.min()),
                      "mask": "author getMask assignment order, retained unchanged",
                      "grid": "end-anchored final tile, full coverage, no unnecessary all-zero trailing tiles",
                      "normalization": normalization, "restored_shape": list(restored.shape)}


def load_author_model(config: dict):
    # CPU is deliberate: TensorFlow 2.18 Windows has no native CUDA backend.
    # Small author-model inference is sufficient here; server GPU training is later work.
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    import tensorflow as tf

    tf.config.threading.set_intra_op_parallelism_threads(config["tensorflow_cpu_threads"])
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.keras.utils.set_random_seed(12345)
    path = config["data_root"] / config["checkpoint"]
    checkpoint_hash = sha256(path)
    manifest_path = config["data_root"] / "00_下载记录/GoogleDrive_1q8sAoLJgbhYHRubzyqMi9KkTeZWXWtNd.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = next(item for item in manifest if relative_path(item["local_path"]).name == path.name)
    if record["sha256"] != checkpoint_hash:
        raise ValueError("Checkpoint source checksum mismatch")
    # compile=False avoids loading old optimizer/training loss; inference weights are unchanged.
    model = tf.keras.models.load_model(path, compile=False)
    filters = [layer.filters for layer in model.layers if isinstance(layer, tf.keras.layers.Conv3D)]
    expected = [16, 16, 32, 32, 64, 64, 128, 128, 64, 64, 32, 32, 16, 16, 1]
    if filters != expected or model.count_params() != 1459585:
        raise ValueError("Loaded model does not match author unet3.py / fseg-60")
    checked = 0
    with h5py.File(path, "r") as archive:
        for layer in model.layers:
            if not layer.weights:
                continue
            group = archive["model_weights"][layer.name]
            names = group.attrs["weight_names"]
            stored = [group[name.decode() if isinstance(name, bytes) else name][()] for name in names]
            loaded = layer.get_weights()
            if len(loaded) != len(stored) or not all(np.array_equal(a, b) for a, b in zip(loaded, stored, strict=True)):
                raise ValueError(f"Loaded weight values differ from original HDF5: {layer.name}")
            checked += len(stored)
    probe = np.linspace(-1, 1, 32**3, dtype=np.float32).reshape(32, 32, 32)

    def predict(array):
        with tf.device("/CPU:0"):
            output = model([array[None, ..., None]], training=False).numpy()[0, ..., 0]
        validate_probability(output, array.shape)
        return output

    first, second = predict(probe), predict(probe)
    delta = float(np.max(np.abs(first - second)))
    if delta > 1e-6:
        raise ValueError(f"Repeated author model prediction is unstable: {delta}")
    info = {"framework": "TensorFlow / tf.keras", "tensorflow": tf.__version__, "keras": tf.keras.__version__,
            "device": "CPU", "cpu_threads": config["tensorflow_cpu_threads"], "python": platform.python_version(),
            "platform": platform.platform(), "checkpoint": config["checkpoint"], "sha256": checkpoint_hash,
            "author_entrypoint": "xinwucwp/faultSeg predNew.ipynb, md=60",
            "checkpoint_load": "unchanged original HDF5; load_model(compile=False); no model/weight conversion",
            "parameter_count": model.count_params(), "conv3d_filters": filters,
            "weight_arrays_exactly_compared_to_hdf5": checked, "repeat_probe_max_absolute_difference": delta,
            "normalization": "float32 per-cube z-score from predNew.ipynb; not the older model3.json minmax*255 path",
            "compatibility": "Skip old training optimizer/loss; prediction graph and all 30 weight arrays preserved. No claim of bitwise parity with the unavailable historical runtime."}
    return predict, info


def reference_overlap(labels: np.ndarray, probability: np.ndarray, mask: np.ndarray, threshold: float) -> dict:
    """Technical overlap on a demonstration ROI; not a field-accuracy estimate."""
    if labels.shape != probability.shape or mask.shape != labels.shape or not mask.any():
        raise ValueError("Overlap inputs must align and have nonempty support")
    predicted = probability >= threshold
    truth = labels.astype(bool)
    tp = int(np.count_nonzero(predicted & truth & mask))
    fp = int(np.count_nonzero(predicted & ~truth & mask))
    fn = int(np.count_nonzero(~predicted & truth & mask))
    tn = int(np.count_nonzero(~predicted & ~truth & mask))
    return {"threshold": threshold, "tp": tp, "fp_vs_supplied_reference": fp, "fn": fn, "tn_vs_supplied_reference": tn,
            "dice_vs_supplied_reference": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
            "precision_vs_supplied_reference": tp / (tp + fp) if tp + fp else None,
            "recall_vs_supplied_reference": tp / (tp + fn) if tp + fn else None,
            "support_voxels": int(np.count_nonzero(mask)),
            "usage": "pipeline verification only; no threshold search or checkpoint selection"}


def save_prediction(config: dict, name: str, seismic: np.ndarray, labels: np.ndarray,
                    probability: np.ndarray, mask: np.ndarray, metadata: dict, offsets=(0, 0, 0)) -> dict:
    output = config["output_root"] / "predictions"
    output.mkdir(parents=True, exist_ok=True)
    validate_probability(probability, seismic.shape)
    # Keep both canonical/model order and the author's raw storage order, with explicit names.
    np.save(output / f"{name}_probability_model.npy", probability)
    np.save(output / f"{name}_probability_raw.npy", model_to_raw(probability))
    np.save(output / f"{name}_amplitude_support_model.npy", mask)
    metadata.update(shape_model=list(probability.shape), shape_raw=list(model_to_raw(probability).shape),
                    probability_min=float(probability.min()), probability_max=float(probability.max()),
                    raw_to_model="transpose(2,1,0)", offsets_model=list(offsets),
                    probability_sha256=sha256(output / f"{name}_probability_model.npy"))
    metadata["reference_overlap"] = reference_overlap(labels, probability, mask, config["decision_threshold"])
    write_json(output / f"{name}_metadata.json", metadata)
    three_views(output / f"{name}_comparison.png", seismic, labels, name + "：作者模型小规模预测",
                prediction=probability, offsets=offsets)
    return metadata


def run_predictions(config: dict) -> dict:
    audit = config["output_root"] / "audit/audit_report.json"
    if not audit.exists() or json.loads(audit.read_text(encoding="utf-8"))["status"] != "passed":
        raise ValueError("Complete stage 1 audit before predicting")
    accepted = json.loads(audit.read_text(encoding="utf-8"))
    if accepted.get("data_root") != str(config["data_root"]):
        raise ValueError("Data root changed; repeat the audit on this machine")
    for source in accepted["source_fingerprints"]:
        path = config["data_root"] / relative_path(source["relative_path"])
        stat = path.stat()
        if stat.st_size != source["bytes"] or stat.st_mtime_ns != source["mtime_ns"]:
            raise ValueError(f"Data changed since acceptance: {path}; repeat audit")
    output = config["output_root"] / "predictions"
    output.mkdir(parents=True, exist_ok=True)
    report = {"status": "running", "started": timestamp(), "training_performed": False,
              "test_data_used": False, "cases": []}
    write_json(output / "inference_report.json", report)
    try:
        predictor, model_info = load_author_model(config)
        report["model"] = model_info
        for index in config["synthetic_prediction_ids"]:
            if not 0 <= index < 20:
                raise ValueError("Prediction IDs must be original synthetic validation IDs 0..19")
            seismic = raw_to_model(read_synthetic(config["data_root"], "validation", "seis", index))
            labels = raw_to_model(read_synthetic(config["data_root"], "validation", "fault", index)).astype(bool)
            normalized, stats = standardize(seismic)
            start = time.perf_counter()
            probability = predictor(normalized)
            elapsed = time.perf_counter() - start
            metadata = {"dataset": "FaultSeg3D", "split": "validation", "id": index,
                        "normalization": stats, "elapsed_seconds": round(elapsed, 3), "tiles": 1,
                        "source": str(synthetic_path_for_record(index))}
            report["cases"].append(save_prediction(config, f"faultseg_validation_{index:03}", seismic, labels,
                                                   probability, np.ones(labels.shape, dtype=bool), metadata))
            print(f"FaultSeg3D validation/{index}: prediction passed ({elapsed:.1f}s)", flush=True)
            write_json(output / "inference_report.json", report)
        demo = json.loads((config["cache_root"] / "thebe_validation_demo.json").read_text(encoding="utf-8"))
        with np.load(config["cache_root"] / "thebe_validation_demo.npz", allow_pickle=False) as cache:
            seismic, labels = cache["seismic"], cache["labels"]
        start = time.perf_counter()
        probability, tiling = sliding_predict(seismic, predictor, config["patch_size"], config["stride"])
        elapsed = time.perf_counter() - start
        mask = raw_to_model(valid_trace_envelope(model_to_raw(seismic)))
        metadata = {"dataset": "Thebe", "split": "val", "demo": demo, "tiling": tiling,
                    "elapsed_seconds": round(elapsed, 3),
                    "support_mask_warning": "Local amplitude trace envelope only; not a complete annotation mask. Zeros are retained in saved probabilities.",
                    "evaluation_warning": "Reference-label-rich validation ROI selected for workflow inspection; not the full Thebe benchmark."}
        report["cases"].append(save_prediction(config, "thebe_validation_demo", seismic, labels, probability, mask,
                                               metadata, offsets=(demo["sample_start"], demo["lateral_start"], demo["section_start"])))
        print(f"Thebe validation demo: {tiling['tiles']} tile prediction passed ({elapsed:.1f}s)", flush=True)
        report.update(status="passed", finished=timestamp())
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}", finished=timestamp())
        write_json(output / "inference_report.json", report)
        raise
    write_json(output / "inference_report.json", report)
    lines = ["# 第2步：作者FaultSeg3D小规模预测报告", "", f"完成时间：{report['finished']}", "",
             "结论：使用作者原始fseg-60.hdf5成功预测两个FaultSeg3D验证体，以及一个跨Thebe验证分块的连续区域。没有训练、微调或读取测试区样例。",
             "", "## 实现与原算法的关系", "",
             "- 直接加载作者HDF5模型，没有移植网络或转换权重；1,459,585个参数，30个权重数组与原文件逐值一致。",
             "- 使用TensorFlow 2.18.1的CPU推理，compile=False仅跳过旧优化器和训练损失的加载。没有声称与历史TensorFlow版本逐比特一致。",
             "- 按predNew.ipynb采用float32逐体z-score，按作者transpose转换轴序；没有混用旧model3.json及其minmax×255预处理。",
             "- Thebe采用128³分块、64样点步长及作者getMask的赋值顺序。仅调整末块位置保证完整覆盖，并在输出时还原原尺寸；没有作为改进实验汇报。",
             "- 恒定输入返回零响应，避免除零；本次真实样例是否触发见逐块日志。", "",
             "| 样例 | 范围 | 分块数 | 预测耗时（秒） | 参考重叠Dice |", "|---|---|---:|---:|---:|"]
    for case in report["cases"]:
        name = case["dataset"] + " / " + str(case.get("id", "validation demo"))
        dice = case["reference_overlap"]["dice_vs_supplied_reference"]
        lines.append(f"| {name} | {case['shape_model']} | {case.get('tiles',case.get('tiling',{}).get('tiles'))} | {case['elapsed_seconds']} | {dice:.4f} |")
    lines += ["", "数值仅用于核对参考标签与预测是否对应。Thebe样例是按标签密度选取的验证区域，且其标注并非全区完整负标注，以上数值不能用于宣称工区准确率或基线性能。",
              "", "概率文件同时提供模型轴序和原始轴序；各图均使用相同振幅范围及固定0.5显示阈值。没有根据验证结果调阈值或筛选最优模型。",
              "", "下一步是按既定主线冻结预处理、适用范围与完整区域评价规则，再建立基线。本次没有开始基线训练、算法改进或实际工区迁移。", ""]
    (output / "作者算法预测报告.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def synthetic_path_for_record(index: int) -> Path:
    return Path("01_FaultSeg3D/训练与验证数据/validation/seis") / f"{index}.dat"
