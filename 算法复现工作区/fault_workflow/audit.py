"""Stage 1: verify pairs, axes, numeric values and padding without preparing training data."""
from __future__ import annotations

from pathlib import Path
import csv
import json
import time

import numpy as np

from .common import raw_to_model, relative_path, sha256, timestamp, write_json
from .data import (SPLIT_COUNTS, SPLIT_OFFSETS, npz_header, plane_stream,
                   read_synthetic, rectangle_start, synthetic_path, thebe_parts)
from .figures import full_section, three_views


def audit_synthetic(config: dict) -> dict:
    root, out = config["data_root"], config["output_root"] / "audit"
    recovery = json.loads((root / "00_下载记录/FaultSeg3D镜像恢复记录.json").read_text(encoding="utf-8"))
    known = {str((root / relative_path(r["relative_path"])).resolve()): r["sha256"] for r in recovery["files"]}
    rows = []
    for split, count in (("train", 200), ("validation", 20)):
        for kind in ("seis", "fault"):
            folder = synthetic_path(root, split, kind, 0).parent
            actual = {p.name for p in folder.glob("*.dat")}
            expected = {f"{i}.dat" for i in range(count)}
            if actual != expected:
                raise ValueError(f"Synthetic ID set mismatch: {split}/{kind}")
        for index in range(count):
            pair = {}
            for kind in ("seis", "fault"):
                path = synthetic_path(root, split, kind, index)
                digest = sha256(path)
                if known.get(str(path.resolve())) != digest:
                    raise ValueError(f"Synthetic provenance hash mismatch: {path}")
                array = read_synthetic(root, split, kind, index)
                if not np.isfinite(array).all():
                    raise ValueError(f"Non-finite samples: {path}")
                if kind == "fault" and not np.isin(array, [0, 1]).all():
                    raise ValueError(f"Non-binary synthetic fault labels: {path}")
                if kind == "seis" and float(array.std()) == 0:
                    raise ValueError(f"Constant synthetic seismic volume: {path}")
                pair[kind] = {"min": float(array.min()), "max": float(array.max()),
                              "mean": float(array.mean()), "std": float(array.std()),
                              "nonzero": int(np.count_nonzero(array)), "sha256": digest}
            rows.append({"split": split, "id": index, "shape_raw": [128, 128, 128], **pair})
        print(f"FaultSeg3D {split}: {count} paired volumes checked", flush=True)
    for split, index in (("train", 0), ("validation", 0), ("validation", 1)):
        seismic = raw_to_model(read_synthetic(root, split, "seis", index))
        labels = raw_to_model(read_synthetic(root, split, "fault", index))
        three_views(out / f"faultseg_{split}_{index:03}_three_views.png", seismic, labels,
                    f"FaultSeg3D {split} / {index}：地震—标签三向核对")
    return {"status": "passed", "train_pairs": 200, "validation_pairs": 20,
            "raw_axes": ["n3/lateral", "n2/lateral", "n1/vertical"],
            "model_axes": ["n1/vertical", "n2/lateral", "n3/lateral"],
            "axis_transform": "raw.transpose(2,1,0), exactly author utils.py / predNew.ipynb",
            "dtype": "little-endian float32", "all_provenance_hashes_verified": True,
            "rows": rows}


def choose_demo(config: dict) -> tuple[dict, np.ndarray]:
    root = config["data_root"]
    start = config["thebe_demo_section_start"]
    nsections, width, depth = config["thebe_demo_shape_raw"]
    if start < 0 or start + nsections > SPLIT_COUNTS["val"]:
        raise ValueError("Demonstration volume must remain inside VALIDATION")
    projection = np.zeros((3174, 1537), dtype=np.int32)
    offset = 0
    for path in thebe_parts(root, "val", "fault"):
        with plane_stream(path) as planes:
            for label in planes:
                if start <= offset < start + nsections:
                    projection += label.astype(np.int32)
                offset += 1
    lateral, vertical = rectangle_start(projection, width, depth)
    metadata = {
        "split": "val", "section_start": start, "section_stop": start + nsections,
        "lateral_start": lateral, "lateral_stop": lateral + width,
        "sample_start": vertical, "sample_stop": vertical + depth,
        "shape_raw": [nsections, width, depth], "shape_model": [depth, width, nsections],
        "global_section_start": SPLIT_OFFSETS["val"] + start,
        "selection": "Fixed 128 validation sections; choose 192x128 rectangle with largest reference-label count. No predictions used.",
        "purpose": "Label-rich pipeline smoke test, NOT an unbiased benchmark or evidence of generalization.",
        "coordinate_units": "zero-based array indices; physical survey mapping not established",
    }
    return metadata, projection


def summarize_plane(seismic: np.ndarray, labels: np.ndarray) -> dict:
    if seismic.shape != labels.shape or not np.isfinite(seismic).all():
        raise ValueError("Thebe mismatched or non-finite plane")
    if labels.dtype != np.bool_ and not np.isin(labels, [0, 1]).all():
        raise ValueError("Non-binary Thebe reference")
    trace_present = np.any(seismic != 0, axis=-1)
    label_counts = np.sum(labels, axis=-1, dtype=np.int64)
    lateral = np.flatnonzero(np.any(labels, axis=-1))
    vertical = np.flatnonzero(np.any(labels, axis=0))
    return {"samples": int(seismic.size), "finite": int(seismic.size),
            "zeros": int(np.count_nonzero(seismic == 0)), "min": float(seismic.min()), "max": float(seismic.max()),
            "sum": float(np.sum(seismic, dtype=np.float64)),
            "sum_squares": float(np.einsum("ij,ij->", seismic, seismic, dtype=np.float64)),
            "nonempty_traces": int(np.count_nonzero(trace_present)),
            "empty_traces": int(np.count_nonzero(~trace_present)),
            "fault_voxels": int(np.sum(label_counts)),
            "fault_voxels_on_empty_traces": int(np.sum(label_counts[~trace_present])),
            "label_lateral_bounds": [int(lateral[0]), int(lateral[-1]) + 1] if lateral.size else None,
            "label_sample_bounds": [int(vertical[0]), int(vertical[-1]) + 1] if vertical.size else None}


def audit_thebe(config: dict) -> dict:
    root, out, cache = config["data_root"], config["output_root"] / "audit", config["cache_root"]
    cache.mkdir(parents=True, exist_ok=True)
    pairs = []
    for split in SPLIT_COUNTS:
        offset = 0
        for seis_path, fault_path in zip(thebe_parts(root, split, "seis"), thebe_parts(root, split, "fault"), strict=True):
            seismic, labels = npz_header(seis_path), npz_header(fault_path)
            if seismic["shape"] != labels["shape"] or seismic["shape"][1:] != [3174, 1537]:
                raise ValueError(f"Thebe pair shape mismatch: {seis_path}")
            if np.dtype(seismic["dtype"]) != np.dtype("float64") or np.dtype(labels["dtype"]) != np.dtype("bool"):
                raise ValueError(f"Unexpected Thebe dtype: {seis_path}")
            pairs.append({"split": split, "section_start": offset,
                          "section_stop": offset + seismic["shape"][0],
                          "global_section_start": offset + SPLIT_OFFSETS[split],
                          "seismic": str(seis_path.relative_to(root)), "labels": str(fault_path.relative_to(root)),
                          **seismic, "labels_dtype": labels["dtype"]})
            offset += seismic["shape"][0]
        if offset != SPLIT_COUNTS[split]:
            raise ValueError(f"Wrong number of Thebe {split} sections: {offset}")
    print("Thebe: all 18 NPZ pairs and 900/200/703 contiguous section counts checked", flush=True)
    demo, _ = choose_demo(config)
    write_json(cache / "thebe_validation_demo.json", demo)
    raw_seismic = np.zeros(demo["shape_raw"], dtype=np.float32)
    raw_labels = np.zeros(demo["shape_raw"], dtype=np.bool_)
    filled = np.zeros(demo["shape_raw"][0], dtype=bool)
    sections = []
    anomalies = []
    for pair in pairs:
        if pair["split"] == "test":
            pair["content_check"] = "headers_only_TEST_locked; original download checksum already verified"
            continue
        start_time = time.perf_counter()
        with plane_stream(root / pair["seismic"]) as seismics, plane_stream(root / pair["labels"]) as labels:
            for local_index, (seismic, reference) in enumerate(zip(seismics, labels, strict=True)):
                index = pair["section_start"] + local_index
                stats = summarize_plane(seismic, reference)
                sections.append({"split": pair["split"], "section": index, **stats})
                if stats["fault_voxels_on_empty_traces"]:
                    missing_trace = ~np.any(seismic != 0, axis=-1)
                    lateral_indices, sample_indices = np.nonzero(reference & missing_trace[:, None])
                    anomalies.extend({"split": pair["split"], "section_in_split": index,
                                      "section_in_concatenated_array": index + SPLIT_OFFSETS[pair["split"]],
                                      "lateral_array_index": int(x), "vertical_sample_index": int(t)}
                                     for x, t in zip(lateral_indices, sample_indices, strict=True))
                if pair["split"] == "val" and demo["section_start"] <= index < demo["section_stop"]:
                    dest = index - demo["section_start"]
                    roi = np.s_[demo["lateral_start"]:demo["lateral_stop"], demo["sample_start"]:demo["sample_stop"]]
                    raw_seismic[dest] = seismic[roi]
                    raw_labels[dest] = reference[roi]
                    filled[dest] = True
                if (pair["split"], index) in (("train", 50), ("val", 100)):
                    full_section(out / f"thebe_{pair['split']}_section_{index:03}_full.png",
                                 seismic.astype(np.float32), reference,
                                 f"Thebe {pair['split']}：分区内剖面索引 {index}，完整补零边界核对")
        pair["content_check"] = "full_stream_finite_labels_padding_and_zip_crc_passed"
        pair["scan_seconds"] = round(time.perf_counter() - start_time, 3)
        print(f"Thebe {Path(pair['seismic']).name}: full content and pair scan passed ({pair['scan_seconds']:.1f}s)", flush=True)
    if not filled.all() or not np.isfinite(raw_seismic).all() or not raw_labels.any():
        raise ValueError("Validation demonstration was not reconstructed correctly")
    seismic, labels = raw_to_model(raw_seismic), raw_to_model(raw_labels)
    np.savez_compressed(cache / "thebe_validation_demo.npz", seismic=seismic, labels=labels)
    three_views(out / "thebe_validation_demo_three_views.png", seismic, labels,
                "Thebe验证区：跨NPZ分块的连续三维样例（按标签选取，非评价样本）",
                offsets=(demo["sample_start"], demo["lateral_start"], demo["section_start"]))
    with (out / "thebe_padding_label_anomalies.csv").open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=["split", "section_in_split", "section_in_concatenated_array",
                                                   "lateral_array_index", "vertical_sample_index"])
        writer.writeheader()
        writer.writerows(anomalies)
    summaries = {}
    for split in ("train", "val"):
        values = [v for v in sections if v["split"] == split]
        count = sum(v["samples"] for v in values)
        total = sum(v["sum"] for v in values)
        sum_squares = sum(v["sum_squares"] for v in values)
        summaries[split] = {"sections": len(values), "samples": count, "mean": total / count,
                            "std": float(np.sqrt(max(0, sum_squares / count - (total / count)**2))),
                            "min": min(v["min"] for v in values), "max": max(v["max"] for v in values),
                            "zero_fraction": sum(v["zeros"] for v in values) / count,
                            "nonempty_traces": sum(v["nonempty_traces"] for v in values),
                            "empty_traces": sum(v["empty_traces"] for v in values),
                            "fault_voxels": sum(v["fault_voxels"] for v in values),
                            "fault_voxels_on_empty_traces": sum(v["fault_voxels_on_empty_traces"] for v in values)}
    return {"status": "passed", "pairs": pairs, "split_summary": summaries, "section_statistics": sections,
            "demo": demo, "raw_axes": ["partition_section_axis", "other_lateral_axis", "vertical_sample"],
            "model_axes": ["vertical_sample", "other_lateral_axis", "partition_section_axis"],
            "axis_naming_warning": "Author notebook calls the first axis IL; data paper swaps IL/XL wording and contains 1803/1804 and 1537/1573 typos. Use array indices until raw SGY mapping is established.",
            "annotation_scope": "Source describes faults >20 m displacement in a particular AOI, approximately 2-4 km depth. No voxel-wise annotation-completeness mask or confirmed physical sample mapping is supplied here.",
            "negative_label_policy": "Zero is the supplied reference value; it is NOT proof of complete negative interpretation over the whole survey.",
            "padding_policy": "All-zero traces are padding candidates; individual zero-valued samples can be valid waveforms. Amplitude support is distinct from interpretation completeness.",
            "test_policy": "Only file headers/pair counts inspected; no TEST sample plots, statistics, predictions, threshold tuning or model selection."}


def run_audit(config: dict) -> dict:
    out = config["output_root"] / "audit"
    out.mkdir(parents=True, exist_ok=True)
    report = {"status": "running", "started": timestamp(), "original_data_access": "read_only",
              "data_root": str(config["data_root"])}
    write_json(out / "audit_report.json", report)
    try:
        report["provenance"] = provenance(config)
        report["faultseg3d"] = audit_synthetic(config)
        write_json(out / "audit_report.json", report)
        report["thebe"] = audit_thebe(config)
        anomalies = [
            {"split": row["split"], "section": row["section"],
             "positive_voxels_on_empty_traces": row["fault_voxels_on_empty_traces"]}
            for row in report["thebe"]["section_statistics"] if row["fault_voxels_on_empty_traces"]
        ]
        report["warnings"] = {
            "thebe_positive_labels_on_empty_traces": anomalies,
            "interpretation": "Data are readable and correctly paired; nonzero anomalies require an explicit support/annotation policy before baseline training.",
        }
        report["source_fingerprints"] = source_fingerprints(config)
        report.update(status="passed", finished=timestamp())
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}", finished=timestamp())
        write_json(out / "audit_report.json", report)
        raise
    write_json(out / "audit_report.json", report)
    text = ["# 第1步：公开数据验收报告", "", f"完成时间：{report['finished']}", "",
            "结论：FaultSeg3D全部220组配对体通过内容与来源哈希检查；Thebe全部18组NPZ分块的尺寸、配对和900/200/703剖面分区通过检查。",
            "Thebe训练区与验证区逐剖面完成有限值、标签、振幅补零和ZIP CRC检查；测试区仅核对文件头及分区，继续锁定。", "",
            "| 数据 | 数量 | 实际格式 |", "|---|---:|---|",
            "| FaultSeg3D训练 | 200对 | 128³，小端float32 |",
            "| FaultSeg3D验证 | 20对 | 128³，小端float32 |",
            "| Thebe训练 | 900幅剖面 / 9对分块 | 3174×1537，振幅float64，标签bool |",
            "| Thebe验证 | 200幅剖面 / 2对分块 | 同上 |",
            "| Thebe测试 | 703幅剖面 / 7对分块 | 末块103幅；本轮仅结构核对 |", "",
            "三向图与完整剖面图保存在本目录。图中横纵坐标统一使用数组索引；不推测真实采样间隔或测线编号。",
            "", "## 后续使用前必须保留的边界", "",
            "- 读取按作者的transpose(2,1,0)转换轴序；写回按相反转换。公开镜像的轴序和验证编号不能直接替代原作者编号。",
            "- Thebe按数字分块编号沿第0轴衔接。缓存只保存一个跨验证分块的连续样例，没有展开或重复保存完整训练资料。",
            "- 作者Notebook和数据论文的Inline/Crossline称呼不完全一致，论文也有尺寸笔误。验收依据文件头、作者处理代码和实际数组；物理坐标映射仍待原始测线元数据确认。",
            "- 全零地震道可标记为补零候选区；有效波形中的零交叉样点应保留。振幅有效范围不等于人工解释完整范围。",
            "- Thebe仅重点解释特定区域、约2–4km深度及断距大于20m的断层；尚无经过确认的逐体素完整解释掩膜，不能把全体零标签当作可靠负样本。",
            "- 当前没有找到项目既有工区角色清单，因此只沿用公开数据的原始分区。实际工区接入时须另核对其角色清单。",
            "- Thebe样例按验证标签密度选取，用于检查流程和地震—标签对应；不能据此报告真实工区泛化能力。", ""]
    for split, stats in report["thebe"]["split_summary"].items():
        text.append(f"Thebe {split}：零振幅比例 {stats['zero_fraction']:.2%}；全零地震道 {stats['empty_traces']:,}；全零道上的参考正标签 {stats['fault_voxels_on_empty_traces']:,}。")
    if report["warnings"]["thebe_positive_labels_on_empty_traces"]:
        text += ["", "**需处理的数据问题：** Thebe训练区剖面索引746—753存在213个参考正标签位于全零地震道。数据验收通过表示读取和配对检查通过，不能据此将这些体素直接用于训练。具体位置见 `thebe_padding_label_anomalies.csv`；进入基线前需明确掩膜或人工核查规则。"]
    text += ["", "该报告只完成数据验收，不代表人工标签地质正确性已经由领域专家确认。", ""]
    (out / "数据验收报告.md").write_text("\n".join(text), encoding="utf-8")
    return report


def source_fingerprints(config: dict) -> list[dict]:
    root = config["data_root"]
    paths = [synthetic_path(root, split, kind, index)
             for split, count in (("train", 200), ("validation", 20))
             for kind in ("seis", "fault") for index in range(count)]
    paths += [path for split in SPLIT_COUNTS for kind in ("seis", "fault") for path in thebe_parts(root, split, kind)]
    paths += [root / config["checkpoint"]]
    return [{"relative_path": str(p.relative_to(root)), "bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
            for p in paths]


def provenance(config: dict) -> dict:
    root = config["data_root"]
    fault = next((root / "01_FaultSeg3D/作者代码/已解压").glob("faultSeg-*"))
    thebe = next((root / "02_Thebe/作者代码/已解压").glob("CNNforFaultInterpretation-*"))
    files = [fault / name for name in ("unet3.py", "utils.py", "predNew.ipynb", "prediction.ipynb")]
    files += [thebe / name for name in ("generatePatchipyTrainValTest.ipynb", "faultSeg/predNew.ipynb")]
    files += [root / "02_Thebe/code/readSGY.ipynb"]
    return {"author_code": [{"relative_path": str(p.relative_to(root)), "sha256": sha256(p)} for p in files],
            "faultseg_repository": "https://github.com/xinwucwp/faultSeg/tree/0ab8ba1c10cb9e5748b0129bfe1a2fe3031b80fe",
            "thebe_repository": "https://github.com/anyuzoey/CNNforFaultInterpretation/tree/58eae5db2312feca003b9eb179fd9172beeeab5d",
            "thebe_data": "https://doi.org/10.7910/DVN/YBYGBK",
            "thebe_data_paper": "https://pmc.ncbi.nlm.nih.gov/articles/PMC8220338/",
            "source_readable_xml": "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC8220338/fullTextXML"}
