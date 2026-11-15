"""Re-render fixed audit views after server font discovery; no new selections."""
from .baseline import require_audit
from .common import load_config, WORKSPACE, raw_to_model, relative_path, timestamp, write_json
from .data import plane_stream, read_synthetic
from .figures import full_section, three_views, plt
import numpy as np


def main():
    config = load_config(WORKSPACE / "configs/baseline_balanced.json")
    audit = require_audit(config)
    out = config["output_root"] / "audit"
    files = []
    for split, index in (("train", 0), ("validation", 0), ("validation", 1)):
        filename = f"faultseg_{split}_{index:03}_three_views.png"
        three_views(out / filename, raw_to_model(read_synthetic(config["data_root"], split, "seis", index)),
                    raw_to_model(read_synthetic(config["data_root"], split, "fault", index)),
                    f"FaultSeg3D {split} / {index}：地震—标签三向核对")
        files.append(filename)
    demo = audit["thebe"]["demo"]
    with np.load(config["cache_root"] / "thebe_validation_demo.npz") as cache:
        three_views(out / "thebe_validation_demo_three_views.png", cache["seismic"], cache["labels"],
                    "Thebe验证区：跨NPZ分块的连续三维样例（按标签选取，非评价样本）",
                    offsets=(demo["sample_start"], demo["lateral_start"], demo["section_start"]))
    files.append("thebe_validation_demo_three_views.png")
    for split, target in (("train", 50), ("val", 100)):
        pair = next(p for p in audit["thebe"]["pairs"] if p["split"] == split and p["section_start"] <= target < p["section_stop"])
        with plane_stream(config["data_root"] / relative_path(pair["seismic"])) as seismics, \
                plane_stream(config["data_root"] / relative_path(pair["labels"])) as references:
            for local, (seismic, labels) in enumerate(zip(seismics, references, strict=True)):
                if pair["section_start"] + local == target:
                    filename = f"thebe_{split}_section_{target:03}_full.png"
                    full_section(out / filename, seismic.astype(np.float32), labels,
                                 f"Thebe {split}：分区内剖面索引 {target}，完整补零边界核对")
                    files.append(filename)
    write_json(out / "font_render_verification.json", {"status": "passed", "finished": timestamp(),
               "font_family": plt.rcParams["font.sans-serif"][0], "fixed_views_regenerated": files,
               "source_coordinates_changed": False, "predictions_changed": False})


if __name__ == "__main__":
    main()
