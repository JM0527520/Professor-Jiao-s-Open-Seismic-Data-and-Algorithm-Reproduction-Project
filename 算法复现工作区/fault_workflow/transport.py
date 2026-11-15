"""Opaque transfer checksums; no TEST samples are decoded or evaluated."""
import hashlib
import json

from .common import timestamp, write_json


def verify_thebe_transport(config):
    root = config["data_root"]
    metadata = json.loads((root / "00_下载记录/Thebe_Dataverse.json").read_text(encoding="utf-8"))
    expected = {v["dataFile"]["filename"]: v["dataFile"] for v in metadata["data"]["latestVersion"]["files"]
                if v["dataFile"]["filename"].endswith(".npz")}
    if len(expected) != 36:
        raise ValueError("Expected the registered 36 original Thebe NPZ files")
    rows = []
    for index, (name, entry) in enumerate(expected.items(), 1):
        kind = "seis" if name.startswith("seis") else "fault"
        path = root / "02_Thebe/data/npzfiles" / kind / name
        if entry["checksum"]["type"] != "MD5" or path.stat().st_size != entry["filesize"]:
            raise ValueError(f"Thebe transfer size or registered checksum type changed: {name}")
        digest = hashlib.md5()  # Registered provider integrity checksum, not a security primitive.
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024**2), b""):
                digest.update(block)
        if digest.hexdigest() != entry["checksum"]["value"]:
            raise ValueError(f"Thebe transfer provider checksum mismatch: {name}")
        rows.append({"relative_path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size,
                     "provider_md5": digest.hexdigest()})
        print(f"Opaque provider checksum {index}/36: {name}", flush=True)
    report = {"status": "passed", "finished": timestamp(), "files": rows,
              "test_samples_decoded": False, "test_used_for_evaluation": False,
              "purpose": "byte-stream integrity against registered provider version only"}
    write_json(config["output_root"] / "audit/thebe_transport_checksums.json", report)
    return report
