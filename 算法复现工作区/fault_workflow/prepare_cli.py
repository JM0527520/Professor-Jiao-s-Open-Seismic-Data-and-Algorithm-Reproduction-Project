"""Preparation CLI keeps deferred server work separate from local checks."""
import argparse
from pathlib import Path

from .common import load_config


def main():
    parser = argparse.ArgumentParser(description="本机准备与服务器基线入口；TEST保持锁定")
    parser.add_argument("stage", choices=("prepare", "smoke", "preflight", "gpu-smoke", "train", "evaluate-synthetic", "evaluate-thebe"))
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "configs/baseline.json")
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu", help="preflight设备；smoke固定CPU，完整训练与评价固定GPU")
    parser.add_argument("--model", choices=("author", "trained"), default="author")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--server-work", action="store_true", help="在已准备的GPU服务器上显式执行暂存的大任务")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.stage == "prepare":
        from .preparation import run_preparation
        run_preparation(config)
    elif args.stage == "smoke":
        from .training import run_smoke
        run_smoke(config)
    elif args.stage == "preflight":
        from .baseline import record_preflight
        record_preflight(config, args.device)
    elif args.stage == "train":
        from .training import run_training
        run_training(config, args.resume, args.server_work)
    elif args.stage == "gpu-smoke":
        from .training import run_gpu_smoke
        run_gpu_smoke(config, args.server_work)
    else:
        from .evaluation import run_evaluation
        run_evaluation(config, args.stage.removeprefix("evaluate-"), args.model, args.server_work)


if __name__ == "__main__":
    main()
