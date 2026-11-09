"""Run stages independently: python -m fault_workflow audit / predict / all."""
import argparse
from pathlib import Path

from .common import load_config


def main():
    parser = argparse.ArgumentParser(description="公开数据验收与作者FaultSeg3D小规模预测；不训练模型")
    parser.add_argument("stage", choices=["audit", "predict", "all"])
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.stage in ("audit", "all"):
        from .audit import run_audit
        run_audit(config)
    if args.stage in ("predict", "all"):
        from .inference import run_predictions
        run_predictions(config)


if __name__ == "__main__":
    main()
