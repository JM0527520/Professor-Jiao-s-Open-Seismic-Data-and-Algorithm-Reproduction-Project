#!/usr/bin/env bash
# One explicit stage per invocation. Never silently runs the complete workload.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
if [[ -x .venv-server/bin/python ]]; then
    task_python=.venv-server/bin/python
else
    task_python=python
fi
stage="${1:-help}"
if [[ $# -gt 0 ]]; then shift; fi
mkdir -p outputs/logs
case "$stage" in
    audit)
        "$task_python" -u -m fault_workflow audit --config configs/baseline.json "$@" 2>&1 | tee outputs/logs/server-audit.log
        ;;
    prepare|smoke|preflight|gpu-smoke|train|evaluate-synthetic|evaluate-thebe)
        "$task_python" -u -m fault_workflow.prepare_cli "$stage" "$@" 2>&1 | tee "outputs/logs/server-${stage}-$(date +%Y%m%d-%H%M%S).log"
        ;;
    *)
        echo 'Stages: audit, prepare, smoke, preflight, gpu-smoke, train, evaluate-synthetic, evaluate-thebe'
        echo 'Full train/evaluate require --server-work and a TensorFlow GPU; see docs/服务器运行手册.md.'
        exit 2
        ;;
esac
