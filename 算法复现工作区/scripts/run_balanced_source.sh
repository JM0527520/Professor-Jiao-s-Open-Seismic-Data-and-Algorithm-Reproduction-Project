#!/usr/bin/env bash
# Source-only acceptance is explicit; this does not certify any Thebe data.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
task_python=.venv-server/bin/python
task_config=configs/baseline_balanced.json
if [[ -e outputs/server_balanced_v2/training/faultseg_balanced_v2/protocol.json || -e outputs/server_balanced_v2/baseline/faultseg_balanced_v2/author/synthetic/evaluation_report.json ]]; then
    echo 'Existing run detected. Preserve it and resume a completed checkpoint explicitly; do not re-run the initialization pipeline.' >&2
    exit 2
fi
mkdir -p outputs/server_balanced_v2/logs
"$task_python" -u -m fault_workflow.prepare_cli prepare-source --config "$task_config"
"$task_python" -u -m fault_workflow.prepare_cli preflight --device gpu --config "$task_config"
"$task_python" -u -m fault_workflow.prepare_cli smoke --config "$task_config"
"$task_python" -u -m fault_workflow.prepare_cli gpu-smoke --server-work --config "$task_config"
"$task_python" -u -m fault_workflow.prepare_cli evaluate-synthetic --model author --server-work --config "$task_config"
"$task_python" -u -m fault_workflow.prepare_cli train --server-work --config "$task_config"
"$task_python" -u -m fault_workflow.prepare_cli evaluate-synthetic --model trained --server-work --config "$task_config"
