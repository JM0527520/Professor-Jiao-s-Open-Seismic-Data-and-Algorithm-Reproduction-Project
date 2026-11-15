#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
task_python=.venv-server/bin/python
task_config=configs/baseline_balanced.json
"$task_python" -u - <<'PY'
import time
from pathlib import Path
marker = Path('outputs/deployment/payload_complete.json')
print('Waiting for the complete registered dataset transfer marker', flush=True)
while not marker.exists():
    time.sleep(20)
print('Transfer marker found; verifying provider checksums', flush=True)
from fault_workflow.common import load_config, WORKSPACE
from fault_workflow.transport import verify_thebe_transport
verify_thebe_transport(load_config(WORKSPACE / 'configs/baseline_balanced.json'))
PY
"$task_python" -u -m fault_workflow audit --config "$task_config"
"$task_python" -u -m fault_workflow.prepare_cli prepare --config "$task_config"
"$task_python" -u -m fault_workflow.prepare_cli evaluate-thebe --model author --server-work --gpu-memory-limit-mb 1152 --config "$task_config"
"$task_python" -u - <<'PY'
import json, os, time
from pathlib import Path
path = Path('outputs/server_balanced_v2/training/faultseg_balanced_v2/training_report.json')
pid_path = Path('outputs/logs/source-pipeline.pid')
print('Author Thebe evaluation completed; waiting for source training', flush=True)
while True:
    report = json.loads(path.read_text())
    if report['status'] == 'passed':
        break
    if report['status'] == 'failed':
        raise RuntimeError('Source training failed; preserve records and diagnose before continuing')
    os.kill(int(pid_path.read_text().strip()), 0)
    time.sleep(30)
PY
"$task_python" -u -m fault_workflow.prepare_cli evaluate-thebe --model trained --server-work --gpu-memory-limit-mb 1152 --config "$task_config"
"$task_python" -u -m fault_workflow.prepare_cli evaluate-thebe --model trained --fusion separable --server-work --gpu-memory-limit-mb 1152 --config "$task_config"
"$task_python" -u -m fault_workflow.run_review --config "$task_config"
