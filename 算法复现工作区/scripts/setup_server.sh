#!/usr/bin/env bash
# Run inside the algorithm workspace on Linux. Does not launch training.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
if [[ "$(uname -s)" != "Linux" ]]; then
    echo 'Use the existing local environment on Windows; this installer is for the future Linux server.' >&2
    exit 2
fi
python3.12 -m venv .venv-server
.venv-server/bin/python -m pip install -r server/requirements-gpu.txt
mkdir -p outputs/preparation
.venv-server/bin/python -m pip freeze > outputs/preparation/server-requirements-lock.txt
.venv-server/bin/python -m pip check
echo 'Environment installed. Repeat audit and preparation in this data location, then run GPU preflight. Training has not started.'
