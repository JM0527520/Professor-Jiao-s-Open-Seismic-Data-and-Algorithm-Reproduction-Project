#!/usr/bin/env bash
# Run inside the algorithm workspace on Linux. Does not launch training.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
if [[ "$(uname -s)" != "Linux" ]]; then
    echo 'Use the existing local environment on Windows; this installer is for the future Linux server.' >&2
    exit 2
fi
mkdir -p outputs/deployment outputs/logs
if [[ ! -x .venv-server/bin/python ]]; then
    python3.12 -m venv --without-pip .venv-server
fi
if ! .venv-server/bin/python -m pip --version >/dev/null 2>&1; then
    if ! .venv-server/bin/python -m ensurepip --upgrade; then
        curl --fail --location --retry 2 https://bootstrap.pypa.io/get-pip.py -o outputs/deployment/get-pip.py
        sha256sum outputs/deployment/get-pip.py > outputs/deployment/get-pip.sha256
        .venv-server/bin/python outputs/deployment/get-pip.py > outputs/logs/pip-bootstrap.log 2>&1
    fi
fi
.venv-server/bin/python -m pip install -r server/requirements-gpu.txt
mkdir -p outputs/preparation
.venv-server/bin/python -m pip freeze > outputs/preparation/server-requirements-lock.txt
.venv-server/bin/python -m pip check
echo 'Environment installed. Repeat audit and preparation in this data location, then run GPU preflight. Training has not started.'
