FROM python:3.12-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /project/workspace
COPY requirements.txt requirements-dev.txt ./
COPY server/requirements-gpu.txt ./server/requirements-gpu.txt
RUN python -m pip install --no-cache-dir -r server/requirements-gpu.txt
ENV PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=2
CMD ["python", "-m", "fault_workflow.prepare_cli", "--help"]
