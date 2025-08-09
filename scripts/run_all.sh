#!/usr/bin/env bash
set -e

python - <<'PY'
from src.train import run_from_config
run_from_config("configs/raw_13e.yaml")
PY