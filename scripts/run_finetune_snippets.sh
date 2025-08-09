#!/usr/bin/env bash
set -e

python - <<'PY'
print("ResNet-50 fine-tuning snippets (U-Net & RA-UNet) — interface check only")
from src.finetune_unet_resnet50_snippet import run as run_unet
from src.finetune_raunet_resnet50_snippet import run as run_raunet
run_unet("configs/unet_resnet50_finetune.yaml")
run_raunet("configs/raunet_resnet50_finetune.yaml")
print("\n[Note] Full fine-tuning code and training loop are available upon request.")
PY