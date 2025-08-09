"""
Public snippet runner (U-Net + ResNet-50)
Data pipeline, callbacks (e.g., ECE), and logging are intentionally withheld.
"""

import yaml
from pathlib import Path
from models.unet.finetune_resnet50_snippet import (
    build_resnet50_unet_head, prepare_for_finetune, compile_for_finetune
)

def run(cfg_path="configs/unet_resnet50_finetune.yaml"):
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    print("[Snippet] U-Net ResNet-50 FT config:", {k: cfg.get(k) for k in ("input_size","freeze_until","epochs","lr")})
    try:
        model = build_resnet50_unet_head(tuple(cfg["input_size"]))
        prepare_for_finetune(model, freeze_until=cfg.get("freeze_until", 140))
        compile_for_finetune(model, lr=cfg.get("lr", 1e-4))
        print("[Snippet] Model ready for fine-tuning (training loop redacted).")
    except NotImplementedError as e:
        print(e)

if __name__ == "__main__":
    run()