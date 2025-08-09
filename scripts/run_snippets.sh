#!/usr/bin/env bash
set -e

python - <<'PY'
import yaml
from models.unet.snippets import build_unet_skeleton
from models.raunet.snippets import build_raunet_skeleton

u_cfg = yaml.safe_load(open("configs/unet_snippet.yaml"))
r_cfg = yaml.safe_load(open("configs/raunet_snippet.yaml"))

u = build_unet_skeleton(input_shape=(u_cfg["img_height"], u_cfg["img_width"], u_cfg["channels"]),
                        base_filters=u_cfg["base_filters"], depth=u_cfg["depth"])
r = build_raunet_skeleton(input_shape=(r_cfg["img_height"], r_cfg["img_width"], r_cfg["channels"]),
                          base_filters=r_cfg["base_filters"], depth=r_cfg["depth"])

print("U-Net skeleton summary:")
u.summary()
print("\nRA-UNet skeleton summary:")
r.summary()
print("\n[Note] Snippet-only demo. Full training/evaluation available upon request.")
PY