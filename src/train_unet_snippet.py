"""
Public interface for U-Net (snippet-only).
Full training loop, optimizer, callbacks, and data pipeline are withheld by design.
"""
import yaml
from models.unet.snippets import build_unet_skeleton

def get_unet_model_from_config(cfg: dict):
    H = cfg.get("img_height", 256)
    W = cfg.get("img_width", 256)
    C = cfg.get("channels", 1)
    base_filters = cfg.get("base_filters", 64)
    depth = cfg.get("depth", 5)
    return build_unet_skeleton((H, W, C), base_filters=base_filters, depth=depth)

if __name__ == "__main__":
    cfg = yaml.safe_load(open("configs/unet_snippet.yaml"))
    model = get_unet_model_from_config(cfg)
    model.summary()
    print("\n[Note] Snippet-only model skeleton; full training available upon request.")