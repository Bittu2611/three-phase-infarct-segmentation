"""
Public interface for RA-UNet (snippet-only).
Full training loop, optimizer, callbacks, and data pipeline are withheld by design.
"""
import yaml
from models.raunet.snippets import build_raunet_skeleton

def get_raunet_model_from_config(cfg: dict):
    H = cfg.get("img_height", 256)
    W = cfg.get("img_width", 256)
    C = cfg.get("channels", 1)
    base_filters = cfg.get("base_filters", 32)
    depth = cfg.get("depth", 4)
    return build_raunet_skeleton((H, W, C), base_filters=base_filters, depth=depth)

if __name__ == "__main__":
    cfg = yaml.safe_load(open("configs/raunet_snippet.yaml"))
    model = get_raunet_model_from_config(cfg)
    model.summary()
    print("\n[Note] Snippet-only model skeleton; full training available upon request.")