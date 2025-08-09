import os, cv2, numpy as np

def load_pairs(img_dir, mask_dir, size=(256, 256)):
    exts = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
    ips = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.lower().endswith(exts)])
    mps = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.lower().endswith(exts)])
    X, Y = [], []
    for ip, mp in zip(ips, mps):
        im = cv2.imread(ip, cv2.IMREAD_GRAYSCALE)
        mk = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        if im is None or mk is None:
            continue
        im = cv2.resize(im, size) / 255.0
        mk = cv2.resize(mk, size) / 255.0
        X.append(im[..., None]); Y.append(mk[..., None])
    return np.array(X), np.array(Y)

def dataset_from_name(name, paths, size=(256, 256)):
    if name == "raw":
        return load_pairs(paths["raw_img"], paths["raw_mask"], size)
    if name == "aug1":
        return load_pairs(paths["aug1_img"], paths["aug1_mask"], size)
    if name == "aug2":
        return load_pairs(paths["aug2_img"], paths["aug2_mask"], size)
    if name == "aug3":
        return load_pairs(paths["aug3_img"], paths["aug3_mask"], size)
    raise ValueError(f"Unknown dataset: {name}")