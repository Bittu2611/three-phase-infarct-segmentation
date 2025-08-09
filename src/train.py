import os, yaml, pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
from .data import dataset_from_name
from .models.unet import build_unet
from .callbacks import ECECallback
from .utils.plotting import save_history_plot

def run_from_config(cfg_path):
    cfg = yaml.safe_load(open(cfg_path))
    H, W = cfg['img_height'], cfg['img_width']
    bs, epochs = cfg['batch_size'], cfg['epochs']
    paths = cfg['paths']; out = cfg['out_dir']
    os.makedirs(out, exist_ok=True)

    master_csv = os.path.join(out, "master_training_metrics.csv")
    for name in cfg['datasets']:
        X, Y = dataset_from_name(name, paths, (W, H))
        Xt, Xv, yt, yv = train_test_split(X, Y, test_size=cfg['val_split'], random_state=cfg['seed'])
        model = build_unet((H, W, 1))
        expid = f"{name}_{epochs}e"

        ckpt = ModelCheckpoint(os.path.join(out, f"best_{expid}.keras"),
                               monitor='val_loss', save_best_only=True, verbose=1)
        ece  = ECECallback((Xv, yv), num_bins=10)

        hist = model.fit(Xt, yt, validation_data=(Xv, yv),
                         batch_size=bs, epochs=epochs, callbacks=[ece, ckpt])

        hist.history['val_ece'] = ece.val_ece
        model.save(os.path.join(out, f"model_{expid}.keras"))
        save_history_plot(hist, os.path.join(out, f"plot_{expid}.png"))

        df = pd.DataFrame(hist.history); df['experiment'] = expid
        if os.path.exists(master_csv):
            pd.concat([pd.read_csv(master_csv), df], ignore_index=True).to_csv(master_csv, index=False)
        else:
            df.to_csv(master_csv, index=False)

if __name__ == "__main__":
    # default run: use the provided config path
    run_from_config("configs/raw_13e.yaml")