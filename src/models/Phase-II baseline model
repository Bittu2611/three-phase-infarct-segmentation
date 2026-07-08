# Phase-II Baseline

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D,
    Conv2DTranspose, concatenate,
    Dropout, BatchNormalization,
)
from tensorflow.keras.optimizers import Adam
from scipy.ndimage import gaussian_filter, map_coordinates
from tensorflow.keras.callbacks import Callback, ModelCheckpoint
from sklearn.model_selection import train_test_split

mps_gpus = tf.config.experimental.list_physical_devices("GPU")
if mps_gpus:
    try:
        for gpu in mps_gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
print("GPUs loaded:", tf.config.list_physical_devices("GPU"))

DATA_ROOT = "/Users/abhishekjha/Desktop/MRI training/MRI 05.16.2025/1818 images_trained model and data"
RAW_IMG_DIR = os.path.join(DATA_ROOT, "raw_images_1818")
RAW_MASK_DIR = os.path.join(DATA_ROOT, "Binary_masked_1818")

AUG_DIRS = {
    1: (os.path.join(DATA_ROOT, "augmented_1/raw_images"),
        os.path.join(DATA_ROOT, "augmented_1/Binary_masked")),
    2: (os.path.join(DATA_ROOT, "augmented_2/raw_images"),
        os.path.join(DATA_ROOT, "augmented_2/Binary_masked")),
    3: (os.path.join(DATA_ROOT, "augmented_3/raw_images"),
        os.path.join(DATA_ROOT, "augmented_3/Binary_masked")),
}

EXPERIMENTS_DIR = os.path.join(DATA_ROOT, "Experiments")
os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
MASTER_METRICS_CSV = os.path.join(EXPERIMENTS_DIR, "master_training_metrics.csv")

IMG_HEIGHT, IMG_WIDTH = 256, 256
BATCH_SIZE = 16
DATASET_SIZES = [447, 950, 1818]
AUG_LEVELS = [0, 1, 2, 3]
EPOCH_SETTINGS = [5, 8, 13]
TRAIN_VAL_SPLIT = 0.2
RANDOM_STATE = 42

def iou_metric(y_true, y_pred):
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    inter = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true + y_pred, axis=[1, 2, 3]) - inter
    return tf.reduce_mean((inter + 1e-10) / (union + 1e-10))

def dice_coef(y_true, y_pred, smooth=1e-6):
    yt = tf.reshape(y_true, [-1])
    yp = tf.reshape(y_pred, [-1])
    inter = tf.reduce_sum(yt * yp)
    return (2 * inter + smooth) / (tf.reduce_sum(yt) + tf.reduce_sum(yp) + smooth)

def expected_calibration_error(y_true, y_pred, num_bins=10):
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    bins = np.linspace(0, 1, num_bins + 1)
    ece = 0.0
    for i in range(num_bins):
        mask = (y_pred >= bins[i]) & (y_pred < bins[i + 1])
        if np.any(mask):
            acc = np.mean(y_true[mask])
            conf = np.mean(y_pred[mask])
            ece += abs(conf - acc) * np.sum(mask) / len(y_pred)
    return ece

class ECECallback(Callback):
    def __init__(self, validation_data, num_bins=10):
        super().__init__()
        self.validation_data = validation_data
        self.num_bins = num_bins
        self.val_ece = []

    def on_epoch_end(self, epoch, logs=None):
        Xv, yv = self.validation_data
        preds = self.model.predict(Xv, verbose=0)
        ece = expected_calibration_error(yv, preds, self.num_bins)
        logs["val_ece"] = ece
        self.val_ece.append(ece)
        print(f" - val_ece: {ece:.4f}")

def load_images(img_dir, mask_dir):
    exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
    ips = sorted(
        os.path.join(img_dir, f)
        for f in os.listdir(img_dir)
        if f.lower().endswith(exts)
    )
    mps = sorted(
        os.path.join(mask_dir, f)
        for f in os.listdir(mask_dir)
        if f.lower().endswith(exts)
    )
    X, Y = [], []
    for ip, mp in zip(ips, mps):
        im = cv2.imread(ip, cv2.IMREAD_GRAYSCALE)
        mk = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        if im is None or mk is None:
            print(f"Skipped: {ip}")
            continue
        im = cv2.resize(im, (IMG_WIDTH, IMG_HEIGHT)) / 255.0
        mk = cv2.resize(mk, (IMG_WIDTH, IMG_HEIGHT)) / 255.0
        im = np.repeat(im[..., None], 3, axis=-1)
        X.append(im)
        Y.append(mk[..., None])
    print(f"Loaded {len(X)} pairs from {img_dir}")
    return np.array(X), np.array(Y)

def subset_dataset(X, y, n_images, seed=RANDOM_STATE):
    if len(X) <= n_images:
        return X, y
    idx = np.arange(len(X))
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    idx = idx[:n_images]
    return X[idx], y[idx]

def _elastic_deform_gray(image, alpha, sigma):
    shape = image.shape
    rng = np.random.default_rng()
    dx = gaussian_filter(rng.uniform(-1, 1, shape), sigma) * alpha
    dy = gaussian_filter(rng.uniform(-1, 1, shape), sigma) * alpha
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = (y + dy).reshape(-1), (x + dx).reshape(-1)
    return map_coordinates(image, indices, order=1, mode="reflect").reshape(shape)

def _static_intensity_augment(gray):
    img = gray.astype(np.float32)
    if np.random.rand() < 0.5:
        img = np.clip(img + np.random.uniform(-0.15, 0.15), 0.0, 1.0)
    if np.random.rand() < 0.5:
        img = np.clip(np.power(np.clip(img, 1e-6, 1.0), np.random.uniform(0.8, 1.2)), 0.0, 1.0)
    if np.random.rand() < 0.5:
        img = gaussian_filter(img, sigma=float(np.random.uniform(0.3, 1.0)))
    if np.random.rand() < 0.5:
        img = np.clip(img + np.random.normal(0.0, np.random.uniform(0.01, 0.05), img.shape), 0.0, 1.0)
    if np.random.rand() < 0.5:
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        img = np.clip(cv2.filter2D(img, -1, kernel), 0.0, 1.0)
    if np.random.rand() < 0.5:
        img = np.clip(_elastic_deform_gray(img, np.random.uniform(8.0, 12.0), 3.0), 0.0, 1.0)
    return img

def _static_geometric_augment(img_gray, msk_gray):
    img = img_gray.astype(np.float32)
    msk = msk_gray.astype(np.float32)
    if np.random.rand() < 0.5:
        img = np.fliplr(img)
        msk = np.fliplr(msk)
    if np.random.rand() < 0.5:
        img = np.flipud(img)
        msk = np.flipud(msk)
    if np.random.rand() < 0.5:
        angle = float(np.random.uniform(-10.0, 10.0))
        h, w = img.shape[:2]
        matrix = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        img = cv2.warpAffine(
            img, matrix, (w, h),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101,
        )
        msk = cv2.warpAffine(
            msk, matrix, (w, h),
            flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
        )
    return img, msk

def static_random_augment_pair(img_gray, msk_gray):
    img, msk = _static_geometric_augment(img_gray, msk_gray)
    img = _static_intensity_augment(img)
    return img, msk

def save_augmented_data(images, masks, factor, img_dir, mask_dir):
    for i, (im, mk) in enumerate(zip(images, masks)):
        gray = np.clip(im[..., 0], 0.0, 1.0)
        mgray = np.clip(mk[..., 0] if mk.ndim == 3 else mk, 0.0, 1.0)
        for j in range(factor):
            ai, am = static_random_augment_pair(gray, mgray)
            cv2.imwrite(
                os.path.join(img_dir, f"aug_{i}_{j}.png"),
                (ai * 255).astype(np.uint8),
            )
            cv2.imwrite(
                os.path.join(mask_dir, f"aug_{i}_{j}.png"),
                (am * 255).astype(np.uint8),
            )
    print("saved aug", factor, img_dir)

def create_augmented_dataset(factor, img_dir, mask_dir, baseX, baseY):
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    if not os.listdir(img_dir):
        save_augmented_data(baseX, baseY, factor, img_dir, mask_dir)
    else:
        print("aug exists", factor)

def build_unet_model(input_size=(IMG_HEIGHT, IMG_WIDTH, 3)):
    def conv_block(x, f):
        x = Conv2D(f, 3, activation="relu", padding="same")(x)
        x = BatchNormalization()(x)
        x = Conv2D(f, 3, activation="relu", padding="same")(x)
        x = BatchNormalization()(x)
        return x

    inputs = Input(input_size)
    c1 = conv_block(inputs, 64)
    p1 = MaxPooling2D()(c1)
    c2 = conv_block(p1, 128)
    p2 = MaxPooling2D()(c2)
    c3 = conv_block(p2, 256)
    p3 = MaxPooling2D()(c3)
    c4 = conv_block(p3, 512)
    p4 = MaxPooling2D()(c4)
    c5 = conv_block(p4, 1024)
    d5 = Dropout(0.3)(c5)

    u6 = Conv2DTranspose(512, 2, strides=2, padding="same")(d5)
    u6 = concatenate([u6, c4])
    c6 = conv_block(u6, 512)
    d6 = Dropout(0.2)(c6)

    u7 = Conv2DTranspose(256, 2, strides=2, padding="same")(d6)
    u7 = concatenate([u7, c3])
    c7 = conv_block(u7, 256)
    d7 = Dropout(0.2)(c7)

    u8 = Conv2DTranspose(128, 2, strides=2, padding="same")(d7)
    u8 = concatenate([u8, c2])
    c8 = conv_block(u8, 128)
    d8 = Dropout(0.1)(c8)

    u9 = Conv2DTranspose(64, 2, strides=2, padding="same")(d8)
    u9 = concatenate([u9, c1])
    c9 = conv_block(u9, 64)

    outputs = Conv2D(1, 1, activation="sigmoid")(c9)
    model = Model(inputs, outputs)
    model.compile(
        optimizer=Adam(),
        loss="binary_crossentropy",
        metrics=["accuracy", iou_metric, dice_coef],
    )
    return model

def plot_and_save_history(history, save_path):
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    axs[0].plot(history.history["loss"], label="Train Loss")
    axs[0].plot(history.history["val_loss"], label="Val Loss")
    axs[0].set_title("Loss")
    axs[0].legend()

    axs[1].plot(history.history["dice_coef"], label="Train Dice")
    axs[1].plot(history.history["val_dice_coef"], label="Val Dice")
    axs[1].set_title("Dice Coefficient")
    axs[1].legend()

    axs[2].plot(history.history["iou_metric"], label="Train IoU")
    axs[2].plot(history.history["val_iou_metric"], label="Val IoU")
    axs[2].set_title("IoU")
    axs[2].legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    

def save_history_to_excel(history, experiment_id, save_folder):
    df_hist = pd.DataFrame(history.history)
    df_hist["experiment"] = experiment_id
    excel_path = os.path.join(save_folder, f"training_metrics_{experiment_id}.xlsx")
    df_hist.to_excel(excel_path, index=False)
    

def append_history_to_master_csv(history, experiment_id):
    df_hist = pd.DataFrame(history.history)
    df_hist["experiment"] = experiment_id
    if os.path.exists(MASTER_METRICS_CSV):
        df_existing = pd.read_csv(MASTER_METRICS_CSV)
        df_all = pd.concat([df_existing, df_hist], ignore_index=True)
    else:
        df_all = df_hist
    df_all.to_csv(MASTER_METRICS_CSV, index=False)
    

def train_experiment(X, y, experiment_id, epochs):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TRAIN_VAL_SPLIT, random_state=RANDOM_STATE
    )
    model = build_unet_model()
    print(experiment_id, epochs)

    ece_callback = ECECallback(validation_data=(X_val, y_val), num_bins=10)
    ckpt_dice = ModelCheckpoint(
        filepath=os.path.join(EXPERIMENTS_DIR, f"best_dice_{experiment_id}.keras"),
        monitor="val_dice_coef",
        mode="max",
        save_best_only=True,
        verbose=1,
    )
    ckpt_loss = ModelCheckpoint(
        filepath=os.path.join(EXPERIMENTS_DIR, f"best_loss_{experiment_id}.keras"),
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        verbose=1,
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        batch_size=BATCH_SIZE,
        epochs=epochs,
        callbacks=[ece_callback, ckpt_dice, ckpt_loss],
    )
    history.history["val_ece"] = ece_callback.val_ece

    plot_and_save_history(
        history, os.path.join(EXPERIMENTS_DIR, f"training_plot_{experiment_id}.png")
    )
    model.save(os.path.join(EXPERIMENTS_DIR, f"final_{experiment_id}.keras"))
    save_history_to_excel(history, experiment_id, EXPERIMENTS_DIR)
    append_history_to_master_csv(history, experiment_id)
    

if __name__ == "__main__":
    rawX, rawY = load_images(RAW_IMG_DIR, RAW_MASK_DIR)
    print("raw:", len(rawX))

    for factor in (1, 2, 3):
        img_dir, mask_dir = AUG_DIRS[factor]
        create_augmented_dataset(factor, img_dir, mask_dir, rawX, rawY)

    aug_datasets = {}
    for factor in (1, 2, 3):
        img_dir, mask_dir = AUG_DIRS[factor]
        aug_datasets[factor] = load_images(img_dir, mask_dir)

    total_configs = len(DATASET_SIZES) * len(AUG_LEVELS) * len(EPOCH_SETTINGS)
    print("configs:", total_configs)

    for n_images in DATASET_SIZES:
        for aug_level in AUG_LEVELS:
            if aug_level == 0:
                Xd, Yd = subset_dataset(rawX, rawY, n_images)
            else:
                X_aug, Y_aug = aug_datasets[aug_level]
                Xd, Yd = subset_dataset(X_aug, Y_aug, n_images)

            for epochs in EPOCH_SETTINGS:
                experiment_id = f"{n_images}_{epochs}_{aug_level}"
                print("run:", experiment_id)
                train_experiment(Xd, Yd, experiment_id, epochs)

    print("all experiments done")


