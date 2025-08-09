"""
U-Net (ResNet-50 encoder) – fine-tuning snippet
Full decoder wiring, loss/metrics, and callbacks are withheld.
Contact: abhishekjha2611@gmail.com
"""

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
# from ..metrics import dice_coef, iou_metric  # (intentionally not exposed here)

def _encoder_resnet50(input_size=(256, 256, 3)):
    base = ResNet50(weights="imagenet", include_top=False, input_tensor=Input(input_size))
    # Typical feature taps (names may vary by TF build); kept here for interface only.
    feats = {
        "c1": base.get_layer("conv1_relu").output,          # 128×128
        "c2": base.get_layer("conv2_block3_out").output,    # 64×64
        "----------------------------------------------,    # 32×32
        "----------------------------------------------,    # 16×16
        "--------------------------------------------    # 8×8
    }
    return base.input, feats

def build_resnet50_unet_head(input_size=(256, 256, 3)) -> Model:
    inp, enc = _encoder_resnet50(input_size)
    # DECODER REDACTED: upsampling blocks, skip concatenations, norm/dropout, 1×1 sigmoid head
    # Example:
    #   x = _up_block(enc["c5"], enc["c4"]); ...
    #   out = Conv2D(1, 1, activation="sigmoid")(x)
    raise NotImplementedError(
        "U-Net decoder/head are redacted in the public snippet. "
        "Full ResNet-50 U-Net is available on request."
    )

def prepare_for_finetune(model: Model, freeze_until: int = 140):
    # Freeze early layers for stable FT; index threshold is dataset-dependent.
    for i, layer in enumerate(model.layers):
        layer.trainable = (i >= freeze_until)

def compile_for_finetune(model: Model, lr=1e-4):
    # Loss/metrics/callbacks omitted by design; minimal compile shown.
    model.compile(optimizer=Adam(learning_rate=lr), loss="binary_crossentropy", metrics=["accuracy"])