"""
RA-UNet (ResNet-50 encoder) – fine-tuning snippet
Residual attention blocks, gating, decoder wiring, and full training stack withheld.
Contact: abhishek-jha@uiowa.edu
"""

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def _encoder_resnet50(input_size=(256, 256, 3)):
    base = ResNet50(weights="imagenet", include_top=False, input_tensor=Input(input_size))
    feats = {
        "c1": base.get_layer("conv1_relu").output,
        "c2": base.get_layer("conv2_block3_out").output,
        "............................................,
        "---------------------------------------------,
        "---------------------------------------------,
    }
    return base.input, feats

def build_resnet50_raunet_head(input_size=(256, 256, 3)) -> Model:
    inp, enc = _encoder_resnet50(input_size)
    # REDACTED:
    #   - residual attention gates on skip connections
    #   - upsampling path with RA blocks
    #   - normalization/dropout strategy and final 1×1 sigmoid
    raise NotImplementedError(
        "RA-UNet attention/decoder are redacted in the public snippet. "
        "Full ResNet-50 RA-UNet is available on request."
    )

def prepare_for_finetune(model: Model, freeze_until: int = 140):
    for i, layer in enumerate(model.layers):
        layer.trainable = (i >= freeze_until)

def compile_for_finetune(model: Model, lr=1e-4):
    model.compile(optimizer=Adam(learning_rate=lr), loss="binary_crossentropy", metrics=["accuracy"])
