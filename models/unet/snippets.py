"""
U-Net snippets (selected fragments only).
Full training and data pipeline withheld; available upon request.
"""

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, Concatenate, BatchNormalization, Dropout, Input
from tensorflow.keras import Model

def conv_block(x, f: int):
    x = Conv2D(f, 3, padding="same", activation="relu")(x); x = BatchNormalization()(x)
    x = Conv2D(f, 3, padding="same", activation="relu")(x); x = BatchNormalization()(x)
    return x

def down_block(x, f: int):
    c = conv_block(x, f)
    p = MaxPooling2D()(c)
    return c, p

def up_block(x, skip, f: int, drop: float | None = None):
    x = Conv2DTranspose(f, 2, strides=2, padding="same")(x)
    x = Concatenate()([x, skip])
    x = conv_block(x, f)
    if drop:
        x = Dropout(drop)(x)
    return x

def build_unet_skeleton(input_shape=(256, 256, 1), base_filters=64, depth=5, drops=(0.3, 0.2, 0.2, 0.1)):
    """Skeleton only: returns an uncompiled U-Net-like Keras Model."""
    inputs = Input(input_shape)
    skips, x = [], inputs
    f = base_filters
    for _ in range(depth - 1):
        c, x = down_block(x, f)
        skips.append(c); f *= 2
    x = conv_block(x, f)  # bottleneck
    for i, s in enumerate(reversed(skips)):
        f //= 2
        drop = drops[i] if i < len(drops) else None
        x = up_block(x, s, f, drop)
    outputs = Conv2D(1, 1, activation="sigmoid")(x)
    return Model(inputs, outputs, name="U-Net_skeleton")