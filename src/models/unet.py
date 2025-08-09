import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from ..metrics import iou_metric, predictive_entropy_metric, dice_coef, perplexity_metric

def _conv_block(x, f):
    x = Conv2D(f, 3, activation='relu', padding='same')(x); x = BatchNormalization()(x)
    x = Conv2D(f, 3, activation='relu', padding='same')(x); x = BatchNormalization()(x)
    return x

def build_unet(input_size=(256, 256, 1)):
    inputs = Input(input_size)
    c1 = _conv_block(inputs, 64);  p1 = MaxPooling2D()(c1)
    c2 = _conv_block(p1, 128);     p2 = MaxPooling2D()(c2)
    c3 = _conv_block(p2, 256);     p3 = MaxPooling2D()(c3)
    c4 = _conv_block(p3, 512);     p4 = MaxPooling2D()(c4)
    c5 = _conv_block(p4, 1024);    d5 = Dropout(0.3)(c5)

    u6 = Conv2DTranspose(512, 2, strides=2, padding='same')(d5)
    u6 = concatenate([u6, c4]); c6 = _conv_block(u6, 512); d6 = Dropout(0.2)(c6)
    u7 = Conv2DTranspose(256, 2, strides=2, padding='same')(d6)
    u7 = concatenate([u7, c3]); c7 = _conv_block(u7, 256); d7 = Dropout(0.2)(c7)
    u8 = Conv2DTranspose(128, 2, strides=2, padding='same')(d7)
    u8 = concatenate([u8, c2]); c8 = _conv_block(u8, 128); d8 = Dropout(0.1)(c8)
    u9 = Conv2DTranspose(64, 2, strides=2, padding='same')(d8)
    u9 = concatenate([u9, c1]); c9 = _conv_block(u9, 64)

    outputs = Conv2D(1, 1, activation='sigmoid')(c9)
    model = Model(inputs, outputs)
    model.compile(
        optimizer=Adam(),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            iou_metric, predictive_entropy_metric, dice_coef,
            tf.keras.metrics.KLDivergence(name='kldiv'),
            perplexity_metric
        ]
    )
    return model