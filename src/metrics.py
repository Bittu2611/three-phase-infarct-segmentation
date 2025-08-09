import tensorflow as tf
import numpy as np

def iou_metric(y_true, y_pred):
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    inter  = tf.reduce_sum(y_true * y_pred, axis=[1,2,3])
    union  = tf.reduce_sum(y_true + y_pred, axis=[1,2,3]) - inter
    return tf.reduce_mean((inter + 1e-10)/(union + 1e-10))

def predictive_entropy_metric(y_true, y_pred):
    eps = 1e-7
    ent = -(y_pred*tf.math.log(y_pred+eps) + (1-y_pred)*tf.math.log(1-y_pred+eps))
    return tf.reduce_mean(ent)

def dice_coef(y_true, y_pred, smooth=1e-6):
    yt = tf.reshape(y_true,[-1]); yp = tf.reshape(y_pred,[-1])
    inter = tf.reduce_sum(yt*yp)
    return (2*inter+smooth)/(tf.reduce_sum(yt)+tf.reduce_sum(yp)+smooth)

def perplexity_metric(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return tf.reduce_mean(tf.exp(bce))

def expected_calibration_error(y_true, y_pred, num_bins=10):
    y_pred = y_pred.flatten(); y_true = y_true.flatten()
    bins = np.linspace(0,1,num_bins+1)
    ece = 0.0
    for i in range(num_bins):
        mask = (y_pred>=bins[i])&(y_pred<bins[i+1])
        if np.any(mask):
            acc = np.mean(y_true[mask]); conf = np.mean(y_pred[mask])
            ece += abs(conf-acc)*np.sum(mask)/len(y_pred)
    return float(ece)