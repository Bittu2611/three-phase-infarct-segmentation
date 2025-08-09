from tensorflow.keras.callbacks import Callback
from .metrics import expected_calibration_error

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
        if logs is not None:
            logs['val_ece'] = ece
        self.val_ece.append(ece)
        print(f" - val_ece: {ece:.4f}")