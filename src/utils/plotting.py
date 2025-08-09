import matplotlib.pyplot as plt
import pandas as pd

def save_history_plot(history, save_path):
    h = pd.DataFrame(history.history)
    cols = [c for c in h.columns if any(k in c for k in [
        'accuracy','loss','auc','precision','recall',
        'iou_metric','predictive_entropy_metric','dice_coef',
        'kldiv','perplexity_metric','val_ece'
    ])]
    fig, axs = plt.subplots(3, 4, figsize=(20, 15))
    axs = axs.ravel()
    for i, col in enumerate(cols[:12]):
        axs[i].plot(h[col], label=col)
        axs[i].set_title(col); axs[i].legend()
    for j in range(len(cols), 12):
        axs[j].axis('off')
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)