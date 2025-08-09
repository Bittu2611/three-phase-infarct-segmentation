## A Three-Phase Deep Learning Framework for Automated Infarct Segmentation in Preclinical Stroke Imaging

Public Snippets Only. This repository intentionally publishes selected fragments (model skeletons and minimal training interfaces) for U-Net, Deep U-Net, RA-UNet, and ResNet-50–based fine-tuning  to support peer review and reproducibility claims.  

### Models Covered (snippet-only)
- Deep U-Net
- U-Net – deeper encoder/decoder scaffold (dropouts, BN, skip wiring withheld)
- RA-UNet– residual attention gate interfaces (attention/gating internals withheld)
- U-Net (ResNet-50 encoder) – fine-tuning interface (decoder & training loop withheld)
- RA-UNet (ResNet-50 encoder) – fine-tuning interface (attention & training loop withheld)


## Baseline Training (Snippet-Only)-Phae-I
- U-Net snippets: `models/unet/snippets.py`  
  Runner: `src/train_unet_snippet.py`  
  Config: `configs/unet_snippet.yaml`

- RA-UNet snippets: `models/raunet/snippets.py`  
  Runner: `src/train_raunet_snippet.py`  
  Config: `configs/raunet_snippet.yaml`

## Fine-tuning (Snippet-Only)
This repository includes **snippet-only** fine-tuning interfaces:

- U-Net (ResNet-50 encoder): `models/unet/finetune_resnet50_snippet.py`  
  Runner: `src/finetune_unet_resnet50_snippet.py`  
  Config: `configs/unet_resnet50_finetune.yaml`

- RA-UNet (ResNet-50 encoder): `models/raunet/finetune_resnet50_snippet.py`  
  Runner: `src/finetune_raunet_resnet50_snippet.py`  
  Config: `configs/raunet_resnet50_finetune.yaml`

# Full Factorial design of experiment based model development (Phase -II)
-Deep U-Net (5-level / 9-layer, Full-Factorial DOE)
- Model file: `models/unet/deep_unet_snippet.py`  
- Runner: `src/train_deep_unet_snippet.py`  
- Config:`configs/deep_unet_snippet.yaml`  

Note: Full implementations (data pipeline, complete training loops, augmentation policy, metrics, callbacks incl. ECE, and evaluation) are withheld and available upon request under the Custom Research License.

Contact: **abhishekjha2611@gmail.com**
