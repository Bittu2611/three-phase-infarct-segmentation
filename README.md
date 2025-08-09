# A Three-Phase Deep Learning Framework for Automated Infarct Segmentation in Preclinical Stroke Imaging

This repository contains modular code snippets for the training and validation of multi-model U-Net-based infarct segmentation pipelines in preclinical stroke MRI, guided by a Design of Experiments (DOE) approach.

## Features
- 5-level U-Net architecture
- Calibration-aware metrics (IoU, Dice, entropy, KL divergence, perplexity)
- Expected Calibration Error (ECE) callback
- Deterministic loading for raw and augmented datasets (1×, 2×, 3×)
- Per-experiment plots and master CSV logging

## Installation
```bash
conda env create -f environment.yml
conda activate infarct-seg